"""Honesty gate: passive maker fills track queue position + aggressors.

Pins slice #17 Task 14 / SPEC §9.5 venue-fidelity sub-rule.

A passive (non-marketable) LIMIT order sits in the book queue. It
fills only when (a) an aggressor on the opposite side trades AT or
through the maker's limit price, AND (b) the maker's queue position
has been consumed by prior aggressors.

`test_passive_maker_realism` drives a maker SELL at a price that the
real BTCUSDT 30-min trade fixture actually crosses. Assert at least
one fill lands, with realistic partial-fill semantics (max single
fill ≤ aggressor qty, total fill ≤ order qty).

`test_sanity_maker_no_fill` (Task 15) pins the dual: a maker SELL
at +1% above the fixture's max price NEVER fills.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

from backtest_simulator.honesty.maker_fill import ImmediateFill, MakerFillModel
from backtest_simulator.venue.types import PendingOrder

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / 'fixtures' / 'market' / 'btcusdt_trades_30min.parquet'
)


def _load_trades() -> pl.DataFrame:
    assert _FIXTURE.is_file(), f'fixture missing at {_FIXTURE}'
    return pl.read_parquet(_FIXTURE)


def _maker(
    *,
    side: str,
    limit_price: Decimal,
    qty: Decimal,
    submit_time: datetime,
) -> PendingOrder:
    return PendingOrder(
        order_id='maker-fixture',
        side=side,
        order_type='LIMIT',
        qty=qty,
        limit_price=limit_price,
        stop_price=None,
        time_in_force='GTC',
        submit_time=submit_time,
        symbol='BTCUSDT',
    )


def test_passive_maker_realism() -> None:
    """A maker at a touched price level partial-fills against aggressors.

    Setup: take the real fixture's first hour, find the median
    price, and place a maker SELL at that price. Window: the
    remainder of the fixture. A maker SELL fills against BUY
    aggressors — `is_buyer_maker == 0` rows that price at or above
    the limit. Initial queue is set to zero (best-case maker) so
    fills are not blocked by queue.
    """
    trades = _load_trades()
    median_price = Decimal(str(trades['price'].median()))
    submit_time = trades['datetime'][0]
    cutoff = submit_time + timedelta(minutes=5)
    window = trades.filter(pl.col('datetime') >= cutoff)

    model = MakerFillModel.calibrate(trades=trades, lookback_minutes=10)
    order = _maker(
        side='SELL', limit_price=median_price,
        qty=Decimal('0.5'), submit_time=submit_time,
    )
    fills = model.evaluate(order=order, trades_in_window=window)

    assert isinstance(fills, list), (
        f'evaluate must return a list, got {type(fills)}'
    )
    assert fills, (
        'no fills produced; the maker SELL at median price should '
        'have caught at least one BUY aggressor crossing.'
    )
    for fill in fills:
        assert isinstance(fill, ImmediateFill)
        assert fill.fill_qty > Decimal('0'), (
            f'zero-qty fill at {fill.fill_time}; not a real fill'
        )
        assert fill.fill_price == median_price, (
            f'maker fills at limit price; got {fill.fill_price} != '
            f'limit {median_price}'
        )
    total_filled = sum((f.fill_qty for f in fills), Decimal('0'))
    assert total_filled <= order.qty, (
        f'over-fill: total {total_filled} > order qty {order.qty}'
    )


def test_sanity_maker_no_fill() -> None:
    """Far-away maker never fills, regardless of window length.

    Slice #17 Task 15. A maker SELL at +1% above the fixture's max
    price has no aggressor that ever crosses it. The model must
    return an empty fill list.
    """
    trades = _load_trades()
    max_price = Decimal(str(trades['price'].max()))
    far_limit = max_price * Decimal('1.01')  # +1% above any trade
    submit_time = trades['datetime'][0]

    model = MakerFillModel.calibrate(trades=trades, lookback_minutes=10)
    order = _maker(
        side='SELL', limit_price=far_limit,
        qty=Decimal('1.0'), submit_time=submit_time,
    )
    fills = model.evaluate(order=order, trades_in_window=trades)
    assert fills == [], (
        f'far-away maker SELL at {far_limit} (max trade price '
        f'{max_price}) must NEVER fill. Got {len(fills)} fills.'
    )


def test_maker_fill_rejects_market_order() -> None:
    """`evaluate` raises on a non-LIMIT order — passive logic only.

    Pins the contract that MakerFillModel only handles passive
    (limit_price set) orders. Routing a MARKET order through it
    is a programmer mistake the gate must catch loudly.
    """
    submit_time = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    market_order = PendingOrder(
        order_id='market-fixture',
        side='BUY',
        order_type='MARKET',
        qty=Decimal('1'),
        limit_price=None,
        stop_price=None,
        time_in_force='IOC',
        submit_time=submit_time,
        symbol='BTCUSDT',
    )
    model = MakerFillModel(_lookback_minutes=10)
    empty_window = pl.DataFrame({
        'datetime': pl.Series('datetime', [], dtype=pl.Datetime('us', 'UTC')),
        'price': pl.Series('price', [], dtype=pl.Float64),
        'quantity': pl.Series('quantity', [], dtype=pl.Float64),
        'is_buyer_maker': pl.Series('is_buyer_maker', [], dtype=pl.UInt8),
    })
    with pytest.raises(ValueError, match='no limit_price'):
        model.evaluate(order=market_order, trades_in_window=empty_window)


def test_maker_fill_calibrate_rejects_zero_lookback() -> None:
    """`lookback_minutes <= 0` is a misconfiguration; raise loud."""
    trades = _load_trades()
    with pytest.raises(ValueError, match='lookback_minutes must be positive'):
        MakerFillModel.calibrate(trades=trades, lookback_minutes=0)


def test_maker_fill_calibrate_rejects_empty_tape() -> None:
    """Empty trade DataFrame at calibrate time raises loud."""
    empty = pl.DataFrame({
        'datetime': pl.Series('datetime', [], dtype=pl.Datetime('us', 'UTC')),
        'price': pl.Series('price', [], dtype=pl.Float64),
        'quantity': pl.Series('quantity', [], dtype=pl.Float64),
        'is_buyer_maker': pl.Series('is_buyer_maker', [], dtype=pl.UInt8),
    })
    with pytest.raises(ValueError, match='empty trade tape'):
        MakerFillModel.calibrate(trades=empty, lookback_minutes=10)


def test_maker_fill_queue_position_blocks_fills() -> None:
    """A deep initial queue prevents the maker from filling small aggressors.

    The lookback estimate of queue position must actually gate
    fills. A SELL maker that arrives behind 1.0 BTC of prior SELL-
    maker liquidity (is_buyer_maker=0) at its price level and sees
    0.5 BTC of subsequent BUY aggression should NOT fill — the
    aggressor only consumed half the queue. Same-side filtering
    (codex T14 R1) means the seed prior trades carry the SELL
    maker's own side flag.
    """
    submit_time = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    limit = Decimal('42700')
    # Pre-submit lookback: 1.0 BTC of SELL-maker liquidity at the
    # limit price (is_buyer_maker=0 → buyer was taker → seller was
    # maker → SELL maker queue).
    pre = pl.DataFrame({
        'datetime': [submit_time - timedelta(minutes=1)],
        'price': [float(limit)],
        'quantity': [1.0],
        'is_buyer_maker': [0],
    })
    # In-window: 0.5 BTC of BUY-aggressor at limit price (would
    # fill a SELL maker if queue=0, but queue=1.0 absorbs it).
    window = pl.DataFrame({
        'datetime': [submit_time + timedelta(seconds=10)],
        'price': [float(limit)],
        'quantity': [0.5],
        'is_buyer_maker': [0],
    })
    order = _maker(
        side='SELL', limit_price=limit,
        qty=Decimal('1.0'), submit_time=submit_time,
    )
    model = MakerFillModel(_lookback_minutes=10)
    fills = model.evaluate(
        order=order, trades_in_window=window,
        trades_pre_submit=pre,
    )
    assert fills == [], (
        f'queue=1.0 BTC must absorb 0.5 BTC aggressor; maker should '
        f'NOT fill yet. Got {len(fills)} fills.'
    )


def test_maker_fill_partial_fill_sizing_is_deterministic() -> None:
    """Each fill size = min(remaining_order_qty, post-queue aggressor qty).

    Codex Task 14 round 1 pinned the gap: an impl that fills the full
    order on the first eligible aggressor (regardless of aggressor
    size) would pass the live-tape MVC. Build a synthetic case with
    queue=0 and three aggressors of known sizes (0.3, 0.4, 0.5) and a
    1.0-BTC SELL maker. Expected: three partial fills of 0.3, 0.4,
    0.3 (last truncated to remaining qty).
    """
    submit_time = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    limit = Decimal('42700')
    aggressor_qtys = [0.3, 0.4, 0.5]
    times = [submit_time + timedelta(seconds=i) for i in range(1, 4)]
    window = pl.DataFrame({
        'datetime': times,
        'price': [float(limit)] * 3,
        'quantity': aggressor_qtys,
        # is_buyer_maker=0 → BUY aggressor → fills SELL maker.
        'is_buyer_maker': [0, 0, 0],
    })
    order = _maker(
        side='SELL', limit_price=limit,
        qty=Decimal('1.0'), submit_time=submit_time,
    )
    model = MakerFillModel(_lookback_minutes=10)
    # No pre-submit slice → queue=0. Each aggressor fills the maker
    # for min(aggressor_qty, remaining_order_qty).
    fills = model.evaluate(
        order=order, trades_in_window=window,
        trades_pre_submit=pl.DataFrame(),
    )
    assert len(fills) == 3, (
        f'expected 3 partial fills (0.3 + 0.4 + 0.3), got {len(fills)}'
    )
    expected_qtys = [Decimal('0.3'), Decimal('0.4'), Decimal('0.3')]
    for i, (fill, expected) in enumerate(zip(fills, expected_qtys, strict=True)):
        assert fill.fill_qty == expected, (
            f'fill #{i}: expected qty {expected}, got {fill.fill_qty}. '
            f'Maker should partial-fill against each aggressor at '
            f'min(remaining, aggressor_qty), not consume the entire '
            f'order on the first eligible aggressor.'
        )
    total = sum((f.fill_qty for f in fills), Decimal('0'))
    assert total == order.qty, (
        f'total filled {total} must equal order qty {order.qty}'
    )


def test_maker_fill_evaluate_uses_calibration_tape_for_queue() -> None:
    """`evaluate(order, trades_in_window)` (no trades_pre_submit) inherits queue.

    Codex Task 14 round 1 pinned the spec-call-path gap: the public
    API does not expose `trades_pre_submit`, so the model must
    derive the queue automatically from its calibration tape's
    `[submit_time - lookback, submit_time)` slice of same-side
    liquidity at the limit price.
    """
    submit_time = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    limit = Decimal('42700')
    # Calibration tape: 1.0 BTC of SELL-maker liquidity (queue) +
    # one BUY-aggressor 0.5 BTC trade in-window.
    tape = pl.DataFrame({
        'datetime': [
            submit_time - timedelta(minutes=2),
            submit_time + timedelta(seconds=30),
        ],
        'price': [float(limit), float(limit)],
        'quantity': [1.0, 0.5],
        'is_buyer_maker': [0, 0],
    })
    model = MakerFillModel.calibrate(trades=tape, lookback_minutes=10)
    window = tape.filter(pl.col('datetime') >= submit_time)
    order = _maker(
        side='SELL', limit_price=limit,
        qty=Decimal('0.4'), submit_time=submit_time,
    )
    # Spec call path: no trades_pre_submit. Queue comes from tape.
    fills = model.evaluate(order=order, trades_in_window=window)
    assert fills == [], (
        f'spec call path must derive queue from calibration tape — '
        f'1.0 BTC prior SELL-maker liquidity should absorb the 0.5 '
        f'BTC aggressor, leaving the maker unfilled. Got {len(fills)} '
        f'fills.'
    )
