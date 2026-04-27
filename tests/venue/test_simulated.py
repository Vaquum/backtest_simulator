"""Pin SimulatedVenueAdapter.submit_order on the zero-fill terminal status.

Pre-fix: a validated order that walked to zero fills (no liquidity in
the window) was reported as `OrderStatus.REJECTED`, conflating venue-
rejection with no-fill-in-window. This made `query_open_orders()`
permanently terminal for any future resting LIMIT and lied to lifecycle
consumers about why the order didn't fill. Post-fix: the validation-
rejection path still returns REJECTED (validation actually said no),
but the no-liquidity walk outcome returns EXPIRED (window closed
without execution).
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import cast

import polars as pl
from praxis.core.domain.enums import OrderSide, OrderStatus, OrderType

from backtest_simulator.feed.protocol import VenueFeed
from backtest_simulator.venue.fees import FeeSchedule
from backtest_simulator.venue.filters import BinanceSpotFilters
from backtest_simulator.venue.simulated import SimulatedVenueAdapter


class _EmptyFeed:
    """VenueFeed that returns no trades for any window — exercises the no-fill path."""

    def _empty_frame(self) -> pl.DataFrame:
        # `walk_trades` filters on a `time` column comparing against a
        # tz-aware UTC literal — the empty frame must declare the same
        # tz-aware Datetime dtype so the comparison's schema validates.
        return pl.DataFrame(schema={
            'time': pl.Datetime(time_zone='UTC'),
            'price': pl.Float64,
            'qty': pl.Float64,
            'trade_id': pl.Int64,
        })

    def get_trades(self, symbol: str, start: datetime, end: datetime) -> pl.DataFrame:
        del symbol, start, end
        return self._empty_frame()

    def _get_trades_for_venue(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        venue_lookahead_seconds: int,
    ) -> pl.DataFrame:
        del symbol, start, end, venue_lookahead_seconds
        return self._empty_frame()


def _adapter() -> SimulatedVenueAdapter:
    return SimulatedVenueAdapter(
        feed=cast(VenueFeed, _EmptyFeed()),
        filters=BinanceSpotFilters.binance_spot('BTCUSDT'),
        fees=FeeSchedule(),
        trade_window_seconds=60,
    )


def test_zero_fill_returns_expired_not_rejected() -> None:
    # Validated MARKET BUY that finds no liquidity in the empty feed
    # must terminate as EXPIRED. REJECTED is reserved for orders that
    # failed the filter pre-walk.
    adapter = _adapter()
    adapter.register_account('acct-1', 'k', 's')
    result = asyncio.run(adapter.submit_order(
        'acct-1', 'BTCUSDT', OrderSide.BUY,
        OrderType.MARKET,
        Decimal('0.001'),
    ))
    assert result.status == OrderStatus.EXPIRED, (
        f'expected EXPIRED for no-fill-in-window outcome, got {result.status}. '
        f'Pre-fix this returned REJECTED, conflating no-fill with venue-rejection.'
    )
    assert result.immediate_fills == ()


def test_zero_fill_status_routing() -> None:
    """`_I.zero_fill_status` picks OPEN for the GTC family, EXPIRED elsewhere.

    Live Binance keeps any GTC non-MARKET order on the book until it
    triggers / crosses / is cancelled. The simulator reports OPEN so
    `query_open_orders` surfaces still-live orders correctly (M1 only
    emits MARKET, so this isn't exercised end-to-end, but the status
    mapping must be correct for backtest ≡ paper ≡ live status parity).
    MARKET (no resting concept) and IOC / FOK (terminate on no
    immediate execution) map to EXPIRED.
    """
    from backtest_simulator.venue._adapter_internals import zero_fill_status

    # GTC family stays OPEN.
    assert zero_fill_status('LIMIT', 'GTC') == OrderStatus.OPEN
    assert zero_fill_status('LIMIT', 'gtc') == OrderStatus.OPEN  # case-insensitive
    assert zero_fill_status('STOP_LOSS', 'GTC') == OrderStatus.OPEN
    assert zero_fill_status('STOP_LOSS_LIMIT', 'GTC') == OrderStatus.OPEN
    assert zero_fill_status('TAKE_PROFIT', 'GTC') == OrderStatus.OPEN
    # IOC / FOK terminate on no immediate execution.
    assert zero_fill_status('LIMIT', 'IOC') == OrderStatus.EXPIRED
    assert zero_fill_status('LIMIT', 'FOK') == OrderStatus.EXPIRED
    assert zero_fill_status('STOP_LOSS_LIMIT', 'IOC') == OrderStatus.EXPIRED
    # MARKET never rests — always EXPIRED on no fill, even with GTC.
    assert zero_fill_status('MARKET', 'GTC') == OrderStatus.EXPIRED
    assert zero_fill_status('MARKET', 'IOC') == OrderStatus.EXPIRED


def test_validation_rejection_still_rejects() -> None:
    # Below min_qty → fails the filter pre-walk → REJECTED. This path
    # is unchanged by the zero-fill semantic correction.
    adapter = _adapter()
    adapter.register_account('acct-1', 'k', 's')
    result = asyncio.run(adapter.submit_order(
        'acct-1', 'BTCUSDT', OrderSide.BUY,
        OrderType.MARKET,
        Decimal('0.000001'),  # below min_qty 0.00001
    ))
    assert result.status == OrderStatus.REJECTED


# Market-impact wiring tests — pin Task 13's submit_order →
# MarketImpactModel.evaluate → bps + flag aggregation chain.
class _SyntheticImpactFeed:
    """Feed with one calibration bucket spanning a known time window.

    The bucket carries 10 BTC of total volume and a 5 bps price
    range — well above the LIMIT order's qty so the impact_bps
    falls out as `qty / 10 * 5 bps`. Used to pin the per-order
    recording path without relying on live ClickHouse.
    """

    def __init__(self, bucket_start: datetime) -> None:
        from datetime import timedelta
        # 10 trades, 1 BTC each, prices ramping 70000 → 70035 over the bucket.
        self._frame = pl.DataFrame({
            'time': [
                bucket_start + timedelta(seconds=i * 5)
                for i in range(10)
            ],
            'price': [70000.0 + i * 5 for i in range(10)],
            'qty': [1.0] * 10,
            'is_buyer_maker': [i % 2 for i in range(10)],
            'trade_id': list(range(10)),
        }).with_columns(
            pl.col('time').cast(pl.Datetime(time_zone='UTC')),
        )

    def get_trades(self, symbol: str, start: datetime, end: datetime) -> pl.DataFrame:
        del symbol
        return self._frame.filter(
            (pl.col('time') >= start) & (pl.col('time') < end),
        )

    def _get_trades_for_venue(
        self, symbol: str, start: datetime, end: datetime,
        *, venue_lookahead_seconds: int,
    ) -> pl.DataFrame:
        del symbol, venue_lookahead_seconds
        return self._frame.filter(
            (pl.col('time') >= start) & (pl.col('time') < end),
        )


def _impact_adapter(
    bucket_start: datetime,
    *,
    strict: bool = False,
):
    """Build an adapter wired for per-submit strict-causal impact.

    `bucket_start` defines the synthetic feed's bucket window;
    submits at `bucket_start + N seconds` (with N in `(0, 60)`)
    will see a populated pre-submit slice and produce calibrated
    samples. `strict=True` engages the pre-fill rejection gate.
    """
    feed = _SyntheticImpactFeed(bucket_start)
    return SimulatedVenueAdapter(
        feed=cast(VenueFeed, feed),
        filters=BinanceSpotFilters.binance_spot('BTCUSDT'),
        fees=FeeSchedule(),
        trade_window_seconds=60,
        market_impact_bucket_minutes=1,
        market_impact_threshold_fraction=Decimal('0.1'),
        strict_impact_policy=strict,
    )


def test_market_impact_records_bps_per_order() -> None:
    """A submit whose timestamp hits a calibrated bucket records bps + flag.

    Mutation proof: if `_record_market_impact` regresses (early
    return, wrong field name, etc.), `n_samples` stays at 0 and
    the JSON `market_impact_realised_bps` reads None — the
    audit's "feature ornamental" failure mode.
    """
    from datetime import UTC, timedelta

    from freezegun import freeze_time
    bucket_start = datetime(2024, 1, 3, 12, 0, tzinfo=UTC)
    adapter = _impact_adapter(bucket_start)
    adapter.register_account('acct-1', 'k', 's')
    # Submit at a wall-clock time inside the bucket. Use freezegun so
    # `_now()` returns a deterministic timestamp the bucket-finder
    # matches.
    with freeze_time(bucket_start + timedelta(seconds=20)):
        asyncio.run(adapter.submit_order(
            'acct-1', 'BTCUSDT', OrderSide.BUY,
            OrderType.MARKET,
            Decimal('0.001'),  # 0.001 of 10 BTC bucket = 0.01% of vol
        ))
    assert adapter.market_impact_n_samples == 1, (
        f'expected 1 calibrated impact sample, got '
        f'{adapter.market_impact_n_samples}.'
    )
    assert adapter.market_impact_n_uncalibrated == 0
    assert adapter.market_impact_n_flagged == 0, (
        f'order qty 0.001 << 10% of 10 BTC bucket vol; should not flag. '
        f'Got n_flagged={adapter.market_impact_n_flagged}.'
    )
    impact_bps = adapter.market_impact_realised_bps
    assert impact_bps is not None and impact_bps > Decimal('0'), (
        f'impact_bps must be positive (qty * range / total_volume); '
        f'got {impact_bps}.'
    )


def test_market_impact_flags_oversize_orders() -> None:
    """An order > threshold_fraction of bucket volume sets `flag=True`."""
    from datetime import UTC, timedelta

    from freezegun import freeze_time
    bucket_start = datetime(2024, 1, 3, 12, 0, tzinfo=UTC)
    adapter = _impact_adapter(bucket_start)
    adapter.register_account('acct-1', 'k', 's')
    # 5 BTC against a 10 BTC bucket = 50% of volume → way above
    # the 10% threshold.
    with freeze_time(bucket_start + timedelta(seconds=20)):
        asyncio.run(adapter.submit_order(
            'acct-1', 'BTCUSDT', OrderSide.BUY,
            OrderType.MARKET,
            Decimal('5'),
        ))
    assert adapter.market_impact_n_flagged == 1, (
        f'5 BTC of 10 BTC bucket exceeds 10% threshold but '
        f'n_flagged={adapter.market_impact_n_flagged}.'
    )


def test_market_impact_uncalibrated_when_no_bucket_match() -> None:
    """Submit outside the calibration window increments n_uncalibrated.

    The model returns `concurrent_volume=0` when no bucket
    contains `t`; the venue records this as uncalibrated rather
    than as a zero-impact sample so the operator sees the
    calibration gap distinctly.
    """
    from datetime import UTC, timedelta

    from freezegun import freeze_time
    bucket_start = datetime(2024, 1, 3, 12, 0, tzinfo=UTC)
    adapter = _impact_adapter(bucket_start)
    adapter.register_account('acct-1', 'k', 's')
    # Submit a year later — far outside the bucket.
    with freeze_time(bucket_start + timedelta(days=365)):
        asyncio.run(adapter.submit_order(
            'acct-1', 'BTCUSDT', OrderSide.BUY,
            OrderType.MARKET,
            Decimal('0.001'),
        ))
    assert adapter.market_impact_n_samples == 0
    assert adapter.market_impact_n_uncalibrated == 1, (
        f'submit outside calibration window should land in '
        f'n_uncalibrated, not n_samples. Got '
        f'n_samples={adapter.market_impact_n_samples} '
        f'n_uncalibrated={adapter.market_impact_n_uncalibrated}.'
    )


def test_market_impact_strict_policy_rejects_oversize_orders() -> None:
    """`strict_impact_policy=True` routes flagged orders to REJECTED.

    Pre-fill gate semantics: under strict mode, an ENTER order
    flagged as exceeding `threshold_fraction` of the per-submit
    bucket's volume is denied BEFORE walk_trades runs. The
    SubmitResult carries `OrderStatus.REJECTED` and zero
    immediate_fills, and `market_impact_n_rejected` increments.
    The default observability mode (strict=False, covered by
    `test_market_impact_flags_oversize_orders` above) records
    the flag without rejecting.

    Mutation proof: if the gate regresses to record-only,
    `result.status` returns FILLED/EXPIRED instead of REJECTED
    and `n_rejected` stays at 0.
    """
    from datetime import UTC, timedelta

    from freezegun import freeze_time
    bucket_start = datetime(2024, 1, 3, 12, 0, tzinfo=UTC)
    adapter = _impact_adapter(bucket_start, strict=True)
    adapter.register_account('acct-1', 'k', 's')
    with freeze_time(bucket_start + timedelta(seconds=20)):
        result = asyncio.run(adapter.submit_order(
            'acct-1', 'BTCUSDT', OrderSide.BUY,
            OrderType.MARKET,
            Decimal('5'),  # 50% of 10 BTC bucket -> flagged
        ))
    assert result.status == OrderStatus.REJECTED, (
        f'strict policy should reject the flagged order, got '
        f'status={result.status}.'
    )
    assert result.immediate_fills == ()
    assert adapter.market_impact_n_rejected == 1
    assert adapter.market_impact_n_flagged == 1, (
        f'rejection counts as flagged too — got n_flagged='
        f'{adapter.market_impact_n_flagged}.'
    )


def test_market_impact_strict_policy_passes_below_threshold() -> None:
    """`strict_impact_policy=True` does NOT reject orders under threshold.

    A small order (well below 10% of bucket volume) carries
    `flag=False` from the model; the gate's `flag and strict`
    condition is False; the order proceeds to walk_trades. This
    pins the gate's specificity — strict mode rejects ONLY
    flagged orders, not all submits.
    """
    from datetime import UTC, timedelta

    from freezegun import freeze_time
    bucket_start = datetime(2024, 1, 3, 12, 0, tzinfo=UTC)
    adapter = _impact_adapter(bucket_start, strict=True)
    adapter.register_account('acct-1', 'k', 's')
    with freeze_time(bucket_start + timedelta(seconds=20)):
        result = asyncio.run(adapter.submit_order(
            'acct-1', 'BTCUSDT', OrderSide.BUY,
            OrderType.MARKET,
            Decimal('0.001'),
        ))
    # Not REJECTED by the impact gate.
    assert result.status != OrderStatus.REJECTED, (
        f'small order should clear the strict-impact gate, got '
        f'status={result.status}.'
    )
    assert adapter.market_impact_n_rejected == 0
    assert adapter.market_impact_n_samples == 1


def test_market_impact_off_when_model_none() -> None:
    """`market_impact_realised_bps` is None when the feature is off.

    Distinguishes "feature off" (`bucket_minutes is None`) from
    "feature on, no data" (zero samples). The JSON consumer
    reads None as "feature disabled" and the per-run line
    skips the `imp` column.
    """
    adapter = _adapter()
    adapter.register_account('acct-1', 'k', 's')
    result = asyncio.run(adapter.submit_order(
        'acct-1', 'BTCUSDT', OrderSide.BUY,
        OrderType.MARKET, Decimal('0.001'),
    ))
    del result
    assert adapter.market_impact_realised_bps is None, (
        f'feature off but realised_bps={adapter.market_impact_realised_bps}'
    )
    assert adapter.market_impact_n_samples == 0
