"""Honesty gate: slippage model calibrates against the real trade tape.

Pins slice #17 Task 12 / SPEC §9.5 venue-fidelity sub-rule.

Slippage is the realised drift from the rolling mid at submit time.
A SlippageModel that hard-codes 5 bps would pass on a narrow tape
and fail catastrophically on a wide one — calibration is the point.

`test_slippage_calibration_against_tape` runs `SlippageModel.calibrate`
on a real 30-minute BTCUSDT trade fixture (`tests/fixtures/market/
btcusdt_trades_30min.parquet`, 16,702 trades from 2024-01-01 12:00 to
12:30 UTC, queried directly from `origo.binance_daily_spot_trades`)
and asserts the calibrated bps match the structural pattern any
honest market produces:

  - BUY-aggressor median bps is positive (taker pays above mid).
  - SELL-aggressor median bps is negative (taker receives below mid).
  - Magnitudes are roughly symmetric (a venue that systematically
    favours one side over the other is the regression to catch).
  - At least one bucket per side is non-empty (calibration sample
    coverage).

The test does NOT assert against itself — calibrate-then-verify-on-
the-same-data would be a tautology. Instead it pins structural
properties that any real-market trade tape must produce.
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import polars as pl
from nexus.core.domain.enums import OrderSide

from backtest_simulator.honesty.slippage import SlippageBucket, SlippageModel

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / 'fixtures' / 'market' / 'btcusdt_trades_30min.parquet'
)


def _load_trades() -> pl.DataFrame:
    assert _FIXTURE.is_file(), (
        f'Trade fixture missing at {_FIXTURE}. Regenerate from '
        f'origo.binance_daily_spot_trades via the ClickHouse tunnel.'
    )
    return pl.read_parquet(_FIXTURE)


def test_slippage_calibration_against_tape() -> None:
    """Calibrated slippage matches structural reality of a BUY/SELL tape.

    Real BTCUSDT in 30 minutes of activity (16,702 trades) produces
    BUY aggressors paying above mid and SELL aggressors receiving
    below mid. Magnitudes are within 1.5x of each other (a wider
    asymmetry would mean the venue is systematically biased — the
    honesty regression). Every defined qty bucket has at least one
    sample (calibration coverage).
    """
    trades = _load_trades()
    side_buckets = (Decimal('0.01'), Decimal('0.1'), Decimal('1.0'))
    model = SlippageModel.calibrate(
        trades=trades,
        side_buckets=side_buckets,
        dt_seconds=10,
    )

    buy_buckets = model.buckets_for(OrderSide.BUY)
    sell_buckets = model.buckets_for(OrderSide.SELL)

    assert buy_buckets, (
        'no BUY buckets calibrated; the fixture has zero is_buyer_maker=0 '
        'rows or the rolling-mid window dropped them all.'
    )
    assert sell_buckets, (
        'no SELL buckets calibrated; the fixture has zero is_buyer_maker=1 '
        'rows or the rolling-mid window dropped them all.'
    )

    # Each bucket carries a non-trivial sample count. The fixture
    # has ~33 trades/sec so even the rare large-qty buckets hold
    # tens of samples over 30 minutes.
    for bucket in (*buy_buckets, *sell_buckets):
        assert isinstance(bucket, SlippageBucket), (
            f'expected SlippageBucket, got {type(bucket)}'
        )
        assert bucket.n_samples > 0, (
            f'bucket {bucket.side.name} '
            f'[{bucket.qty_min}, {bucket.qty_max}) has 0 samples; '
            f'calibration coverage gap.'
        )

    # Volume-weighted aggregate bps per side: weight each bucket's
    # median by its sample count. This collapses the qty dimension
    # so the BUY-vs-SELL sign asymmetry is isolated. Checking only
    # the smallest bucket would let a high-qty bias slip through.
    def _agg_bps(buckets: tuple[SlippageBucket, ...]) -> Decimal:
        total_n = sum(b.n_samples for b in buckets)
        weighted = sum(b.median_bps * Decimal(b.n_samples) for b in buckets)
        return weighted / Decimal(total_n)

    buy_agg_bps = _agg_bps(buy_buckets)
    sell_agg_bps = _agg_bps(sell_buckets)

    # BUY-aggressor median must be positive: takers pay above mid.
    # An honest tape with non-zero spread cannot produce a negative
    # BUY-aggressor median; if it does, the calibration sign is
    # inverted or the rolling-mid leaks future prices.
    assert buy_agg_bps > Decimal('0'), (
        f'BUY-aggressor aggregate bps must be positive (taker pays '
        f'above mid); got {buy_agg_bps}. Calibration sign is inverted '
        f'or the rolling mid leaks future trade prices.'
    )
    # SELL-aggressor median must be negative.
    assert sell_agg_bps < Decimal('0'), (
        f'SELL-aggressor aggregate bps must be negative (taker '
        f'receives below mid); got {sell_agg_bps}. Calibration sign '
        f'is inverted or the rolling mid leaks future prices.'
    )

    # Side-magnitude symmetry: a healthy venue's bid-ask spread is
    # symmetric around mid, so |buy_bps| and |sell_bps| should be
    # within 2x. A dramatic asymmetry (10x) would mean the venue
    # is systematically biased — exactly the regression we catch.
    buy_mag = abs(buy_agg_bps)
    sell_mag = abs(sell_agg_bps)
    larger = max(buy_mag, sell_mag)
    smaller = min(buy_mag, sell_mag)
    assert smaller > Decimal('0'), (
        f'one side has zero magnitude; calibration is degenerate. '
        f'BUY={buy_agg_bps} SELL={sell_agg_bps}'
    )
    assert larger / smaller <= Decimal('2.0'), (
        f'BUY/SELL slippage magnitudes are asymmetric beyond 2x: '
        f'BUY={buy_agg_bps} SELL={sell_agg_bps} '
        f'ratio={larger / smaller}. The venue is systematically '
        f'biased toward one side, OR the calibration is broken.'
    )


def test_slippage_apply_returns_bucket_median() -> None:
    """`apply` looks up the matching bucket and returns its median bps.

    Pins the apply-side contract so a regression that swaps median
    for mean (or returns 0 always) is caught. The test bypasses
    real calibration by constructing a model with two known buckets
    and asserting the lookup picks the right one.
    """
    buy_small = SlippageBucket(
        side=OrderSide.BUY,
        qty_min=Decimal('0'),
        qty_max=Decimal('0.1'),
        median_bps=Decimal('1.5'),
        n_samples=100,
    )
    buy_large = SlippageBucket(
        side=OrderSide.BUY,
        qty_min=Decimal('0.1'),
        qty_max=None,
        median_bps=Decimal('7.5'),
        n_samples=20,
    )
    model = SlippageModel(
        _buckets_by_side={
            OrderSide.BUY: (buy_small, buy_large),
            OrderSide.SELL: (),
        },
        _dt_seconds=10,
    )

    # qty in small bucket → 1.5 bps.
    from datetime import UTC, datetime
    t = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    bps_small = model.apply(
        side=OrderSide.BUY, qty=Decimal('0.05'),
        mid=Decimal('70000'), t=t,
    )
    assert bps_small == Decimal('1.5'), (
        f'qty=0.05 should fall in [0, 0.1) → 1.5 bps; got {bps_small}'
    )

    # qty in large bucket → 7.5 bps.
    bps_large = model.apply(
        side=OrderSide.BUY, qty=Decimal('5.0'),
        mid=Decimal('70000'), t=t,
    )
    assert bps_large == Decimal('7.5'), (
        f'qty=5.0 should fall in [0.1, ∞) → 7.5 bps; got {bps_large}'
    )

    # SELL with no calibrated buckets → loud raise, NOT silent zero.
    # Silent fallback would create a backtest-to-live fidelity hole
    # where an uncalibrated side trades with zero slippage. Codex
    # Task 12 round 1 pinned this gap.
    import pytest
    with pytest.raises(ValueError, match='no calibrated buckets'):
        model.apply(
            side=OrderSide.SELL, qty=Decimal('0.05'),
            mid=Decimal('70000'), t=t,
        )


def test_slippage_apply_raises_on_unmatched_qty() -> None:
    """qty outside every calibrated bucket raises rather than return 0.

    A negative or absurdly large qty that no bucket covers means
    the calibration is the wrong fit for this order. Loud rejection
    forces the caller to widen the window or split the order;
    silent zero would let an uncalibrated qty trade live without
    any slippage cost.
    """
    bucket = SlippageBucket(
        side=OrderSide.BUY,
        qty_min=Decimal('0.01'),
        qty_max=Decimal('0.1'),
        median_bps=Decimal('1.5'),
        n_samples=100,
    )
    model = SlippageModel(
        _buckets_by_side={OrderSide.BUY: (bucket,), OrderSide.SELL: ()},
        _dt_seconds=10,
    )
    from datetime import UTC, datetime
    t = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    import pytest
    # qty=0.005 is below the smallest bucket's qty_min=0.01.
    with pytest.raises(ValueError, match='outside every calibrated bucket'):
        model.apply(
            side=OrderSide.BUY, qty=Decimal('0.005'),
            mid=Decimal('70000'), t=t,
        )


def test_slippage_calibration_rejects_empty_tape() -> None:
    """An empty trade DataFrame is a calibration failure, not a zero model.

    A model fit on no data would silently apply zero slippage to
    every order, hiding the absence of calibration. Loud rejection
    forces the operator to widen the window or fix the feed.
    """
    empty = pl.DataFrame({
        'datetime': pl.Series('datetime', [], dtype=pl.Datetime('us', 'UTC')),
        'price': pl.Series('price', [], dtype=pl.Float64),
        'quantity': pl.Series('quantity', [], dtype=pl.Float64),
        'is_buyer_maker': pl.Series('is_buyer_maker', [], dtype=pl.UInt8),
    })
    import pytest
    with pytest.raises(ValueError, match='empty trade tape'):
        SlippageModel.calibrate(
            trades=empty, side_buckets=(Decimal('0.1'),), dt_seconds=10,
        )
