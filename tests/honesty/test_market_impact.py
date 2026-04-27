"""Honesty gate: market impact scales with order size relative to volume.

Pins slice #17 Task 13 / SPEC §9.5 venue-fidelity sub-rule.

An order's impact bps must scale with `order_qty / concurrent_volume`,
not be a fixed constant. A simulator that fills 1000 BTC at mid in
a 30-minute window with 50 BTC of trades is silently giving the
strategy a free ride that would never replicate live. The model
captures this via per-`bucket_minutes` calibration of (total_volume,
price_range) and a linear-interpolation `impact_bps` that grows with
the qty/volume ratio.

`test_market_impact_size_vs_volume` calibrates on the real BTCUSDT
30-minute trade fixture and asserts:
  - A small order (qty << bucket volume) produces small impact.
  - A large order (qty > threshold_fraction * volume) produces
    proportionally larger impact AND raises the flag.
  - Doubling qty roughly doubles impact_bps within the same
    bucket (linearity).
"""
from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

from backtest_simulator.honesty.market_impact import (
    ImpactBucket,
    MarketImpactDecision,
    MarketImpactModel,
)

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / 'fixtures' / 'market' / 'btcusdt_trades_30min.parquet'
)


def _load_trades() -> pl.DataFrame:
    assert _FIXTURE.is_file(), (
        f'Trade fixture missing at {_FIXTURE}.'
    )
    return pl.read_parquet(_FIXTURE)


def test_market_impact_size_vs_volume() -> None:
    """Impact bps scales monotonically with qty / concurrent_volume.

    Real BTCUSDT 30-minute fixture: ~16,702 trades. Calibrate at
    1-minute buckets, threshold at 10% of bucket volume.

    Within one bucket:
      - qty = 1% of volume → impact tiny, flag=False.
      - qty = 50% of volume → impact ~50x the 1% case, flag=True.
      - doubling qty doubles impact_bps (linearity).
    """
    trades = _load_trades()
    model = MarketImpactModel.calibrate(
        trades=trades,
        bucket_minutes=1,
        threshold_fraction=Decimal('0.1'),
    )

    # Pick a non-empty mid-tape bucket with a non-zero price range —
    # the first bucket may be under-sampled at fixture boundaries,
    # and a flat-price bucket has price_range_bps==0 which would
    # let a hard-coded zero-impact implementation slip through the
    # linearity check below.
    middle_buckets = [
        b for b in model.buckets
        if b.total_volume > Decimal('0.5') and b.price_range_bps > Decimal('0')
    ]
    assert middle_buckets, (
        'no buckets with > 0.5 BTC volume AND non-zero price range; '
        'fixture too thin or too flat to test impact'
    )
    bucket = middle_buckets[len(middle_buckets) // 2]
    bucket_volume = bucket.total_volume
    bucket_t = bucket.bucket_start
    mid = Decimal('42700')  # close to the fixture's actual price

    # Small order: 1% of volume.
    small_qty = bucket_volume * Decimal('0.01')
    small = model.evaluate(qty=small_qty, mid=mid, t=bucket_t)
    assert isinstance(small, MarketImpactDecision)
    assert small.flag is False, (
        f'qty=1% of {bucket_volume} should not trip the 10% '
        f'threshold flag; got flag={small.flag}'
    )
    # The bucket has non-zero range, so a non-trivial qty MUST
    # produce non-zero impact. A hard-coded zero-impact impl would
    # pass the flag/volume assertions but trip this. Codex Task 13
    # round 1 pinned this gap.
    assert small.impact_bps > Decimal('0'), (
        f'small qty (1% of bucket volume) at non-flat price range '
        f'must produce non-zero impact_bps; got {small.impact_bps}. '
        f'A hard-coded zero-impact implementation would pass the '
        f'flag/volume checks but trip this.'
    )

    # Large order: 50% of volume.
    large_qty = bucket_volume * Decimal('0.5')
    large = model.evaluate(qty=large_qty, mid=mid, t=bucket_t)
    assert large.flag is True, (
        f'qty=50% of {bucket_volume} should trip the 10% threshold '
        f'flag; got flag={large.flag}'
    )
    # Large impact must exceed small impact monotonically.
    assert large.impact_bps > small.impact_bps, (
        f'large impact ({large.impact_bps}) must exceed small impact '
        f'({small.impact_bps}); the model is not monotone in qty.'
    )

    # Linearity: 50%/1% = 50x. Linear-interpolation impact scales
    # linearly so impact_bps should be exactly 50x.
    ratio = large.impact_bps / small.impact_bps
    assert Decimal('45') < ratio < Decimal('55'), (
        f'impact_bps does not scale linearly with qty: '
        f'small={small.impact_bps}, large={large.impact_bps}, '
        f'ratio={ratio}; expected ~50.'
    )

    # Exact-value check: a volume-blind impl like
    # `impact_bps = qty * k` would pass small>0, large>small,
    # AND the 50x ratio. The honest formula is
    # `impact_bps = (qty / total_volume) * price_range_bps`.
    # Assert directly that small.impact_bps == 0.01 * range_bps
    # (within floating-point Decimal tolerance) and similarly for
    # large = 0.5 * range_bps. Codex Task 13 round 2 pinned this.
    expected_small = bucket.price_range_bps * Decimal('0.01')
    expected_large = bucket.price_range_bps * Decimal('0.5')
    # Tolerance: 1% relative (Decimal multiply / divide accumulates
    # tiny rounding when bucket_volume is fractional).
    tol_small = expected_small * Decimal('0.01')
    tol_large = expected_large * Decimal('0.01')
    assert abs(small.impact_bps - expected_small) <= tol_small, (
        f'small impact must equal 0.01 * range_bps={expected_small}; '
        f'got {small.impact_bps}. The model is not actually scaling '
        f'qty by concurrent volume — a volume-blind impl would land '
        f'here too with an incorrect value.'
    )
    assert abs(large.impact_bps - expected_large) <= tol_large, (
        f'large impact must equal 0.5 * range_bps={expected_large}; '
        f'got {large.impact_bps}.'
    )

    # Volume-as-denominator check: same qty across buckets with
    # different volumes must produce impact_bps proportional to
    # 1/volume (with the bucket's own range_bps). A volume-blind
    # impl that uses `qty * range_bps` would scale with range_bps
    # but NOT with volume — so the != check from round-2 wasn't
    # enough. Codex Task 13 round 3 pinned this gap. Pick another
    # bucket with materially different volume AND non-zero range,
    # assert the exact formula against IT, and require
    # concurrent_volume to equal the new bucket's volume.
    other_bucket = next(
        (b for b in middle_buckets if b is not bucket
         and b.total_volume != bucket.total_volume
         and b.price_range_bps > Decimal('0')),
        None,
    )
    assert other_bucket is not None, (
        'fixture must include at least two buckets with different '
        'total_volume AND non-zero price_range_bps so the volume-as-'
        'denominator dimension can be tested.'
    )
    same_qty = small_qty  # 1% of original bucket
    other = model.evaluate(
        qty=same_qty, mid=mid, t=other_bucket.bucket_start,
    )
    # Honest formula: impact_bps = qty / other.total_volume *
    # other.price_range_bps. A volume-blind impl using
    # `qty * range_bps` (or anything not normalised by volume)
    # would produce a different value here.
    expected_other = (
        same_qty / other_bucket.total_volume * other_bucket.price_range_bps
    )
    tol_other = abs(expected_other) * Decimal('0.01') + Decimal('1e-12')
    assert abs(other.impact_bps - expected_other) <= tol_other, (
        f'cross-bucket impact must equal qty/other.total_volume * '
        f'other.price_range_bps={expected_other}; got {other.impact_bps}. '
        f'A volume-blind implementation would land here too with an '
        f'incorrect value.'
    )
    assert other.concurrent_volume == other_bucket.total_volume, (
        f'cross-bucket concurrent_volume must equal '
        f'{other_bucket.total_volume}; got {other.concurrent_volume}'
    )

    # concurrent_volume must surface for both decisions; it's the
    # bucket's total volume.
    assert small.concurrent_volume == bucket_volume
    assert large.concurrent_volume == bucket_volume


def test_market_impact_unknown_time_flags_loud() -> None:
    """A `t` outside the calibration window flags=True with zero data.

    The operator must see "no calibration here" — silently returning
    impact_bps=0 with flag=False would let an unaware strategy
    submit at any time and look fine.
    """
    model = MarketImpactModel(
        _bucket_minutes=1,
        _threshold_fraction=Decimal('0.1'),
        _buckets=(
            ImpactBucket(
                bucket_start=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
                total_volume=Decimal('10'),
                price_range_bps=Decimal('5'),
            ),
        ),
    )
    far_future = datetime(2026, 1, 1, tzinfo=UTC)
    decision = model.evaluate(
        qty=Decimal('1'), mid=Decimal('70000'), t=far_future,
    )
    assert decision.impact_bps == Decimal('0')
    assert decision.concurrent_volume == Decimal('0')
    assert decision.flag is True, (
        f'unknown bucket must set flag=True so the operator sees '
        f'the calibration gap; got flag={decision.flag}'
    )


def test_market_impact_calibration_rejects_empty_tape() -> None:
    """Empty trades raises loudly, not a no-impact silently."""
    empty = pl.DataFrame({
        'datetime': pl.Series('datetime', [], dtype=pl.Datetime('us', 'UTC')),
        'price': pl.Series('price', [], dtype=pl.Float64),
        'quantity': pl.Series('quantity', [], dtype=pl.Float64),
    })
    with pytest.raises(ValueError, match='empty trade tape'):
        MarketImpactModel.calibrate(trades=empty, bucket_minutes=1)


def test_market_impact_calibration_rejects_zero_bucket_minutes() -> None:
    """`bucket_minutes <= 0` raises rather than silently divide-by-zero."""
    trades = _load_trades()
    with pytest.raises(ValueError, match='bucket_minutes must be positive'):
        MarketImpactModel.calibrate(trades=trades, bucket_minutes=0)


def test_market_impact_calibration_rejects_threshold_out_of_range() -> None:
    """`threshold_fraction` outside [0, 1] is a misconfiguration."""
    trades = _load_trades()
    with pytest.raises(ValueError, match='threshold_fraction must'):
        MarketImpactModel.calibrate(
            trades=trades, bucket_minutes=1,
            threshold_fraction=Decimal('1.5'),
        )
    with pytest.raises(ValueError, match='threshold_fraction must'):
        MarketImpactModel.calibrate(
            trades=trades, bucket_minutes=1,
            threshold_fraction=Decimal('-0.1'),
        )


# `evaluate_rolling` — the rolling-slice contract used by the bts
# venue path. Centralising the qty-to-bps math on the model means
# these tests protect SimulatedVenueAdapter._record_market_impact_pre_fill
# from drift: any future change to the math fails here, not silently
# in the venue. The audit on commit fe00024 explicitly required this.

def _slice_from_bucket(
    trades: pl.DataFrame, bucket_start: datetime, bucket_minutes: int,
) -> pl.DataFrame:
    """Extract a per-minute slice in the convention `evaluate_rolling` expects."""
    from datetime import timedelta
    end = bucket_start + timedelta(minutes=bucket_minutes)
    return trades.filter(
        (pl.col('datetime') >= bucket_start)
        & (pl.col('datetime') < end),
    )


def test_market_impact_rolling_empty_slice_is_uncalibrated() -> None:
    """Empty slice returns None — distinct from a zero-impact decision."""
    empty = pl.DataFrame({
        'datetime': pl.Series('datetime', [], dtype=pl.Datetime('us', 'UTC')),
        'price': pl.Series('price', [], dtype=pl.Float64),
        'quantity': pl.Series('quantity', [], dtype=pl.Float64),
    })
    decision = MarketImpactModel.evaluate_rolling(
        qty=Decimal('1'), trades_pre_submit=empty,
        threshold_fraction=Decimal('0.1'),
    )
    assert decision is None, (
        f'empty slice must return None (uncalibrated signal); got {decision}'
    )


def test_market_impact_rolling_zero_volume_is_uncalibrated() -> None:
    """Slice with zero summed volume returns None."""
    zero_vol = pl.DataFrame({
        'datetime': [datetime(2024, 1, 1, 12, 0, tzinfo=UTC)],
        'price': [70000.0],
        'quantity': [0.0],
    })
    decision = MarketImpactModel.evaluate_rolling(
        qty=Decimal('1'), trades_pre_submit=zero_vol,
        threshold_fraction=Decimal('0.1'),
    )
    assert decision is None


def test_market_impact_rolling_non_positive_first_price_is_uncalibrated() -> None:
    """First-row non-positive price returns None — guards divide-by-zero."""
    bad_price = pl.DataFrame({
        'datetime': [datetime(2024, 1, 1, 12, 0, tzinfo=UTC)],
        'price': [0.0],
        'quantity': [1.0],
    })
    decision = MarketImpactModel.evaluate_rolling(
        qty=Decimal('1'), trades_pre_submit=bad_price,
        threshold_fraction=Decimal('0.1'),
    )
    assert decision is None


def test_market_impact_rolling_small_qty_below_threshold() -> None:
    """qty=1% of slice volume → impact > 0, flag=False."""
    trades = _load_trades()
    model = MarketImpactModel.calibrate(
        trades=trades, bucket_minutes=1,
        threshold_fraction=Decimal('0.1'),
    )
    middle_buckets = [
        b for b in model.buckets
        if b.total_volume > Decimal('0.5') and b.price_range_bps > Decimal('0')
    ]
    assert middle_buckets
    bucket = middle_buckets[len(middle_buckets) // 2]
    sl = _slice_from_bucket(trades, bucket.bucket_start, 1)
    qty = bucket.total_volume * Decimal('0.01')
    decision = MarketImpactModel.evaluate_rolling(
        qty=qty, trades_pre_submit=sl,
        threshold_fraction=Decimal('0.1'),
    )
    assert decision is not None
    assert decision.flag is False
    assert decision.impact_bps > Decimal('0')


def test_market_impact_rolling_oversize_qty_flags() -> None:
    """qty > threshold_fraction * volume → flag=True."""
    trades = _load_trades()
    model = MarketImpactModel.calibrate(
        trades=trades, bucket_minutes=1,
        threshold_fraction=Decimal('0.1'),
    )
    middle_buckets = [
        b for b in model.buckets
        if b.total_volume > Decimal('0.5') and b.price_range_bps > Decimal('0')
    ]
    assert middle_buckets
    bucket = middle_buckets[len(middle_buckets) // 2]
    sl = _slice_from_bucket(trades, bucket.bucket_start, 1)
    qty = bucket.total_volume * Decimal('0.5')
    decision = MarketImpactModel.evaluate_rolling(
        qty=qty, trades_pre_submit=sl,
        threshold_fraction=Decimal('0.1'),
    )
    assert decision is not None
    assert decision.flag is True


def test_market_impact_rolling_doubling_qty_doubles_impact() -> None:
    """Linear interpolation: 2x qty → 2x impact_bps within one slice."""
    trades = _load_trades()
    model = MarketImpactModel.calibrate(
        trades=trades, bucket_minutes=1,
        threshold_fraction=Decimal('0.1'),
    )
    middle_buckets = [
        b for b in model.buckets
        if b.total_volume > Decimal('0.5') and b.price_range_bps > Decimal('0')
    ]
    assert middle_buckets
    bucket = middle_buckets[len(middle_buckets) // 2]
    sl = _slice_from_bucket(trades, bucket.bucket_start, 1)
    qty1 = bucket.total_volume * Decimal('0.01')
    qty2 = qty1 * Decimal('2')
    d1 = MarketImpactModel.evaluate_rolling(
        qty=qty1, trades_pre_submit=sl, threshold_fraction=Decimal('0.1'),
    )
    d2 = MarketImpactModel.evaluate_rolling(
        qty=qty2, trades_pre_submit=sl, threshold_fraction=Decimal('0.1'),
    )
    assert d1 is not None and d2 is not None
    ratio = d2.impact_bps / d1.impact_bps
    assert Decimal('1.99') < ratio < Decimal('2.01'), (
        f'doubling qty must double impact_bps; ratio={ratio}'
    )


def test_market_impact_rolling_volume_denominator() -> None:
    """Same qty across two slices with different volumes → impact ∝ 1/volume.

    A volume-blind implementation that uses `qty * range_bps`
    would scale with the slice's range but not with volume,
    failing this. The honest formula is
    `impact_bps = qty / total_volume * price_range_bps`.
    """
    trades = _load_trades()
    model = MarketImpactModel.calibrate(
        trades=trades, bucket_minutes=1,
        threshold_fraction=Decimal('0.1'),
    )
    middle_buckets = [
        b for b in model.buckets
        if b.total_volume > Decimal('0.5') and b.price_range_bps > Decimal('0')
    ]
    assert len(middle_buckets) >= 2
    b1 = middle_buckets[len(middle_buckets) // 2]
    b2 = next(
        (b for b in middle_buckets if b is not b1
         and b.total_volume != b1.total_volume),
        None,
    )
    assert b2 is not None
    sl1 = _slice_from_bucket(trades, b1.bucket_start, 1)
    sl2 = _slice_from_bucket(trades, b2.bucket_start, 1)
    qty = min(b1.total_volume, b2.total_volume) * Decimal('0.01')
    d1 = MarketImpactModel.evaluate_rolling(
        qty=qty, trades_pre_submit=sl1, threshold_fraction=Decimal('0.1'),
    )
    d2 = MarketImpactModel.evaluate_rolling(
        qty=qty, trades_pre_submit=sl2, threshold_fraction=Decimal('0.1'),
    )
    assert d1 is not None and d2 is not None
    expected1 = qty / b1.total_volume * b1.price_range_bps
    expected2 = qty / b2.total_volume * b2.price_range_bps
    tol1 = abs(expected1) * Decimal('0.01') + Decimal('1e-12')
    tol2 = abs(expected2) * Decimal('0.01') + Decimal('1e-12')
    assert abs(d1.impact_bps - expected1) <= tol1
    assert abs(d2.impact_bps - expected2) <= tol2
    # concurrent_volume must surface the slice's actual volume,
    # not a cached or class-level value.
    assert d1.concurrent_volume == b1.total_volume
    assert d2.concurrent_volume == b2.total_volume


def test_market_impact_rolling_matches_evaluate_when_slice_equals_bucket() -> None:
    """Rolling on a slice equal to one calibrated bucket equals `evaluate(t)`.

    Bridge contract: a slice that spans exactly one bucket-minute
    aligned at the bucket boundary must produce the same
    decision as the wall-clock `evaluate(t)` at that bucket.
    Anchors `evaluate_rolling` against the existing `evaluate`
    behaviour the bts venue path used to call.
    """
    trades = _load_trades()
    bucket_minutes = 1
    model = MarketImpactModel.calibrate(
        trades=trades, bucket_minutes=bucket_minutes,
        threshold_fraction=Decimal('0.1'),
    )
    middle_buckets = [
        b for b in model.buckets
        if b.total_volume > Decimal('0.5') and b.price_range_bps > Decimal('0')
    ]
    assert middle_buckets
    bucket = middle_buckets[len(middle_buckets) // 2]
    sl = _slice_from_bucket(trades, bucket.bucket_start, bucket_minutes)
    qty = bucket.total_volume * Decimal('0.05')
    rolling = MarketImpactModel.evaluate_rolling(
        qty=qty, trades_pre_submit=sl,
        threshold_fraction=Decimal('0.1'),
    )
    via_evaluate = model.evaluate(
        qty=qty, mid=Decimal('42700'), t=bucket.bucket_start,
    )
    assert rolling is not None
    # Both paths share `_impact_from_bucket`, so the
    # impact_bps / flag / concurrent_volume must match exactly.
    assert rolling.impact_bps == via_evaluate.impact_bps
    assert rolling.concurrent_volume == via_evaluate.concurrent_volume
    assert rolling.flag == via_evaluate.flag
