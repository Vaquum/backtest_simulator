"""Market impact model — per-order bps penalty derived from tape volume."""
from __future__ import annotations

# Slice #17 Task 13: an order of size `qty` at midpoint `mid` would,
# if it actually traded, push the book by some bps as it walks
# liquidity. The same simulator that fills 1000 BTC at mid would be
# off by orders of magnitude live. Calibrate the empirical
# qty-to-bps relationship from the same symbol's trade tape and
# return the bps along with a `flag` for orders that would consume
# more than `threshold_fraction` of concurrent trade volume — those
# are the orders the operator must size down before live.
#
# The calibration is per-`bucket_minutes` window: each bucket holds
# total volume + the qty-quantile-to-bps mapping. `evaluate` looks
# up the bucket containing `t`, computes order_qty / concurrent_volume,
# and returns the impact_bps + flag.
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal

import polars as pl

_BPS = Decimal('10000')


@dataclass(frozen=True)
class ImpactBucket:
    """One time-window's volume distribution.

    `bucket_start` is inclusive; the bucket ends at
    `bucket_start + bucket_minutes`. `total_volume` is the summed
    qty traded in the window. `price_range_bps` is the (high-low)
    range in bps over the window — the upper bound on impact for
    an order that consumed all the volume.
    """

    bucket_start: datetime
    total_volume: Decimal
    price_range_bps: Decimal


@dataclass(frozen=True)
class MarketImpactDecision:
    """Outcome of `MarketImpactModel.evaluate`.

    `impact_bps` is the linear-interpolation estimate: order_qty
    relative to the bucket's total volume, scaled by the bucket's
    price range. An order consuming the entire bucket's volume
    would move price by the full range; smaller orders move
    proportionally less. `concurrent_volume` is the total volume in
    the matching bucket; `flag=True` when order_qty exceeds
    `threshold_fraction` of concurrent_volume — the operator-
    visible "this order is too big" signal.
    """

    impact_bps: Decimal
    concurrent_volume: Decimal
    flag: bool


@dataclass
class MarketImpactModel:
    """Calibrated qty-to-bps impact estimator with size-vs-volume flag."""

    _bucket_minutes: int = 0
    _threshold_fraction: Decimal = field(default_factory=lambda: Decimal('0.1'))
    _buckets: tuple[ImpactBucket, ...] = field(default_factory=tuple)

    @classmethod
    def calibrate(
        cls,
        *,
        trades: pl.DataFrame,
        bucket_minutes: int,
        threshold_fraction: Decimal = Decimal('0.1'),
    ) -> MarketImpactModel:
        """Fit the model over `trades` partitioned by `bucket_minutes`.

        `trades` must carry `datetime`, `price`, `quantity`. Each
        bucket records the total qty traded and the price range —
        the calibration's basis for converting order_qty/volume into
        bps.

        `threshold_fraction` is the size-vs-volume flag threshold.
        Default 0.1 (10%) — an order bigger than 10% of a
        bucket-minute's volume is the operator's "down-size before
        live" trigger.
        """
        if trades.is_empty():
            msg = (
                'MarketImpactModel.calibrate: empty trade tape; cannot '
                'fit any bucket. Widen the calibration window.'
            )
            raise ValueError(msg)
        if bucket_minutes <= 0:
            msg = (
                f'MarketImpactModel.calibrate: bucket_minutes must be '
                f'positive, got {bucket_minutes}.'
            )
            raise ValueError(msg)
        if threshold_fraction < Decimal('0') or threshold_fraction > Decimal('1'):
            msg = (
                f'MarketImpactModel.calibrate: threshold_fraction must '
                f'be in [0, 1]; got {threshold_fraction}.'
            )
            raise ValueError(msg)
        sorted_trades = trades.sort('datetime')
        # Group by bucket — fixed-width windows starting at minute 0.
        bucketed = sorted_trades.with_columns(
            pl.col('datetime').dt.truncate(f'{bucket_minutes}m').alias(
                'bucket_start',
            ),
        )
        agg = bucketed.group_by('bucket_start').agg(
            pl.col('quantity').sum().alias('total_volume'),
            pl.col('price').max().alias('price_max'),
            pl.col('price').min().alias('price_min'),
            pl.col('price').first().alias('price_first'),
        ).sort('bucket_start')
        buckets: list[ImpactBucket] = []
        for row in agg.iter_rows(named=True):
            price_first = Decimal(str(row['price_first']))
            if price_first <= Decimal('0'):
                # Pathological row — skip rather than divide by zero.
                continue
            price_range = (
                Decimal(str(row['price_max']))
                - Decimal(str(row['price_min']))
            )
            price_range_bps = price_range / price_first * _BPS
            buckets.append(ImpactBucket(
                bucket_start=row['bucket_start'],
                total_volume=Decimal(str(row['total_volume'])),
                price_range_bps=price_range_bps,
            ))
        if not buckets:
            msg = (
                'MarketImpactModel.calibrate: every bucket was '
                'pathological (zero or negative price); cannot fit.'
            )
            raise ValueError(msg)
        return cls(
            _bucket_minutes=bucket_minutes,
            _threshold_fraction=threshold_fraction,
            _buckets=tuple(buckets),
        )

    @property
    def bucket_minutes(self) -> int:
        return self._bucket_minutes

    @property
    def threshold_fraction(self) -> Decimal:
        return self._threshold_fraction

    @property
    def buckets(self) -> tuple[ImpactBucket, ...]:
        return self._buckets

    def evaluate(
        self,
        *,
        qty: Decimal,
        mid: Decimal,
        t: datetime,
    ) -> MarketImpactDecision:
        del mid  # mid reserved for future quote-anchored calibration
        bucket = self._find_bucket(t)
        if bucket is None:
            # No bucket covers `t`. The honest answer is "no
            # calibration data" — return zero impact + flag=True so
            # the operator sees the gap.
            return MarketImpactDecision(
                impact_bps=Decimal('0'),
                concurrent_volume=Decimal('0'),
                flag=True,
            )
        if bucket.total_volume <= Decimal('0'):
            # Empty-volume bucket: return flag=True for the same
            # reason as no-bucket above.
            return MarketImpactDecision(
                impact_bps=Decimal('0'),
                concurrent_volume=Decimal('0'),
                flag=True,
            )
        # Linear-interpolation impact: order consuming the whole
        # bucket's volume moves price by the full range; smaller
        # orders move proportionally.
        impact_bps = (
            qty / bucket.total_volume * bucket.price_range_bps
        )
        flag = qty > self._threshold_fraction * bucket.total_volume
        return MarketImpactDecision(
            impact_bps=impact_bps,
            concurrent_volume=bucket.total_volume,
            flag=flag,
        )

    def _find_bucket(self, t: datetime) -> ImpactBucket | None:
        # Linear scan. Production should switch to a sorted
        # bisect when the bucket count grows; for ≤ 1 day of
        # 1-minute buckets (~1440) this is fine.
        window = timedelta(minutes=self._bucket_minutes)
        for b in self._buckets:
            if b.bucket_start <= t < b.bucket_start + window:
                return b
        return None
