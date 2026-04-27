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
        return _impact_from_bucket(
            qty=qty,
            total_volume=bucket.total_volume,
            price_range_bps=bucket.price_range_bps,
            threshold_fraction=self._threshold_fraction,
        )

    @classmethod
    def evaluate_rolling(
        cls,
        *,
        qty: Decimal,
        trades_pre_submit: pl.DataFrame,
        threshold_fraction: Decimal = Decimal('0.1'),
    ) -> MarketImpactDecision | None:
        """Strict-causal evaluation against a pre-fetched rolling slice.

        Treats `trades_pre_submit` as a single bucket — no
        wall-clock truncation. The caller (the bts venue path
        in `SimulatedVenueAdapter._record_market_impact_pre_fill`)
        fetches `[submit_time - bucket_minutes, submit_time)` of
        tape, applies a strict `time < submit_time` post-fetch
        filter (the feed's range query may be inclusive at one
        end), renames columns to the model's convention
        (`datetime`, `quantity`, `price`), and passes the result
        here. The model owns the qty-to-bps math; the venue owns
        the slice's strict-causal contract.

        Returns `None` when the slice is empty, zero-volume, or
        has a non-positive first price. `None` is the
        "uncalibrated" signal — distinct from a zero-impact
        decision (which only arises when `qty == 0` against a
        well-formed bucket). Callers translate `None` into their
        own uncalibrated counter; they MUST NOT treat it as a
        zero-impact sample, otherwise the realised aggregate is
        silently weighted down by calibration gaps.

        `evaluate_rolling` shares the linear-interpolation core
        with `evaluate`: an order consuming the entire slice's
        volume would move price by the slice's full range; a
        smaller order moves proportionally. `flag=True` when
        `qty > threshold_fraction * total_volume`.

        Required columns on `trades_pre_submit`: `quantity`,
        `price`. The `datetime` column is not read here — the
        slice's time-bounding is the caller's contract.
        """
        if trades_pre_submit.is_empty():
            return None
        total_volume = Decimal(str(trades_pre_submit['quantity'].sum()))
        if total_volume <= Decimal('0'):
            return None
        price_first_raw = trades_pre_submit.head(1)['price'].item()
        if price_first_raw is None or price_first_raw <= 0:
            return None
        price_first = Decimal(str(price_first_raw))
        price_max = Decimal(str(trades_pre_submit['price'].max()))
        price_min = Decimal(str(trades_pre_submit['price'].min()))
        price_range_bps = (price_max - price_min) / price_first * _BPS
        return _impact_from_bucket(
            qty=qty,
            total_volume=total_volume,
            price_range_bps=price_range_bps,
            threshold_fraction=threshold_fraction,
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


def _impact_from_bucket(
    *,
    qty: Decimal,
    total_volume: Decimal,
    price_range_bps: Decimal,
    threshold_fraction: Decimal,
) -> MarketImpactDecision:
    """Linear-interpolation impact + flag from a one-bucket summary.

    Single source of truth for the qty-to-bps math, shared by
    `MarketImpactModel.evaluate` (wall-clock bucket lookup) and
    `MarketImpactModel.evaluate_rolling` (rolling pre-submit
    slice). Centralising here is what the audit's "no two sources
    of truth" requirement demands — both paths now diverge only
    in HOW the `(total_volume, price_range_bps)` pair is built,
    not in what they do with it.
    """
    impact_bps = qty / total_volume * price_range_bps
    flag = qty > threshold_fraction * total_volume
    return MarketImpactDecision(
        impact_bps=impact_bps,
        concurrent_volume=total_volume,
        flag=flag,
    )
