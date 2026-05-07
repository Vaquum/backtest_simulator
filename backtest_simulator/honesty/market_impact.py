"""Market impact model — per-order bps penalty derived from tape volume."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal

import polars as pl

_BPS = Decimal('10000')

@dataclass(frozen=True)
class ImpactBucket:
    bucket_start: datetime
    total_volume: Decimal
    price_range_bps: Decimal

@dataclass(frozen=True)
class MarketImpactDecision:
    impact_bps: Decimal
    concurrent_volume: Decimal
    flag: bool

@dataclass
class MarketImpactModel:
    _bucket_minutes: int = 0
    _threshold_fraction: Decimal = field(default_factory=lambda: Decimal('0.1'))
    _buckets: tuple[ImpactBucket, ...] = field(default_factory=tuple)

    @classmethod
    def calibrate(cls, *, trades: pl.DataFrame, bucket_minutes: int, threshold_fraction: Decimal=Decimal('0.1')) -> MarketImpactModel:
        if trades.is_empty():
            msg = 'MarketImpactModel.calibrate: empty trade tape; cannot fit any bucket. Widen the calibration window.'
            raise ValueError(msg)
        if bucket_minutes <= 0:
            msg = f'MarketImpactModel.calibrate: bucket_minutes must be positive, got {bucket_minutes}.'
            raise ValueError(msg)
        if threshold_fraction < Decimal('0') or threshold_fraction > Decimal('1'):
            msg = f'MarketImpactModel.calibrate: threshold_fraction must be in [0, 1]; got {threshold_fraction}.'
            raise ValueError(msg)
        sorted_trades = trades.sort('datetime')
        bucketed = sorted_trades.with_columns(pl.col('datetime').dt.truncate(f'{bucket_minutes}m').alias('bucket_start'))
        agg = bucketed.group_by('bucket_start').agg(pl.col('quantity').sum().alias('total_volume'), pl.col('price').max().alias('price_max'), pl.col('price').min().alias('price_min'), pl.col('price').first().alias('price_first')).sort('bucket_start')
        buckets: list[ImpactBucket] = []
        for row in agg.iter_rows(named=True):
            price_first = Decimal(str(row['price_first']))
            if price_first <= Decimal('0'):
                continue
            price_range = Decimal(str(row['price_max'])) - Decimal(str(row['price_min']))
            price_range_bps = price_range / price_first * _BPS
            buckets.append(ImpactBucket(bucket_start=row['bucket_start'], total_volume=Decimal(str(row['total_volume'])), price_range_bps=price_range_bps))
        if not buckets:
            msg = 'MarketImpactModel.calibrate: every bucket was pathological (zero or negative price); cannot fit.'
            raise ValueError(msg)
        return cls(_bucket_minutes=bucket_minutes, _threshold_fraction=threshold_fraction, _buckets=tuple(buckets))

    def evaluate(self, *, qty: Decimal, mid: Decimal, t: datetime) -> MarketImpactDecision:
        del mid
        bucket = self._find_bucket(t)
        if bucket is None:
            return MarketImpactDecision(impact_bps=Decimal('0'), concurrent_volume=Decimal('0'), flag=True)
        if bucket.total_volume <= Decimal('0'):
            return MarketImpactDecision(impact_bps=Decimal('0'), concurrent_volume=Decimal('0'), flag=True)
        return _impact_from_bucket(qty=qty, total_volume=bucket.total_volume, price_range_bps=bucket.price_range_bps, threshold_fraction=self._threshold_fraction)

    @classmethod
    def evaluate_rolling(cls, *, qty: Decimal, trades_pre_submit: pl.DataFrame, threshold_fraction: Decimal=Decimal('0.1')) -> MarketImpactDecision | None:
        if trades_pre_submit.is_empty():
            return None
        total_volume = Decimal(str(trades_pre_submit['quantity'].sum()))
        price_first_raw = trades_pre_submit.head(1)['price'].item()
        if total_volume <= Decimal('0') or price_first_raw is None or price_first_raw <= 0:
            return None
        price_first = Decimal(str(price_first_raw))
        price_range_bps = (Decimal(str(trades_pre_submit['price'].max())) - Decimal(str(trades_pre_submit['price'].min()))) / price_first * _BPS
        return _impact_from_bucket(qty=qty, total_volume=total_volume, price_range_bps=price_range_bps, threshold_fraction=threshold_fraction)

    def _find_bucket(self, t: datetime) -> ImpactBucket | None:
        window = timedelta(minutes=self._bucket_minutes)
        for b in self._buckets:
            if b.bucket_start <= t < b.bucket_start + window:
                return b
        return None

def _impact_from_bucket(*, qty: Decimal, total_volume: Decimal, price_range_bps: Decimal, threshold_fraction: Decimal) -> MarketImpactDecision:
    return MarketImpactDecision(impact_bps=qty / total_volume * price_range_bps, concurrent_volume=total_volume, flag=qty > threshold_fraction * total_volume)
