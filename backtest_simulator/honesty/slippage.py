"""Empirical slippage model — calibrated from a symbol's trade tape."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

import polars as pl
from nexus.core.domain.enums import OrderSide

_BPS = Decimal('10000')

@dataclass(frozen=True)
class SlippageBucket:
    side: OrderSide
    qty_min: Decimal
    qty_max: Decimal | None
    median_bps: Decimal
    n_samples: int

@dataclass
class SlippageModel:
    _buckets_by_side: dict[OrderSide, tuple[SlippageBucket, ...]] = field(default_factory=lambda: {})
    _dt_seconds: int = 0

    @classmethod
    def calibrate(cls, *, trades: pl.DataFrame, side_buckets: Sequence[Decimal], dt_seconds: int) -> SlippageModel:
        sorted_trades = trades.sort('datetime')
        rolled = sorted_trades.with_columns(pl.col('price').rolling_median_by('datetime', window_size=f'{dt_seconds}s', closed='left').alias('mid_proxy')).drop_nulls('mid_proxy')
        rolled = rolled.with_columns(((pl.col('price') - pl.col('mid_proxy')) / pl.col('mid_proxy') * float(_BPS)).alias('slippage_bps'))
        bucket_thresholds = sorted({Decimal(str(q)) for q in side_buckets})
        bucket_thresholds_sorted = sorted(bucket_thresholds)
        buckets_by_side: dict[OrderSide, list[SlippageBucket]] = {OrderSide.BUY: [], OrderSide.SELL: []}
        for is_buyer_maker_raw, side in ((0, OrderSide.BUY), (1, OrderSide.SELL)):
            side_rows = rolled.filter(pl.col('is_buyer_maker') == is_buyer_maker_raw)
            qty_min = Decimal('0')
            for threshold in bucket_thresholds_sorted:
                bucket = side_rows.filter((pl.col('quantity') >= float(qty_min)) & (pl.col('quantity') < float(threshold)))
                if not bucket.is_empty():
                    median_bps = Decimal(str(bucket['slippage_bps'].median()))
                    buckets_by_side[side].append(SlippageBucket(side=side, qty_min=qty_min, qty_max=threshold, median_bps=median_bps, n_samples=len(bucket)))
                qty_min = threshold
            tail = side_rows.filter(pl.col('quantity') >= float(qty_min))
            if not tail.is_empty():
                median_bps = Decimal(str(tail['slippage_bps'].median()))
                buckets_by_side[side].append(SlippageBucket(side=side, qty_min=qty_min, qty_max=None, median_bps=median_bps, n_samples=len(tail)))
        return cls(_buckets_by_side={k: tuple(v) for k, v in buckets_by_side.items()}, _dt_seconds=dt_seconds)

    @property
    def dt_seconds(self) -> int:
        return self._dt_seconds

    def apply(self, *, side: OrderSide, qty: Decimal, mid: Decimal, t: datetime) -> Decimal:
        del mid, t
        side_buckets = self._buckets_by_side.get(side, ())
        for bucket in side_buckets:
            in_lower = qty >= bucket.qty_min
            in_upper = bucket.qty_max is None or qty < bucket.qty_max
            if in_lower and in_upper:
                return bucket.median_bps
        msg = f'SlippageModel.apply: qty={qty} falls outside every calibrated bucket for side={side.name}. The calibration tape did not include this qty range; widen the window or supply a smaller order.'
        raise ValueError(msg)
