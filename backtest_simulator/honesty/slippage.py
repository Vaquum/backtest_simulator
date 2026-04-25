"""Empirical slippage model — calibrated from a symbol's trade tape."""
from __future__ import annotations

# Slice #17 Task 12: when a strategy submits an order at price `mid`,
# the realised fill price drifts by some bps. The drift comes from
# the bid-ask spread, queue position, and the latency between submit
# and execution. Calibrate the empirical distribution of that drift
# from the same symbol's historical trades, bucketed by (side, qty).
#
# Sign convention: `apply` returns *signed* bps in price-space.
#   - BUY aggressor pays above mid → positive bps.
#   - SELL aggressor receives below mid → negative bps.
# To get the expected fill price the caller scales bps by mid:
# `fill_price = mid * (Decimal(1) + bps / Decimal(10000))`. Adding
# bps directly to mid would be dimensionally wrong (mixing absolute
# price units with bps). Codex Task 12 round 4 pinned the
# docstring/contract clarity gap.
#
# Aggressor side comes from Binance's `is_buyer_maker`:
#   - is_buyer_maker == 0 → buyer was taker → BUY aggressor.
#   - is_buyer_maker == 1 → seller was taker → SELL aggressor.
#
# `mid` per trade is a rolling median of the preceding `dt_seconds`
# of trades, excluding the current trade itself (otherwise the
# trade's own price contaminates its own mid).
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

import polars as pl
from nexus.core.domain.enums import OrderSide

_BPS = Decimal('10000')


@dataclass(frozen=True)
class SlippageBucket:
    """One (side, qty-bucket) entry of the calibrated slippage table.

    `qty_min` is inclusive, `qty_max` is exclusive. The final bucket
    has `qty_max=None` ("and above").

    `median_bps` is the operator-visible median realised slippage
    in price-space basis points; `n_samples` is the calibration
    sample count (low n means the bucket is untrustworthy and the
    operator should widen the calibration window).
    """

    side: OrderSide
    qty_min: Decimal
    qty_max: Decimal | None
    median_bps: Decimal
    n_samples: int


@dataclass
class SlippageModel:
    """Per-(side, qty-bucket) calibrated slippage in signed bps.

    `calibrate` walks the trade tape and computes per-bucket median
    drift from a rolling-mid over `dt_seconds`. `apply` looks up the
    bucket for `(side, qty)` and returns the median bps. Raises
    `ValueError` when the side has no calibrated buckets at all OR
    when `qty` falls outside every bucket on the side. Silent
    zero-fallback would create a backtest-to-live fidelity hole;
    the calibration must cover every (side, qty) combination the
    caller intends to query.
    """

    _buckets_by_side: dict[OrderSide, tuple[SlippageBucket, ...]] = field(
        default_factory=dict,
    )
    _dt_seconds: int = 0

    @classmethod
    def calibrate(
        cls,
        *,
        trades: pl.DataFrame,
        side_buckets: Sequence[Decimal],
        dt_seconds: int,
    ) -> SlippageModel:
        """Return a fitted SlippageModel from `trades`.

        `trades` must carry columns:
          - `datetime` (Datetime, UTC),
          - `price` (Float64),
          - `quantity` (Float64),
          - `is_buyer_maker` (UInt8 / 0 or 1).

        `side_buckets` defines qty thresholds. e.g. `[0.001, 0.01,
        0.1]` produces four buckets: [0, 0.001), [0.001, 0.01),
        [0.01, 0.1), [0.1, ∞). Both BUY and SELL sides get the
        same bucket boundaries — different magnitude per side is
        the signal the calibration captures.

        `dt_seconds` is the rolling-mid lookback. Each trade's mid
        is the median of trade prices in the preceding `dt_seconds`,
        excluding the current trade. A trade with no preceding
        window (start-of-tape) is dropped.
        """
        if trades.is_empty():
            msg = (
                'SlippageModel.calibrate: empty trade tape; cannot fit '
                'any bucket. Widen the calibration window.'
            )
            raise ValueError(msg)
        # Sort by datetime — prerequisite for the rolling mid.
        sorted_trades = trades.sort('datetime')
        # Rolling median of `price` over the preceding `dt_seconds`.
        # Polars's `rolling_median_by` gives a left-inclusive window
        # which includes the current row; we subtract its
        # contribution by taking the median of the prior window
        # rolled forward by one step. The simplest correct path:
        # compute rolling median over (-dt_seconds, -1ns] explicitly
        # by shifting the window one row back.
        rolled = sorted_trades.with_columns(
            pl.col('price').rolling_median_by(
                'datetime',
                window_size=f'{dt_seconds}s',
                closed='left',  # excludes the current trade's price
            ).alias('mid_proxy'),
        ).drop_nulls('mid_proxy')
        if rolled.is_empty():
            msg = (
                f'SlippageModel.calibrate: rolling-mid window '
                f'{dt_seconds}s leaves zero usable rows. The window '
                f'is wider than the calibration tape.'
            )
            raise ValueError(msg)
        # Slippage in bps, signed by aggressor:
        #   BUY aggressor (is_buyer_maker=0) → price - mid (>= 0 in
        #     a rising window).
        #   SELL aggressor (is_buyer_maker=1) → price - mid (<= 0).
        # Both feed into per-side buckets directly without sign
        # inversion: apply() returns the bucket's median, which is
        # already signed by side.
        rolled = rolled.with_columns(
            ((pl.col('price') - pl.col('mid_proxy'))
             / pl.col('mid_proxy') * float(_BPS)).alias('slippage_bps'),
        )
        bucket_thresholds = sorted({Decimal(str(q)) for q in side_buckets})
        bucket_thresholds_sorted = sorted(bucket_thresholds)
        buckets_by_side: dict[OrderSide, list[SlippageBucket]] = {
            OrderSide.BUY: [],
            OrderSide.SELL: [],
        }
        # Group: aggressor side + qty bucket.
        for is_buyer_maker_raw, side in (
            (0, OrderSide.BUY), (1, OrderSide.SELL),
        ):
            side_rows = rolled.filter(
                pl.col('is_buyer_maker') == is_buyer_maker_raw,
            )
            qty_min = Decimal('0')
            for threshold in bucket_thresholds_sorted:
                bucket = side_rows.filter(
                    (pl.col('quantity') >= float(qty_min))
                    & (pl.col('quantity') < float(threshold)),
                )
                if not bucket.is_empty():
                    median_bps = Decimal(str(bucket['slippage_bps'].median()))
                    buckets_by_side[side].append(SlippageBucket(
                        side=side,
                        qty_min=qty_min,
                        qty_max=threshold,
                        median_bps=median_bps,
                        n_samples=len(bucket),
                    ))
                qty_min = threshold
            # Final open-ended bucket.
            tail = side_rows.filter(pl.col('quantity') >= float(qty_min))
            if not tail.is_empty():
                median_bps = Decimal(str(tail['slippage_bps'].median()))
                buckets_by_side[side].append(SlippageBucket(
                    side=side, qty_min=qty_min, qty_max=None,
                    median_bps=median_bps, n_samples=len(tail),
                ))
        return cls(
            _buckets_by_side={
                k: tuple(v) for k, v in buckets_by_side.items()
            },
            _dt_seconds=dt_seconds,
        )

    @property
    def dt_seconds(self) -> int:
        return self._dt_seconds

    def buckets_for(self, side: OrderSide) -> tuple[SlippageBucket, ...]:
        return self._buckets_by_side.get(side, ())

    def apply(
        self,
        *,
        side: OrderSide,
        qty: Decimal,
        mid: Decimal,
        t: datetime,
    ) -> Decimal:
        del mid, t  # static lookup; mid/t reserved for future
        # time-varying calibration (e.g. per-hour buckets).
        side_buckets = self._buckets_by_side.get(side, ())
        if not side_buckets:
            # No calibration for this side. The honest answer is
            # "we have no data" — silently returning 0 bps would
            # let an uncalibrated path trade live with zero
            # slippage (backtest-to-live fidelity hole). Codex
            # Task 12 round 1 pinned this gap.
            msg = (
                f'SlippageModel.apply: no calibrated buckets for '
                f'side={side.name}. The calibration window did not '
                f'cover this side; widen the window or recalibrate.'
            )
            raise ValueError(msg)
        for bucket in side_buckets:
            in_lower = qty >= bucket.qty_min
            in_upper = bucket.qty_max is None or qty < bucket.qty_max
            if in_lower and in_upper:
                return bucket.median_bps
        msg = (
            f'SlippageModel.apply: qty={qty} falls outside every '
            f'calibrated bucket for side={side.name}. The '
            f'calibration tape did not include this qty range; '
            f'widen the window or supply a smaller order.'
        )
        raise ValueError(msg)
