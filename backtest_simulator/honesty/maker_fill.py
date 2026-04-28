"""Passive maker-fill realism — queue position + partial fills + aggressor bound."""
from __future__ import annotations

# Slice #17 Task 14: a passive (LIMIT-not-marketable) order sits in
# the book queue at its limit price. It fills only when the book
# trades AT the limit price AND the order's queue position is
# reached. The simulator that fills every limit order at submit
# price is silently producing alpha that cannot replicate live.
#
# Realism factors captured here:
#   - Queue position: a maker arriving at a price level joins the
#     back of the queue. Initial queue_position = sum of trade qty
#     at the same price level over `lookback_minutes` preceding
#     submit time (proxy for "qty already resting at this level").
#   - Aggressor matching: each counter-side trade at the maker's
#     limit price reduces queue_position by the aggressor qty.
#   - Partial fills: when queue_position reaches zero, the maker
#     starts filling against subsequent aggressors. Each aggressor
#     fills min(remaining_qty, aggressor_qty); the maker may
#     partial-fill across multiple aggressors.
#   - Aggressor-side rule: a BUY maker fills against SELL
#     aggressors (is_buyer_maker=1 trades). A SELL maker fills
#     against BUY aggressors (is_buyer_maker=0 trades).
#
# The model deliberately omits price-time priority bumps,
# self-trade prevention, and book replenishment — those are M3
# refinements. The MVC pin is that a far-away maker NEVER fills
# (Task 15) and a near-touch maker fills realistically (Task 14).
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import cast

import polars as pl

from backtest_simulator.venue.types import PendingOrder


@dataclass(frozen=True)
class ImmediateFill:
    """One partial / full fill produced by `MakerFillModel.evaluate`."""

    fill_time: datetime
    fill_price: Decimal
    fill_qty: Decimal


@dataclass
class MakerFillModel:
    """Queue-position tracker for passive LIMIT orders.

    `lookback_minutes` is the window used to estimate the initial
    queue position when a maker order arrives. Larger windows give
    more conservative (deeper) initial queues; smaller windows
    give more optimistic ones. The operator picks based on the
    symbol's typical book depth dynamics.

    `_trades` carries the calibration tape so `evaluate()` can
    compute the pre-submit lookback slice automatically — the
    spec call path (`evaluate(*, order, trades_in_window)`) does
    not pass `trades_pre_submit` itself. Codex Task 14 round 1
    pinned this gap.
    """

    _lookback_minutes: int = 0
    _trades: pl.DataFrame = field(default_factory=pl.DataFrame)

    @classmethod
    def calibrate(
        cls,
        *,
        trades: pl.DataFrame,
        lookback_minutes: int,
    ) -> MakerFillModel:
        """Calibrate the lookback window and store the tape for evaluate.

        The stored `trades` is sliced per-evaluate to compute the
        `[submit_time - lookback_minutes, submit_time)` window of
        same-side trades that establish the maker's initial queue
        position. Queue-realistic behavior requires the calibration
        tape to be available at evaluate time without depending on
        an out-of-spec second argument.
        """
        if lookback_minutes <= 0:
            msg = (
                f'MakerFillModel.calibrate: lookback_minutes must be '
                f'positive, got {lookback_minutes}.'
            )
            raise ValueError(msg)
        if trades.is_empty():
            msg = (
                'MakerFillModel.calibrate: empty trade tape. The '
                'lookback estimate would have no data to work from.'
            )
            raise ValueError(msg)
        return cls(
            _lookback_minutes=lookback_minutes,
            _trades=trades.sort('datetime'),
        )

    @property
    def lookback_minutes(self) -> int:
        return self._lookback_minutes

    def evaluate(
        self,
        *,
        order: PendingOrder,
        trades_in_window: pl.DataFrame,
        trades_pre_submit: pl.DataFrame | None = None,
    ) -> list[ImmediateFill]:
        """Walk `trades_in_window` and return the maker's fill schedule.

        `order.limit_price` is the maker's resting price.
        `trades_in_window` are trades from `submit_time` until
        cancel/expiry; assumed sorted by datetime.
        `trades_pre_submit` is the pre-submit lookback used to
        estimate the initial queue position. If None, queue starts
        at zero (best-case maker — used by sanity tests that pin
        far-away non-fills).
        """
        if order.limit_price is None:
            msg = (
                f'MakerFillModel.evaluate: order {order.order_id} has '
                f'no limit_price; passive-maker logic only applies to '
                f'LIMIT orders.'
            )
            raise ValueError(msg)
        limit = order.limit_price
        aggressor_is_buyer_maker = self._aggressor_flag_for(order.side)
        # Caller-supplied pre-submit slice wins — that's the path the
        # tests use to inject deterministic queue depths. Otherwise
        # fall back to the calibration tape automatically: same-side
        # trades at the limit price within the lookback window.
        # Codex Task 14 round 1 pinned the spec-call-path gap (the
        # spec evaluate signature has no trades_pre_submit, so the
        # public API must self-serve from the stored tape).
        if trades_pre_submit is not None and not trades_pre_submit.is_empty():
            pre = trades_pre_submit
        elif not self._trades.is_empty():
            from datetime import timedelta
            window_start = order.submit_time - timedelta(
                minutes=self._lookback_minutes,
            )
            pre = self._trades.filter(
                (pl.col('datetime') >= window_start)
                & (pl.col('datetime') < order.submit_time),
            )
        else:
            pre = pl.DataFrame()
        if not pre.is_empty():
            queue = self._queue_position_from_lookback(
                pre, limit, aggressor_is_buyer_maker,
            )
        else:
            queue = Decimal('0')
        remaining = order.qty
        fills: list[ImmediateFill] = []
        if trades_in_window.is_empty():
            return fills
        for row in trades_in_window.iter_rows(named=True):
            if remaining <= Decimal('0'):
                break
            queue, remaining, fill = self._step_one_trade(
                row=row, order_side=order.side, limit=limit,
                aggressor_flag=aggressor_is_buyer_maker,
                queue=queue, remaining=remaining,
            )
            if fill is not None:
                fills.append(fill)
        return fills

    @staticmethod
    def _aggressor_flag_for(side: str) -> int:
        # Aggressor side that fills this maker.
        # BUY maker → fills against SELL aggressors → is_buyer_maker=1.
        # SELL maker → fills against BUY aggressors → is_buyer_maker=0.
        if side == 'BUY':
            return 1
        if side == 'SELL':
            return 0
        msg = f'unknown side {side!r}; expected BUY/SELL'
        raise ValueError(msg)

    @staticmethod
    def _step_one_trade(
        *,
        row: dict[str, object],
        order_side: str,
        limit: Decimal,
        aggressor_flag: int,
        queue: Decimal,
        remaining: Decimal,
    ) -> tuple[Decimal, Decimal, ImmediateFill | None]:
        # Maker only fills when:
        #   - aggressor side matches, AND
        #   - the trade price has reached the maker's limit:
        #     BUY maker fills if trade_price <= limit;
        #     SELL maker fills if trade_price >= limit.
        trade_price = Decimal(str(row['price']))
        trade_qty = Decimal(str(row['quantity']))
        trade_aggressor = int(str(row['is_buyer_maker']))
        side_matches = trade_aggressor == aggressor_flag
        price_matches = (
            (order_side == 'BUY' and trade_price <= limit)
            or (order_side == 'SELL' and trade_price >= limit)
        )
        if not (side_matches and price_matches):
            return queue, remaining, None
        if queue > Decimal('0'):
            consumed = min(queue, trade_qty)
            queue -= consumed
            trade_qty -= consumed
            if trade_qty <= Decimal('0'):
                return queue, remaining, None
        fill_qty = min(remaining, trade_qty)
        return (
            queue,
            remaining - fill_qty,
            ImmediateFill(
                fill_time=cast('datetime', row['datetime']),
                fill_price=limit,
                fill_qty=fill_qty,
            ),
        )

    def _queue_position_from_lookback(
        self,
        trades: pl.DataFrame,
        limit: Decimal,
        aggressor_flag: int,
    ) -> Decimal:
        # Initial queue: sum of SAME-SIDE liquidity at the maker's
        # limit price. A SELL maker inherits prior seller-maker
        # liquidity (is_buyer_maker=0); a BUY maker inherits prior
        # buyer-maker liquidity (is_buyer_maker=1). The maker side's
        # is_buyer_maker is the *opposite* of the aggressor flag we
        # match in evaluate(): BUY maker has aggressor_flag=1 and
        # inherits is_buyer_maker=1 prior trades. Codex Task 14
        # round 1 pinned the side-blind sum gap.
        same_side_flag = aggressor_flag
        same_price = trades.filter(
            (pl.col('price') == float(limit))
            & (pl.col('is_buyer_maker') == same_side_flag),
        )
        if same_price.is_empty():
            return Decimal('0')
        total = same_price['quantity'].sum()
        return Decimal(str(total))
