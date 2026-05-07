"""Passive maker-fill realism — queue position + partial fills + aggressor bound."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import cast

import polars as pl

from backtest_simulator.venue.types import PendingOrder


@dataclass(frozen=True)
class ImmediateFill:

    fill_time: datetime
    fill_price: Decimal
    fill_qty: Decimal

@dataclass
class MakerFillModel:

    _lookback_minutes: int = 0
    _trades: pl.DataFrame = field(default_factory=pl.DataFrame)

    @classmethod
    def calibrate(
        cls,
        *,
        trades: pl.DataFrame,
        lookback_minutes: int,
    ) -> MakerFillModel:
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
        if order.limit_price is None:
            msg = (
                f'MakerFillModel.evaluate: order {order.order_id} has '
                f'no limit_price; passive-maker logic only applies to '
                f'LIMIT orders.'
            )
            raise ValueError(msg)
        limit = order.limit_price
        aggressor_is_buyer_maker = self._aggressor_flag_for(order.side)
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
        same_side_flag = aggressor_flag
        same_price = trades.filter(
            (pl.col('price') == float(limit))
            & (pl.col('is_buyer_maker') == same_side_flag),
        )
        if same_price.is_empty():
            return Decimal('0')
        total = same_price['quantity'].sum()
        return Decimal(str(total))
