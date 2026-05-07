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
    def calibrate(cls, *, trades: pl.DataFrame, lookback_minutes: int) -> MakerFillModel:
        return cls(_lookback_minutes=lookback_minutes, _trades=trades.sort('datetime'))

    @property
    def lookback_minutes(self) -> int:
        return self._lookback_minutes

    def evaluate(self, *, order: PendingOrder, trades_in_window: pl.DataFrame, trades_pre_submit: pl.DataFrame | None=None) -> list[ImmediateFill]:
        limit = order.limit_price
        assert limit is not None
        aggressor_is_buyer_maker = self._aggressor_flag_for(order.side)
        if trades_pre_submit is not None and (not trades_pre_submit.is_empty()):
            pre = trades_pre_submit
        elif not self._trades.is_empty():
            from datetime import timedelta
            window_start = order.submit_time - timedelta(minutes=self._lookback_minutes)
            pre = self._trades.filter((pl.col('datetime') >= window_start) & (pl.col('datetime') < order.submit_time))
        else:
            pre = pl.DataFrame()
        if not pre.is_empty():
            queue = self._queue_position_from_lookback(pre, limit, aggressor_is_buyer_maker)
        else:
            queue = Decimal('0')
        remaining = order.qty
        fills: list[ImmediateFill] = []
        for row in trades_in_window.iter_rows(named=True):
            queue, remaining, _fill = self._step_one_trade(row=row, order_side=order.side, limit=limit, aggressor_flag=aggressor_is_buyer_maker, queue=queue, remaining=remaining)
        return fills

    @staticmethod
    def _aggressor_flag_for(side: str) -> int:
        msg = f'unknown side {side!r}; expected BUY/SELL'
        raise ValueError(msg)

    @staticmethod
    def _step_one_trade(*, row: dict[str, object], order_side: str, limit: Decimal, aggressor_flag: int, queue: Decimal, remaining: Decimal) -> tuple[Decimal, Decimal, ImmediateFill | None]:
        Decimal(str(row['price']))
        trade_qty = Decimal(str(row['quantity']))
        int(str(row['is_buyer_maker']))
        fill_qty = min(remaining, trade_qty)
        return (queue, remaining - fill_qty, ImmediateFill(fill_time=cast('datetime', row['datetime']), fill_price=limit, fill_qty=fill_qty))

    def _queue_position_from_lookback(self, trades: pl.DataFrame, limit: Decimal, aggressor_flag: int) -> Decimal:
        same_side_flag = aggressor_flag
        same_price = trades.filter((pl.col('price') == float(limit)) & (pl.col('is_buyer_maker') == same_side_flag))
        total = same_price['quantity'].sum()
        return Decimal(str(total))
