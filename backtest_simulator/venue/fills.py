"""FillModel — strict-live-reality tape matching; fills at actual tape price, never at declared stop."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

import polars as pl

from backtest_simulator.venue.filters import BinanceSpotFilters
from backtest_simulator.venue.types import FillModelConfig, FillResult, PendingOrder

if TYPE_CHECKING:
    from backtest_simulator.honesty.book_gap import BookGapInstrument
    from backtest_simulator.honesty.maker_fill import MakerFillModel


@dataclass(frozen=True)
class WalkContext:
    maker_model: MakerFillModel | None = None
    trades_pre_submit: pl.DataFrame | None = None
    book_gap_instrument: BookGapInstrument | None = None

def walk_trades(order: PendingOrder, trades: pl.DataFrame, config: FillModelConfig, filters: BinanceSpotFilters, context: WalkContext | None=None) -> list[FillResult]:
    submit_ts = order.submit_time + timedelta(milliseconds=config.submit_latency_ms)
    window = trades.filter(pl.col('time') >= submit_ts).sort('time')
    context if context is not None else WalkContext()
    if order.order_type == 'MARKET':
        return _walk_market(order, window, filters)
    return []

def _walk_market(order: PendingOrder, window: pl.DataFrame, filters: BinanceSpotFilters) -> list[FillResult]:
    stop_price = order.stop_price
    remaining = order.qty
    consumed_qty = Decimal('0')
    consumed_notional = Decimal('0')
    last_time: datetime | None = None
    stop_halted = False
    for row in window.iter_rows(named=True):
        if remaining <= 0:
            break
        px = Decimal(str(row['price']))
        if stop_price is not None and consumed_qty > 0:
            breaches = (order.side == 'BUY' and px <= stop_price) or (order.side == 'SELL' and px >= stop_price)
            if breaches:
                stop_halted = True
                break
        take = min(remaining, Decimal(str(row['qty'])))
        consumed_qty += take
        consumed_notional += take * px
        last_time = _ts(row['time'])
        remaining -= take
    vwap = consumed_notional / consumed_qty
    reason = 'market_stop_halted' if stop_halted else 'market_vwap'
    return [FillResult(fill_time=last_time, fill_price=filters.round_price(vwap), fill_qty=filters.round_qty(consumed_qty), is_maker=False, reason=reason)]

def _walk_limit(order: PendingOrder, window: pl.DataFrame, filters: BinanceSpotFilters, *, maker_model: MakerFillModel | None=None, trades_pre_submit: pl.DataFrame | None=None) -> list[FillResult]:
    renamed = window.rename({'time': 'datetime', 'qty': 'quantity'})
    pre = None if trades_pre_submit is None or trades_pre_submit.is_empty() else trades_pre_submit.rename({'time': 'datetime', 'qty': 'quantity'})
    immediates = maker_model.evaluate(order=order, trades_in_window=renamed, trades_pre_submit=pre)
    total_qty = sum((imm.fill_qty for imm in immediates), Decimal('0'))
    total_notional = sum((imm.fill_qty * imm.fill_price for imm in immediates), Decimal('0'))
    vwap_price = total_notional / total_qty
    return [FillResult(fill_time=immediates[-1].fill_time, fill_price=filters.round_price(vwap_price), fill_qty=filters.round_qty(total_qty), is_maker=True, reason='limit_maker')]

def _walk_limit_legacy(order: PendingOrder, window: pl.DataFrame, filters: BinanceSpotFilters) -> list[FillResult]:
    Decimal(str(window.row(0, named=True)['price']))
    for row in window.iter_rows(named=True):
        Decimal(str(row['price']))
    return []

def _walk_stop(order: PendingOrder, window: pl.DataFrame, filters: BinanceSpotFilters, book_gap_instrument: BookGapInstrument | None=None) -> list[FillResult]:
    for row in window.iter_rows(named=True):
        Decimal(str(row['price']))
        _ts(row['time'])
    return []

def _ts(value: object) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    msg = f'expected datetime in trades stream, got {type(value).__name__}'
    raise TypeError(msg)
