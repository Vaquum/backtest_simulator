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

def walk_trades(
    order: PendingOrder,
    trades: pl.DataFrame,
    config: FillModelConfig,
    filters: BinanceSpotFilters,
    context: WalkContext | None = None,
) -> list[FillResult]:
    submit_ts = order.submit_time + timedelta(milliseconds=config.submit_latency_ms)
    window = trades.filter(pl.col('time') >= submit_ts).sort('time')
    if window.is_empty():
        return []
    ctx = context if context is not None else WalkContext()
    if order.order_type == 'MARKET':
        return _walk_market(order, window, filters)
    if order.order_type == 'LIMIT':
        return _walk_limit(
            order, window, filters,
            maker_model=ctx.maker_model,
            trades_pre_submit=ctx.trades_pre_submit,
        )
    if order.order_type in {'STOP_LOSS', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT'}:
        return _walk_stop(order, window, filters, ctx.book_gap_instrument)
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
            breaches = (
                (order.side == 'BUY' and px <= stop_price)
                or (order.side == 'SELL' and px >= stop_price)
            )
            if breaches:
                stop_halted = True
                break
        take = min(remaining, Decimal(str(row['qty'])))
        consumed_qty += take
        consumed_notional += take * px
        last_time = _ts(row['time'])
        remaining -= take
    if consumed_qty == 0 or last_time is None:
        return []
    vwap = consumed_notional / consumed_qty
    reason = 'market_stop_halted' if stop_halted else 'market_vwap'
    return [FillResult(
        fill_time=last_time,
        fill_price=filters.round_price(vwap),
        fill_qty=filters.round_qty(consumed_qty),
        is_maker=False, reason=reason,
    )]

def _walk_limit(
    order: PendingOrder,
    window: pl.DataFrame,
    filters: BinanceSpotFilters,
    *,
    maker_model: MakerFillModel | None = None,
    trades_pre_submit: pl.DataFrame | None = None,
) -> list[FillResult]:
    if order.limit_price is None:
        return []
    if maker_model is None:
        return _walk_limit_legacy(order, window, filters)
    renamed = window.rename({'time': 'datetime', 'qty': 'quantity'})
    pre = (
        None if trades_pre_submit is None or trades_pre_submit.is_empty()
        else trades_pre_submit.rename({'time': 'datetime', 'qty': 'quantity'})
    )
    immediates = maker_model.evaluate(
        order=order, trades_in_window=renamed,
        trades_pre_submit=pre,
    )
    if not immediates:
        return []
    total_qty = sum((imm.fill_qty for imm in immediates), Decimal('0'))
    if total_qty <= Decimal('0'):
        return []
    total_notional = sum(
        (imm.fill_qty * imm.fill_price for imm in immediates),
        Decimal('0'),
    )
    vwap_price = total_notional / total_qty
    return [FillResult(
        fill_time=immediates[-1].fill_time,
        fill_price=filters.round_price(vwap_price),
        fill_qty=filters.round_qty(total_qty),
        is_maker=True, reason='limit_maker',
    )]

def _walk_limit_legacy(
    order: PendingOrder,
    window: pl.DataFrame,
    filters: BinanceSpotFilters,
) -> list[FillResult]:
    if order.limit_price is None:
        return []
    first_px = Decimal(str(window.row(0, named=True)['price']))
    crossed = (order.side == 'BUY' and first_px <= order.limit_price) or (
        order.side == 'SELL' and first_px >= order.limit_price
    )
    if crossed:
        taker_fills = _walk_market(order, window, filters)
        return [
            FillResult(
                fill_time=f.fill_time, fill_price=f.fill_price,
                fill_qty=f.fill_qty, is_maker=False,
                reason='limit_taker',
            )
            for f in taker_fills
        ]
    for row in window.iter_rows(named=True):
        px = Decimal(str(row['price']))
        cross = (order.side == 'BUY' and px <= order.limit_price) or (
            order.side == 'SELL' and px >= order.limit_price
        )
        if cross:
            return [FillResult(
                fill_time=_ts(row['time']),
                fill_price=filters.round_price(order.limit_price),
                fill_qty=filters.round_qty(order.qty),
                is_maker=True, reason='limit_maker',
            )]
    return []

def _walk_stop(
    order: PendingOrder,
    window: pl.DataFrame,
    filters: BinanceSpotFilters,
    book_gap_instrument: BookGapInstrument | None = None,
) -> list[FillResult]:
    if order.stop_price is None:
        return []
    stop = order.stop_price
    prev_time: datetime | None = None
    for row in window.iter_rows(named=True):
        px = Decimal(str(row['price']))
        triggered = (order.side == 'SELL' and px <= stop) or (order.side == 'BUY' and px >= stop)
        if triggered:
            row_time = _ts(row['time'])
            if book_gap_instrument is not None:
                book_gap_instrument.record_stop_cross(
                    t_cross=prev_time if prev_time is not None else row_time,
                    t_first_trade=row_time,
                )
            return [FillResult(
                fill_time=row_time,
                fill_price=filters.round_price(px),
                fill_qty=filters.round_qty(order.qty), is_maker=False, reason='stop_trigger',
            )]
        prev_time = _ts(row['time'])
    return []

def _ts(value: object) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    msg = f'expected datetime in trades stream, got {type(value).__name__}'
    raise TypeError(msg)
