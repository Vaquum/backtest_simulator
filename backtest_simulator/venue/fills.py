"""FillModel — historical-trade matching with strict stop enforcement."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import polars as pl

from backtest_simulator.venue.filters import BinanceSpotFilters
from backtest_simulator.venue.types import FillModelConfig, FillResult, PendingOrder


def walk_trades(
    order: PendingOrder,
    trades: pl.DataFrame,
    config: FillModelConfig,
    filters: BinanceSpotFilters,
) -> list[FillResult]:
    """Match `order` against the historical `trades` stream. Stop enforcement is strict.

    MARKET walks trades from `submit_time + submit_latency` until qty is filled.
    LIMIT goes taker if crossed at submit, otherwise maker-queued; an incoming
    trade later at or beyond the limit fills the maker leg.
    STOP_LOSS / STOP_LOSS_LIMIT / TAKE_PROFIT: the trade stream triggers the
    stop the FIRST time it crosses `stop_price`. The fill price is
    `stop_price` rounded to tick_size; we DO NOT slide to wherever the bar
    closed. R = |entry - stop| * qty is honest only if this function's output
    never silently substitutes a different price.
    """
    submit_ts = order.submit_time + timedelta(milliseconds=config.submit_latency_ms)
    window = trades.filter(pl.col('time') >= submit_ts).sort('time')
    if window.is_empty():
        return []
    if order.order_type == 'MARKET':
        return _walk_market(order, window, filters)
    if order.order_type == 'LIMIT':
        return _walk_limit(order, window, filters)
    if order.order_type in {'STOP_LOSS', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT'}:
        return _walk_stop(order, window, filters)
    return []


def _walk_market(order: PendingOrder, window: pl.DataFrame, filters: BinanceSpotFilters) -> list[FillResult]:
    # Aggregate the tape walk into ONE VWAP fill rather than emitting
    # one FillResult per consumed tape tick. Two reasons:
    #   1. Real venues return a handful of fills for a typical MARKET
    #      order (one per price level hit on the book), not hundreds;
    #      emitting one per tape tick overstates fill granularity.
    #   2. Each FillResult generates a `FillReceived` event_spine append
    #      in Praxis — 283 appends per order blows past any reasonable
    #      drain budget. Aggregating keeps the backtest honest on total
    #      cost/qty (VWAP preserves both) while staying realistic on
    #      fill-event shape.
    remaining = order.qty
    consumed_qty = Decimal('0')
    consumed_notional = Decimal('0')
    last_time: datetime | None = None
    for row in window.iter_rows(named=True):
        if remaining <= 0:
            break
        take = min(remaining, Decimal(str(row['qty'])))
        px = Decimal(str(row['price']))
        consumed_qty += take
        consumed_notional += take * px
        last_time = _ts(row['time'])
        remaining -= take
    if consumed_qty == 0 or last_time is None:
        return []
    vwap = consumed_notional / consumed_qty
    return [FillResult(
        fill_time=last_time,
        fill_price=filters.round_price(vwap),
        fill_qty=filters.round_qty(consumed_qty),
        is_maker=False, reason='market_vwap',
    )]


def _walk_limit(order: PendingOrder, window: pl.DataFrame, filters: BinanceSpotFilters) -> list[FillResult]:
    if order.limit_price is None:
        return []
    first_px = Decimal(str(window.row(0, named=True)['price']))
    crossed = (order.side == 'BUY' and first_px <= order.limit_price) or (
        order.side == 'SELL' and first_px >= order.limit_price
    )
    if crossed:
        return _walk_market(order, window, filters)
    for row in window.iter_rows(named=True):
        px = Decimal(str(row['price']))
        cross = (order.side == 'BUY' and px <= order.limit_price) or (
            order.side == 'SELL' and px >= order.limit_price
        )
        if cross:
            return [FillResult(
                fill_time=_ts(row['time']),
                fill_price=filters.round_price(order.limit_price),
                fill_qty=filters.round_qty(order.qty), is_maker=True, reason='limit_maker',
            )]
    return []


def _walk_stop(order: PendingOrder, window: pl.DataFrame, filters: BinanceSpotFilters) -> list[FillResult]:
    if order.stop_price is None:
        return []
    stop = order.stop_price
    for row in window.iter_rows(named=True):
        px = Decimal(str(row['price']))
        triggered = (order.side == 'SELL' and px <= stop) or (order.side == 'BUY' and px >= stop)
        if triggered:
            return [FillResult(
                fill_time=_ts(row['time']),
                fill_price=filters.round_price(stop),
                fill_qty=filters.round_qty(order.qty), is_maker=False, reason='stop_trigger',
            )]
    return []


def _ts(value: object) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    msg = f'expected datetime in trades stream, got {type(value).__name__}'
    raise TypeError(msg)
