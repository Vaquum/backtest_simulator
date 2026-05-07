"""Internal helpers for `SimulatedVenueAdapter` — account bookkeeping + validators."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal

from praxis.core.domain.enums import OrderSide, OrderStatus, OrderType
from praxis.infrastructure.venue_adapter import ImmediateFill, VenueOrder, VenueTrade

from backtest_simulator.venue.fees import FeeSchedule
from backtest_simulator.venue.filters import BinanceSpotFilters
from backtest_simulator.venue.types import FillResult, PendingOrder

TYPE_MAP: dict[OrderType, str] = {OrderType.MARKET: 'MARKET', OrderType.LIMIT: 'LIMIT', OrderType.LIMIT_IOC: 'LIMIT', OrderType.STOP: 'STOP_LOSS', OrderType.STOP_LIMIT: 'STOP_LOSS_LIMIT', OrderType.TAKE_PROFIT: 'TAKE_PROFIT', OrderType.TP_LIMIT: 'TAKE_PROFIT', OrderType.OCO: 'LIMIT'}

@dataclass
class Account:
    account_id: str
    api_key: str
    api_secret: str
    balances: dict[str, Decimal] = field(default_factory=lambda: dict[str, Decimal]())
    locked: dict[str, Decimal] = field(default_factory=lambda: dict[str, Decimal]())
    trades: list[VenueTrade] = field(default_factory=lambda: list[VenueTrade]())
    orders: dict[str, VenueOrder] = field(default_factory=lambda: dict[str, VenueOrder]())

def quote_asset(symbol: str) -> str:
    return 'USDT' if symbol.endswith('USDT') else symbol[-3:]

def window_seconds(seconds: int) -> timedelta:
    return timedelta(seconds=seconds)

def reject_reason(order: PendingOrder, filters: BinanceSpotFilters, price: Decimal | None) -> str | None:
    reason = filters.validate(order.qty, price)
    if reason is not None:
        return reason
    if order.stop_price is not None and order.order_type != 'MARKET':
        reason = filters.validate(order.qty, order.stop_price)
        if reason is not None:
            return reason
    if price is None and order.order_type == 'MARKET' and (order.stop_price is not None):
        notional = order.qty * order.stop_price
        if notional < filters.min_notional:
            return f'MIN_NOTIONAL: {notional} < {filters.min_notional}'
    return None

def record_rejection(account: Account, order: PendingOrder, client_order_id: str, side: OrderSide, order_type: OrderType, price: Decimal | None) -> None:
    account.orders[order.order_id] = VenueOrder(venue_order_id=order.order_id, client_order_id=client_order_id, status=OrderStatus.REJECTED, symbol=order.symbol, side=side, order_type=order_type, qty=order.qty, filled_qty=Decimal('0'), price=price)

@dataclass(frozen=True)
class OrderIdentity:
    venue_order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide

def record_fills(account: Account, fees: FeeSchedule, identity: OrderIdentity, fills: list[FillResult], next_trade_id_fn: Callable[[], str]) -> tuple[ImmediateFill, ...]:
    recorded: list[ImmediateFill] = []
    symbol = identity.symbol
    for fill in fills:
        fee = fees.fee(symbol, fill.fill_price * fill.fill_qty, is_maker=fill.is_maker)
        trade_id = next_trade_id_fn()
        recorded.append(ImmediateFill(venue_trade_id=trade_id, qty=fill.fill_qty, price=fill.fill_price, fee=fee, fee_asset=quote_asset(symbol), is_maker=fill.is_maker))
        account.trades.append(VenueTrade(venue_trade_id=trade_id, venue_order_id=identity.venue_order_id, client_order_id=identity.client_order_id, symbol=symbol, side=identity.side, qty=fill.fill_qty, price=fill.fill_price, fee=fee, fee_asset=quote_asset(symbol), is_maker=fill.is_maker, timestamp=fill.fill_time))
    return tuple(recorded)
