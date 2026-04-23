"""Internal helpers for `SimulatedVenueAdapter` — account bookkeeping + validators."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal
from typing import Any

from praxis.core.domain.enums import OrderSide, OrderStatus, OrderType
from praxis.infrastructure.venue_adapter import (
    ImmediateFill,
    NotFoundError,
    VenueOrder,
    VenueTrade,
)

from backtest_simulator.venue.fees import FeeSchedule
from backtest_simulator.venue.filters import BinanceSpotFilters
from backtest_simulator.venue.types import PendingOrder

TYPE_MAP: dict[OrderType, str] = {
    OrderType.MARKET: 'MARKET',
    OrderType.LIMIT: 'LIMIT',
    OrderType.LIMIT_IOC: 'LIMIT',
    OrderType.STOP: 'STOP_LOSS',
    OrderType.STOP_LIMIT: 'STOP_LOSS_LIMIT',
    OrderType.TAKE_PROFIT: 'TAKE_PROFIT',
    OrderType.TP_LIMIT: 'TAKE_PROFIT',
    OrderType.OCO: 'LIMIT',
}


@dataclass
class Account:
    """Per-account bookkeeping kept by the simulated venue."""

    account_id: str
    api_key: str
    api_secret: str
    balances: dict[str, Decimal] = field(default_factory=dict)
    locked: dict[str, Decimal] = field(default_factory=dict)
    trades: list[VenueTrade] = field(default_factory=list)
    orders: dict[str, VenueOrder] = field(default_factory=dict)


def quote_asset(symbol: str) -> str:
    """USDT-quoted pairs end with USDT; else the last 3 chars are the quote."""
    return 'USDT' if symbol.endswith('USDT') else symbol[-3:]


def window_seconds(seconds: int) -> timedelta:
    return timedelta(seconds=seconds)


def resolve_order(
    account: Account, venue_order_id: str | None, client_order_id: str | None,
) -> VenueOrder:
    if venue_order_id is not None and venue_order_id in account.orders:
        return account.orders[venue_order_id]
    if client_order_id is not None:
        for o in account.orders.values():
            if o.client_order_id == client_order_id:
                return o
    msg = f'order not found: venue_order_id={venue_order_id} client_order_id={client_order_id}'
    raise NotFoundError(msg)


def reject_reason(
    order: PendingOrder, filters: BinanceSpotFilters, price: Decimal | None,
) -> str | None:
    reference = price or order.stop_price
    if reference is not None:
        return filters.validate(order.qty, reference)
    if order.qty < filters.min_qty:
        return f'LOT_SIZE: qty {order.qty} < min_qty {filters.min_qty}'
    if order.qty > filters.max_qty:
        return f'LOT_SIZE: qty {order.qty} > max_qty {filters.max_qty}'
    return None


def record_rejection(
    account: Account, order: PendingOrder, client_order_id: str,
    side: OrderSide, order_type: OrderType, price: Decimal | None,
) -> None:
    account.orders[order.order_id] = VenueOrder(
        venue_order_id=order.order_id, client_order_id=client_order_id,
        status=OrderStatus.REJECTED, symbol=order.symbol, side=side,
        order_type=order_type, qty=order.qty,
        filled_qty=Decimal('0'), price=price,
    )


def record_fills(  # noqa: PLR0913 - recorder; every arg carries schema meaning
    account: Account, fees: FeeSchedule, venue_order_id: str, client_order_id: str,
    symbol: str, side: OrderSide, fills: list[Any], next_trade_id_fn: Callable[[], str],
) -> tuple[ImmediateFill, ...]:
    recorded: list[ImmediateFill] = []
    for fill in fills:
        fee = fees.fee(symbol, fill.fill_price * fill.fill_qty, is_maker=fill.is_maker)
        trade_id = next_trade_id_fn()
        recorded.append(ImmediateFill(
            venue_trade_id=trade_id, qty=fill.fill_qty, price=fill.fill_price,
            fee=fee, fee_asset=quote_asset(symbol), is_maker=fill.is_maker,
        ))
        account.trades.append(VenueTrade(
            venue_trade_id=trade_id, venue_order_id=venue_order_id,
            client_order_id=client_order_id, symbol=symbol, side=side,
            qty=fill.fill_qty, price=fill.fill_price,
            fee=fee, fee_asset=quote_asset(symbol),
            is_maker=fill.is_maker, timestamp=fill.fill_time,
        ))
    return tuple(recorded)
