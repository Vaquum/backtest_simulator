"""Internal helpers for `SimulatedVenueAdapter` — account bookkeeping + validators."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal

from praxis.core.domain.enums import OrderSide, OrderStatus, OrderType
from praxis.infrastructure.venue_adapter import (
    ImmediateFill,
    NotFoundError,
    VenueOrder,
    VenueTrade,
)

from backtest_simulator.venue.fees import FeeSchedule
from backtest_simulator.venue.filters import BinanceSpotFilters
from backtest_simulator.venue.types import FillResult, PendingOrder

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


def zero_fill_status(order_type: str, time_in_force: str) -> OrderStatus:
    """Pick the terminal status for a validated-but-unfilled order.

    Live Binance keeps any GTC order on the book until it triggers,
    crosses, or is cancelled — that includes LIMIT, STOP_LOSS,
    STOP_LOSS_LIMIT, TAKE_PROFIT (and family). Only MARKET (which
    never rests by definition) and IOC / FOK / GTX (which expire on
    no immediate execution) terminate as EXPIRED on a zero-fill
    window. Returning OPEN for the GTC family lets `query_open_orders`
    surface still-live orders correctly; the simulator does NOT yet
    re-attempt fills on subsequent windows for resting orders, but
    the status reporting must match live's contract for backtest ≡
    paper ≡ live status parity.
    """
    if order_type == 'MARKET':
        return OrderStatus.EXPIRED
    if time_in_force.upper() == 'GTC':
        return OrderStatus.OPEN
    return OrderStatus.EXPIRED


@dataclass
class Account:
    """Per-account bookkeeping kept by the simulated venue."""

    account_id: str
    api_key: str
    api_secret: str
    balances: dict[str, Decimal] = field(
        default_factory=lambda: dict[str, Decimal](),
    )
    locked: dict[str, Decimal] = field(
        default_factory=lambda: dict[str, Decimal](),
    )
    trades: list[VenueTrade] = field(
        default_factory=lambda: list[VenueTrade](),
    )
    orders: dict[str, VenueOrder] = field(
        default_factory=lambda: dict[str, VenueOrder](),
    )


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
    # `filters.validate(qty, price)` does the qty-side LOT_SIZE checks
    # always, plus the price-side PRICE_FILTER + MIN_NOTIONAL when a
    # `price` is given. The stop_price branch below is split by
    # order_type because Binance treats stop_price differently:
    #   - MARKET: stop_price is a backtest-side risk anchor (the fill
    #     engine halts on breach). The venue NEVER sees it as a quoted
    #     price, so PRICE_FILTER's tick_size rule does not apply — only
    #     a min-notional sanity check. (Pre-fix `reference =
    #     price or order.stop_price` would reject a MARKET BUY just
    #     because its kelly-derived risk stop fell on a sub-tick
    #     boundary like 69649.9876.)
    #   - STOP_LOSS / STOP_LOSS_LIMIT / TAKE_PROFIT (and the LIMIT
    #     variants): stop_price IS the venue-quoted trigger, so it
    #     must be tick-aligned. Live Binance rejects a sub-tick
    #     stopPrice; backtest must too, otherwise paper/live diverge.
    reason = filters.validate(order.qty, price)
    if reason is not None:
        return reason
    # Venue-quoted stop trigger: must be tick-aligned (and notional-sane
    # on its own) for STOP_LOSS / STOP_LOSS_LIMIT / TAKE_PROFIT / TP_LIMIT.
    # The check runs whether or not a limit price was also given —
    # STOP_LOSS_LIMIT carries BOTH a tick-validated price AND a tick-
    # validated stop_price, and live Binance rejects either if sub-tick.
    if order.stop_price is not None and order.order_type != 'MARKET':
        reason = filters.validate(order.qty, order.stop_price)
        if reason is not None:
            return reason
    # MARKET-with-stop: stop is a risk anchor, no tick. When no limit
    # price was given, still apply a min-notional sanity check via the
    # stop reference.
    if price is None and order.order_type == 'MARKET' and order.stop_price is not None:
        notional = order.qty * order.stop_price
        if notional < filters.min_notional:
            return f'MIN_NOTIONAL: {notional} < {filters.min_notional}'
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


@dataclass(frozen=True)
class OrderIdentity:
    """Identifying fields a recorded fill ties to.

    Bundles the four per-order identifier/side fields so `record_fills`
    takes one argument for them instead of four.
    """

    venue_order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide


def record_fills(
    account: Account, fees: FeeSchedule, identity: OrderIdentity,
    fills: list[FillResult], next_trade_id_fn: Callable[[], str],
) -> tuple[ImmediateFill, ...]:
    recorded: list[ImmediateFill] = []
    symbol = identity.symbol
    for fill in fills:
        fee = fees.fee(symbol, fill.fill_price * fill.fill_qty, is_maker=fill.is_maker)
        trade_id = next_trade_id_fn()
        recorded.append(ImmediateFill(
            venue_trade_id=trade_id, qty=fill.fill_qty, price=fill.fill_price,
            fee=fee, fee_asset=quote_asset(symbol), is_maker=fill.is_maker,
        ))
        account.trades.append(VenueTrade(
            venue_trade_id=trade_id, venue_order_id=identity.venue_order_id,
            client_order_id=identity.client_order_id,
            symbol=symbol, side=identity.side,
            qty=fill.fill_qty, price=fill.fill_price,
            fee=fee, fee_asset=quote_asset(symbol),
            is_maker=fill.is_maker, timestamp=fill.fill_time,
        ))
    return tuple(recorded)
