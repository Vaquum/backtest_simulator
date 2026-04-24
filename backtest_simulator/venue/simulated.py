"""SimulatedVenueAdapter — real Praxis VenueAdapter Protocol against historical trades."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal

from praxis.core.domain.enums import OrderSide, OrderStatus, OrderType
from praxis.core.domain.health_snapshot import HealthSnapshot
from praxis.infrastructure.venue_adapter import (
    BalanceEntry,
    CancelResult,
    ExecutionReport,
    NotFoundError,
    OrderBookLevel,
    OrderBookSnapshot,
    SubmitResult,
    VenueOrder,
    VenueTrade,
)
from praxis.infrastructure.venue_adapter import (
    SymbolFilters as PraxisSymbolFilters,
)

from backtest_simulator.feed.protocol import HistoricalFeed
from backtest_simulator.venue import _adapter_internals as _I
from backtest_simulator.venue.fees import FeeSchedule
from backtest_simulator.venue.fills import walk_trades
from backtest_simulator.venue.filters import BinanceSpotFilters
from backtest_simulator.venue.types import FillModelConfig, PendingOrder


class SimulatedVenueAdapter:
    """VenueAdapter Protocol implementation backed by historical trades."""

    def __init__(
        self,
        feed: HistoricalFeed,
        filters: BinanceSpotFilters,
        fees: FeeSchedule,
        fill_config: FillModelConfig | None = None,
        trade_window_seconds: int = 3600,
    ) -> None:
        self._feed = feed
        self._filters = filters
        self._fees = fees
        self._fill_config = fill_config or FillModelConfig()
        self._trade_window_seconds = trade_window_seconds
        self._accounts: dict[str, _I.Account] = {}
        self._symbol_filters: dict[str, BinanceSpotFilters] = {filters.symbol: filters}
        self._next_order_seq = 1
        self._next_trade_seq = 1
        self._history: dict[str, _I.Account] = {}

    def register_account(self, account_id: str, api_key: str, api_secret: str) -> None:
        # If the same account_id was registered + unregistered + re-registered,
        # recover the prior Account so fill/order history survives the cycle.
        # Praxis's shutdown path unregisters the account during normal
        # teardown; post-run inspection needs the trades to still exist.
        account = self._history.pop(account_id, None) or _I.Account(
            account_id=account_id, api_key=api_key, api_secret=api_secret,
        )
        self._accounts[account_id] = account

    def unregister_account(self, account_id: str) -> None:
        if account_id not in self._accounts:
            msg = f'account_id {account_id!r} not registered'
            raise KeyError(msg)
        self._history[account_id] = self._accounts.pop(account_id)

    def history(self, account_id: str) -> _I.Account:
        """Return the Account (orders + trades) whether currently registered or not."""
        if account_id in self._accounts:
            return self._accounts[account_id]
        if account_id in self._history:
            return self._history[account_id]
        msg = f'account_id {account_id!r} never registered'
        raise KeyError(msg)

    async def submit_order(
        self,
        account_id: str,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        qty: Decimal,
        *,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        stop_limit_price: Decimal | None = None,
        client_order_id: str | None = None,
        time_in_force: str | None = None,
    ) -> SubmitResult:
        del stop_limit_price  # OCO/stop-limit path not implemented in the simulated fill engine
        account = self._require_account(account_id)
        venue_order_id = self._mint_order_id()
        coid = client_order_id or f'BTS-{venue_order_id}'
        order = PendingOrder(
            order_id=venue_order_id, side=side.name, order_type=_I.TYPE_MAP[order_type],
            qty=qty, limit_price=price, stop_price=stop_price,
            time_in_force=time_in_force or 'GTC', submit_time=self._now(), symbol=symbol,
        )
        if _I.reject_reason(order, self._filters, price) is not None:
            _I.record_rejection(account, order, coid, side, order_type, price)
            return SubmitResult(
                venue_order_id=venue_order_id, status=OrderStatus.REJECTED, immediate_fills=(),
            )
        trades = self._feed.get_trades(
            symbol, order.submit_time,
            order.submit_time + _I.window_seconds(self._trade_window_seconds),
            venue_lookahead_seconds=self._trade_window_seconds,
        )
        fills = walk_trades(order, trades, self._fill_config, self._filters)
        immediate = _I.record_fills(
            account, self._fees,
            _I.OrderIdentity(
                venue_order_id=venue_order_id, client_order_id=coid,
                symbol=symbol, side=side,
            ),
            fills, self._mint_trade_id,
        )
        filled_qty = sum((f.qty for f in immediate), Decimal('0'))
        status = (
            OrderStatus.FILLED if filled_qty >= qty
            else OrderStatus.PARTIALLY_FILLED if filled_qty > 0
            else OrderStatus.REJECTED
        )
        account.orders[venue_order_id] = VenueOrder(
            venue_order_id=venue_order_id, client_order_id=coid,
            status=status, symbol=symbol, side=side, order_type=order_type,
            qty=qty, filled_qty=filled_qty, price=price,
        )
        return SubmitResult(
            venue_order_id=venue_order_id, status=status, immediate_fills=immediate,
        )

    async def cancel_order(
        self, account_id: str, symbol: str,
        *, venue_order_id: str | None = None, client_order_id: str | None = None,
    ) -> CancelResult:
        account = self._require_account(account_id)
        vo = _I.resolve_order(account, venue_order_id, client_order_id)
        terminal = {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED}
        if vo.status in terminal:
            return CancelResult(venue_order_id=vo.venue_order_id, status=vo.status)
        account.orders[vo.venue_order_id] = VenueOrder(
            venue_order_id=vo.venue_order_id, client_order_id=vo.client_order_id,
            status=OrderStatus.CANCELED, symbol=vo.symbol, side=vo.side,
            order_type=vo.order_type, qty=vo.qty, filled_qty=vo.filled_qty, price=vo.price,
        )
        return CancelResult(venue_order_id=vo.venue_order_id, status=OrderStatus.CANCELED)

    async def cancel_order_list(
        self, account_id: str, symbol: str,
        *, venue_order_id: str | None = None, client_order_id: str | None = None,
    ) -> CancelResult:
        return await self.cancel_order(
            account_id, symbol, venue_order_id=venue_order_id, client_order_id=client_order_id,
        )

    async def query_order(
        self, account_id: str, symbol: str,
        *, venue_order_id: str | None = None, client_order_id: str | None = None,
    ) -> VenueOrder:
        return _I.resolve_order(
            self._require_account(account_id), venue_order_id, client_order_id,
        )

    async def query_open_orders(self, account_id: str, symbol: str) -> list[VenueOrder]:
        account = self._require_account(account_id)
        live = {OrderStatus.SUBMITTING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED}
        return [o for o in account.orders.values() if o.symbol == symbol and o.status in live]

    async def query_balance(
        self, account_id: str, assets: frozenset[str],
    ) -> list[BalanceEntry]:
        account = self._require_account(account_id)
        return [
            BalanceEntry(
                asset=a, free=account.balances.get(a, Decimal('0')),
                locked=account.locked.get(a, Decimal('0')),
            )
            for a in sorted(assets)
        ]

    async def query_trades(
        self, account_id: str, symbol: str,
        *, start_time: datetime | None = None,
    ) -> list[VenueTrade]:
        account = self._require_account(account_id)
        return [
            t for t in account.trades
            if t.symbol == symbol and (start_time is None or t.timestamp >= start_time)
        ]

    async def get_exchange_info(self, symbol: str) -> PraxisSymbolFilters:
        f = self._symbol_filters.get(symbol)
        if f is None:
            msg = f'exchange_info: symbol {symbol!r} not loaded; call load_filters first'
            raise NotFoundError(msg)
        return PraxisSymbolFilters(
            symbol=f.symbol, tick_size=f.tick_size, lot_step=f.step_size,
            lot_min=f.min_qty, lot_max=f.max_qty, min_notional=f.min_notional,
        )

    async def query_order_book(self, symbol: str, *, limit: int = 20) -> OrderBookSnapshot:
        # One-level book sourced from most recent trade. Real depth-20 needs a
        # live book; this passes the Protocol so no hot-path consumer of the
        # book exists in the backtest. HonestyStatus flags as ESTIMATED.
        del limit
        now = self._now()
        trades = self._feed.get_trades(symbol, now - _I.window_seconds(60), now)
        if trades.is_empty():
            return OrderBookSnapshot(bids=(), asks=(), last_update_id=0)
        last = trades.tail(1).row(0, named=True)
        px, qty = Decimal(str(last['price'])), Decimal(str(last['qty']))
        return OrderBookSnapshot(
            bids=(OrderBookLevel(price=px, qty=qty),),
            asks=(OrderBookLevel(price=px, qty=qty),),
            last_update_id=int(last.get('trade_id', 0)),
        )

    async def get_server_time(self) -> int:
        return int(self._now().timestamp() * 1000)

    def get_health_snapshot(self, account_id: str) -> HealthSnapshot:
        # Simulated venue has no network, retries, or drift. Zeros are honest.
        return HealthSnapshot()

    async def load_filters(self, symbols: Sequence[str]) -> None:
        for sym in symbols:
            if sym not in self._symbol_filters:
                self._symbol_filters[sym] = BinanceSpotFilters.binance_spot(sym)

    def parse_execution_report(self, data: Mapping[str, object]) -> ExecutionReport:
        del data
        msg = (
            'SimulatedVenueAdapter.parse_execution_report: WebSocket path is '
            'unused in backtest; fills return inline via SubmitResult.'
        )
        raise NotImplementedError(msg)

    def _require_account(self, account_id: str) -> _I.Account:
        account = self._accounts.get(account_id)
        if account is None:
            msg = f'account_id {account_id!r} not registered'
            raise KeyError(msg)
        return account

    def _mint_order_id(self) -> str:
        oid = f'SIM-O-{self._next_order_seq:08d}'
        self._next_order_seq += 1
        return oid

    def _mint_trade_id(self) -> str:
        tid = f'SIM-T-{self._next_trade_seq:08d}'
        self._next_trade_seq += 1
        return tid

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)
