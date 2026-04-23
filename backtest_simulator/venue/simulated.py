"""SimulatedVenueAdapter — implements praxis VenueAdapter against a parquet feed."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import polars as pl

from backtest_simulator.feed.protocol import HistoricalFeed
from backtest_simulator.venue.fees import FeeSchedule
from backtest_simulator.venue.fills import walk_trades
from backtest_simulator.venue.filters import SymbolFilters
from backtest_simulator.venue.types import (
    Account,
    FillModelConfig,
    FillResult,
    PendingOrder,
    SubmitResult,
)


@dataclass
class SimulatedVenueAdapter:
    """Minimal Praxis VenueAdapter implementation for backtesting.

    Stop enforcement is delegated to `walk_trades`, which fills at the
    declared stop price (rounded to tick_size) — never a post-bar-close
    substitute. That property is what makes `R = |entry - stop| * qty`
    mechanically truthful; it is the single load-bearing invariant of
    this layer.
    """

    feed: HistoricalFeed
    fees: FeeSchedule
    filters: SymbolFilters
    fill_config: FillModelConfig = field(default_factory=FillModelConfig)
    trade_window_seconds: int = 3600
    _accounts: dict[str, Account] = field(default_factory=dict)
    _open_orders: dict[str, PendingOrder] = field(default_factory=dict)
    _next_order_id: int = 1

    def register_account(self, account_id: str, api_key: str, api_secret: str) -> None:
        self._accounts[account_id] = Account(account_id, api_key, api_secret)

    def unregister_account(self, account_id: str) -> None:
        self._accounts.pop(account_id, None)

    def attach_ws_sink(
        self, account_id: str,
        sink: Callable[[str, dict[str, object]], Awaitable[None]],
    ) -> None:
        acct = self._accounts.get(account_id)
        if acct is not None:
            acct.ws_sink = sink

    async def submit_order(  # noqa: PLR0913 - Praxis VenueAdapter Protocol; kwargs-only
        self, *,
        symbol: str, side: str, order_type: str,
        qty: Decimal,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
        time_in_force: str = 'GTC',
        submit_time: datetime | None = None,
    ) -> SubmitResult:
        order_id = self._allocate_order_id()
        now = submit_time or datetime.now(UTC)
        # For LIMIT / STOP orders we can validate notional up-front. MARKET
        # orders skip MIN_NOTIONAL here and rely on post-fill validation via
        # the fill-closure + fee accounting gates; a full VenueAdapter
        # implementation with book access would validate against best bid/
        # ask at submit time. The LOT_SIZE check still runs (qty alone).
        reference_price = limit_price or stop_price
        if reference_price is not None:
            rejection = self.filters.validate(qty, reference_price)
            if rejection is not None:
                return SubmitResult(order_id=order_id, accepted=False, reject_reason=rejection, fills=[], fees_quote=Decimal('0'))
        elif qty < self.filters.min_qty or qty > self.filters.max_qty:
            reason = f'LOT_SIZE: qty {qty} outside [{self.filters.min_qty}, {self.filters.max_qty}]'
            return SubmitResult(order_id=order_id, accepted=False, reject_reason=reason, fills=[], fees_quote=Decimal('0'))
        order = PendingOrder(
            order_id=order_id, side=side, order_type=order_type,
            qty=qty, limit_price=limit_price, stop_price=stop_price,
            time_in_force=time_in_force, submit_time=now, symbol=symbol,
        )
        self._open_orders[order_id] = order
        fills = walk_trades(order, self._fetch_trades(symbol, now), self.fill_config, self.filters)
        await asyncio.sleep(0)
        fees_total = self._apply_fees(symbol, fills)
        if fills:
            self._open_orders.pop(order_id, None)
        return SubmitResult(order_id=order_id, accepted=True, reject_reason=None, fills=fills, fees_quote=fees_total)

    async def cancel_order(self, *, order_id: str) -> bool:
        return self._open_orders.pop(order_id, None) is not None

    def _allocate_order_id(self) -> str:
        oid = f'SIM{self._next_order_id:08d}'
        self._next_order_id += 1
        return oid

    def _apply_fees(self, symbol: str, fills: list[FillResult]) -> Decimal:
        return sum(
            (self.fees.fee(symbol, f.fill_price * f.fill_qty, is_maker=f.is_maker) for f in fills),
            Decimal('0'),
        )

    def _fetch_trades(self, symbol: str, submit_time: datetime) -> pl.DataFrame:
        start = submit_time
        end = submit_time + timedelta(seconds=self.trade_window_seconds)
        try:
            return self.feed.get_trades(symbol, start, end)
        except Exception:  # noqa: BLE001 - feed may legitimately have no trades for this window
            return pl.DataFrame(schema={'time': pl.Datetime, 'price': pl.Float64, 'qty': pl.Float64})
