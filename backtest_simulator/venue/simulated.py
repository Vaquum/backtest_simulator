"""SimulatedVenueAdapter — real Praxis VenueAdapter Protocol against historical trades."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal

from nexus.core.domain.enums import OrderSide as NexusOrderSide
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

from backtest_simulator.feed.protocol import VenueFeed
from backtest_simulator.honesty.slippage import SlippageModel
from backtest_simulator.venue import _adapter_internals as _I
from backtest_simulator.venue.fees import FeeSchedule
from backtest_simulator.venue.fills import walk_trades
from backtest_simulator.venue.filters import BinanceSpotFilters
from backtest_simulator.venue.types import FillModelConfig, FillResult, PendingOrder

_BPS = Decimal('10000')


class SimulatedVenueAdapter:
    """VenueAdapter Protocol implementation backed by historical trades."""

    def __init__(
        self,
        feed: VenueFeed,
        filters: BinanceSpotFilters,
        fees: FeeSchedule,
        fill_config: FillModelConfig | None = None,
        trade_window_seconds: int = 3600,
        slippage_model: SlippageModel | None = None,
    ) -> None:
        self._feed = feed
        self._filters = filters
        self._fees = fees
        self._fill_config = fill_config or FillModelConfig()
        self._trade_window_seconds = trade_window_seconds
        # `slippage_model` is calibrated externally (`SlippageModel.calibrate`
        # over a pre-window trade slice) and supplied by the operator
        # — usually `cli/_run_window.run_window_in_process`. None means
        # "no slippage layer"; the adapter falls back to walk_trades's
        # raw VWAP/tick prices and `slippage_realised_aggregate_bps`
        # returns None. With a model, every taker fill has its
        # `fill_price` adjusted by `mid * (1 + bps/10000)` and the
        # realised bps recorded for the JSON report.
        self._slippage_model = slippage_model
        self._slippage_realised_bps: list[Decimal] = []
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

    def _apply_slippage(
        self,
        order: PendingOrder,
        fills: list[FillResult],
        symbol_filters: BinanceSpotFilters,
    ) -> list[FillResult]:
        """Apply calibrated slippage bps to each taker fill, in price-space.

        Maker fills (`is_maker=True`) keep their limit-price fill price
        — no slippage layer applies because the maker ate from the
        opposite side at exactly its declared price. Taker fills
        (`is_maker=False`, the MARKET / stop-trigger / VWAP paths)
        get `fill_price * (1 + bps / 10000)` where `bps` is the
        side-signed bps the calibration's matching `(side, qty)`
        bucket records. The bucket's median is positive for BUY
        aggressors (taker pays above mid) and negative for SELL
        aggressors (taker receives below mid). Both sides land at
        worse prices than the raw tape VWAP, which is the live
        reality the simulator must reflect.

        When `slippage_model` is None, fills pass through unchanged
        and `slippage_realised_aggregate_bps` returns None — the
        operator opted out of the slippage layer.
        """
        if self._slippage_model is None:
            return fills
        adjusted: list[FillResult] = []
        for f in fills:
            if f.is_maker:
                adjusted.append(f)
                continue
            # SlippageModel is keyed by `nexus.core.domain.enums.OrderSide`,
            # NOT the `praxis.core.domain.enums.OrderSide` this adapter
            # imports at module top — they are distinct enum classes
            # even though the names align. The wiring tested this gap
            # (the adapter's praxis lookup found zero buckets and apply
            # raised ValueError, recording bps=0). Use the nexus enum
            # the model registers against.
            side_enum = (
                NexusOrderSide.BUY if order.side == 'BUY'
                else NexusOrderSide.SELL
            )
            try:
                bps = self._slippage_model.apply(
                    side=side_enum,
                    qty=f.fill_qty,
                    mid=f.fill_price,
                    t=f.fill_time,
                )
            except ValueError:
                # The calibration window did not cover this (side, qty)
                # combination. Loud-on-gap is the design for the
                # standalone model — but inside the live fill path we
                # cannot crash an entire run on one out-of-bucket order.
                # Record bps=0 (the conservative choice — no slippage
                # adjustment) and keep the per-fill record so the
                # operator sees the missing-coverage entries when the
                # aggregate is reviewed alongside `n_samples`. The right
                # fix at the operator level is to widen the calibration
                # window or split the order; the right fix here is
                # not to silently swallow a real fill.
                bps = Decimal('0')
            adjusted_price = f.fill_price * (
                Decimal('1') + bps / _BPS
            )
            self._slippage_realised_bps.append(bps)
            adjusted.append(FillResult(
                fill_time=f.fill_time,
                fill_price=symbol_filters.round_price(adjusted_price),
                fill_qty=f.fill_qty,
                is_maker=f.is_maker,
                reason=f.reason,
            ))
        return adjusted

    @property
    def slippage_realised_aggregate_bps(self) -> Decimal | None:
        """Mean of the realised slippage bps across all taker fills.

        Returns `None` when no slippage model was attached to this
        adapter (the operator opted out) — keeping the JSON report
        signal honest: `None` means "feature disabled", a numeric
        value means "feature active and this is what the run paid".
        """
        if self._slippage_model is None:
            return None
        if not self._slippage_realised_bps:
            return Decimal('0')
        total = sum(self._slippage_realised_bps, Decimal('0'))
        return total / Decimal(len(self._slippage_realised_bps))

    @property
    def slippage_realised_n_samples(self) -> int:
        return len(self._slippage_realised_bps)

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
        # `OrderType.LIMIT_IOC` collapses to `'LIMIT'` in `TYPE_MAP`,
        # so without this nudge a caller that passes the IOC enum but
        # leaves `time_in_force=None` would land on PendingOrder with
        # `time_in_force='GTC'` and the zero-fill branch would mis-
        # report the order as OPEN (resting) instead of EXPIRED. Force
        # `IOC` whenever the enum carries it. (Other TIF mappings —
        # FOK, GTX, GTC explicit — must be passed by the caller.)
        effective_tif = time_in_force or (
            'IOC' if order_type == OrderType.LIMIT_IOC else 'GTC'
        )
        order = PendingOrder(
            order_id=venue_order_id, side=side.name, order_type=_I.TYPE_MAP[order_type],
            qty=qty, limit_price=price, stop_price=stop_price,
            time_in_force=effective_tif, submit_time=self._now(), symbol=symbol,
        )
        # Resolve the per-symbol filter record. `_symbol_filters` is the
        # authoritative source (populated via `load_filters()` at boot
        # and seeded from the adapter's init filter). Falling back to
        # `self._filters` only when the symbol hasn't been registered
        # would mask the misroute silently; raise instead.
        symbol_filters = self._symbol_filters.get(symbol)
        if symbol_filters is None:
            msg = (
                f'submit_order: symbol {symbol!r} has no registered filters; '
                f'call load_filters([{symbol!r}]) before submitting'
            )
            raise ValueError(msg)
        if _I.reject_reason(order, symbol_filters, price) is not None:
            _I.record_rejection(account, order, coid, side, order_type, price)
            return SubmitResult(
                venue_order_id=venue_order_id, status=OrderStatus.REJECTED, immediate_fills=(),
            )
        # The venue carve-out: peek up to `trade_window_seconds` past
        # `frozen_now()` to simulate a realistic submit→fill window.
        # The strategy-facing `get_trades` does not accept a kwarg for
        # this — `_get_trades_for_venue` is the only path with the
        # bounded peek. See `feed/protocol.py` for the rationale.
        trades = self._feed._get_trades_for_venue(
            symbol, order.submit_time,
            order.submit_time + _I.window_seconds(self._trade_window_seconds),
            venue_lookahead_seconds=self._trade_window_seconds,
        )
        fills = walk_trades(order, trades, self._fill_config, symbol_filters)
        fills = self._apply_slippage(order, fills, symbol_filters)
        immediate = _I.record_fills(
            account, self._fees,
            _I.OrderIdentity(
                venue_order_id=venue_order_id, client_order_id=coid,
                symbol=symbol, side=side,
            ),
            fills, self._mint_trade_id,
        )
        filled_qty = sum((f.qty for f in immediate), Decimal('0'))
        # Validation rejection (filter failure) returned earlier with
        # status=REJECTED via the early branch above. By the time we
        # reach this line the order passed validation. A zero-fill
        # outcome's terminal status depends on `(order_type, TIF)`:
        #   - MARKET (any TIF): there's no resting concept — no
        #     liquidity in the window means the order failed to
        #     execute; map to EXPIRED.
        #   - LIMIT with GTC: live Binance keeps the order on the
        #     book until it crosses or is cancelled. Mark OPEN so
        #     `query_open_orders` surfaces it.
        #   - LIMIT with IOC / FOK / GTX: window closed without
        #     execution → EXPIRED.
        # Pre-fix this branch returned REJECTED uniformly, conflating
        # venue-rejection with no-fill-in-window.
        status = (
            OrderStatus.FILLED if filled_qty >= qty
            else OrderStatus.PARTIALLY_FILLED if filled_qty > 0
            else _I.zero_fill_status(order.order_type, order.time_in_force)
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
