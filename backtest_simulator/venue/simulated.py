"""SimulatedVenueAdapter — real Praxis VenueAdapter Protocol against historical trades."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal

import polars as pl
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
        # — usually `cli/_run_window.run_window_in_process`. The
        # adapter does NOT adjust `fill_price` based on this model:
        # `walk_trades` already prices fills against the actual
        # historical trade prints, which IS the live taker reality.
        # The model contributes its `dt_seconds` (rolling-mid window)
        # to a per-fill *measurement* of realised bps — what the fill
        # paid relative to mid — for operator-visible reporting
        # (signed, adverse, per-side aggregates). When None, the
        # measurement layer is off and aggregate properties return
        # None as the "feature disabled" signal.
        self._slippage_model = slippage_model
        self._slippage_realised_bps: list[Decimal] = []
        self._slippage_realised_sides: list[NexusOrderSide] = []
        self._slippage_n_excluded: int = 0
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

    def _record_slippage(
        self,
        order: PendingOrder,
        fills: list[FillResult],
        trades_window: pl.DataFrame,
    ) -> None:
        """Record realised slippage bps per taker fill — measure, do NOT adjust.

        The previous wiring multiplied `f.fill_price` by `(1 + bps/10000)`
        on top of `walk_trades`'s already-tape-priced fill, which
        double-counts the spread/drift effect (the audit's P1 #1).
        `walk_trades` returns realistic taker prices because it walks
        actual historical trade prints — that IS the price the strategy
        pays in live. Slippage here is observability only: for each
        taker fill, record the deviation from the rolling mid over
        `slippage_model.dt_seconds` preceding `fill_time`. Maker fills
        (`is_maker=True`) are recorded too — they fill at limit, so
        their realised "slippage" against mid is meaningful (often
        negative cost = price improvement).

        The signed bps is stored alongside the side so the aggregator
        can report per-side and adverse (|bps|) means — a plain signed
        mean would let a round trip cancel to zero even though the
        strategy paid spread on both legs (the audit's P1 #3).

        Fills whose preceding `dt_seconds` window has zero trades
        (start-of-tape, halt) are recorded under `n_excluded` rather
        than counted as zero — silent zeros would let an empty mid
        masquerade as "no slippage paid" (the audit's P1 #2 in its
        measurement form).

        When `slippage_model` is None this method is a no-op and the
        aggregate properties return None.
        """
        if self._slippage_model is None:
            return
        if trades_window.is_empty():
            self._slippage_n_excluded += len(
                [f for f in fills if not f.is_maker],
            )
            return
        from datetime import timedelta
        dt = timedelta(seconds=self._slippage_model.dt_seconds)
        for f in fills:
            if f.is_maker:
                # Maker fills land at limit — measuring against mid
                # is a future-tense feature (price improvement
                # reporting); for now skip to keep the signal scoped
                # to taker fills where the calibration semantics
                # match. Track separately if it's needed.
                continue
            window_start = f.fill_time - dt
            preceding = trades_window.filter(
                (pl.col('time') >= window_start)
                & (pl.col('time') < f.fill_time),
            )
            if preceding.is_empty():
                self._slippage_n_excluded += 1
                continue
            mid_value = preceding['price'].median()
            if mid_value is None or mid_value <= 0:
                self._slippage_n_excluded += 1
                continue
            mid = Decimal(str(mid_value))
            bps = (f.fill_price - mid) / mid * _BPS
            side = (
                NexusOrderSide.BUY if order.side == 'BUY'
                else NexusOrderSide.SELL
            )
            self._slippage_realised_bps.append(bps)
            self._slippage_realised_sides.append(side)

    def _aggregate_bps_when_active(
        self,
        sample_filter: object = None,
    ) -> Decimal | None:
        """Mean of recorded bps under `sample_filter`; None when slippage is off."""
        if self._slippage_model is None:
            return None
        if sample_filter is None:
            samples = list(self._slippage_realised_bps)
        else:
            samples = [
                bps for bps, side in zip(
                    self._slippage_realised_bps,
                    self._slippage_realised_sides,
                    strict=True,
                )
                if sample_filter(bps, side)
            ]
        if not samples:
            return Decimal('0')
        return sum(samples, Decimal('0')) / Decimal(len(samples))

    @property
    def slippage_realised_aggregate_bps(self) -> Decimal | None:
        """Signed mean realised bps across all taker fills.

        Positive when the population of fills paid above mid (more
        BUY aggressors); negative when below. NOT a cost metric on
        its own — a balanced round-trip can register near zero even
        when each leg paid spread. Pair with `slippage_realised_adverse_bps`
        for the gross cost view. None when no slippage model attached.
        """
        return self._aggregate_bps_when_active()

    @property
    def slippage_realised_adverse_bps(self) -> Decimal | None:
        """Mean of `|bps|` — the round-trip-honest cost metric.

        Always non-negative: every taker fill contributes its absolute
        deviation from mid regardless of side. This is the bps the
        operator should cite when answering "what did slippage cost
        this run". None when slippage off.
        """
        if self._slippage_model is None:
            return None
        if not self._slippage_realised_bps:
            return Decimal('0')
        total = sum((abs(bps) for bps in self._slippage_realised_bps), Decimal('0'))
        return total / Decimal(len(self._slippage_realised_bps))

    @property
    def slippage_realised_buy_bps(self) -> Decimal | None:
        """Mean realised bps over BUY-aggressor fills (positive = paid above mid)."""
        return self._aggregate_bps_when_active(
            lambda _bps, side: side == NexusOrderSide.BUY,
        )

    @property
    def slippage_realised_sell_bps(self) -> Decimal | None:
        """Mean realised bps over SELL-aggressor fills (negative = received below mid)."""
        return self._aggregate_bps_when_active(
            lambda _bps, side: side == NexusOrderSide.SELL,
        )

    @property
    def slippage_realised_n_samples(self) -> int:
        return len(self._slippage_realised_bps)

    @property
    def slippage_realised_n_excluded(self) -> int:
        """Taker fills excluded because the preceding mid window was empty.

        Honest separation between "measured zero" and "could not measure":
        a sparse-tape window at run start may produce excluded fills
        without any signal — the operator sees this count and knows
        to widen the calibration / pre-window slice. The standalone
        `SlippageModel.apply` raises on uncalibrated buckets; this
        adapter does not call apply on the load-bearing path (we
        measure directly), so that loud-vs-silent gap collapses to
        the n_excluded counter here.
        """
        return self._slippage_n_excluded

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
        #
        # When slippage measurement is active, extend the START of
        # the fetch by `dt_seconds` so the rolling-mid window for a
        # fill at the very first post-submit tick has pre-submit
        # tape to work with. Codex pinned this gap: the rolling
        # mid for a fill at t = submit_time + ε needs trades from
        # [t - dt_seconds, t), most of which sit before submit_time.
        # `walk_trades` itself filters internally with
        # `pl.col('time') >= submit_ts`, so the pre-submit prefix
        # never reaches the fill computation — it's measurement-only.
        # The lookahead carve-out is on `end`, not `start`, so
        # widening the start is unrestricted.
        if self._slippage_model is not None:
            from datetime import timedelta as _td
            fetch_start = order.submit_time - _td(
                seconds=self._slippage_model.dt_seconds,
            )
        else:
            fetch_start = order.submit_time
        trades = self._feed._get_trades_for_venue(
            symbol, fetch_start,
            order.submit_time + _I.window_seconds(self._trade_window_seconds),
            venue_lookahead_seconds=self._trade_window_seconds,
        )
        fills = walk_trades(order, trades, self._fill_config, symbol_filters)
        # Measure realised slippage against rolling mid; do NOT
        # adjust `fills` — the audit's P1 was that adjusting on top
        # of tape-priced fills double-counts spread.
        self._record_slippage(order, fills, trades)
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
