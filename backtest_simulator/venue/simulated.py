"""SimulatedVenueAdapter — real Praxis VenueAdapter Protocol against historical trades."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
from backtest_simulator.honesty.book_gap import BookGapInstrument, BookGapMetric
from backtest_simulator.honesty.maker_fill import MakerFillModel
from backtest_simulator.honesty.slippage import SlippageModel
from backtest_simulator.venue import _adapter_internals as _I
from backtest_simulator.venue.fees import FeeSchedule
from backtest_simulator.venue.fills import WalkContext, walk_trades
from backtest_simulator.venue.filters import BinanceSpotFilters
from backtest_simulator.venue.types import FillModelConfig, FillResult, PendingOrder

_BPS = Decimal('10000')

class SimulatedVenueAdapter:

    def __init__(
        self,
        feed: VenueFeed,
        filters: BinanceSpotFilters,
        fees: FeeSchedule,
        fill_config: FillModelConfig | None = None,
        trade_window_seconds: int = 3600,
        slippage_model: SlippageModel | None = None,
        maker_fill_model: MakerFillModel | None = None,
        market_impact_bucket_minutes: int | None = None,
        market_impact_threshold_fraction: Decimal = Decimal('0.1'),
        strict_impact_policy: bool = False,
        window_end_clamp: datetime | None = None,
    ) -> None:
        self._feed = feed
        self._filters = filters
        self._fees = fees
        self._fill_config = fill_config or FillModelConfig()
        self._trade_window_seconds = trade_window_seconds
        self._window_end_clamp = window_end_clamp
        self._slippage_model = slippage_model
        self._slippage_realised_bps: list[Decimal] = []
        self._slippage_realised_sides: list[NexusOrderSide] = []
        self._slippage_predicted_bps: list[Decimal | None] = []
        self._slippage_n_excluded: int = 0
        self._slippage_n_uncalibrated_predict: int = 0
        self._maker_fill_model = maker_fill_model
        self._n_limit_submitted: int = 0
        self._n_limit_filled_full: int = 0
        self._n_limit_filled_partial: int = 0
        self._n_limit_filled_zero: int = 0
        self._n_limit_marketable_taker: int = 0
        self._maker_fill_efficiencies: list[Decimal] = []
        self._market_impact_bucket_minutes = market_impact_bucket_minutes
        self._market_impact_threshold_fraction = (
            market_impact_threshold_fraction
        )
        self._strict_impact_policy = strict_impact_policy
        self._market_impact_bps_samples: list[Decimal] = []
        self._market_impact_n_flagged: int = 0
        self._market_impact_n_uncalibrated: int = 0
        self._market_impact_n_rejected: int = 0
        self._book_gap_instrument = BookGapInstrument()
        self._accounts: dict[str, _I.Account] = {}
        self._symbol_filters: dict[str, BinanceSpotFilters] = {filters.symbol: filters}
        self._next_order_seq = 1
        self._next_trade_seq = 1
        self._history: dict[str, _I.Account] = {}

    def touch_for_symbol(self, symbol: str) -> Decimal | None:
        from datetime import timedelta
        now = self._now()
        trades = self._feed.get_trades_for_venue(
            symbol, now - timedelta(minutes=1), now,
            venue_lookahead_seconds=0,
        )
        if trades.is_empty():
            return None
        return Decimal(str(trades.tail(1)['price'].item()))

    def tick_for_symbol(self, symbol: str) -> Decimal:
        filters = self._symbol_filters.get(symbol)
        if filters is None:
            msg = (
                f'tick_for_symbol: symbol {symbol!r} not registered; '
                f'call load_filters([{symbol!r}]) first.'
            )
            raise KeyError(msg)
        return filters.tick_size

    def register_account(self, account_id: str, api_key: str, api_secret: str) -> None:
        account = self._history.pop(account_id, None) or _I.Account(
            account_id=account_id, api_key=api_key, api_secret=api_secret,
        )
        self._accounts[account_id] = account

    def unregister_account(self, account_id: str) -> None:
        if account_id not in self._accounts:
            msg = f'account_id {account_id!r} not registered'
            raise KeyError(msg)
        self._history[account_id] = self._accounts.pop(account_id)

    async def close(self) -> None:
        pass

    def _record_slippage(
        self,
        order: PendingOrder,
        fills: list[FillResult],
        trades_window: pl.DataFrame,
    ) -> None:
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
            if not isinstance(mid_value, (int, float)) or mid_value <= 0:
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
            try:
                predicted_bps = self._slippage_model.apply(
                    side=side, qty=f.fill_qty, mid=mid, t=f.fill_time,
                )
            except ValueError:
                self._slippage_predicted_bps.append(None)
                self._slippage_n_uncalibrated_predict += 1
            else:
                self._slippage_predicted_bps.append(predicted_bps)

    def _aggregate_bps_when_active(
        self,
        sample_filter: Callable[[Decimal, NexusOrderSide], bool] | None = None,
    ) -> Decimal | None:
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
        return self._aggregate_bps_when_active()

    @property
    def slippage_realised_cost_bps(self) -> Decimal | None:
        if self._slippage_model is None:
            return None
        if not self._slippage_realised_bps:
            return Decimal('0')
        cost_samples = [
            bps if side == NexusOrderSide.BUY else -bps
            for bps, side in zip(
                self._slippage_realised_bps,
                self._slippage_realised_sides,
                strict=True,
            )
        ]
        return sum(cost_samples, Decimal('0')) / Decimal(len(cost_samples))

    @property
    def slippage_realised_buy_bps(self) -> Decimal | None:
        return self._aggregate_bps_when_active(
            lambda _bps, side: side == NexusOrderSide.BUY,
        )

    @property
    def slippage_realised_sell_bps(self) -> Decimal | None:
        return self._aggregate_bps_when_active(
            lambda _bps, side: side == NexusOrderSide.SELL,
        )

    @property
    def slippage_predicted_cost_bps(self) -> Decimal | None:
        if self._slippage_model is None:
            return None
        paired = [
            (pred, side)
            for pred, side in zip(
                self._slippage_predicted_bps,
                self._slippage_realised_sides,
                strict=True,
            )
            if pred is not None
        ]
        if not paired:
            return Decimal('0')
        cost_samples = [
            pred if side == NexusOrderSide.BUY else -pred
            for pred, side in paired
        ]
        return sum(cost_samples, Decimal('0')) / Decimal(len(cost_samples))

    @property
    def slippage_predict_vs_realised_gap_bps(self) -> Decimal | None:
        if self._slippage_model is None:
            return None
        gaps: list[Decimal] = []
        for realised, pred, side in zip(
            self._slippage_realised_bps,
            self._slippage_predicted_bps,
            self._slippage_realised_sides,
            strict=True,
        ):
            if pred is None:
                continue
            realised_cost = (
                realised if side == NexusOrderSide.BUY else -realised
            )
            predicted_cost = pred if side == NexusOrderSide.BUY else -pred
            gaps.append(realised_cost - predicted_cost)
        if not gaps:
            return Decimal('0')
        return sum(gaps, Decimal('0')) / Decimal(len(gaps))

    @property
    def slippage_n_uncalibrated_predict(self) -> int:
        return self._slippage_n_uncalibrated_predict

    def _record_market_impact_pre_fill(
        self,
        order: PendingOrder,
        symbol: str,
        submit_time: datetime,
    ) -> bool:
        if self._market_impact_bucket_minutes is None:
            return False
        from datetime import timedelta

        from backtest_simulator.honesty.market_impact import (
            MarketImpactModel,
        )
        bucket = self._market_impact_bucket_minutes
        raw = self._feed.get_trades_for_venue(
            symbol, submit_time - timedelta(minutes=bucket),
            submit_time,
            venue_lookahead_seconds=0,
        )
        pre = raw.filter(pl.col('time') < submit_time).rename({
            'time': 'datetime', 'qty': 'quantity',
        })
        decision = MarketImpactModel.evaluate_rolling(
            qty=order.qty,
            trades_pre_submit=pre,
            threshold_fraction=self._market_impact_threshold_fraction,
        )
        if decision is None:
            self._market_impact_n_uncalibrated += 1
            return False
        self._market_impact_bps_samples.append(decision.impact_bps)
        if decision.flag:
            self._market_impact_n_flagged += 1
            if self._strict_impact_policy and order.side == 'BUY':
                self._market_impact_n_rejected += 1
                return True
        return False

    @property
    def market_impact_realised_bps(self) -> Decimal | None:
        if self._market_impact_bucket_minutes is None:
            return None
        if not self._market_impact_bps_samples:
            return Decimal('0')
        return sum(
            self._market_impact_bps_samples, Decimal('0'),
        ) / Decimal(len(self._market_impact_bps_samples))

    @property
    def market_impact_n_samples(self) -> int:
        return len(self._market_impact_bps_samples)

    @property
    def market_impact_n_flagged(self) -> int:
        return self._market_impact_n_flagged

    @property
    def market_impact_n_uncalibrated(self) -> int:
        return self._market_impact_n_uncalibrated

    @property
    def market_impact_n_rejected(self) -> int:
        return self._market_impact_n_rejected

    def _record_limit_outcome(
        self,
        order: PendingOrder,
        fills: list[FillResult],
    ) -> None:
        if order.order_type != 'LIMIT':
            return
        self._n_limit_submitted += 1
        if not fills:
            self._n_limit_filled_zero += 1
            self._maker_fill_efficiencies.append(Decimal('0'))
            return
        all_taker = all(not f.is_maker for f in fills)
        if all_taker:
            self._n_limit_marketable_taker += 1
            return
        total_filled = sum((f.fill_qty for f in fills), Decimal('0'))
        if total_filled >= order.qty:
            self._n_limit_filled_full += 1
        elif total_filled > Decimal('0'):
            self._n_limit_filled_partial += 1
        else:
            self._n_limit_filled_zero += 1
        if order.qty > Decimal('0'):
            self._maker_fill_efficiencies.append(
                total_filled / order.qty,
            )

    @property
    def n_limit_orders_submitted(self) -> int:
        return self._n_limit_submitted

    @property
    def n_limit_filled_full(self) -> int:
        return self._n_limit_filled_full

    @property
    def n_limit_filled_partial(self) -> int:
        return self._n_limit_filled_partial

    @property
    def n_limit_filled_zero(self) -> int:
        return self._n_limit_filled_zero

    @property
    def n_limit_marketable_taker(self) -> int:
        return self._n_limit_marketable_taker

    @property
    def maker_fill_efficiency_p50(self) -> Decimal | None:
        if not self._maker_fill_efficiencies:
            return None
        ordered = sorted(self._maker_fill_efficiencies)
        n = len(ordered)
        if n % 2 == 1:
            return ordered[n // 2]
        return (ordered[n // 2 - 1] + ordered[n // 2]) / Decimal('2')

    @property
    def maker_fill_efficiency_mean(self) -> Decimal | None:
        if not self._maker_fill_efficiencies:
            return None
        return sum(
            self._maker_fill_efficiencies, Decimal('0'),
        ) / Decimal(len(self._maker_fill_efficiencies))

    @property
    def n_passive_limits(self) -> int:
        return len(self._maker_fill_efficiencies)

    @property
    def slippage_n_predicted_samples(self) -> int:
        return sum(
            1 for pred in self._slippage_predicted_bps if pred is not None
        )

    @property
    def slippage_realised_n_samples(self) -> int:
        return len(self._slippage_realised_bps)

    @property
    def slippage_realised_n_excluded(self) -> int:
        return self._slippage_n_excluded

    def book_gap_snapshot(self) -> BookGapMetric:
        return self._book_gap_instrument.snapshot()

    def history(self, account_id: str) -> _I.Account:
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
        del stop_limit_price
        account = self._require_account(account_id)
        venue_order_id = self._mint_order_id()
        coid = client_order_id or f'BTS-{venue_order_id}'
        effective_tif = time_in_force or (
            'IOC' if order_type == OrderType.LIMIT_IOC else 'GTC'
        )
        symbol_filters = self._symbol_filters.get(symbol)
        if symbol_filters is None:
            msg = (
                f'submit_order: symbol {symbol!r} has no registered filters; '
                f'call load_filters([{symbol!r}]) before submitting'
            )
            raise ValueError(msg)
        submit_time = self._now()
        from datetime import timedelta as _td
        fetch_start = submit_time
        if self._slippage_model is not None:
            fetch_start = min(
                fetch_start,
                submit_time - _td(seconds=self._slippage_model.dt_seconds),
            )
        if self._maker_fill_model is not None:
            fetch_start = min(
                fetch_start,
                submit_time - _td(
                    minutes=self._maker_fill_model.lookback_minutes,
                ),
            )
        walk_end = submit_time + _I.window_seconds(self._trade_window_seconds)
        if self._window_end_clamp is not None:
            walk_end = min(walk_end, self._window_end_clamp)
        effective_lookahead_seconds = max(
            0, int((walk_end - submit_time).total_seconds()),
        )
        trades = self._feed.get_trades_for_venue(
            symbol, fetch_start,
            walk_end,
            venue_lookahead_seconds=effective_lookahead_seconds,
        )
        order = PendingOrder(
            order_id=venue_order_id, side=side.name, order_type=_I.TYPE_MAP[order_type],
            qty=qty, limit_price=price, stop_price=stop_price,
            time_in_force=effective_tif, submit_time=submit_time, symbol=symbol,
        )
        if _I.reject_reason(order, symbol_filters, price) is not None:
            _I.record_rejection(account, order, coid, side, order_type, price)
            return SubmitResult(
                venue_order_id=venue_order_id, status=OrderStatus.REJECTED, immediate_fills=(),
            )
        if self._record_market_impact_pre_fill(order, symbol, submit_time):
            _I.record_rejection(account, order, coid, side, order_type, price)
            return SubmitResult(
                venue_order_id=venue_order_id,
                status=OrderStatus.REJECTED,
                immediate_fills=(),
            )
        if self._maker_fill_model is not None:
            trades_pre_submit = trades.filter(pl.col('time') < submit_time)
        else:
            trades_pre_submit = None
        fills = walk_trades(
            order, trades, self._fill_config, symbol_filters,
            WalkContext(
                maker_model=self._maker_fill_model,
                trades_pre_submit=trades_pre_submit,
                book_gap_instrument=self._book_gap_instrument,
            ),
        )
        self._record_slippage(order, fills, trades)
        self._record_limit_outcome(order, fills)
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
        del limit
        now = self._now()
        trades = self._feed.get_trades(symbol, now - _I.window_seconds(60), now)
        if trades.is_empty():
            return OrderBookSnapshot(bids=(), asks=(), last_update_id=0)
        last = trades.tail(1).row(0, named=True)
        px = Decimal(str(last['price']))
        depth_qty = Decimal(str(trades['qty'].sum()))
        return OrderBookSnapshot(
            bids=(OrderBookLevel(price=px, qty=depth_qty),),
            asks=(OrderBookLevel(price=px, qty=depth_qty),),
            last_update_id=int(last.get('trade_id', 0)),
        )

    async def get_server_time(self) -> int:
        return int(self._now().timestamp() * 1000)

    def get_health_snapshot(self, account_id: str) -> HealthSnapshot:
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
