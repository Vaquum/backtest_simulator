"""BacktestLauncher — real praxis.launcher.Launcher subclass, historical seams."""
from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import queue
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import cast

import polars as pl
from limen import HistoricalData
from nexus.core.domain.capital_state import CapitalState
from nexus.core.domain.enums import OperationalMode
from nexus.core.domain.instance_state import InstanceState
from nexus.core.outcome_loop import ActionSubmitter, OutcomeLoop
from nexus.core.validator.pipeline_models import ValidationDecision, ValidationRequestContext
from nexus.infrastructure.manifest import load_manifest
from nexus.infrastructure.praxis_connector import trade_outcome as _nexus_trade_outcome_mod
from nexus.infrastructure.praxis_connector.praxis_inbound import PraxisInbound
from nexus.infrastructure.praxis_connector.praxis_outbound import PraxisOutbound
from nexus.infrastructure.state_store import StateStore
from nexus.instance_config import InstanceConfig as NexusInstanceConfig
from nexus.startup.sequencer import StartupSequencer, WiredSensor
from nexus.startup.shutdown_sequencer import ShutdownSequencer
from nexus.strategy.action import Action
from nexus.strategy.context import StrategyContext
from nexus.strategy.predict_loop import PredictLoop
from nexus.strategy.runner import StrategyRunner
from nexus.strategy.timer_loop import TimerLoop
from praxis.core.domain.enums import OrderSide, OrderType
from praxis.core.domain.trade_outcome import TradeOutcome
from praxis.infrastructure.event_spine import EventSpine
from praxis.infrastructure.venue_adapter import SubmitResult, VenueAdapter
from praxis.launcher import InstanceConfig, Launcher
from praxis.trading_config import TradingConfig

from backtest_simulator.honesty import (
    CapitalLifecycleTracker,
    assert_conservation,
    build_validation_pipeline,
    capital_totals,
)
from backtest_simulator.launcher.action_submitter import SubmitterBindings, build_action_submitter
from backtest_simulator.launcher.clock import accelerated_clock
from backtest_simulator.launcher.poller import BacktestMarketDataPoller
from backtest_simulator.launcher.replay_clock import ReplayClock

NexusTradeOutcome = _nexus_trade_outcome_mod.TradeOutcome
_log = logging.getLogger(__name__)
_BOOT_TIMEOUT_SECONDS = 60
_NEXUS_RUN_TIMEOUT_SECONDS = 120
_POLL_INTERVAL_SECONDS = 0.05
_REAL_TIME_CAP_SECONDS = 600
_SHUTDOWN_TIMEOUT_SECONDS = 30
_REAL_MONOTONIC = time.monotonic
_DRAIN_POLL_INTERVAL_SECONDS = 0.02
_DRAIN_TIMEOUT_SECONDS = 15.0
_DRAIN_SLOW_WARN_SECONDS = 2.0

class DrainTimeoutError(RuntimeError):
    pass

class CapitalOvershootError(RuntimeError):
    pass

class CapitalPartialFillError(RuntimeError):
    pass

def _extract_declared_stop_price(action: Action) -> Decimal | None:
    from collections.abc import Mapping
    params = action.execution_params
    if not isinstance(params, Mapping):
        return None
    raw = params.get('stop_price')
    if raw is None or str(raw).strip() in ('', 'None'):
        return None
    return Decimal(str(raw))

@dataclass(frozen=True)
class _LifecycleContext:
    tracker: CapitalLifecycleTracker
    capital_state: CapitalState
    initial_pool: Decimal

def _check_tracker_match_required(pre_match_command_id: str | None, side: OrderSide, client_order_id: str | None) -> None:
    if pre_match_command_id is not None:
        return
    if side.name != 'BUY':
        return
    msg = f'capital lifecycle: BUY submit for client_order_id={client_order_id!r} has no matching reservation in tracker. Part 2 honesty requires every BUY to clear the CAPITAL stage BEFORE dispatch; missing match means the action-submitter skipped the reservation OR the client_order_id format has changed.'
    raise RuntimeError(msg)

def _maybe_inject_declared_stop(pre_match_command_id: str | None, client_order_id: str | None, stop_price: Decimal | None, tracker: CapitalLifecycleTracker, declared_stops: dict[str, Decimal], order_type: OrderType) -> Decimal | None:
    if pre_match_command_id is None or stop_price is not None:
        return stop_price
    declared_stop = tracker.declared_stop_for_command(pre_match_command_id)
    if declared_stop is None:
        return stop_price
    if client_order_id is not None:
        declared_stops[client_order_id] = declared_stop
    if order_type == OrderType.MARKET:
        return declared_stop
    return stop_price

def _sum_fill_totals(result: SubmitResult) -> tuple[Decimal, Decimal]:
    fill_notional = Decimal('0')
    fill_fees = Decimal('0')
    for fill in result.immediate_fills:
        fill_notional += Decimal(str(fill.qty)) * Decimal(str(fill.price))
        fill_fees += Decimal(str(fill.fee))
    return (fill_notional, fill_fees)

def _sum_fill_qty(result: SubmitResult) -> Decimal:
    total = Decimal('0')
    for fill in result.immediate_fills:
        total += Decimal(str(fill.qty))
    return total

def _finalize_successful_fill(ctx: _LifecycleContext, command_id: str, result: SubmitResult, venue_order_id: str, status_name: str) -> None:
    ctx.tracker.record_sent(command_id, venue_order_id)
    assert_conservation(ctx.capital_state, ctx.initial_pool, context=f'send_order command_id={command_id}')
    fill_notional, fill_fees = _sum_fill_totals(result)
    reservation_notional = ctx.tracker.declared_reservation_for_command(command_id)
    if reservation_notional is not None and fill_notional > reservation_notional:
        slippage = fill_notional - reservation_notional
        msg = f'capital overshoot: fill_notional={fill_notional} exceeds reserved={reservation_notional} for command_id={command_id} by {slippage}. Raise `NOTIONAL_RESERVATION_BUFFER` in backtest_simulator.launcher.action_submitter to absorb this slippage honestly; silently capping here would bypass CAPITAL gating.'
        raise CapitalOvershootError(msg)
    release_residual = status_name == 'PARTIALLY_FILLED'
    if release_residual:
        _log.info('capital lifecycle: command_id=%s PARTIALLY_FILLED — driving order_fill(%s) + order_cancel(%s) to release unfilled reservation residual back to available capital.', command_id, fill_notional, venue_order_id)
    pending_strategy_id = ctx.tracker.strategy_id_for_pending(command_id)
    ctx.tracker.record_ack_and_fill(command_id, venue_order_id, fill_notional, fill_fees, release_residual=release_residual)
    assert_conservation(ctx.capital_state, ctx.initial_pool, context=f'order_fill command_id={command_id}')
    fill_qty = _sum_fill_qty(result)
    ctx.tracker.record_open_position(command_id=command_id, strategy_id=pending_strategy_id or 'unknown', cost_basis=fill_notional, entry_fees=fill_fees, entry_qty=fill_qty)
    _log.info('capital lifecycle: command_id=%s reserve->send->ack->fill%s venue_order_id=%s fill_notional=%s fees=%s open_positions=%d', command_id, '+cancel' if release_residual else '', venue_order_id, fill_notional, fill_fees, ctx.tracker.open_position_count)

def _finalize_sell_close(ctx: _LifecycleContext, result: SubmitResult, sell_command_id: str) -> None:
    sell_proceeds, sell_fees = _sum_fill_totals(result)
    sell_qty = _sum_fill_qty(result)
    realized_pnl, closed_record = ctx.tracker.record_close_position(ctx.capital_state, sell_command_id=sell_command_id, sell_qty=sell_qty, sell_proceeds=sell_proceeds, sell_fees=sell_fees)
    assert_conservation(ctx.capital_state, ctx.initial_pool, context=f'sell_close sell_command_id={sell_command_id}')
    _log.info('capital lifecycle: SELL close sell_command_id=%s paired_buy_command_id=%s released_cost_basis=%s released_entry_fees=%s sell_qty=%s sell_proceeds=%s sell_fees=%s realized_pnl=%s open_positions=%d', sell_command_id, closed_record.command_id, closed_record.cost_basis, closed_record.entry_fees, sell_qty, sell_proceeds, sell_fees, realized_pnl, ctx.tracker.open_position_count)

def _make_outcome_router(outcome_queues: dict[str, queue.Queue[NexusTradeOutcome]], on_routed: Callable[[], None] | None=None) -> Callable[[TradeOutcome], Awaitable[None]]:

    async def _route(praxis_outcome: TradeOutcome) -> None:
        nexus_outcome = _translate_praxis_outcome(praxis_outcome)
        if nexus_outcome is None:
            return
        account_queue = outcome_queues.get(praxis_outcome.account_id)
        if account_queue is None:
            return
        account_queue.put_nowait(nexus_outcome)
        if on_routed is not None:
            on_routed()
    return _route

def _translate_praxis_outcome(praxis_outcome: TradeOutcome) -> NexusTradeOutcome | None:
    from nexus.infrastructure.praxis_connector.trade_outcome_type import TradeOutcomeType
    from praxis.core.domain.enums import TradeStatus
    status = praxis_outcome.status
    type_map = {TradeStatus.FILLED: TradeOutcomeType.FILLED, TradeStatus.PARTIAL: TradeOutcomeType.PARTIAL, TradeStatus.REJECTED: TradeOutcomeType.REJECTED, TradeStatus.EXPIRED: TradeOutcomeType.EXPIRED, TradeStatus.CANCELED: TradeOutcomeType.CANCELED, TradeStatus.PENDING: TradeOutcomeType.EXPIRED}
    nexus_type = type_map.get(status)
    if nexus_type is None:
        return None
    if status == TradeStatus.EXPIRED and praxis_outcome.filled_qty > 0:
        nexus_type = TradeOutcomeType.PARTIAL
    is_fill = nexus_type.is_fill
    fill_size = praxis_outcome.filled_qty if is_fill else None
    fill_price = praxis_outcome.avg_fill_price if is_fill else None
    fill_notional = praxis_outcome.filled_qty * praxis_outcome.avg_fill_price if is_fill and praxis_outcome.avg_fill_price is not None else None
    actual_fees = Decimal('0') if is_fill else None
    if nexus_type == TradeOutcomeType.REJECTED:
        reject_reason = praxis_outcome.reason if praxis_outcome.reason else f'translated-from-praxis-{status.value}'
    else:
        reject_reason = None
    return NexusTradeOutcome(outcome_id=f'{praxis_outcome.command_id}-{status.value}', command_id=praxis_outcome.command_id, outcome_type=nexus_type, timestamp=praxis_outcome.created_at, fill_size=fill_size, fill_price=fill_price, fill_notional=fill_notional, actual_fees=actual_fees, reject_reason=reject_reason)

def _build_outcome_loop(*, runner: StrategyRunner, praxis_inbound: PraxisInbound, state: InstanceState, context_provider: Callable[[str], StrategyContext], wired_sensors: Sequence[WiredSensor], action_submit: ActionSubmitter) -> OutcomeLoop:
    if not wired_sensors:
        msg = 'BacktestLauncher: no wired sensors after sequencer.start(); OutcomeLoop has nothing to resolve outcomes against.'
        raise RuntimeError(msg)
    strategy_ids = {s.strategy_id for s in wired_sensors}
    if len(strategy_ids) != 1:
        msg = f'BacktestLauncher: multiple strategy_ids wired ({sorted(strategy_ids)}) — outcome resolution requires a per-command registry, not yet implemented.'
        raise RuntimeError(msg)
    single = next(iter(strategy_ids))

    def _resolve(_outcome: NexusTradeOutcome) -> str | None:
        return single
    return OutcomeLoop(runner=runner, praxis_inbound=praxis_inbound, state=state, context_provider=context_provider, resolve_strategy_id=_resolve, action_submit=action_submit)

def _install_capital_adapter_wrapper(adapter: VenueAdapter, tracker: CapitalLifecycleTracker, capital_state: CapitalState, initial_pool: Decimal, declared_stops: dict[str, Decimal]) -> None:
    original_submit = adapter.submit_order
    ctx = _LifecycleContext(tracker=tracker, capital_state=capital_state, initial_pool=initial_pool)

    async def wrapped_submit(account_id: str, symbol: str, side: OrderSide, order_type: OrderType, qty: Decimal, *, price: Decimal | None=None, stop_price: Decimal | None=None, stop_limit_price: Decimal | None=None, client_order_id: str | None=None, time_in_force: str | None=None) -> SubmitResult:
        pre_match_command_id = _match_command_id(tracker, client_order_id)
        _check_tracker_match_required(pre_match_command_id, side, client_order_id)
        stop_price = _maybe_inject_declared_stop(pre_match_command_id, client_order_id, stop_price, tracker, declared_stops, order_type)
        try:
            result = await original_submit(account_id, symbol, side, order_type, qty, price=price, stop_price=stop_price, stop_limit_price=stop_limit_price, client_order_id=client_order_id, time_in_force=time_in_force)
        except Exception:
            if pre_match_command_id is not None:
                tracker.record_rejection(pre_match_command_id, '')
                assert_conservation(capital_state, initial_pool, context=f'adapter_raised command_id={pre_match_command_id}')
            raise
        command_id = _match_command_id(tracker, client_order_id)
        venue_order_id = result.venue_order_id
        status_name = result.status.name
        if command_id is None:
            if side.name == 'SELL' and status_name in ('FILLED', 'PARTIALLY_FILLED') and result.immediate_fills:
                _finalize_sell_close(ctx, result, client_order_id or 'unknown')
            return result
        if status_name in ('REJECTED', 'EXPIRED'):
            if status_name == 'EXPIRED':
                tracker.record_sent(command_id, venue_order_id)
            tracker.record_rejection(command_id, venue_order_id)
            assert_conservation(capital_state, initial_pool, context=f'order_reject command_id={command_id}')
            _log.info('capital lifecycle: command_id=%s %s (reservation released)', command_id, status_name)
            return result
        if status_name == 'OPEN':
            tracker.record_sent(command_id, venue_order_id)
            tracker.record_rejection(command_id, venue_order_id)
            assert_conservation(capital_state, initial_pool, context=f'limit_open_no_fill command_id={command_id}')
            _log.info('capital lifecycle: command_id=%s OPEN (LIMIT no-fill in lookahead window — reservation released)', command_id)
            return result
        _finalize_successful_fill(ctx, command_id, result, venue_order_id, status_name)
        return result
    setattr(adapter, 'submit_order', wrapped_submit)

def _match_command_id(tracker: CapitalLifecycleTracker, client_order_id: str | None) -> str | None:
    if client_order_id is None:
        return None
    parts = client_order_id.split('-')
    if len(parts) < 3:
        return None
    prefix = parts[1]
    return tracker.match_pending_by_prefix(prefix)

def _wipe_event_spine_artifacts(db_path: Path) -> None:
    for suffix in ('', '-wal', '-shm'):
        target = db_path if suffix == '' else db_path.with_name(db_path.name + suffix)
        target.unlink(missing_ok=True)

class BacktestLauncher(Launcher):

    def __init__(self, trading_config: TradingConfig, instances: list[InstanceConfig], venue_adapter: VenueAdapter, *, event_spine: EventSpine | None=None, db_path: Path | None=None, historical_data: HistoricalData | None=None, max_allocation_per_trade_pct: Decimal | None=None, clock_tick_seconds: int | None=None) -> None:
        if event_spine is None and db_path is not None:
            _wipe_event_spine_artifacts(db_path)
        super().__init__(trading_config=trading_config, instances=instances, event_spine=event_spine, db_path=db_path, venue_adapter=venue_adapter, healthz_port=None)
        self._historical_data = historical_data or HistoricalData()
        if clock_tick_seconds is not None and clock_tick_seconds <= 0:
            raise ValueError(f'clock_tick_seconds must be > 0 when provided, got {clock_tick_seconds!r}.')
        if clock_tick_seconds is not None:
            _log.info('clock_tick_seconds=%s ignored — ReplayClock reads cadence from wired_sensor.interval_seconds (slice 0; #64)', clock_tick_seconds)
        self._submitted_commands = 0
        self._submit_lock = threading.Lock()
        self._venue_adapter = venue_adapter
        self._max_allocation_per_trade_pct = max_allocation_per_trade_pct
        self._routed_outcomes = 0
        self._predict_loop: PredictLoop | None = None
        self._outcome_loop: OutcomeLoop | None = None
        self._wired_sensors: tuple[WiredSensor, ...] | None = None

    def _start_trading(self) -> None:
        super()._start_trading()
        if self._trading is None:
            msg = 'BacktestLauncher._start_trading: super()._start_trading returned without populating self._trading.'
            raise RuntimeError(msg)
        trading = self._trading
        router = _make_outcome_router(self._outcome_queues, on_routed=self._record_routed_outcome)
        trading.execution_manager.set_on_trade_outcome(router)

    def _record_routed_outcome(self) -> None:
        with self._submit_lock:
            self._routed_outcomes += 1

    def _start_poller(self) -> None:
        kline_intervals = self._resolve_kline_intervals_from_manifests()
        params_by_kline_size = self._resolve_data_source_params_by_kline_size()
        poller = BacktestMarketDataPoller(kline_intervals=kline_intervals, historical_data=self._historical_data, params_by_kline_size=params_by_kline_size)
        setattr(self, '_poller', poller)
        poller.start()
        _log.info('backtest poller started', extra={'kline_sizes': sorted(kline_intervals)})

    def _resolve_data_source_params_by_kline_size(self) -> dict[int, dict[str, object]]:
        params_by_kline: dict[int, dict[str, object]] = {}
        for inst in self._instances:
            manifest = load_manifest(inst.manifest_path)
            for spec in manifest.strategies:
                for sensor_spec in spec.sensors:
                    cfg_params = self._data_source_params_from_experiment_dir(sensor_spec.experiment_dir)
                    kline_size_obj = cfg_params['kline_size']
                    if not isinstance(kline_size_obj, int):
                        msg = f'data_source params kline_size for {sensor_spec.experiment_dir} must be int, got {type(kline_size_obj).__name__}={kline_size_obj!r}'
                        raise TypeError(msg)
                    prior = params_by_kline.get(kline_size_obj)
                    if prior is not None and prior != cfg_params:
                        msg = f'conflicting data_source.params for kline_size={kline_size_obj}: prior={prior!r} new={cfg_params!r} (from {sensor_spec.experiment_dir}). Two sensors sharing a kline_size must declare identical data_source.params; otherwise the poller cannot serve a single coherent fetch.'
                        raise ValueError(msg)
                    params_by_kline[kline_size_obj] = cfg_params
        return params_by_kline

    @staticmethod
    def _data_source_params_from_experiment_dir(experiment_dir: Path) -> dict[str, object]:
        metadata_path = experiment_dir / 'metadata.json'
        if not metadata_path.is_file():
            msg = f'metadata.json not found at {metadata_path}'
            raise FileNotFoundError(msg)
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        sfd_module_name = metadata['sfd_module']
        sfd = importlib.import_module(sfd_module_name)
        limen_manifest = sfd.manifest()
        return dict(limen_manifest.data_source_config.params)

    def _resolve_kline_intervals_from_manifests(self) -> dict[int, int]:
        intervals: dict[int, int] = {}
        for inst in self._instances:
            manifest = load_manifest(inst.manifest_path)
            for spec in manifest.strategies:
                for sensor_spec in spec.sensors:
                    kline_size = self._kline_size_from_experiment_dir(sensor_spec.experiment_dir)
                    current = intervals.get(kline_size)
                    if current is None or sensor_spec.interval_seconds < current:
                        intervals[kline_size] = sensor_spec.interval_seconds
        return intervals

    @classmethod
    def _kline_size_from_experiment_dir(cls, experiment_dir: Path) -> int:
        params = cls._data_source_params_from_experiment_dir(experiment_dir)
        kline_size_obj = params['kline_size']
        if not isinstance(kline_size_obj, int):
            msg = f'data_source params kline_size for {experiment_dir} must be int, got {type(kline_size_obj).__name__}={kline_size_obj!r}'
            raise TypeError(msg)
        return kline_size_obj

    def _shutdown(self) -> None:
        import asyncio
        self._stop_healthz()
        for thread in self._nexus_threads:
            thread.join(timeout=30)
            if thread.is_alive():
                _log.warning('nexus thread did not finish within timeout', extra={'thread': thread.name})
        if self._poller is not None:
            self._poller.stop()
        if self._trading is not None and self._loop is not None:
            stop_future = asyncio.run_coroutine_threadsafe(self._trading.stop(), self._loop)
            stop_future.result(timeout=30)
        if self._owns_spine and self._db_conn is not None and (self._loop is not None):
            commit_future = asyncio.run_coroutine_threadsafe(self._db_conn.commit(), self._loop)
            commit_future.result(timeout=10)
            close_future = asyncio.run_coroutine_threadsafe(self._db_conn.close(), self._loop)
            close_future.result(timeout=10)
            self._db_conn = None
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5)
        if self._loop is not None and (not self._loop.is_closed()):
            self._loop.close()
        self._loop = None
        self._loop_thread = None
        _log.info('shutdown complete')

    def request_stop(self) -> None:
        self._stop_event.set()

    def _start_nexus_instances(self) -> None:
        if self._trading is None or self._loop is None:
            msg = 'trading not started'
            raise RuntimeError(msg)
        for inst in self._instances:
            t = threading.Thread(target=self._run_my_nexus_instance, args=(inst, self._outcome_queues[inst.account_id]), daemon=True, name=f'nexus-{inst.account_id}')
            self._nexus_threads.append(t)
            t.start()

    def _run_my_nexus_instance(self, inst: InstanceConfig, outcome_queue: queue.Queue[TradeOutcome]) -> None:
        if self._trading is None or self._loop is None:
            msg = 'BacktestLauncher._run_nexus_instance: trading/loop not initialised'
            raise RuntimeError(msg)
        state_store = StateStore(inst.state_dir)
        praxis_outbound = PraxisOutbound(submit_fn=self._trading.submit_command, loop=self._loop, register_fn=self._trading.register_account, unregister_fn=self._trading.unregister_account, pull_positions_fn=self._trading.pull_positions)
        sequencer = StartupSequencer(state_store=state_store, manifest_path=inst.manifest_path, strategies_base_path=inst.strategies_base_path, strategy_state_path=inst.strategy_state_path, praxis_outbound=praxis_outbound)
        runner = sequencer.start()
        nexus_config = self._build_nexus_instance_config(inst)
        manifest = load_manifest(inst.manifest_path)
        allocated_capital = manifest.allocated_capital
        pipeline, controller, capital_state = build_validation_pipeline(nexus_config=nexus_config, capital_pool=allocated_capital, max_allocation_per_trade_pct=self._max_allocation_per_trade_pct)
        state = InstanceState(capital=capital_state)
        self._capital_state = capital_state
        self._capital_initial_pool = capital_state.capital_pool
        self._capital_initial_total = capital_totals(capital_state).total
        self._capital_tracker = CapitalLifecycleTracker(controller)
        self._declared_stops: dict[str, Decimal] = {}
        _install_capital_adapter_wrapper(adapter=self._venue_adapter, tracker=self._capital_tracker, capital_state=capital_state, initial_pool=self._capital_initial_pool, declared_stops=self._declared_stops)
        from backtest_simulator.venue.simulated import SimulatedVenueAdapter
        sva = self._venue_adapter if isinstance(self._venue_adapter, SimulatedVenueAdapter) else None
        touch_provider = sva.touch_for_symbol if sva is not None else None
        tick_provider = sva.tick_for_symbol if sva is not None else None
        action_submit = build_action_submitter(SubmitterBindings(nexus_config=nexus_config, state=state, praxis_outbound=praxis_outbound, validation_pipeline=pipeline, capital_controller=controller, strategy_budget=allocated_capital, touch_provider=touch_provider, tick_provider=tick_provider), on_reservation=self._record_reservation, on_submit=self._record_submitted_command)

        def market_data_provider(kline_size: int) -> pl.DataFrame:
            if self._poller is None:
                msg = 'BacktestLauncher.market_data_provider: poller not initialised'
                raise RuntimeError(msg)
            return self._poller.get_market_data(kline_size)

        def context_provider(_strategy_id: str) -> StrategyContext:
            return StrategyContext(positions=(), capital_available=Decimal('0'), operational_mode=OperationalMode.ACTIVE)
        if sequencer.timer_specs:
            raise RuntimeError(f'BacktestLauncher: sequencer declares non-empty timer_specs ({list(sequencer.timer_specs)}); TimerLoop is not supported in the synchronous replay path. Strategy-authored on_timer callbacks require a follow-up RFC adding tick_once-style entry points to TimerLoop.')
        predict_loop = PredictLoop(runner=runner, wired_sensors=sequencer.wired_sensors, market_data_provider=market_data_provider, context_provider=context_provider, action_submit=action_submit)
        nexus_outcome_queue: queue.Queue[NexusTradeOutcome] = cast('queue.Queue[NexusTradeOutcome]', outcome_queue)
        praxis_inbound = PraxisInbound(outcome_queue=nexus_outcome_queue, poll_timeout=0.0)
        outcome_loop = _build_outcome_loop(runner=runner, praxis_inbound=praxis_inbound, state=state, context_provider=context_provider, wired_sensors=sequencer.wired_sensors, action_submit=action_submit)
        self._predict_loop = predict_loop
        self._outcome_loop = outcome_loop
        self._wired_sensors = tuple(sequencer.wired_sensors)
        self._nexus_running.set()
        self._stop_event.wait()
        timer_loop: TimerLoop | None = None
        sequencer_manifest = sequencer.manifest
        sequencer_state = sequencer.instance_state
        if sequencer_manifest is None or sequencer_state is None:
            msg = f'StartupSequencer did not populate manifest/state after start(): manifest={sequencer_manifest!r} state={sequencer_state!r}'
            raise RuntimeError(msg)
        shutdown = ShutdownSequencer(runner=runner, manifest=sequencer_manifest, state_store=state_store, state=sequencer_state, strategy_state_path=inst.strategy_state_path or inst.state_dir / 'strategy_state', predict_loop=predict_loop, timer_loop=timer_loop, praxis_outbound=praxis_outbound, praxis_inbound=praxis_inbound, account_id=inst.account_id)
        shutdown.shutdown()

    @staticmethod
    def _build_nexus_instance_config(inst: InstanceConfig) -> NexusInstanceConfig:
        return NexusInstanceConfig(account_id=inst.account_id, venue='binance_spot_simulated')

    def run_window(self, start: datetime, end: datetime) -> None:
        if end <= start:
            msg = f'run_window: end {end} must be after start {start}'
            raise ValueError(msg)
        self._nexus_running = threading.Event()
        launch_thread = threading.Thread(target=self.launch, daemon=True, name='backtest-launch')
        launch_thread.start()
        self._wait_until_trading_ready()
        self._wait_until_all_nexus_running()
        self._install_real_loop_time()
        try:
            with accelerated_clock(start) as freezer:
                try:
                    pl, ol, ws = (self._predict_loop, self._outcome_loop, self._wired_sensors)
                    if pl is None or ol is None or ws is None:
                        raise RuntimeError('run_window: predict_loop / outcome_loop / wired_sensors were not stashed by _run_my_nexus_instance')
                    ReplayClock().drive_window(window_start=start, window_end=end, wired_sensors=ws, predict_loop=pl, outcome_loop=ol, drain_pending_submits=self._drain_pending_submits, freezer=freezer)
                finally:
                    self.request_stop()
        finally:
            launch_thread.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
            if launch_thread.is_alive():
                msg = f'backtest launch thread did not terminate within {_SHUTDOWN_TIMEOUT_SECONDS}s of shutdown'
                raise RuntimeError(msg)

    def _install_real_loop_time(self) -> None:
        if self._loop is None:
            msg = '_install_real_loop_time: event loop not running yet; call only after _wait_until_trading_ready'
            raise RuntimeError(msg)
        setattr(self._loop, 'time', _REAL_MONOTONIC)

    def _wait_until_trading_ready(self) -> None:
        start = os.times()[4]
        while os.times()[4] - start < _BOOT_TIMEOUT_SECONDS:
            if self._trading is not None and self._trading.started:
                return
            time.sleep(_POLL_INTERVAL_SECONDS)
        msg = f'Trading did not start within {_BOOT_TIMEOUT_SECONDS}s of real wall time'
        raise RuntimeError(msg)

    def _wait_until_all_nexus_running(self) -> None:
        if not self._nexus_running.wait(timeout=_NEXUS_RUN_TIMEOUT_SECONDS):
            msg = f'Nexus instance did not reach running within {_NEXUS_RUN_TIMEOUT_SECONDS}s of real wall time'
            raise RuntimeError(msg)

    def _record_submitted_command(self, command_id: str) -> None:
        del command_id
        with self._submit_lock:
            self._submitted_commands += 1

    def _record_reservation(self, command_id: str, decision: ValidationDecision, context: ValidationRequestContext, action: Action) -> None:
        reservation = decision.reservation
        if reservation is None:
            return
        declared_stop_price = _extract_declared_stop_price(action)
        self._capital_tracker.record_reservation(command_id=command_id, reservation_id=reservation.reservation_id, strategy_id=context.strategy_id, notional=reservation.notional, estimated_fees=reservation.estimated_fees, declared_stop_price=declared_stop_price)
        _log.info('capital lifecycle: check_and_reserve command_id=%s reservation_id=%s notional=%s declared_stop=%s', command_id, reservation.reservation_id, reservation.notional, declared_stop_price)
        assert_conservation(self._capital_state, self._capital_initial_pool, context=f'check_and_reserve command_id={command_id}')

    def _delivered_command_count(self) -> int:
        adapter = self._venue_adapter
        active = getattr(adapter, '_accounts', {})
        history = getattr(adapter, '_history', {})
        return sum(len(a.orders) for a in active.values()) + sum(len(a.orders) for a in history.values())

    def _drain_pending_submits(self) -> None:
        drain_start = os.times()[4]
        while True:
            with self._submit_lock:
                submitted = self._submitted_commands
                routed = self._routed_outcomes
            delivered = self._delivered_command_count()
            if delivered >= submitted and routed >= delivered:
                elapsed = os.times()[4] - drain_start
                if elapsed > _DRAIN_SLOW_WARN_SECONDS:
                    _log.warning('drain slow: submitted=%d delivered=%d routed=%d wallclock=%.2fs', submitted, delivered, routed, elapsed)
                return
            if os.times()[4] - drain_start > _DRAIN_TIMEOUT_SECONDS:
                diag = self._praxis_queue_sizes()
                msg = f'drain timeout: submitted={submitted} delivered={delivered} routed={routed} after {_DRAIN_TIMEOUT_SECONDS:.2f}s; praxis_state={diag}'
                raise DrainTimeoutError(msg)
            self._yield_to_loop_once()

    def _yield_to_loop_once(self) -> None:
        if self._loop is None:
            time.sleep(_DRAIN_POLL_INTERVAL_SECONDS)
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(asyncio.sleep(0), self._loop)
            fut.result(timeout=_DRAIN_POLL_INTERVAL_SECONDS)
        except TimeoutError:
            _log.debug('yield-to-loop timed out; event loop busy')

    def _praxis_queue_sizes(self) -> dict[str, dict[str, object]]:
        if self._trading is None:
            return {}
        exec_mgr = getattr(self._trading, '_execution_manager', None)
        if exec_mgr is None:
            return {}
        accounts = getattr(exec_mgr, '_accounts', {})
        out: dict[str, dict[str, object]] = {}
        for aid, rt in accounts.items():
            task = getattr(rt, 'task', None)
            entry: dict[str, object] = {'cmd_q': rt.command_queue.qsize(), 'ws_q': rt.ws_event_queue.qsize(), 'prio_q': rt.priority_queue.qsize()}
            if task is not None:
                entry['task_done'] = task.done()
                if task.done():
                    exc = task.exception() if not task.cancelled() else 'cancelled'
                    entry['task_exc'] = repr(exc)
                else:
                    frames = task.get_stack(limit=3)
                    entry['await_at'] = [f'{f.f_code.co_name}:{f.f_lineno}' for f in frames]
            out[aid] = entry
        return out
