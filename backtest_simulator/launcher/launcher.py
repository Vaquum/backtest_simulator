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
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import FrameType

import polars as pl
from limen import HistoricalData
from nexus.core.domain.capital_state import CapitalState
from nexus.core.domain.enums import OperationalMode
from nexus.core.domain.instance_state import InstanceState
from nexus.core.validator.pipeline_models import (
    ValidationDecision,
    ValidationRequestContext,
)
from nexus.infrastructure.manifest import load_manifest
from nexus.infrastructure.praxis_connector.praxis_inbound import PraxisInbound
from nexus.infrastructure.praxis_connector.praxis_outbound import PraxisOutbound
from nexus.infrastructure.state_store import StateStore
from nexus.instance_config import InstanceConfig as NexusInstanceConfig
from nexus.startup.sequencer import StartupSequencer
from nexus.startup.shutdown_sequencer import ShutdownSequencer
from nexus.strategy.action import Action
from nexus.strategy.context import StrategyContext
from nexus.strategy.predict_loop import PredictLoop
from nexus.strategy.timer_loop import TimerLoop
from praxis.core.domain.enums import OrderSide, OrderType
from nexus.infrastructure.praxis_connector.trade_outcome import TradeOutcome
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
from backtest_simulator.launcher.action_submitter import (
    SubmitterBindings,
    build_action_submitter,
)
from backtest_simulator.launcher.clock import accelerated_clock
from backtest_simulator.launcher.poller import BacktestMarketDataPoller

_log = logging.getLogger(__name__)

_BOOT_TIMEOUT_SECONDS = 60
_NEXUS_RUN_TIMEOUT_SECONDS = 120
_POLL_INTERVAL_SECONDS = 0.05
_REAL_TIME_CAP_SECONDS = 600
_SHUTDOWN_TIMEOUT_SECONDS = 30

# Capture `time.monotonic` at module load — BEFORE any `freeze_time`
# block could patch it — so the real monotonic reference survives the
# freeze. `asyncio.BaseEventLoop.time` defaults to `time.monotonic`,
# which under freezegun returns frozen monotonic; that makes every
# `asyncio.sleep(delay)` callback scheduled for `frozen_now + delay`
# wait against a loop.time that stays at `frozen_now` forever — the
# deadline never passes, the sleep never fires. Rebinding `loop.time`
# to this captured real clock keeps the asyncio scheduler on real
# wall time, regardless of what freezegun has patched globally.
_REAL_MONOTONIC = time.monotonic
# Main clock tick: 120 frozen seconds per iteration with a real-time
# yield so the asyncio account_loop + Timer threads get CPU between
# ticks. Strategy sklearn inference (~50ms real) keeps main ticking
# during that window; the tinier the pause, the bigger the frozen-time
# drift per strategy tick (drift ~= sklearn_time x tick_seconds / pause).
# With 120s ticks at 0.01s pause, drift is ~600s = 10 min per tick —
# small enough that PredictLoop Timer firings still land at the intended
# frozen schedule, and the decoder's proven `preds 0→1→0→1` transition
# cluster on the 14:00-04:00 window fires cleanly. 120s also halves the
# main iteration count vs 60s ticks, cutting clock-advance real time
# from ~8.4s to ~4.2s on a 14h window — the biggest single lever to
# keep the e2e profile inside the 10s budget.
_CLOCK_TICK_SECONDS = timedelta(seconds=120)
_CLOCK_TICK_REAL_PAUSE_SECONDS = 0.01
# Drain settings: Praxis's account_loop polls its queue every 0.1s. The
# drain wakes up at 0.02s granularity to catch fresh dispatches with
# minimal delay after they complete; it bounds total drain real time so
# a stuck command can't run the 10s budget into the ground.
_DRAIN_POLL_INTERVAL_SECONDS = 0.02
# Drain must stay well inside the 10s total e2e budget. 1.5s is enough
# for `_process_command` to complete its ClickHouse round-trip
# (`query_order_book` for slippage + `submit_order` walking the trade
# window) even on a slow network. Anything longer is a real bug — the
# whole run aborts so the operator sees the failure on the next line,
# instead of letting frozen-minute ticks each spend seconds on drain
# and silently blow the budget into the minutes.
_DRAIN_TIMEOUT_SECONDS = 1.5


class DrainTimeoutError(RuntimeError):
    """Raised when a submitted command doesn't dispatch to the venue.

    Carries the per-account diagnostic so the e2e log surfaces the exact
    state (command queue size, async task's current await, etc.) at the
    moment of failure. The backtest then unwinds cleanly instead of
    spinning on repeat-warnings for the rest of the frozen window.
    """


class CapitalOvershootError(RuntimeError):
    """Raised when a fill's realised notional exceeds the reservation.

    The action-submitter reserves `reference_price * size * (1 + buffer)`
    at validation time. Actual fills that overshoot that buffered
    amount indicate the strategy used more capital than the CAPITAL
    stage gated — silently capping the ledger fill to the reservation
    would bypass the gate. Raising here surfaces the overshoot so
    the operator can raise the buffer or tighten the strategy.
    """


class CapitalPartialFillError(RuntimeError):
    """Raised when a submit returns `PARTIALLY_FILLED`.

    Part 2 lifecycle expects terminal FILLED or REJECTED — the VWAP
    MARKET walk produces one or the other. A PARTIALLY_FILLED status
    means residual working-notional will remain after we pop the
    lifecycle, silently under-counting committed capital.
    """


def _extract_declared_stop_price(action: Action) -> Decimal | None:
    """Pull `declared_stop_price` from the Nexus Action.

    Reads `action.execution_params['stop_price']` and returns None
    when absent or blank.

    The strategy template writes `execution_params['stop_price']` as a
    string for YAML roundtrip; this function coerces to Decimal.
    `execution_params` is a `MappingProxyType`, not a `dict`, so we
    use the `collections.abc.Mapping` protocol.
    """
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
    """Bundled state the adapter-wrapper helpers share.

    Used instead of passing `tracker`, `capital_state`, `initial_pool`
    as three separate positional args on every helper call.
    """

    tracker: CapitalLifecycleTracker
    capital_state: CapitalState
    initial_pool: Decimal


def _check_tracker_match_required(
    pre_match_command_id: str | None,
    side: OrderSide,
    client_order_id: str | None,
) -> None:
    """Raise when a BUY submit has no tracker match.

    Part 2 SELL-as-close convention: the action_submitter skips
    CAPITAL reservation for SELL actions (they are exits, not new
    reservations). Those show up here with side=SELL and no tracker
    entry — that's expected, not an error.
    """
    if pre_match_command_id is not None:
        return
    if side.name != 'BUY':
        return
    msg = (
        f'capital lifecycle: BUY submit for '
        f'client_order_id={client_order_id!r} has no matching '
        f'reservation in tracker. Part 2 honesty requires '
        f'every BUY to clear the CAPITAL stage BEFORE dispatch; '
        f'missing match means the action-submitter skipped the '
        f'reservation OR the client_order_id format has changed.'
    )
    raise RuntimeError(msg)


def _maybe_inject_declared_stop(
    pre_match_command_id: str | None,
    client_order_id: str | None,
    stop_price: Decimal | None,
    tracker: CapitalLifecycleTracker,
    declared_stops: dict[str, Decimal],
) -> Decimal | None:
    """Return the stop_price to forward to `submit_order`.

    Praxis's `validate_trade_command` strips stop_price from MARKET
    orders before the command reaches here. We look up the command
    via its client_order_id and inject the honest declared stop so
    FillModel.apply_stop can enforce it during the trade walk.
    """
    if pre_match_command_id is None or stop_price is not None:
        return stop_price
    declared_stop = tracker.declared_stop_for_command(pre_match_command_id)
    if declared_stop is None:
        return stop_price
    if client_order_id is not None:
        declared_stops[client_order_id] = declared_stop
    return declared_stop


def _sum_fill_totals(result: SubmitResult) -> tuple[Decimal, Decimal]:
    """Sum the result's immediate_fills into (notional, fees)."""
    fill_notional = Decimal('0')
    fill_fees = Decimal('0')
    for fill in result.immediate_fills:
        fill_notional += Decimal(str(fill.qty)) * Decimal(str(fill.price))
        fill_fees += Decimal(str(fill.fee))
    return fill_notional, fill_fees


def _finalize_successful_fill(
    ctx: _LifecycleContext,
    command_id: str,
    result: SubmitResult,
    venue_order_id: str,
    status_name: str,
) -> None:
    """Fire record_sent + record_ack_and_fill with conservation checks.

    Fails loud on capital overshoot or PARTIALLY_FILLED status — Part 2
    honesty requires terminal FILLED/REJECTED only, and reservation
    buffering must absorb any real slippage without silent capping.
    """
    ctx.tracker.record_sent(command_id, venue_order_id)
    assert_conservation(
        ctx.capital_state, ctx.initial_pool,
        context=f'send_order command_id={command_id}',
    )
    fill_notional, fill_fees = _sum_fill_totals(result)
    reservation_notional = ctx.tracker.declared_reservation_for_command(command_id)
    if reservation_notional is not None and fill_notional > reservation_notional:
        slippage = fill_notional - reservation_notional
        msg = (
            f'capital overshoot: fill_notional={fill_notional} exceeds '
            f'reserved={reservation_notional} for command_id={command_id} '
            f'by {slippage}. Raise `_NOTIONAL_RESERVATION_BUFFER` in '
            f'backtest_simulator.launcher.action_submitter to absorb this '
            f'slippage honestly; silently capping here would bypass '
            f'CAPITAL gating.'
        )
        raise CapitalOvershootError(msg)
    if status_name == 'PARTIALLY_FILLED':
        msg = (
            f'PARTIALLY_FILLED command_id={command_id} venue_order_id='
            f'{venue_order_id}: Part 2 lifecycle expects terminal '
            f'FILLED or REJECTED. Partial residual working-notional '
            f'would not be drained and the capital ledger would '
            f'under-count. Investigate why the MARKET walk did not '
            f'fully fill the order.'
        )
        raise CapitalPartialFillError(msg)
    ctx.tracker.record_ack_and_fill(
        command_id, venue_order_id, fill_notional, fill_fees,
    )
    assert_conservation(
        ctx.capital_state, ctx.initial_pool,
        context=f'order_fill command_id={command_id}',
    )
    _log.info(
        'capital lifecycle: command_id=%s reserve->send->ack->fill '
        'venue_order_id=%s fill_notional=%s fees=%s',
        command_id, venue_order_id, fill_notional, fill_fees,
    )


def _install_capital_adapter_wrapper(
    adapter: VenueAdapter,
    tracker: CapitalLifecycleTracker,
    capital_state: CapitalState,
    initial_pool: Decimal,
    declared_stops: dict[str, Decimal],
) -> None:
    """Wrap `adapter.submit_order` so the capital lifecycle completes in-line.

    The 4-step lifecycle is `check_and_reserve → send_order → order_ack
    → order_fill`. `check_and_reserve` happens in the action_submitter
    (CAPITAL stage); the remaining three must fire once the venue has
    produced its SubmitResult. Because `SimulatedVenueAdapter.submit_order`
    is synchronous from the caller's point of view — fills are returned
    inline on the same await — we collapse send_order + order_ack +
    order_fill into one wrapper around submit_order.

    The original `submit_order` is called normally; we inspect the
    `SubmitResult`, look up the pending reservation by `client_order_id`
    (which embeds the Nexus `command_id`), and feed the lifecycle
    events into the tracker in order. Conservation is asserted after
    each transition so any drift fails loud at the offending boundary.
    """
    original_submit = adapter.submit_order
    ctx = _LifecycleContext(
        tracker=tracker, capital_state=capital_state, initial_pool=initial_pool,
    )

    async def wrapped_submit(
        account_id: str, symbol: str, side: OrderSide, order_type: OrderType,
        qty: Decimal, *,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        stop_limit_price: Decimal | None = None,
        client_order_id: str | None = None,
        time_in_force: str | None = None,
    ) -> SubmitResult:
        pre_match_command_id = _match_command_id(tracker, client_order_id)
        _check_tracker_match_required(pre_match_command_id, side, client_order_id)
        stop_price = _maybe_inject_declared_stop(
            pre_match_command_id, client_order_id, stop_price,
            tracker, declared_stops,
        )
        # If the original adapter raises mid-submit we MUST release
        # the capital reservation — otherwise the reservation stays
        # locked forever and the lifecycle tracker never pops it,
        # failing the `pending_count == 0` terminal check.
        try:
            result = await original_submit(
                account_id, symbol, side, order_type, qty,
                price=price, stop_price=stop_price,
                stop_limit_price=stop_limit_price,
                client_order_id=client_order_id,
                time_in_force=time_in_force,
            )
        except Exception:
            if pre_match_command_id is not None:
                tracker.record_rejection(pre_match_command_id, '')
                assert_conservation(
                    capital_state, initial_pool,
                    context=f'adapter_raised command_id={pre_match_command_id}',
                )
            raise
        command_id = _match_command_id(tracker, client_order_id)
        if command_id is None:
            _log.info(
                'capital lifecycle: no pending command matches '
                'client_order_id=%s (expected for SELL exits); '
                'pending_count=%d',
                client_order_id, tracker.pending_count,
            )
            return result
        venue_order_id = result.venue_order_id
        status_name = result.status.name
        if status_name == 'REJECTED':
            tracker.record_rejection(command_id, venue_order_id)
            assert_conservation(
                capital_state, initial_pool,
                context=f'order_reject command_id={command_id}',
            )
            _log.info(
                'capital lifecycle: command_id=%s REJECTED (reservation released)',
                command_id,
            )
            return result
        _finalize_successful_fill(
            ctx, command_id, result, venue_order_id, status_name,
        )
        return result

    setattr(adapter, 'submit_order', wrapped_submit)


def _match_command_id(
    tracker: CapitalLifecycleTracker, client_order_id: str | None,
) -> str | None:
    """Map Nexus's client_order_id back to the tracker's command_id.

    The Nexus form is `SS-<command_prefix>-<seq>`; we map it back to
    the full `command_id` the tracker recorded.

    Nexus's `generate_client_order_id` takes the original `command_id`
    and strips dashes, then uses the first 16 hex chars as the prefix.
    We match via the tracker's public `match_pending_by_prefix`
    accessor instead of reaching into `_pending`.
    """
    if client_order_id is None:
        return None
    parts = client_order_id.split('-')
    if len(parts) < 3:
        return None
    prefix = parts[1]
    return tracker.match_pending_by_prefix(prefix)


def _wipe_event_spine_artifacts(db_path: Path) -> None:
    """Remove the sqlite db + WAL/SHM siblings if they exist.

    Left-over WAL/SHM from a previously hard-killed process can hold a
    lock that blocks new writes ("database is locked"). A backtest is
    meant to produce an event trail for THIS run only, so we wipe the
    slate deterministically. `missing_ok=True` on `unlink` keeps this
    idempotent across first-run and re-run.
    """
    for suffix in ('', '-wal', '-shm'):
        target = db_path if suffix == '' else db_path.with_name(db_path.name + suffix)
        target.unlink(missing_ok=True)


class BacktestLauncher(Launcher):
    """Praxis `Launcher` subclass that swaps live seams for historical ones.

    Three overrides — the boundary seams this shim is allowed to swap:
      1. `_start_poller` — `BacktestMarketDataPoller` reads klines from
         Limen HistoricalData instead of Binance REST.
      2. `_signal_handler` — no-op; backtests terminate on window end
         rather than SIGINT/SIGTERM.
      3. Venue seam — passed via `venue_adapter=` in Launcher's existing
         constructor; our `SimulatedVenueAdapter` plugs in unchanged.

    Nothing else is reimplemented. `_start_trading`, `_start_nexus_instances`,
    `_run_nexus_instance`, StartupSequencer, PredictLoop, TimerLoop, and
    PraxisOutbound all run from upstream Nexus / Praxis unchanged — the
    apples-to-apples guarantee requires that production plumbing is the
    same plumbing the backtest drives.
    """

    # `_poller` is inherited from praxis.Launcher with type
    # `MarketDataPoller | None`. The backtest substitutes a duck-typed
    # `BacktestMarketDataPoller` (same public surface — start/stop/
    # running/get_market_data/add_kline_size/remove_kline_size). We
    # don't re-annotate here: pyright's variance rules treat any
    # subclass-side narrowing of a mutable class-level attribute as
    # `reportIncompatibleVariableOverride`. Instead, every consumer
    # site that calls `self._poller.start()` etc. is wrapped in an
    # `assert self._poller is not None` so reportOptionalMemberAccess
    # is satisfied without changing the parent's type.

    def __init__(
        self,
        trading_config: TradingConfig,
        instances: list[InstanceConfig],
        venue_adapter: VenueAdapter,
        *,
        event_spine: EventSpine | None = None,
        db_path: Path | None = None,
        historical_data: HistoricalData | None = None,
    ) -> None:
        # A backtest run is one window, one set of events — not a live
        # service that needs durable state across processes. Starting
        # from a fresh EventSpine sqlite prevents `sqlite3.OperationalError:
        # database is locked` when a prior run left the WAL/SHM sidecars
        # in a half-open state (e.g. after a hard kill). The `db_path`
        # parent directory is preserved; only the db file + `-wal`/`-shm`
        # siblings are removed. A pre-built `event_spine` takes precedence
        # over `db_path`, so callers that truly want persistence can hand
        # one in and opt out of the wipe.
        if event_spine is None and db_path is not None:
            _wipe_event_spine_artifacts(db_path)
        super().__init__(
            trading_config=trading_config,
            instances=instances,
            event_spine=event_spine,
            db_path=db_path,
            venue_adapter=venue_adapter,
            healthz_port=None,
        )
        self._historical_data = historical_data or HistoricalData()
        # Synchronous-drain counters. `_submitted_commands` is bumped by
        # the action_submitter's `on_submit` callback after every
        # successful `praxis_outbound.send_command`; the main clock loop
        # then blocks until the venue adapter has delivered an equal
        # number of orders (filled, partial, or rejected) before
        # advancing the next frozen tick.
        self._submitted_commands = 0
        self._submit_lock = threading.Lock()
        self._venue_adapter = venue_adapter

    def _start_poller(self) -> None:
        kline_intervals = self._resolve_kline_intervals_from_manifests()
        self._poller = BacktestMarketDataPoller(
            kline_intervals=kline_intervals,
            historical_data=self._historical_data,
        )
        self._poller.start()
        _log.info(
            'backtest poller started',
            extra={'kline_sizes': sorted(kline_intervals)},
        )

    def _resolve_kline_intervals_from_manifests(self) -> dict[int, int]:
        # Praxis's inherited `_collect_kline_intervals` reads
        # `sensor._limen_manifest.data_source_config.params['kline_size']`,
        # but that attribute is never attached to the frozen `SensorSpec`
        # dataclass returned from `load_manifest` — it's an artifact of
        # an earlier wiring step that no longer happens in this release.
        # Result: the inherited extractor returns `{}` and the poller
        # fetches no klines, so the strategy's `on_signal` gets called
        # but `signal.values` is empty (no probability, no close).
        #
        # We read the kline_size straight from the experiment_dir's
        # `metadata.json` → sfd_module → `sfd.manifest().data_source_config`
        # instead. That matches what Limen's `Trainer` does when it
        # reconstructs the sensor, so the poller's data source and
        # Trainer's data source stay aligned.
        intervals: dict[int, int] = {}
        for inst in self._instances:
            manifest = load_manifest(inst.manifest_path)
            for spec in manifest.strategies:
                for sensor_spec in spec.sensors:
                    kline_size = self._kline_size_from_experiment_dir(
                        sensor_spec.experiment_dir,
                    )
                    current = intervals.get(kline_size)
                    if current is None or sensor_spec.interval_seconds < current:
                        intervals[kline_size] = sensor_spec.interval_seconds
        return intervals

    @staticmethod
    def _kline_size_from_experiment_dir(experiment_dir: Path) -> int:
        metadata_path = experiment_dir / 'metadata.json'
        if not metadata_path.is_file():
            msg = f'metadata.json not found at {metadata_path}'
            raise FileNotFoundError(msg)
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        sfd_module_name = metadata['sfd_module']
        sfd = importlib.import_module(sfd_module_name)
        limen_manifest = sfd.manifest()
        return int(limen_manifest.data_source_config.params['kline_size'])

    def _signal_handler(self, _signum: int, _frame: FrameType | None) -> None:
        # Backtests aren't daemons; termination is driven by the outer
        # harness calling `request_stop()` when the window ends, not by
        # SIGINT/SIGTERM landing mid-run.
        _log.info('backtest launcher ignoring external signal')

    def request_stop(self) -> None:
        """Ask `launch()` to return. Outer harness uses this at window end."""
        self._stop_event.set()

    def _run_nexus_instance(
        self, inst: InstanceConfig, outcome_queue: queue.Queue[TradeOutcome],
    ) -> None:
        """Mirror praxis.Launcher._run_nexus_instance but wire `action_submit`.

        Upstream Launcher leaves `PredictLoop(action_submit=None)`, which
        per Nexus's docstring means "returned actions are discarded
        (back-compat for tests that do not exercise the submission
        path)". For a real backtest we need every ENTER/EXIT from
        on_signal / on_timer to flow through translate + PraxisOutbound
        into Trading + SimulatedVenueAdapter. This method is a copy of
        the upstream body with `action_submit=build_action_submitter(...)`
        injected into both `PredictLoop` and `TimerLoop`.
        """
        if self._trading is None or self._loop is None:
            msg = 'BacktestLauncher._run_nexus_instance: trading/loop not initialised'
            raise RuntimeError(msg)
        state_store = StateStore(inst.state_dir)
        praxis_outbound = PraxisOutbound(
            submit_fn=self._trading.submit_command,
            loop=self._loop,
            register_fn=self._trading.register_account,
            unregister_fn=self._trading.unregister_account,
            pull_positions_fn=self._trading.pull_positions,
        )
        sequencer = StartupSequencer(
            state_store=state_store,
            manifest_path=inst.manifest_path,
            strategies_base_path=inst.strategies_base_path,
            strategy_state_path=inst.strategy_state_path,
            praxis_outbound=praxis_outbound,
        )
        runner = sequencer.start()
        nexus_config = self._build_nexus_instance_config(inst)
        # Part 2: real CAPITAL ValidationPipeline + 4-step lifecycle. The
        # `capital_pool` is the manifest's `allocated_capital` (that is
        # the total capital this backtest is allowed to deploy on the
        # account). The InstanceState holds the SAME CapitalState the
        # CAPITAL validator guards — Nexus's `validate_capital_stage`
        # reads via `context.state.capital`, so we wire both sides to
        # one shared `CapitalState` object.
        manifest = load_manifest(inst.manifest_path)
        allocated_capital = manifest.allocated_capital
        pipeline, controller, capital_state = build_validation_pipeline(
            capital_pool=allocated_capital,
        )
        state = InstanceState(capital=capital_state)
        self._capital_state = capital_state
        self._capital_initial_pool = capital_state.capital_pool
        self._capital_initial_total = capital_totals(capital_state).total
        self._capital_tracker = CapitalLifecycleTracker(controller)
        # Per-trade honesty record: client_order_id -> declared_stop_price.
        # Populated at reservation time; read by `compute_r_per_trade`
        # post-run to produce the Part 2 honest-R metric.
        self._declared_stops: dict[str, Decimal] = {}
        _install_capital_adapter_wrapper(
            adapter=self._venue_adapter,
            tracker=self._capital_tracker,
            capital_state=capital_state,
            initial_pool=self._capital_initial_pool,
            declared_stops=self._declared_stops,
        )
        action_submit = build_action_submitter(
            SubmitterBindings(
                nexus_config=nexus_config, state=state,
                praxis_outbound=praxis_outbound,
                validation_pipeline=pipeline,
                strategy_budget=allocated_capital,
            ),
            on_reservation=self._record_reservation,
            on_submit=self._record_submitted_command,
        )

        def market_data_provider(kline_size: int) -> pl.DataFrame:
            if self._poller is None:
                msg = 'BacktestLauncher.market_data_provider: poller not initialised'
                raise RuntimeError(msg)
            return self._poller.get_market_data(kline_size)

        def context_provider(_strategy_id: str) -> StrategyContext:
            return StrategyContext(
                positions=(),
                capital_available=Decimal('0'),
                operational_mode=OperationalMode.ACTIVE,
            )

        predict_loop = PredictLoop(
            runner=runner, wired_sensors=sequencer.wired_sensors,
            market_data_provider=market_data_provider,
            context_provider=context_provider,
            action_submit=action_submit,
        )
        predict_loop.start()

        timer_loop: TimerLoop | None = None
        if sequencer.timer_specs:
            timer_loop = TimerLoop(
                runner=runner, strategy_timers=sequencer.timer_specs,
                context_provider=context_provider,
                action_submit=action_submit,
            )
            timer_loop.start()

        praxis_inbound = PraxisInbound(outcome_queue=outcome_queue)
        # Direct event signal — no log-message interception. The running
        # event is initialised in `run_window` on `self._nexus_running`.
        self._nexus_running.set()
        self._stop_event.wait()

        # `sequencer.start()` has returned, so `manifest` and
        # `instance_state` are both populated — but the public
        # properties are typed `| None` for the pre-start window. The
        # None-guard is a runtime assertion that the invariant still
        # holds; if it ever fails the shutdown path gets a concrete
        # error instead of a downstream NoneType crash.
        sequencer_manifest = sequencer.manifest
        sequencer_state = sequencer.instance_state
        if sequencer_manifest is None or sequencer_state is None:
            msg = (
                f'StartupSequencer did not populate manifest/state after '
                f'start(): manifest={sequencer_manifest!r} '
                f'state={sequencer_state!r}'
            )
            raise RuntimeError(msg)
        shutdown = ShutdownSequencer(
            runner=runner,
            manifest=sequencer_manifest,
            state_store=state_store,
            state=sequencer_state,
            strategy_state_path=inst.strategy_state_path or inst.state_dir / 'strategy_state',
            predict_loop=predict_loop,
            timer_loop=timer_loop,
            praxis_outbound=praxis_outbound,
            praxis_inbound=praxis_inbound,
            account_id=inst.account_id,
        )
        shutdown.shutdown()

    @staticmethod
    def _build_nexus_instance_config(inst: InstanceConfig) -> NexusInstanceConfig:
        # translate_to_trade_command reads `config.account_id`, `config.venue`,
        # `config.stp_mode` off this object. Default STPMode.CANCEL_TAKER
        # mirrors production; venue string is unused by the simulated path
        # but must be non-empty.
        return NexusInstanceConfig(account_id=inst.account_id, venue='binance_spot_simulated')

    def run_window(self, start: datetime, end: datetime) -> None:
        """Run the backtest from `start` to `end`; boot then accelerate.

        Boot (Trading + Nexus StartupSequencer + Trainer + strategy
        on_startup) runs under REAL wall time. Nexus's Trainer re-streams
        the full BTCUSDT-klines dataset from HuggingFace on every boot
        (Limen has no on-disk cache) and that fetch takes ~20 real
        seconds plus sklearn refit. Running that under `accelerated_clock`
        lets concurrent asyncio.sleep sites tick the frozen clock
        forward during the real-time HTTPS wait and burn through the
        backtest window before PredictLoop ticks at all.

        The flow instead:
          1. Launch thread starts under real time — Trainer fetches at
             full speed, SSL validates against a 2026 real clock, and
             every Nexus instance logs 'nexus instance running' when its
             StartupSequencer completes and PredictLoop.start() has
             been called.
          2. A log handler keyed on that message releases a barrier
             once all instances are up.
          3. We enter `accelerated_clock(start)` at that point. The
             main thread is now the sole driver of frozen time:
             `_advance_clock_until` ticks the freezer by
             `_CLOCK_TICK_SECONDS` per iteration while asyncio sleeps
             and `threading.Timer.run` wait (via the frozen-aware
             patch in `clock.py`) for the frozen clock to reach their
             targets.
          4. Main thread blocks until `datetime.now(UTC) >= end`, then
             calls `request_stop()`. A real-wall-clock cap protects
             against runs that never produce a sleep.
        """
        if end <= start:
            msg = f'run_window: end {end} must be after start {start}'
            raise ValueError(msg)

        # Direct threading.Event — set by `_run_nexus_instance` when the
        # Nexus instance thread is ready. No log-message interception.
        self._nexus_running = threading.Event()

        launch_thread = threading.Thread(
            target=self.launch, daemon=True, name='backtest-launch',
        )
        launch_thread.start()
        self._wait_until_trading_ready()
        self._wait_until_all_nexus_running()

        # Under `freeze_time`, `time.monotonic` is patched to return the
        # frozen monotonic clock. asyncio's event loop schedules callbacks
        # via `loop.call_at(self.time() + delay, ...)`; when `loop.time`
        # delegates to the patched monotonic, every sleep's deadline
        # (`frozen_now + delay`) is compared against a loop time that
        # stays at `frozen_now` forever — the deadline never passes, the
        # sleep's callback never fires, and `_account_loop` hangs at
        # `await asyncio.sleep(_QUEUE_POLL_INTERVAL)`. `freeze_time
        # (real_asyncio=True)` is meant to cover this but doesn't on
        # every freezegun version. We override `loop.time` explicitly
        # with real monotonic; asyncio scheduling then advances in real
        # wall time, independent of the frozen clock.
        self._install_real_loop_time()
        try:
            with accelerated_clock(start) as freezer:
                try:
                    self._advance_clock_until(end, freezer)
                finally:
                    # Always request_stop so the launch thread exits,
                    # even when the clock advancer raised (e.g. the
                    # drain timed out). Without this, DrainTimeoutError
                    # would leak while the launch thread keeps running
                    # until the daemon shuts down — pushing the e2e
                    # well past the 10s budget.
                    self.request_stop()
        finally:
            launch_thread.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
            if launch_thread.is_alive():
                msg = (
                    f'backtest launch thread did not terminate within '
                    f'{_SHUTDOWN_TIMEOUT_SECONDS}s of shutdown'
                )
                raise RuntimeError(msg)

    def _install_real_loop_time(self) -> None:
        # Must run AFTER the Praxis event loop exists (i.e. past
        # `_wait_until_trading_ready`) and BEFORE `accelerated_clock`
        # engages. The captured `time.monotonic` reference was bound at
        # this module's import time — before any `freeze_time` block —
        # so it's the genuine monotonic clock. Binding it onto the loop
        # instance overrides the default `BaseEventLoop.time` which
        # would otherwise look up `time.monotonic` dynamically (and be
        # frozen under freezegun).
        if self._loop is None:
            msg = (
                '_install_real_loop_time: event loop not running yet; '
                'call only after _wait_until_trading_ready'
            )
            raise RuntimeError(msg)
        setattr(self._loop, 'time', _REAL_MONOTONIC)

    def _wait_until_trading_ready(self) -> None:
        # freezegun patches every public `time.*` clock (time.monotonic,
        # time.perf_counter, even freezegun's own cached real_monotonic
        # reference is affected because `_time` is patched at the C
        # module level). `os.times()[4]` is the elapsed wall-clock field
        # from the POSIX `times()` syscall — freezegun does not touch
        # it, so it's the real-time deadline source that survives an
        # active freeze_time block.
        start = os.times()[4]
        while (os.times()[4] - start) < _BOOT_TIMEOUT_SECONDS:
            if self._trading is not None and self._trading.started:
                return
            time.sleep(_POLL_INTERVAL_SECONDS)
        msg = f'Trading did not start within {_BOOT_TIMEOUT_SECONDS}s of real wall time'
        raise RuntimeError(msg)

    def _wait_until_all_nexus_running(self) -> None:
        if not self._nexus_running.wait(timeout=_NEXUS_RUN_TIMEOUT_SECONDS):
            msg = (
                f'Nexus instance did not reach running within '
                f'{_NEXUS_RUN_TIMEOUT_SECONDS}s of real wall time'
            )
            raise RuntimeError(msg)

    def _record_submitted_command(self, command_id: str) -> None:
        # Called by action_submitter's `on_submit` hook. The lock guards
        # the counter only — actual drain coordination happens on the
        # main clock thread in `_advance_clock_until`, which reads the
        # counter and compares it to the adapter's delivered-order count.
        del command_id
        with self._submit_lock:
            self._submitted_commands += 1

    def _record_reservation(
        self,
        command_id: str,
        decision: ValidationDecision,
        context: ValidationRequestContext,
        action: Action,
    ) -> None:
        # Called by action_submitter's `on_reservation` hook after a
        # successful CAPITAL validation. Feeds the
        # `CapitalLifecycleTracker` so the later `adapter.submit_order`
        # callback can match command_id → reservation for
        # `send_order` / `order_ack` / `order_fill`. Conservation is
        # also asserted here — the reservation stage has mutated the
        # CapitalState and the component totals must still balance.
        #
        # `declared_stop_price` is read from the Nexus Action's
        # `execution_params` (not the Praxis cmd, which strips it for
        # MARKET orders) and carried through the lifecycle so the
        # venue adapter's `FillModel.apply_stop` can enforce it during
        # the trade walk.
        reservation = decision.reservation
        if reservation is None:
            return
        declared_stop_price = _extract_declared_stop_price(action)
        self._capital_tracker.record_reservation(
            command_id=command_id,
            reservation_id=reservation.reservation_id,
            strategy_id=context.strategy_id,
            notional=reservation.notional,
            estimated_fees=reservation.estimated_fees,
            declared_stop_price=declared_stop_price,
        )
        _log.info(
            'capital lifecycle: check_and_reserve command_id=%s '
            'reservation_id=%s notional=%s declared_stop=%s',
            command_id, reservation.reservation_id, reservation.notional,
            declared_stop_price,
        )
        assert_conservation(
            self._capital_state,
            self._capital_initial_pool,
            context=f'check_and_reserve command_id={command_id}',
        )

    def _delivered_command_count(self) -> int:
        # Every `adapter.submit_order` call terminates by assigning into
        # `account.orders[venue_order_id]` — whether the order filled,
        # partially filled, or was rejected (`record_rejection` writes
        # the same dict). So the total across all accounts is the count
        # of submit_order coroutines that have completed.
        #
        # We reach into the private `_accounts`/`_history` on the
        # `SimulatedVenueAdapter` rather than inventing a new public
        # accessor — the drain is a backtest-internal coordination and
        # the adapter class is a concrete backtest class, not a real
        # venue.
        adapter = self._venue_adapter
        active = getattr(adapter, '_accounts', {})
        history = getattr(adapter, '_history', {})
        return sum(len(a.orders) for a in active.values()) + \
               sum(len(a.orders) for a in history.values())

    def _drain_pending_submits(self) -> None:
        """Block until every submitted command_id has landed at the adapter.

        Praxis's `account_loop` polls its submit queue on a 0.1s cadence,
        then awaits `adapter.submit_order`. The main clock thread must
        not advance frozen time past a submit that account_loop hasn't
        yet dispatched — otherwise the adapter's trade-window lookahead
        will miss the trades that would have filled the order.

        Simply `time.sleep` between checks releases the GIL but does not
        guarantee the asyncio event loop iterates. With heavy main-thread
        clock ticking, account_loop can be starved by the scheduler. The
        drain instead schedules a no-op coroutine onto Praxis's loop and
        awaits its completion — that guarantees the loop reaches at
        least one full iteration per drain cycle, which is enough for
        account_loop to pick up the queued command and for
        `adapter.submit_order` to write the resulting order entry.

        A truly stuck command (adapter.submit_order raises into the
        task and never stores an order) is bounded by
        `_DRAIN_TIMEOUT_SECONDS`; on timeout the drain raises
        `DrainTimeoutError` carrying the per-account queue state so
        the run aborts instead of silently burning frozen time into
        a grey zone.
        """
        drain_start = os.times()[4]
        while True:
            with self._submit_lock:
                submitted = self._submitted_commands
            delivered = self._delivered_command_count()
            if delivered >= submitted:
                return
            if os.times()[4] - drain_start > _DRAIN_TIMEOUT_SECONDS:
                diag = self._praxis_queue_sizes()
                msg = (
                    f'drain timeout: submitted={submitted} delivered={delivered} '
                    f'after {_DRAIN_TIMEOUT_SECONDS:.2f}s; praxis_state={diag}'
                )
                raise DrainTimeoutError(msg)
            self._yield_to_loop_once()

    def _yield_to_loop_once(self) -> None:
        # Force the event loop to complete one iteration. `asyncio.sleep(0)`
        # scheduled onto the loop returns only after the loop has run
        # pending callbacks — which is when account_loop can wake from
        # its `await asyncio.sleep(_QUEUE_POLL_INTERVAL)` and see the
        # non-empty queue.
        if self._loop is None:
            time.sleep(_DRAIN_POLL_INTERVAL_SECONDS)
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(asyncio.sleep(0), self._loop)
            fut.result(timeout=_DRAIN_POLL_INTERVAL_SECONDS)
        except TimeoutError:
            # Event loop hasn't iterated in `_DRAIN_POLL_INTERVAL_SECONDS`
            # real — it's busy running something blocking. The next drain
            # iteration will try again; the outer timeout bounds total
            # wait. Don't sleep extra here because that would double the
            # delay without giving the loop any new yield point.
            _log.debug('yield-to-loop timed out; event loop busy')

    def _praxis_queue_sizes(self) -> dict[str, dict[str, object]]:
        # Per-account diagnostic. Reports, for each registered account,
        # whether account_loop is alive or crashed, what its coroutine
        # stack looks like (where it's currently awaiting), and the
        # current sizes of the three queues it drains. This is the only
        # way to tell apart:
        #   - "cmd stuck in queue, account_loop never iterates"
        #   - "account_loop task died silently"
        #   - "account_loop spinning on ws_event_queue, never reaches cmd"
        if self._trading is None:
            return {}
        exec_mgr = getattr(self._trading, '_execution_manager', None)
        if exec_mgr is None:
            return {}
        accounts = getattr(exec_mgr, '_accounts', {})
        out: dict[str, dict[str, object]] = {}
        for aid, rt in accounts.items():
            task = getattr(rt, 'task', None)
            entry: dict[str, object] = {
                'cmd_q': rt.command_queue.qsize(),
                'ws_q': rt.ws_event_queue.qsize(),
                'prio_q': rt.priority_queue.qsize(),
            }
            if task is not None:
                entry['task_done'] = task.done()
                if task.done():
                    exc = task.exception() if not task.cancelled() else 'cancelled'
                    entry['task_exc'] = repr(exc)
                else:
                    # Inspect where the task is currently suspended. Limit
                    # stack depth so the log line stays readable.
                    frames = task.get_stack(limit=3)
                    entry['await_at'] = [
                        f'{f.f_code.co_name}:{f.f_lineno}' for f in frames
                    ]
            out[aid] = entry
        return out

    def _advance_clock_until(self, end: datetime, freezer: object) -> None:
        # PredictLoop schedules its ticks via `threading.Timer`, whose
        # wait loop uses real monotonic time regardless of freezegun.
        # `accelerated_clock` patches `threading.Timer.run` to poll the
        # frozen clock instead, so the timer fires when enough frozen
        # time has elapsed — but that requires someone to actually
        # advance the frozen clock. Here we do: tick by
        # `_CLOCK_TICK_SECONDS` each iteration, pause briefly in real
        # time to let the Timer thread + strategy callbacks + venue
        # adapter + reconciliation process the tick, drain any pending
        # submits, repeat until the frozen window end.
        real_start = os.times()[4]
        while datetime.now(UTC) < end:
            if os.times()[4] - real_start > _REAL_TIME_CAP_SECONDS:
                _log.warning(
                    'backtest window exceeded %ds of real wall time without '
                    'reaching end=%s; forcing stop at frozen %s',
                    _REAL_TIME_CAP_SECONDS, end, datetime.now(UTC),
                )
                return
            tick_fn = getattr(freezer, 'tick')
            tick_fn(_CLOCK_TICK_SECONDS)
            time.sleep(_CLOCK_TICK_REAL_PAUSE_SECONDS)
            self._drain_pending_submits()
        # Grace drain: frozen time drifts because multiple threads
        # (main + per-Timer conditional-sleep) tick it concurrently, so
        # the Timer chain can fire a last strategy tick at a frozen
        # instant slightly past `end` — right as the main loop is
        # exiting. Pause briefly for any such late tick to reach
        # `send_command` (the strategy thread blocks on
        # `future.result()` until the coroutine enqueues), then drain
        # so the command lands at the venue adapter before we
        # `request_stop`. Without this, the last ENTER/SELL of the
        # window is lost and the trade summary undercounts.
        time.sleep(_CLOCK_TICK_REAL_PAUSE_SECONDS)
        self._drain_pending_submits()
