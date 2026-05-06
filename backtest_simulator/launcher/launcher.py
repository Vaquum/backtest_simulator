"""BacktestLauncher — real praxis.launcher.Launcher subclass, historical seams."""
from __future__ import annotations

import asyncio
import importlib
import json
import logging
import math
import os
import queue
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import FrameType
from typing import cast

import polars as pl
from limen import HistoricalData
from nexus.core.domain.capital_state import CapitalState
from nexus.core.domain.enums import OperationalMode
from nexus.core.domain.instance_state import InstanceState
from nexus.core.outcome_loop import ActionSubmitter, OutcomeLoop
from nexus.core.validator.pipeline_models import (
    ValidationDecision,
    ValidationRequestContext,
)
from nexus.infrastructure.manifest import load_manifest

# Use module-qualified import to keep pyright's symbol table from
# binding the bare name `TradeOutcome` to nexus's class. The override
# variance check on `_run_nexus_instance` was misresolving the
# parameter to nexus when both imports landed in the same scope.
from nexus.infrastructure.praxis_connector import (
    trade_outcome as _nexus_trade_outcome_mod,
)
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
from backtest_simulator.launcher.action_submitter import (
    SubmitterBindings,
    build_action_submitter,
)
from backtest_simulator.launcher.clock import accelerated_clock
from backtest_simulator.launcher.poller import BacktestMarketDataPoller

NexusTradeOutcome = _nexus_trade_outcome_mod.TradeOutcome

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
# a stuck command can't burn frozen-time ticks into a grey zone.
_DRAIN_POLL_INTERVAL_SECONDS = 0.02
# 15s hard cap absorbs Praxis-pump tail-latency; 2s soft cap logs
# slow-but-passing drains so pump regressions can't hide.
_DRAIN_TIMEOUT_SECONDS = 15.0
_DRAIN_SLOW_WARN_SECONDS = 2.0


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
    """Retained for import-compat; no longer raised by the wrapper.

    Under the strict-live-reality fill model (`_walk_market` halts on
    stop breach and returns a partial fill), `PARTIALLY_FILLED` is a
    legitimate terminal status. The wrapper handles it by driving
    `order_fill(partial_notional) + order_cancel(venue_order_id)` to
    release the unfilled reservation residual back to available
    capital. See `CapitalLifecycleTracker.record_ack_and_fill`'s
    `release_residual` parameter. A future refinement may re-raise
    this class only when the partial fill lacks a recognised reason
    (tape exhaustion vs. stop halt), at which point the wrapper
    should distinguish.
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
    order_type: OrderType,
) -> Decimal | None:
    """Return the stop_price to forward to `submit_order`.

    Praxis's `validate_trade_command` strips stop_price from non-stop
    order types (MARKET / LIMIT) before the command reaches here. We
    look up the command's BTS-declared protective stop via the
    `client_order_id` and stash it in `declared_stops` for the
    R-multiple metric to read later, regardless of `order_type`.

    For MARKET orders we ALSO inject the declared stop back into the
    venue's `stop_price` kwarg so `walk_market` halts on breach and
    the entry walk respects the protective stop in-line.

    For LIMIT orders we do NOT inject. A LIMIT entry's mechanics are
    "fill at limit price if the tape crosses, otherwise rest" — the
    protective stop is a separate STOP_LOSS exit order, not part of
    the LIMIT order. Injecting the declared stop into a LIMIT would
    also fail the venue's PRICE_FILTER tick check (the BTS R-anchor
    is a Kelly-derived fractional price, not a venue-quoted stop
    trigger), and every LIMIT entry would land as REJECTED. This
    was the failure pinned during maker-fill wiring.
    """
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
    """Sum the result's immediate_fills into (notional, fees)."""
    fill_notional = Decimal('0')
    fill_fees = Decimal('0')
    for fill in result.immediate_fills:
        fill_notional += Decimal(str(fill.qty)) * Decimal(str(fill.price))
        fill_fees += Decimal(str(fill.fee))
    return fill_notional, fill_fees


def _sum_fill_qty(result: SubmitResult) -> Decimal:
    """Sum the result's immediate_fills into total filled quantity."""
    total = Decimal('0')
    for fill in result.immediate_fills:
        total += Decimal(str(fill.qty))
    return total


def _finalize_successful_fill(
    ctx: _LifecycleContext,
    command_id: str,
    result: SubmitResult,
    venue_order_id: str,
    status_name: str,
) -> None:
    """Fire record_sent + record_ack_and_fill with conservation checks.

    Fails loud on capital overshoot. On `PARTIALLY_FILLED` — now a
    legitimate terminal outcome under the strict-live-reality fill
    model (MARKET walk halts on stop breach, returning partial fill)
    — drives `record_ack_and_fill(release_residual=True)` so the
    unfilled reservation is cancelled back to available capital.
    Without release the ledger would under-count available capital
    on every halted entry.
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
            f'by {slippage}. Raise `NOTIONAL_RESERVATION_BUFFER` in '
            f'backtest_simulator.launcher.action_submitter to absorb this '
            f'slippage honestly; silently capping here would bypass '
            f'CAPITAL gating.'
        )
        raise CapitalOvershootError(msg)
    release_residual = status_name == 'PARTIALLY_FILLED'
    if release_residual:
        _log.info(
            'capital lifecycle: command_id=%s PARTIALLY_FILLED — '
            'driving order_fill(%s) + order_cancel(%s) to release '
            'unfilled reservation residual back to available capital.',
            command_id, fill_notional, venue_order_id,
        )
    # Capture the strategy_id BEFORE record_ack_and_fill pops the
    # pending entry — record_open_position (called below) needs
    # it to attribute the open position to the right strategy.
    pending_strategy_id = ctx.tracker.strategy_id_for_pending(command_id)
    ctx.tracker.record_ack_and_fill(
        command_id, venue_order_id, fill_notional, fill_fees,
        release_residual=release_residual,
    )
    assert_conservation(
        ctx.capital_state, ctx.initial_pool,
        context=f'order_fill command_id={command_id}',
    )
    # BUY fills open a position; remember the cost basis + entry
    # fees so the matching SELL can release them on close. The
    # capital_state.position_notional already carries this amount
    # (controller.order_fill moved it from working_order_notional);
    # the tracker keeps a parallel ledger so `record_close_position`
    # can subtract the exact cost basis without poking through the
    # controller's internals. Codex round 4 P0 fix.
    fill_qty = _sum_fill_qty(result)
    ctx.tracker.record_open_position(
        command_id=command_id,
        strategy_id=pending_strategy_id or 'unknown',
        cost_basis=fill_notional,
        entry_fees=fill_fees,
        entry_qty=fill_qty,
    )
    _log.info(
        'capital lifecycle: command_id=%s reserve->send->ack->fill%s '
        'venue_order_id=%s fill_notional=%s fees=%s open_positions=%d',
        command_id, '+cancel' if release_residual else '',
        venue_order_id, fill_notional, fill_fees,
        ctx.tracker.open_position_count,
    )


def _finalize_sell_close(
    ctx: _LifecycleContext,
    result: SubmitResult,
    sell_command_id: str,
) -> None:
    """Run the SELL-fill close-position lifecycle.

    Pulls the FIFO-oldest open position via
    `tracker.record_close_position`, which mutates
    `capital_state.position_notional` (releases the proportional
    cost basis + entry fees) and `per_strategy_deployed` (releases
    proportional attribution). `capital_pool` is NOT touched —
    the controller treats it as immutable budget, SELL proceeds
    are realized PnL not new budget. Returns the realized PnL for
    the closed leg and asserts conservation afterwards.

    Partial SELL fills only release the SOLD portion of the open
    position; the residual entry stays open for the next preds=0
    to finish closing (codex round 5 P2).

    The realized PnL is logged on the lifecycle line so the
    operator can audit per-trade economics directly from the run
    log; the trade-pair-level R / return are still computed
    downstream in `cli/_metrics.py::print_run` (those use the
    declared-stop-anchored R, which doesn't depend on the capital
    ledger). Pre-fix the SELL exit silently bypassed this whole
    flow — codex round 4 P0.
    """
    sell_proceeds, sell_fees = _sum_fill_totals(result)
    sell_qty = _sum_fill_qty(result)
    realized_pnl, closed_record = ctx.tracker.record_close_position(
        ctx.capital_state,
        sell_command_id=sell_command_id,
        sell_qty=sell_qty,
        sell_proceeds=sell_proceeds,
        sell_fees=sell_fees,
    )
    assert_conservation(
        ctx.capital_state, ctx.initial_pool,
        context=f'sell_close sell_command_id={sell_command_id}',
    )
    _log.info(
        'capital lifecycle: SELL close sell_command_id=%s '
        'paired_buy_command_id=%s released_cost_basis=%s '
        'released_entry_fees=%s sell_qty=%s sell_proceeds=%s '
        'sell_fees=%s realized_pnl=%s open_positions=%d',
        sell_command_id, closed_record.command_id,
        closed_record.cost_basis, closed_record.entry_fees,
        sell_qty, sell_proceeds, sell_fees, realized_pnl,
        ctx.tracker.open_position_count,
    )


def _make_outcome_router(
    outcome_queues: dict[str, queue.Queue[NexusTradeOutcome]],
) -> Callable[[TradeOutcome], Awaitable[None]]:
    """Return an async router that translates Praxis outcomes into the launcher queue.

    Extracted to module level so the routing path is testable
    without spinning up a real `BacktestLauncher`. The router
    closes over `outcome_queues` (the launcher's per-account
    dict) and:

      1. translates the Praxis outcome via
         `_translate_praxis_outcome` (returns `None` for unknown
         status; PENDING is mapped to EXPIRED so a bounded-
         lookahead non-fill clears `_pending_buy`),
      2. looks up the account's queue,
      3. enqueues the Nexus outcome.

    Outcomes for accounts without a registered queue are dropped;
    that case only arises during shutdown teardown when the
    launcher has already deregistered the account.
    """
    async def _route(praxis_outcome: TradeOutcome) -> None:
        nexus_outcome = _translate_praxis_outcome(praxis_outcome)
        if nexus_outcome is None:
            return
        account_queue = outcome_queues.get(praxis_outcome.account_id)
        if account_queue is None:
            return
        account_queue.put_nowait(nexus_outcome)
    return _route


def _translate_praxis_outcome(
    praxis_outcome: TradeOutcome,
) -> NexusTradeOutcome | None:
    """Convert a Praxis `TradeOutcome` into Nexus's `TradeOutcome` shape.

    The two `TradeOutcome` dataclasses differ:
      - Praxis: `command_id`, `trade_id`, `account_id`, `status`,
        `target_qty`, `filled_qty`, `avg_fill_price`, `created_at`...
      - Nexus: `outcome_id`, `command_id`, `outcome_type`, `timestamp`,
        `fill_size`, `fill_price`, `fill_notional`, `actual_fees`...

    Mapping:
      - `outcome_id` ← `f'{command_id}-{status.value}'` (Praxis carries
        no outcome_id; deterministic synthetic id from command + status
        is enough for the strategy state machine to dedup; the
        post-translation Nexus `outcome_id` therefore reflects the
        Praxis status even when this function overrides
        `outcome_type` — see the EXPIRED→PARTIAL upgrade below).
      - `outcome_type`: Praxis FILLED/PARTIAL/REJECTED/CANCELED map
        1:1. Two backtest-specific upgrades:
        * `Praxis PENDING + filled_qty==0` → `Nexus EXPIRED`. In the
          bounded-lookahead world that's terminal — no further
          updates will arrive — so we hand the strategy a terminal
          and let `_pending_buy` clear (codex P1: leaving PENDING
          dropped silently let zero-fill LIMITs jam the gate
          forever).
        * `Praxis EXPIRED + filled_qty > 0` → `Nexus PARTIAL`. This
          is the "PARTIAL upgraded to EXPIRED on deadline" path
          (`praxis/core/execution_manager.py:1032-1045`); a
          deadline-truncated partial is terminal in our world, and
          Nexus rejects fill fields on `outcome_type.is_fill==False`,
          so we must surface the partial as Nexus PARTIAL for the
          strategy's existing fill branch to reconcile `_entry_qty`.
      - `fill_size` ← `filled_qty` when the OVERRIDDEN `nexus_type`
        is `is_fill==True`, else None (Nexus rejects fill fields on
        non-fill outcomes; the upgrade rule for EXPIRED+fill above
        is what makes those visible).
      - `fill_price` ← `avg_fill_price` (same is_fill gate as
        `fill_size`).
      - `fill_notional` ← `filled_qty * avg_fill_price` (same gate).
      - `actual_fees` ← Decimal('0') on fills, None on non-fills
        (Praxis TradeOutcome carries no fee aggregate; the per-fill
        fees live on the venue's `Account.trades`. The strategy
        state machine doesn't read actual_fees, so 0 is honest
        enough for state reconciliation — pinning it here keeps the
        deeper "real fees on outcomes" slice for a follow-up).
      - `reject_reason` ← Praxis `reason` ONLY when the overridden
        `nexus_type == REJECTED` (or a synthesized fallback when
        Praxis emits an empty reason on REJECTED). Otherwise None.
        Nexus rejects `reject_reason != None` on non-REJECTED
        outcomes (codex P2 caught the post-init blow-up;
        forwarding `reason='deadline exceeded'` from Praxis EXPIRED
        would explode there).
    """
    from nexus.infrastructure.praxis_connector.trade_outcome_type import (
        TradeOutcomeType,
    )
    from praxis.core.domain.enums import TradeStatus
    status = praxis_outcome.status
    type_map = {
        TradeStatus.FILLED: TradeOutcomeType.FILLED,
        TradeStatus.PARTIAL: TradeOutcomeType.PARTIAL,
        TradeStatus.REJECTED: TradeOutcomeType.REJECTED,
        TradeStatus.EXPIRED: TradeOutcomeType.EXPIRED,
        TradeStatus.CANCELED: TradeOutcomeType.CANCELED,
        # Praxis stamps PENDING when the order was accepted but no
        # fill landed within the deadline. In our bounded-lookahead
        # backtest that IS terminal — no later outcome will arrive
        # for the same command. Map to EXPIRED so the strategy
        # clears `_pending_buy` and the next signal can re-enter.
        TradeStatus.PENDING: TradeOutcomeType.EXPIRED,
    }
    nexus_type = type_map.get(status)
    if nexus_type is None:
        return None
    # A `Praxis EXPIRED + filled_qty > 0` is the "PARTIAL upgraded
    # to EXPIRED on deadline" path
    # (`praxis/core/execution_manager.py:1032-1045`): the venue did
    # fill some of the qty, and only the remainder got cancelled.
    # In bts's bounded-lookahead world that's terminal — no further
    # updates will arrive for the command. Translate to a Nexus
    # PARTIAL so the strategy's existing fill branch reconciles
    # `_entry_qty` against the actual partial fill instead of
    # treating it as a zero-fill. Nexus rejects fill fields on
    # `outcome_type.is_fill == False`, so handing the partial
    # through as Nexus EXPIRED is not an option.
    if status == TradeStatus.EXPIRED and praxis_outcome.filled_qty > 0:
        nexus_type = TradeOutcomeType.PARTIAL
    # Derive `is_fill` and `reject_reason` from the OVERRIDDEN
    # `nexus_type`, NOT the Praxis status. Otherwise the
    # deadline-truncated PARTIAL path leaves `fill_size=None` on a
    # Nexus PARTIAL and `__post_init__` raises; symmetrically,
    # forwarding `praxis_outcome.reason='deadline exceeded'` as
    # `reject_reason` on a non-REJECTED Nexus outcome also raises.
    is_fill = nexus_type.is_fill
    fill_size = praxis_outcome.filled_qty if is_fill else None
    fill_price = praxis_outcome.avg_fill_price if is_fill else None
    fill_notional = (
        praxis_outcome.filled_qty * praxis_outcome.avg_fill_price
        if is_fill and praxis_outcome.avg_fill_price is not None
        else None
    )
    actual_fees = Decimal('0') if is_fill else None
    if nexus_type == TradeOutcomeType.REJECTED:
        reject_reason = (
            praxis_outcome.reason if praxis_outcome.reason
            else f'translated-from-praxis-{status.value}'
        )
    else:
        reject_reason = None
    return NexusTradeOutcome(
        outcome_id=f'{praxis_outcome.command_id}-{status.value}',
        command_id=praxis_outcome.command_id,
        outcome_type=nexus_type,
        timestamp=praxis_outcome.created_at,
        fill_size=fill_size,
        fill_price=fill_price,
        fill_notional=fill_notional,
        actual_fees=actual_fees,
        reject_reason=reject_reason,
    )


def _build_outcome_loop(
    *,
    runner: StrategyRunner,
    praxis_inbound: PraxisInbound,
    state: InstanceState,
    context_provider: Callable[[str], StrategyContext],
    wired_sensors: Sequence[WiredSensor],
    action_submit: ActionSubmitter,
) -> OutcomeLoop:
    """Wire up the Nexus `OutcomeLoop` for backtest-runtime outcome dispatch.

    Without an `OutcomeLoop`, `TradeOutcome`s pile up in the queue
    and never reach the strategy's `on_outcome` — which broke
    `--maker` runs where the strategy reconciles `_long` /
    `_entry_qty` from actual fills (a partial-fill BUY LIMIT would
    leave the strategy thinking it's flat forever; a zero-fill BUY
    would let a later preds=0 emit a phantom SELL). Codex P1
    pinned this.

    `resolve_strategy_id` returns the single registered
    strategy_id — backtest manifests today wire one strategy per
    instance, so the resolver doesn't need a per-command registry.
    Multiple strategies on one instance would need that — fail
    loud here so the gap doesn't smuggle the wrong on_outcome
    target into a multi-strategy backtest.
    """
    if not wired_sensors:
        msg = (
            'BacktestLauncher: no wired sensors after sequencer.start(); '
            'OutcomeLoop has nothing to resolve outcomes against.'
        )
        raise RuntimeError(msg)
    strategy_ids = {s.strategy_id for s in wired_sensors}
    if len(strategy_ids) != 1:
        msg = (
            f'BacktestLauncher: multiple strategy_ids wired '
            f'({sorted(strategy_ids)}) — outcome resolution requires '
            f'a per-command registry, not yet implemented.'
        )
        raise RuntimeError(msg)
    single = next(iter(strategy_ids))

    def _resolve(_outcome: NexusTradeOutcome) -> str | None:
        return single

    return OutcomeLoop(
        runner=runner,
        praxis_inbound=praxis_inbound,
        state=state,
        context_provider=context_provider,
        resolve_strategy_id=_resolve,
        action_submit=action_submit,
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
            tracker, declared_stops, order_type,
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
        venue_order_id = result.venue_order_id
        status_name = result.status.name
        if command_id is None:
            # SELL exits don't go through the BUY reservation
            # tracker (no `record_reservation` call) — the
            # action_submitter's SELL fast-path skips
            # validation_pipeline.validate (see TODO Task 27 for
            # the architectural decision: CapitalController has
            # no close_position primitive, so the close
            # lifecycle is BTS-side via
            # `record_close_position`). The CLOSE half releases
            # the matched BUY's `cost_basis + entry_fees` from
            # `position_notional` and decrements
            # `per_strategy_deployed[strategy_id]` by the same
            # amount; `capital_pool` stays untouched (it's the
            # immutable strategy budget, not a cash ledger).
            # Codex round 4 P0: pre-fix this branch just
            # logged-and-returned, so `position_notional` and
            # the per-strategy attribution dict went
            # permanently out of sync after every close.
            # `.name` comparison instead of `is` / `==` against
            # Praxis's OrderSide so the branch still fires when
            # callers pass the Nexus-side OrderSide enum (the
            # adapter Protocol type-annotates Praxis but Nexus's
            # action_submitter and several tests use Nexus's
            # enum — both have `.name == 'SELL'` on the SELL
            # member). Identity comparison on `is`/`==` returns
            # False across the two distinct Enum classes even
            # though they're behaviourally identical.
            if (
                side.name == 'SELL'
                and status_name in ('FILLED', 'PARTIALLY_FILLED')
                and result.immediate_fills
            ):
                # Both FILLED and PARTIALLY_FILLED feed the close
                # flow; `record_close_position` releases the
                # proportional share of the matched BUY's cost
                # basis (sold qty / entry qty) and leaves the
                # residual open if any. The strategy's
                # `_pending_sell` clears on the outcome dispatch
                # so the next preds=0 can finish the residual.
                _finalize_sell_close(
                    ctx, result, client_order_id or 'unknown',
                )
            else:
                _log.info(
                    'capital lifecycle: no pending command matches '
                    'client_order_id=%s; pending_count=%d',
                    client_order_id, tracker.pending_count,
                )
            return result
        # REJECTED (filter rejection, never reached the venue) and
        # EXPIRED (validated, sent to the venue, didn't fill in window)
        # both terminate without a fill, but their lifecycle audit
        # trails differ:
        #   - REJECTED: pre-send terminal. `record_rejection` while
        #     `pending.sent=False` takes the `release_reservation`
        #     path — releases capital before any send/ack lifecycle.
        #   - EXPIRED: post-send terminal. The venue actually saw the
        #     order, just didn't have liquidity. `record_sent` first
        #     so the tracker's `pending.sent=True`, then
        #     `record_rejection` takes the post-send `order_reject`
        #     path (which is the right release method for an
        #     accepted-then-expired order).
        # Without the `record_sent` for EXPIRED, the audit trail would
        # call `release_reservation` on an order that the venue had
        # actually seen, diverging from live's accepted→expired shape.
        if status_name in ('REJECTED', 'EXPIRED'):
            if status_name == 'EXPIRED':
                tracker.record_sent(command_id, venue_order_id)
            tracker.record_rejection(command_id, venue_order_id)
            assert_conservation(
                capital_state, initial_pool,
                context=f'order_reject command_id={command_id}',
                    )
            _log.info(
                'capital lifecycle: command_id=%s %s (reservation released)',
                command_id, status_name,
            )
            return result
        if status_name == 'OPEN':
            # `OPEN` means a GTC LIMIT / STOP_LOSS / STOP_LOSS_LIMIT /
            # TAKE_PROFIT order is resting on the book waiting to
            # trigger or cross. In live this is a long-lived state;
            # in the backtest the venue evaluates the entire post-
            # submit `trade_window_seconds` slice in one shot via
            # `MakerFillModel.evaluate`, so OPEN at return-time means
            # "the maker engine ran and produced zero fills inside
            # the lookahead window." There will be no later fill
            # event — this slice's MakerFillModel does not stream;
            # the order's effective lifetime IS the lookahead.
            # Treat as EXPIRED for capital lifecycle purposes:
            # `record_sent` then `record_rejection` releases the
            # reservation, matching the audit trail of a venue-
            # accepted order that didn't trade. The order remains
            # in `account.orders` with status=OPEN so the operator
            # can see it surfaced in maker-fill telemetry
            # (`n_limit_filled_zero` increments).
            tracker.record_sent(command_id, venue_order_id)
            tracker.record_rejection(command_id, venue_order_id)
            assert_conservation(
                capital_state, initial_pool,
                context=f'limit_open_no_fill command_id={command_id}',
                    )
            _log.info(
                'capital lifecycle: command_id=%s OPEN (LIMIT no-fill in '
                'lookahead window — reservation released)',
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
        max_allocation_per_trade_pct: Decimal | None = None,
        clock_tick_seconds: int | None = None,
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
        # `clock_tick_seconds` controls how coarsely the main thread
        # advances frozen time inside `_advance_clock_until`. The
        # default of 120s was sized for sub-minute Timer cadences; for
        # bts sweeps with kline_size >> 120s (e.g. 4h klines) it's
        # 60-120x finer than necessary — most iterations sleep 10ms +
        # drain to find no new submits. Pass `kline_size` here and
        # the loop iterates only once per strategy tick boundary.
        # `None` keeps the legacy 120s behaviour for callers that
        # haven't been audited.
        if clock_tick_seconds is not None and clock_tick_seconds <= 0:
            # `_advance_clock_until` derives boundary buffers and
            # tick amounts from `_clock_tick_seconds.total_seconds()`;
            # zero or negative would produce zero / negative
            # timedeltas and either spin or freeze (Copilot P1).
            msg = (
                f'clock_tick_seconds must be > 0 when provided, '
                f'got {clock_tick_seconds!r}.'
            )
            raise ValueError(msg)
        self._clock_tick_seconds = (
            timedelta(seconds=clock_tick_seconds)
            if clock_tick_seconds is not None
            else _CLOCK_TICK_SECONDS
        )
        # Synchronous-drain counters. `_submitted_commands` is bumped by
        # the action_submitter's `on_submit` callback after every
        # successful `praxis_outbound.send_command`; the main clock loop
        # then blocks until the venue adapter has delivered an equal
        # number of orders (filled, partial, or rejected) before
        # advancing the next frozen tick.
        self._submitted_commands = 0
        self._submit_lock = threading.Lock()
        self._venue_adapter = venue_adapter
        self._max_allocation_per_trade_pct = max_allocation_per_trade_pct

    def _start_trading(self) -> None:
        """Wire `Trading.route_outcome` into the execution_manager.

        Praxis's `TradingConfig.on_trade_outcome` is None by default
        in our backtest setup. Without it, fill events emitted by
        `ExecutionManager._publish_outcome` never reach
        `Trading.route_outcome`, so per-account outcome queues never
        get populated, `PraxisInbound.receive_outcome` always sees
        empty, and `OutcomeLoop` can't dispatch `on_outcome` to the
        strategy. Codex P1 caught this: with `--maker` the strategy
        reconciles `_long` / `_entry_qty` from actual fills, but
        without the route the outcome never reaches the strategy
        and `_long` stays False forever.

        Praxis's `TradeOutcome` and Nexus's `TradeOutcome` are
        different dataclasses (Nexus expects `outcome_type`,
        `fill_size`, `fill_price`, `fill_notional`, `actual_fees`;
        Praxis emits `status`, `target_qty`, `filled_qty`,
        `avg_fill_price`). The route closure translates Praxis →
        Nexus before pushing into the queue so `OutcomeLoop` can
        dispatch a Nexus-shaped outcome to `Strategy.on_outcome`
        without each strategy author rolling their own translator.

        We can't pass this callback into TradingConfig at
        construction (it lives on the not-yet-created Trading
        instance). After `super()._start_trading()` returns,
        `self._trading` exists; mutate the execution_manager's
        `_on_trade_outcome` to the wrapped translator + router.
        """
        super()._start_trading()
        if self._trading is None:
            msg = (
                'BacktestLauncher._start_trading: super()._start_trading '
                'returned without populating self._trading.'
            )
            raise RuntimeError(msg)
        trading = self._trading
        # Route Praxis outcomes into the launcher's per-account
        # queues — the dict that `OutcomeLoop.PraxisInbound`
        # actually consumes from (populated in
        # `praxis/launcher.py:1122`). `Trading._outcome_queues`
        # is a separate dict only filled when callers invoke
        # `Trading.register_outcome_queue`, which Praxis 0.48.0
        # does not do; reading it would return None and drop the
        # outcome.
        router = _make_outcome_router(self._outcome_queues)
        trading.execution_manager.set_on_trade_outcome(router)

    def _start_poller(self) -> None:
        kline_intervals = self._resolve_kline_intervals_from_manifests()
        params_by_kline_size = self._resolve_data_source_params_by_kline_size()
        # `BacktestMarketDataPoller` is duck-typed against the parent's
        # `MarketDataPoller` Protocol — same public surface, different
        # underlying source. setattr-style assignment dodges the
        # `_poller: MarketDataPoller | None` invariant-mutable-attribute
        # check on the parent class.
        poller = BacktestMarketDataPoller(
            kline_intervals=kline_intervals,
            historical_data=self._historical_data,
            params_by_kline_size=params_by_kline_size,
        )
        setattr(self, '_poller', poller)
        poller.start()
        _log.info(
            'backtest poller started',
            extra={'kline_sizes': sorted(kline_intervals)},
        )

    def _resolve_data_source_params_by_kline_size(
        self,
    ) -> dict[int, dict[str, object]]:
        # Fail-loud on conflicts: if two sensors share a kline_size
        # but disagree on data_source.params (different n_rows /
        # start_date_limit), the silent last-writer-wins produces
        # a fetch that doesn't match either sensor's training. Raise
        # so the bundle author sees the mismatch instead of debugging
        # a parity violation later.
        params_by_kline: dict[int, dict[str, object]] = {}
        for inst in self._instances:
            manifest = load_manifest(inst.manifest_path)
            for spec in manifest.strategies:
                for sensor_spec in spec.sensors:
                    cfg_params = self._data_source_params_from_experiment_dir(
                        sensor_spec.experiment_dir,
                    )
                    kline_size_obj = cfg_params['kline_size']
                    if not isinstance(kline_size_obj, int):
                        msg = (
                            f'data_source params kline_size for '
                            f'{sensor_spec.experiment_dir} must be int, got '
                            f'{type(kline_size_obj).__name__}={kline_size_obj!r}'
                        )
                        raise TypeError(msg)
                    prior = params_by_kline.get(kline_size_obj)
                    if prior is not None and prior != cfg_params:
                        msg = (
                            f'conflicting data_source.params for '
                            f'kline_size={kline_size_obj}: prior={prior!r} '
                            f'new={cfg_params!r} (from '
                            f'{sensor_spec.experiment_dir}). Two sensors '
                            f'sharing a kline_size must declare identical '
                            f'data_source.params; otherwise the poller '
                            f'cannot serve a single coherent fetch.'
                        )
                        raise ValueError(msg)
                    params_by_kline[kline_size_obj] = cfg_params
        return params_by_kline

    @staticmethod
    def _data_source_params_from_experiment_dir(
        experiment_dir: Path,
    ) -> dict[str, object]:
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

    @classmethod
    def _kline_size_from_experiment_dir(cls, experiment_dir: Path) -> int:
        # Reuses the single metadata.json -> sfd_module -> manifest()
        # loader so the two extraction paths cannot drift.
        params = cls._data_source_params_from_experiment_dir(experiment_dir)
        kline_size_obj = params['kline_size']
        if not isinstance(kline_size_obj, int):
            msg = (
                f'data_source params kline_size for {experiment_dir} '
                f'must be int, got '
                f'{type(kline_size_obj).__name__}={kline_size_obj!r}'
            )
            raise TypeError(msg)
        return kline_size_obj

    def _shutdown(self) -> None:
        # Slice #17 Task 18 / auditor round 5 P1: Praxis's parent
        # `_shutdown` closes the aiosqlite connection without
        # commit. EventSpine.append uses the default deferred-
        # transaction mode (sqlite3 isolation_level='') so writes
        # are pending until commit; aiosqlite.close() does NOT
        # auto-commit. Result: the EventSpine sqlite file looks
        # empty to any fresh reader (e.g. our post-run
        # `dump_event_spine_to_jsonl`) because the writes were
        # rolled back at close.
        #
        # Commit must happen AFTER Nexus threads join + Trading.stop
        # (so all post-window appends land first) and BEFORE
        # _db_conn.close. That requires copying the parent's
        # shutdown body and injecting the commit at the precise
        # point — a single-call `super()._shutdown()` followed by
        # commit (which we tried first) misses the close that
        # already happened, and a commit before super() misses
        # appends that fire during Trading.stop. Codex round-5 P1.
        import asyncio
        self._stop_healthz()
        for thread in self._nexus_threads:
            thread.join(timeout=30)
            if thread.is_alive():
                _log.warning(
                    'nexus thread did not finish within timeout',
                    extra={'thread': thread.name},
                )
        if self._poller is not None:
            self._poller.stop()
        if self._trading is not None and self._loop is not None:
            stop_future = asyncio.run_coroutine_threadsafe(
                self._trading.stop(), self._loop,
            )
            stop_future.result(timeout=30)
        if (
            self._owns_spine
            and self._db_conn is not None
            and self._loop is not None
        ):
            # Inject point: all writers have stopped, all events
            # are appended to the connection, but the connection
            # is still open. Commit flushes deferred transactions
            # to the file before close.
            commit_future = asyncio.run_coroutine_threadsafe(
                self._db_conn.commit(), self._loop,
            )
            commit_future.result(timeout=10)
            close_future = asyncio.run_coroutine_threadsafe(
                self._db_conn.close(), self._loop,
            )
            close_future.result(timeout=10)
            self._db_conn = None
        # Parent's tail: stop loop, join loop thread, close loop,
        # null out refs, log. Codex round-6 P1 — the round-5 copy
        # truncated this and let the loop / loop_thread leak.
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5)
        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()
        self._loop = None
        self._loop_thread = None
        _log.info('shutdown complete')

    def _signal_handler(self, _signum: int, _frame: FrameType | None) -> None:
        # Backtests aren't daemons; termination is driven by the outer
        # harness calling `request_stop()` when the window ends, not by
        # SIGINT/SIGTERM landing mid-run.
        _log.info('backtest launcher ignoring external signal')

    def request_stop(self) -> None:
        """Ask `launch()` to return. Outer harness uses this at window end."""
        self._stop_event.set()

    def _start_nexus_instances(self) -> None:
        # Spawn at this level (instead of overriding _run_nexus_instance)
        # to avoid pyright's false-positive invariance check on
        # Queue[TradeOutcome] at the cross-module override site.
        if self._trading is None or self._loop is None:
            msg = 'trading not started'
            raise RuntimeError(msg)
        for inst in self._instances:
            t = threading.Thread(
                target=self._run_my_nexus_instance,
                args=(inst, self._outcome_queues[inst.account_id]),
                daemon=True, name=f'nexus-{inst.account_id}',
            )
            self._nexus_threads.append(t)
            t.start()

    def _run_my_nexus_instance(
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
            nexus_config=nexus_config,
            capital_pool=allocated_capital,
            max_allocation_per_trade_pct=self._max_allocation_per_trade_pct,
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
        # Touch + tick providers feed the action_submitter's
        # LIMIT-touch-refresh hook. The hook rewrites a LIMIT
        # action's `execution_params['price']` to `touch ± tick`
        # before validation. `SimulatedVenueAdapter` exposes both
        # providers; non-sim adapters fall back to None (action
        # submitter's no-touch path).
        from backtest_simulator.venue.simulated import SimulatedVenueAdapter
        sva = self._venue_adapter if isinstance(self._venue_adapter, SimulatedVenueAdapter) else None
        touch_provider = sva.touch_for_symbol if sva is not None else None
        tick_provider = sva.tick_for_symbol if sva is not None else None
        action_submit = build_action_submitter(
            SubmitterBindings(
                nexus_config=nexus_config, state=state,
                praxis_outbound=praxis_outbound,
                validation_pipeline=pipeline,
                capital_controller=controller,
                strategy_budget=allocated_capital,
                touch_provider=touch_provider,
                tick_provider=tick_provider,
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

        # PraxisInbound and praxis.Launcher use distinct TradeOutcome
        # classes from different modules but the same shape; assign
        # to a typed local at the boundary so pyright accepts both
        # ends of the queue.
        nexus_outcome_queue: queue.Queue[NexusTradeOutcome] = cast(
            'queue.Queue[NexusTradeOutcome]', outcome_queue,
        )
        praxis_inbound = PraxisInbound(outcome_queue=nexus_outcome_queue)
        outcome_loop = _build_outcome_loop(
            runner=runner, praxis_inbound=praxis_inbound, state=state,
            context_provider=context_provider,
            wired_sensors=sequencer.wired_sensors,
            action_submit=action_submit,
        )
        outcome_loop.start()
        # Direct event signal — no log-message interception. The running
        # event is initialised in `run_window` on `self._nexus_running`.
        self._nexus_running.set()
        self._stop_event.wait()
        outcome_loop.stop()

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

        Main clock must not advance past a submit Praxis hasn't dispatched.
        Bounded by `_DRAIN_TIMEOUT_SECONDS`; raises `DrainTimeoutError`
        on timeout (with per-account queue state) so the run aborts loudly.
        """
        drain_start = os.times()[4]
        while True:
            with self._submit_lock:
                submitted = self._submitted_commands
            delivered = self._delivered_command_count()
            if delivered >= submitted:
                elapsed = os.times()[4] - drain_start
                if elapsed > _DRAIN_SLOW_WARN_SECONDS:
                    _log.warning('drain slow: submitted=%d delivered=%d wallclock=%.2fs', submitted, delivered, elapsed)
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
        # advance the frozen clock.
        #
        # Two modes:
        #   - Uniform mode (`_clock_tick_seconds <= 120s`): tick by
        #     a fixed step (default 120s), pause 10ms, drain. Safe
        #     for any cadence; default for callers that don't pass
        #     `clock_tick_seconds`. Cost: 720 iters/day at 120s x 10ms
        #     ≈ 7s of pure `time.sleep` per day.
        #   - Smart kline-aware mode (`_clock_tick_seconds > 120s`):
        #     `_clock_tick_seconds` is interpreted as the bundle's
        #     `kline_size`. Far from a kline boundary the loop jumps
        #     big (`margin - 60s`), preserving the 60s buffer; near a
        #     boundary it crosses with a generous 250ms real-time
        #     pause so the `produce_signal → on_signal → submit →
        #     fill` chain has wall-clock time to complete BEFORE the
        #     next freezer tick. Reduces 720 iters/day to ~20-25 for
        #     a 4h kline; pause budget around each boundary stays in
        #     the 250-300ms range that uniform mode delivered via 120
        #     x 10ms.
        real_start = os.times()[4]
        kline_size_s = self._clock_tick_seconds.total_seconds()
        use_smart = kline_size_s > 120.0
        # Buffer before a boundary where we slow down to single-tick
        # crossings; >= the largest realistic asyncio-chain latency we
        # need to absorb (300ms ClickHouse fetch + Praxis pump + drain).
        _BOUNDARY_BUFFER_S = 60
        # Real-time pause used when a tick CROSSES a kline boundary —
        # gives the asyncio event loop, Praxis pump, venue adapter
        # and `_drain_pending_submits` enough wall-clock to ferry the
        # produce_signal → fill chain through before we tick again.
        _BOUNDARY_PAUSE_S = 0.25
        while datetime.now(UTC) < end:
            if os.times()[4] - real_start > _REAL_TIME_CAP_SECONDS:
                _log.warning(
                    'backtest window exceeded %ds of real wall time without '
                    'reaching end=%s; forcing stop at frozen %s',
                    _REAL_TIME_CAP_SECONDS, end, datetime.now(UTC),
                )
                return
            tick_fn = getattr(freezer, 'tick')
            if use_smart:
                frozen_now = datetime.now(UTC)
                secs = frozen_now.timestamp()
                # Next kline boundary aligned to UNIX epoch (matches
                # the cadence the strategy's PredictLoop schedules
                # against — Timer interval = `kline_size`).
                next_boundary = (
                    math.ceil((secs + 1e-6) / kline_size_s) * kline_size_s
                )
                margin_s = next_boundary - secs
                end_dist_s = (end - frozen_now).total_seconds()
                if margin_s > _BOUNDARY_BUFFER_S * 2:
                    # Far from boundary — jump big, leave one buffer
                    # behind so we approach the boundary with margin.
                    tick_amt = margin_s - _BOUNDARY_BUFFER_S
                    pause_s = _CLOCK_TICK_REAL_PAUSE_SECONDS
                else:
                    # In or near the boundary buffer — cross by 1s
                    # and grant the asyncio chain a generous pause.
                    tick_amt = min(margin_s + 1.0, end_dist_s)
                    pause_s = _BOUNDARY_PAUSE_S
                tick_amt = min(tick_amt, max(end_dist_s, 1.0))
                tick_fn(timedelta(seconds=tick_amt))
                time.sleep(pause_s)
            else:
                tick_fn(self._clock_tick_seconds)
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
