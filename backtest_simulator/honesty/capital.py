"""Real six-stage ValidationPipeline + 4-step CapitalController lifecycle driver."""
from __future__ import annotations

# Slice #28 (validator parity): every Nexus pipeline stage runs a real
# `validate_*_stage` call; no `_allow` stubs. The wiring mirrors the
# Praxis paper-trade reference at `praxis/launcher.py::_build_validation_pipeline`
# — same intake-hook-built-once pattern, same MMVP-lenient defaults
# (`RiskStageLimits()`, `PlatformLimitsStageLimits()`, `HealthStagePolicy()`,
# `PriceStageLimits` from config). Operator-supplied limits dial in
# real validator behavior by passing a configured `nexus_config` and
# richer snapshot providers; no new bts-specific knobs are invented.
#
# `build_validation_pipeline` returns the configured pipeline plus its
# `CapitalController`. The controller is the SHARED instance the
# action-submitter and the venue-fill bridge both drive — the
# pipeline's CAPITAL stage calls `check_and_reserve` during validation,
# and the `CapitalLifecycleTracker` feeds `send_order`, `order_ack`,
# and `order_fill` back in as Praxis's event spine produces
# `CommandAccepted`, `OrderSubmitted`, and `FillReceived` events.
import logging
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from threading import Lock

from nexus.core.capital_controller.capital_controller import CapitalController
from nexus.core.domain.capital_state import CapitalState
from nexus.core.validator import ValidationPipeline
from nexus.core.validator.capital_stage import validate_capital_stage
from nexus.core.validator.health_stage import (
    HealthStagePolicy,
    HealthStageSnapshot,
    validate_health_stage,
)
from nexus.core.validator.intake_stage import (
    build_default_intake_hooks,
    validate_intake_stage,
)
from nexus.core.validator.pipeline_models import (
    ValidationDecision,
    ValidationRequestContext,
    ValidationStage,
)
from nexus.core.validator.platform_limits_stage import (
    PlatformLimitsStageLimits,
    PlatformLimitsStageSnapshot,
    validate_platform_limits_stage,
)
from nexus.core.validator.price_stage import (
    PriceCheckSnapshot,
    build_price_stage_limits_from_config,
    validate_price_stage,
)
from nexus.core.validator.risk_stage import RiskStageLimits, validate_risk_stage
from nexus.instance_config import InstanceConfig

_log = logging.getLogger(__name__)


def _default_health_snapshot() -> HealthStageSnapshot:
    """Return a neutral-healthy `HealthStageSnapshot` for MMVP-lenient health.

    Mirrors `praxis/launcher.py::_default_health_snapshot`. With
    `HealthStagePolicy()`'s thresholds all-None this snapshot allows
    every action; only operator-supplied policy + snapshot pair fires
    a denial.
    """
    return HealthStageSnapshot(
        latency_ms=Decimal(0),
        consecutive_failures=Decimal(0),
        failure_rate=Decimal(0),
        rate_limit_headroom=Decimal(1),
        clock_drift_ms=Decimal(0),
    )


def _default_platform_snapshot() -> PlatformLimitsStageSnapshot:
    """Return an empty `PlatformLimitsStageSnapshot` for MMVP defaults."""
    return PlatformLimitsStageSnapshot()


def _default_price_snapshot() -> PriceCheckSnapshot | None:
    """Return `None`; MMVP `PriceStageLimits` are all-unset by default."""
    return None


def build_validation_pipeline(
    *,
    nexus_config: InstanceConfig,
    capital_pool: Decimal,
    reservation_ttl_seconds: int = 86_400,
    risk_limits: RiskStageLimits | None = None,
    health_policy: HealthStagePolicy | None = None,
    platform_limits: PlatformLimitsStageLimits | None = None,
    health_snapshot_provider: Callable[[], HealthStageSnapshot] = _default_health_snapshot,
    platform_snapshot_provider: Callable[[], PlatformLimitsStageSnapshot] = _default_platform_snapshot,
    price_snapshot_provider: Callable[[], PriceCheckSnapshot | None] = _default_price_snapshot,
) -> tuple[ValidationPipeline, CapitalController, CapitalState]:
    """Build the six-stage validator pipeline that runs every backtest action.

    Each stage closure captures stage-specific configuration derived
    once from `nexus_config`; mutable runtime state (health snapshot,
    platform snapshot, price snapshot) is read on every call via the
    supplied providers. MMVP defaults are deliberately lenient — same
    posture Praxis paper-trade ships (`RiskStageLimits()`,
    `PlatformLimitsStageLimits()`, `HealthStagePolicy()` all
    threshold-None; `PriceStageLimits` derived from config which
    inherits the all-unset posture). Operator-supplied limits + a
    configured `nexus_config` dial in real denial behavior.

    `reservation_ttl_seconds` defaults to 86_400 (one day) because
    backtest submission-to-fill spans frozen-minute main ticks that
    can drift further than the 30-second live default. The TTL is
    still finite (a real bug that leaks reservations will still fail
    loud on the next pipeline pass), but it's long enough that the
    expected lifecycle path is never tripped by the test-time
    acceleration.
    """
    state = CapitalState(capital_pool=capital_pool)
    controller = CapitalController(state)

    intake_hooks = build_default_intake_hooks(nexus_config)
    resolved_risk_limits = risk_limits if risk_limits is not None else RiskStageLimits()
    resolved_health_policy = (
        health_policy if health_policy is not None else HealthStagePolicy()
    )
    resolved_platform_limits = (
        platform_limits if platform_limits is not None
        else PlatformLimitsStageLimits()
    )
    price_limits = build_price_stage_limits_from_config(nexus_config)

    def intake(context: ValidationRequestContext) -> ValidationDecision:
        return validate_intake_stage(context, hooks=intake_hooks)

    def risk(context: ValidationRequestContext) -> ValidationDecision:
        return validate_risk_stage(context, resolved_risk_limits)

    def price(context: ValidationRequestContext) -> ValidationDecision:
        return validate_price_stage(
            context, price_limits, price_snapshot_provider(),
        )

    def capital(context: ValidationRequestContext) -> ValidationDecision:
        return validate_capital_stage(
            context, controller, ttl_seconds=reservation_ttl_seconds,
        )

    def health(context: ValidationRequestContext) -> ValidationDecision:
        return validate_health_stage(
            context, health_snapshot_provider(), resolved_health_policy,
        )

    def platform(context: ValidationRequestContext) -> ValidationDecision:
        return validate_platform_limits_stage(
            context, resolved_platform_limits, platform_snapshot_provider(),
        )

    validators: dict[ValidationStage, Callable[[ValidationRequestContext], ValidationDecision]] = {
        ValidationStage.INTAKE: intake,
        ValidationStage.RISK: risk,
        ValidationStage.PRICE: price,
        ValidationStage.CAPITAL: capital,
        ValidationStage.HEALTH: health,
        ValidationStage.PLATFORM_LIMITS: platform,
    }
    pipeline = ValidationPipeline(validators)
    return pipeline, controller, state


@dataclass
class _PendingLifecycle:
    """What the tracker remembers between phases of one command_id."""

    reservation_id: str
    strategy_id: str
    notional: Decimal
    estimated_fees: Decimal
    declared_stop_price: Decimal | None = None
    sent: bool = False
    acked: bool = False


@dataclass
class _OpenPosition:
    """Per-BUY open-position bookkeeping for the SELL-exit lifecycle.

    `cost_basis` is the BUY's `fill_notional` (qty * fill_price), and
    `entry_fees` is the BUY's actual fees. `CapitalController.order_fill`
    moved `cost_basis + entry_fees` into `capital_state.position_notional`,
    so the matching close releases the same `cost_basis + entry_fees`
    on full close. `entry_qty` enables proportional release on partial
    SELLs: a SELL fill of `sell_qty` releases
    `sell_qty / entry_qty * (cost_basis + entry_fees)` and shrinks
    the open position to the residual qty (codex round 5 P2 caught
    the prior shape: full-pop on partial SELL collapsed the residual).
    `command_id` and `strategy_id` echo the BUY's identity for
    audit-trail purposes.
    """

    command_id: str
    strategy_id: str
    cost_basis: Decimal
    entry_fees: Decimal
    entry_qty: Decimal


class CapitalLifecycleTracker:
    """Feed the `CapitalController` the 4-step lifecycle events.

    Part 2 requires the four-step reservation → sent → ack → fill flow.

    The backtest's action-submitter logs the reservation at
    `check_and_reserve` time (stored under `command_id`); the launcher's
    adapter wrapper then calls `record_sent` before `adapter.submit_order`
    and `record_ack_and_fill` after the fills come back. Each method
    is the identity operation on `CapitalController` with the addition
    of a conservation check via `assert_conservation` (imported lazily
    to avoid a cycle between `capital.py` and `conservation.py`).

    Thread-safety: the tracker's own dict is lock-guarded. The
    underlying controller is itself thread-safe per its docstring.
    """

    def __init__(self, controller: CapitalController) -> None:
        self._controller = controller
        self._pending: dict[str, _PendingLifecycle] = {}
        # FIFO queue of open positions; `record_open_position`
        # appends on BUY fill, `record_close_position` pops from
        # the front on SELL fill. Single-position long-only
        # strategies have at most one entry; the FIFO discipline
        # extends naturally to multi-position scenarios without
        # changing the API. See `_OpenPosition`.
        self._open_positions: list[_OpenPosition] = []
        self._lock = Lock()

    def record_reservation(
        self,
        *,
        command_id: str,
        reservation_id: str,
        strategy_id: str,
        notional: Decimal,
        estimated_fees: Decimal,
        declared_stop_price: Decimal | None = None,
    ) -> None:
        with self._lock:
            self._pending[command_id] = _PendingLifecycle(
                reservation_id=reservation_id,
                strategy_id=strategy_id,
                notional=notional,
                estimated_fees=estimated_fees,
                declared_stop_price=declared_stop_price,
            )

    def declared_stop_for_command(self, command_id: str) -> Decimal | None:
        """Lookup the declared stop for a still-pending command_id."""
        with self._lock:
            entry = self._pending.get(command_id)
            return entry.declared_stop_price if entry is not None else None

    def strategy_id_for_pending(self, command_id: str) -> str | None:
        """Lookup the strategy_id for a still-pending command_id.

        The launcher's adapter wrapper calls this BEFORE
        `record_ack_and_fill` (which pops the pending entry) so
        `record_open_position` can capture the strategy_id for
        the open-position ledger.
        """
        with self._lock:
            entry = self._pending.get(command_id)
            return entry.strategy_id if entry is not None else None

    def declared_reservation_for_command(self, command_id: str) -> Decimal | None:
        """Lookup the reserved notional for a still-pending command_id.

        The launcher's adapter wrapper uses this to fail loud on
        capital overshoot without reading the tracker's private dict.
        """
        with self._lock:
            entry = self._pending.get(command_id)
            return entry.notional if entry is not None else None

    def match_pending_by_prefix(self, prefix: str) -> str | None:
        """Return the pending command_id matching the given prefix.

        The dash-stripped form of the command_id must start with
        `prefix`, else None.

        The launcher's adapter wrapper uses this to match Praxis's
        `SS-<command-prefix>-<seq>` client_order_id back to a full
        command_id without reaching into the tracker's private
        `_pending` dict.
        """
        with self._lock:
            for command_id in self._pending:
                if command_id.replace('-', '').startswith(prefix):
                    return command_id
        return None

    def record_sent(self, command_id: str, venue_order_id: str) -> None:
        """Transition reservation → in_flight via `send_order`."""
        with self._lock:
            pending = self._pending.get(command_id)
            if pending is None:
                msg = (
                    f'record_sent: unknown command_id={command_id!r} — '
                    f'record_reservation was not called before submit_order.'
                )
                raise KeyError(msg)
            if pending.sent:
                return  # idempotent — multiple adapter wrappers may race
            result = self._controller.send_order(pending.reservation_id, venue_order_id)
            if not result.success:
                msg = (
                    f'CapitalController.send_order failed for '
                    f'command_id={command_id} venue_order_id={venue_order_id}: '
                    f'reason={result.reason!r} category={result.category}'
                )
                raise RuntimeError(msg)
            pending.sent = True

    def record_ack_and_fill(
        self,
        command_id: str,
        venue_order_id: str,
        fill_notional: Decimal,
        fees: Decimal,
        *,
        release_residual: bool = False,
    ) -> None:
        """Complete the lifecycle: `order_ack` → `order_fill` [→ `order_cancel` on terminal partial].

        In the backtest, `submit_order` returns fills synchronously so
        the ACK and the FILL collapse to the same handler call. We
        still drive the controller in the live order: ack first, then
        fill — Nexus's capital state model expects working orders to
        pass through `working_order_notional` before becoming
        `position_notional`.

        When the fill is a TERMINAL PARTIAL (the backtest's strict-live-
        reality fill model halts MARKET walks on stop breach and returns
        the pre-breach partial fill as the final result), pass
        `release_residual=True`. The tracker then drives an extra
        `order_cancel(venue_order_id)` after `order_fill` so the unfilled
        residual's reservation is released back to available capital.
        Without this, the CapitalController would keep the residual in
        `working_order_notional` indefinitely and the ledger would
        under-count available capital on every halted entry.
        """
        with self._lock:
            pending = self._pending.get(command_id)
            if pending is None:
                msg = (
                    f'record_ack_and_fill: unknown command_id={command_id!r} — '
                    f'record_reservation was not called before submit_order.'
                )
                raise KeyError(msg)
            if not pending.sent:
                msg = (
                    f'record_ack_and_fill: command_id={command_id!r} received '
                    f'ack+fill before send_order was recorded; the 4-step '
                    f'lifecycle is out of order.'
                )
                raise RuntimeError(msg)
            if not pending.acked:
                ack_result = self._controller.order_ack(venue_order_id)
                if not ack_result.success:
                    msg = (
                        f'CapitalController.order_ack failed for '
                        f'command_id={command_id} venue_order_id={venue_order_id}: '
                        f'reason={ack_result.reason!r} '
                        f'category={ack_result.category}'
                    )
                    raise RuntimeError(msg)
                pending.acked = True
            if fill_notional > 0:
                fill_result = self._controller.order_fill(
                    venue_order_id, fill_notional, fees,
                )
                if not fill_result.success:
                    msg = (
                        f'CapitalController.order_fill failed for '
                        f'command_id={command_id} venue_order_id={venue_order_id} '
                        f'fill_notional={fill_notional} fees={fees}: '
                        f'reason={fill_result.reason!r} '
                        f'category={fill_result.category}'
                    )
                    raise RuntimeError(msg)
            if release_residual and fill_notional < pending.notional:
                # Terminal partial: the unfilled residual of the reserved
                # notional stays in `working_order_notional` until we
                # explicitly cancel. `order_cancel` pops the residual
                # TrackedOrder from `_orders` and adds `remaining_total`
                # back to available capital. Idempotent if the order was
                # fully filled (order_fill already removed it).
                cancel_result = self._controller.order_cancel(venue_order_id)
                if not cancel_result.success:
                    # Cancel is expected to succeed only when a working
                    # residual exists. If it doesn't, the fill was
                    # actually terminal-full — not an error condition.
                    # We only raise on structural / invariant-breach
                    # categories; EXPECTED_MISS (order already done)
                    # is silently tolerated.
                    category = cancel_result.category
                    category_name = category.name if category is not None else ''
                    if category_name != 'EXPECTED_MISS':
                        msg = (
                            f'CapitalController.order_cancel failed releasing '
                            f'terminal-partial residual for '
                            f'command_id={command_id} venue_order_id={venue_order_id}: '
                            f'reason={cancel_result.reason!r} '
                            f'category={cancel_result.category}'
                        )
                        raise RuntimeError(msg)
            self._pending.pop(command_id, None)

    def record_rejection(self, command_id: str, venue_order_id: str) -> None:
        """Terminal reject: release the reservation back to the pool.

        Used when `SimulatedVenueAdapter.submit_order` returns status
        `REJECTED` (filter violations, min-notional failures) or when
        the adapter raises mid-submit. The CapitalController's
        release/reject call is checked against `LifecycleResult`;
        a silent failure would leave the tracker "clean" but the
        controller still holding the reservation — explicit
        `RuntimeError` surfaces that.
        """
        with self._lock:
            pending = self._pending.pop(command_id, None)
            if pending is None:
                _log.debug(
                    'record_rejection: no pending lifecycle for command_id=%s',
                    command_id,
                )
                return
            if pending.sent:
                result = self._controller.order_reject(venue_order_id)
                if not result.success:
                    msg = (
                        f'CapitalController.order_reject failed for '
                        f'command_id={command_id} venue_order_id={venue_order_id}: '
                        f'reason={result.reason!r} category={result.category}'
                    )
                    raise RuntimeError(msg)
            else:
                result = self._controller.release_reservation(
                    pending.reservation_id,
                )
                if not result.success:
                    msg = (
                        f'CapitalController.release_reservation failed for '
                        f'command_id={command_id} reservation_id='
                        f'{pending.reservation_id}: reason={result.reason!r} '
                        f'category={result.category}'
                    )
                    raise RuntimeError(msg)

    def record_open_position(
        self,
        *,
        command_id: str,
        strategy_id: str,
        cost_basis: Decimal,
        entry_fees: Decimal,
        entry_qty: Decimal,
    ) -> None:
        """Append an open position after a BUY fill.

        `cost_basis` is the BUY's `fill_notional`. `entry_fees` is
        the BUY's actual fees (controller deployed
        `cost_basis + entry_fees` into `position_notional`).
        `entry_qty` is the BUY's filled quantity — used by
        `record_close_position` to release proportionally on
        partial SELL fills.

        Idempotent under repeated calls with the same
        command_id? No — caller must dedup. The launcher's
        adapter wrapper calls this exactly once per BUY fill
        (after `record_ack_and_fill`).
        """
        with self._lock:
            self._open_positions.append(_OpenPosition(
                command_id=command_id, strategy_id=strategy_id,
                cost_basis=cost_basis, entry_fees=entry_fees,
                entry_qty=entry_qty,
            ))

    def record_close_position(
        self,
        capital_state: CapitalState,
        *,
        sell_command_id: str,
        sell_qty: Decimal,
        sell_proceeds: Decimal,
        sell_fees: Decimal,
    ) -> tuple[Decimal, _OpenPosition]:
        """FIFO-match a SELL fill against the oldest open position.

        Returns `(realized_pnl, closed_position)`. Mutates
        `capital_state` to reverse the BUY's deployment exactly:
          - `position_notional -= (cost_basis + entry_fees)` —
            mirrors the controller's `order_fill` line
            `position_notional += fill_notional + actual_fees`,
            so the close inverts the open one-for-one and
            available-budget capacity (`capital_pool -
            total_deployed`) is restored.
          - `per_strategy_deployed[strategy_id] -=
            (cost_basis + entry_fees)` — the controller's
            attribution dict is keyed on strategy_id and gets
            pruned to zero on perfect close. Without this the
            next BUY's `compute_strategy_budget` sees stale
            deployment and may deny legitimate entries.
          - `capital_pool` is NOT touched. The controller
            treats `capital_pool` as the immutable strategy
            budget; SELL proceeds are NOT new budget. Realized
            PnL is reported via the return value (callers log
            it, and a future compounding slice can feed it
            into `compute_strategy_budget(strategy_realized_pnl=...)`).
            Codex round 5 P1 caught the prior shape: crediting
            `sell_proceeds - sell_fees` to `capital_pool` was
            double-counting (capital_pool was never debited at
            BUY — only `position_notional` grew).

        Realized PnL = `sell_proceeds - cost_basis - entry_fees -
        sell_fees`. Includes BOTH legs' fees so the operator-
        visible PnL matches the audit-trail trade-pair PnL.

        Raises if no open position is available (SELL with no
        prior BUY — the strategy's `_long` gate should prevent
        this; raising surfaces a state-machine bug rather than
        silently corrupting capital).
        """
        with self._lock:
            if not self._open_positions:
                msg = (
                    f'record_close_position: SELL command_id='
                    f'{sell_command_id} has no matching open '
                    f'position. The strategy emitted a SELL while '
                    f'the lifecycle tracker held zero entries — '
                    f'either `_long` gating regressed or a prior '
                    f'BUY fill bypassed `record_open_position`.'
                )
                raise RuntimeError(msg)
            head = self._open_positions[0]
            if sell_qty <= Decimal('0'):
                msg = (
                    f'record_close_position: SELL command_id='
                    f'{sell_command_id} sell_qty={sell_qty} must be '
                    f'positive.'
                )
                raise ValueError(msg)
            if sell_qty > head.entry_qty:
                msg = (
                    f'record_close_position: SELL command_id='
                    f'{sell_command_id} sell_qty={sell_qty} exceeds '
                    f'oldest open entry_qty={head.entry_qty}; '
                    f'cross-position closes are not yet supported '
                    f'(single-position long-only invariant).'
                )
                raise RuntimeError(msg)
            ratio = sell_qty / head.entry_qty
            cost_release = head.cost_basis * ratio
            fee_release = head.entry_fees * ratio
            deployed_release = cost_release + fee_release
            realized_pnl = (
                sell_proceeds - cost_release - fee_release - sell_fees
            )
            capital_state.position_notional -= deployed_release
            current_attr = capital_state.per_strategy_deployed.get(
                head.strategy_id, Decimal('0'),
            )
            new_attr = current_attr - deployed_release
            if new_attr <= Decimal('0'):
                capital_state.per_strategy_deployed.pop(
                    head.strategy_id, None,
                )
            else:
                capital_state.per_strategy_deployed[
                    head.strategy_id
                ] = new_attr
            if sell_qty == head.entry_qty:
                self._open_positions.pop(0)
                return realized_pnl, head
            # Partial close: shrink the head position by the
            # closed share. `entry_qty`, `cost_basis`, and
            # `entry_fees` all decrement proportionally so a
            # subsequent SELL against the residual continues to
            # ratio against the remaining qty correctly. Keeps
            # the FIFO entry alive for the next preds=0 to
            # finish closing.
            head.entry_qty -= sell_qty
            head.cost_basis -= cost_release
            head.entry_fees -= fee_release
            partial_record = _OpenPosition(
                command_id=head.command_id,
                strategy_id=head.strategy_id,
                cost_basis=cost_release,
                entry_fees=fee_release,
                entry_qty=sell_qty,
            )
            return realized_pnl, partial_record

    @property
    def open_position_count(self) -> int:
        """Number of currently-open positions awaiting SELL close."""
        with self._lock:
            return len(self._open_positions)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)
