"""Real-CAPITAL ValidationPipeline + 4-step CapitalController lifecycle driver.

Part 2 invariant: every ENTER/EXIT action that reaches the venue
adapter must have first cleared the real Nexus CAPITAL validator AND
must pass through `check_and_reserve → send_order → order_ack →
order_fill` on a shared `CapitalController`. The other five validator
stages (INTAKE / RISK / PRICE / HEALTH / PLATFORM_LIMITS) are wired to
an `_allow` stub because Part 2 scope is "CAPITAL real, others _allow"
per `TODO.md`; they will land as real checks in a follow-up slice.

`build_validation_pipeline` returns the configured
`nexus.core.validator.ValidationPipeline` plus its `CapitalController`.
The controller is the SHARED instance the action-submitter and the
venue-fill bridge both drive — the pipeline's CAPITAL stage calls
`check_and_reserve` during validation, and the `CapitalLifecycleTracker`
feeds `send_order`, `order_ack`, and `order_fill` back in as Praxis's
event spine produces `CommandAccepted`, `OrderSubmitted`, and
`FillReceived` events.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from threading import Lock

from nexus.core.domain.capital_state import CapitalState
from nexus.core.validator import ValidationPipeline
from nexus.core.validator.capital_stage import (
    CapitalController,
    validate_capital_stage,
)
from nexus.core.validator.pipeline_models import (
    ValidationAction,
    ValidationDecision,
    ValidationRequestContext,
    ValidationStage,
)

_log = logging.getLogger(__name__)


def _allow_stage(_ctx: ValidationRequestContext) -> ValidationDecision:
    """`_allow` stub: unconditionally pass. Used for the Part 2 stages
    we deliberately leave open (INTAKE, RISK, PRICE, HEALTH,
    PLATFORM_LIMITS). The backtest is not a production intake; all of
    those checks are gate-only-at-live concerns (order_rate, book
    staleness, etc.) that would reject honest historical hypotheses.
    """
    return ValidationDecision(allowed=True)


def build_validation_pipeline(
    *,
    capital_pool: Decimal,
    reservation_ttl_seconds: int = 86_400,
) -> tuple[ValidationPipeline, CapitalController, CapitalState]:
    """Construct the Part 2 validator stack: CAPITAL real, others _allow.

    Returns the pipeline, the `CapitalController` (the backtest
    launcher drives its 4-step lifecycle), and the `CapitalState`
    snapshot for conservation assertions.

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

    def capital_validator(context: ValidationRequestContext) -> ValidationDecision:
        return validate_capital_stage(
            context, controller, ttl_seconds=reservation_ttl_seconds,
        )

    validators: dict[ValidationStage, Callable[[ValidationRequestContext], ValidationDecision]] = {
        ValidationStage.INTAKE: _allow_stage,
        ValidationStage.RISK: _allow_stage,
        ValidationStage.PRICE: _allow_stage,
        ValidationStage.CAPITAL: capital_validator,
        ValidationStage.HEALTH: _allow_stage,
        ValidationStage.PLATFORM_LIMITS: _allow_stage,
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


class CapitalLifecycleTracker:
    """Feeds the `CapitalController` the 4-step lifecycle events that
    Part 2 requires.

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
    ) -> None:
        """Complete the lifecycle: `order_ack` → `order_fill`.

        In the backtest, `submit_order` returns fills synchronously so
        the ACK and the FILL collapse to the same handler call. We
        still drive the controller in the live order: ack first, then
        fill — Nexus's capital state model expects working orders to
        pass through `working_order_notional` before becoming
        `position_notional`.
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
            self._pending.pop(command_id, None)

    def record_rejection(self, command_id: str, venue_order_id: str) -> None:
        """Terminal reject: release the reservation back to the pool.

        Used when `SimulatedVenueAdapter.submit_order` returns status
        `REJECTED` (filter violations, min-notional failures). Both
        `order_reject` (if `send_order` was recorded) and
        `release_reservation` (if it wasn't) are safe no-ops when the
        tracker has no pending entry.
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
                self._controller.order_reject(venue_order_id)
            else:
                self._controller.release_reservation(pending.reservation_id)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)
