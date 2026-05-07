"""Real six-stage ValidationPipeline + 4-step CapitalController lifecycle driver."""
from __future__ import annotations

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
from nexus.core.validator.intake_stage import build_default_intake_hooks, validate_intake_stage
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
    return HealthStageSnapshot(latency_ms=Decimal(0), consecutive_failures=Decimal(0), failure_rate=Decimal(0), rate_limit_headroom=Decimal(1), clock_drift_ms=Decimal(0))

def _default_platform_snapshot() -> PlatformLimitsStageSnapshot:
    return PlatformLimitsStageSnapshot()

def _default_price_snapshot() -> PriceCheckSnapshot | None:
    return None

def build_validation_pipeline(*, nexus_config: InstanceConfig, capital_pool: Decimal, reservation_ttl_seconds: int=86400, risk_limits: RiskStageLimits | None=None, health_policy: HealthStagePolicy | None=None, platform_limits: PlatformLimitsStageLimits | None=None, health_snapshot_provider: Callable[[], HealthStageSnapshot]=_default_health_snapshot, platform_snapshot_provider: Callable[[], PlatformLimitsStageSnapshot]=_default_platform_snapshot, price_snapshot_provider: Callable[[], PriceCheckSnapshot | None]=_default_price_snapshot, max_allocation_per_trade_pct: Decimal | None=None) -> tuple[ValidationPipeline, CapitalController, CapitalState]:
    state = CapitalState(capital_pool=capital_pool)
    if max_allocation_per_trade_pct is None:
        controller = CapitalController(state)
    else:
        controller = CapitalController(state, max_allocation_per_trade_pct=max_allocation_per_trade_pct)
    intake_hooks = build_default_intake_hooks(nexus_config)
    resolved_risk_limits = risk_limits if risk_limits is not None else RiskStageLimits()
    resolved_health_policy = health_policy if health_policy is not None else HealthStagePolicy()
    resolved_platform_limits = platform_limits if platform_limits is not None else PlatformLimitsStageLimits()
    price_limits = build_price_stage_limits_from_config(nexus_config)

    def intake(context: ValidationRequestContext) -> ValidationDecision:
        return validate_intake_stage(context, hooks=intake_hooks)

    def risk(context: ValidationRequestContext) -> ValidationDecision:
        return validate_risk_stage(context, resolved_risk_limits)

    def price(context: ValidationRequestContext) -> ValidationDecision:
        return validate_price_stage(context, price_limits, price_snapshot_provider())

    def capital(context: ValidationRequestContext) -> ValidationDecision:
        return validate_capital_stage(context, controller, ttl_seconds=reservation_ttl_seconds)

    def health(context: ValidationRequestContext) -> ValidationDecision:
        return validate_health_stage(context, health_snapshot_provider(), resolved_health_policy)

    def platform(context: ValidationRequestContext) -> ValidationDecision:
        return validate_platform_limits_stage(context, resolved_platform_limits, platform_snapshot_provider())
    validators: dict[ValidationStage, Callable[[ValidationRequestContext], ValidationDecision]] = {ValidationStage.INTAKE: intake, ValidationStage.RISK: risk, ValidationStage.PRICE: price, ValidationStage.CAPITAL: capital, ValidationStage.HEALTH: health, ValidationStage.PLATFORM_LIMITS: platform}
    pipeline = ValidationPipeline(validators)
    return (pipeline, controller, state)

@dataclass
class _PendingLifecycle:
    reservation_id: str
    strategy_id: str
    notional: Decimal
    estimated_fees: Decimal
    declared_stop_price: Decimal | None = None
    sent: bool = False
    acked: bool = False

@dataclass
class _OpenPosition:
    command_id: str
    strategy_id: str
    cost_basis: Decimal
    entry_fees: Decimal
    entry_qty: Decimal

class CapitalLifecycleTracker:

    def __init__(self, controller: CapitalController) -> None:
        self._controller = controller
        self._pending: dict[str, _PendingLifecycle] = {}
        self._open_positions: list[_OpenPosition] = []
        self._lock = Lock()

    def record_reservation(self, *, command_id: str, reservation_id: str, strategy_id: str, notional: Decimal, estimated_fees: Decimal, declared_stop_price: Decimal | None=None) -> None:
        with self._lock:
            self._pending[command_id] = _PendingLifecycle(reservation_id=reservation_id, strategy_id=strategy_id, notional=notional, estimated_fees=estimated_fees, declared_stop_price=declared_stop_price)

    def declared_stop_for_command(self, command_id: str) -> Decimal | None:
        with self._lock:
            entry = self._pending.get(command_id)
            return entry.declared_stop_price if entry is not None else None

    def strategy_id_for_pending(self, command_id: str) -> str | None:
        with self._lock:
            entry = self._pending.get(command_id)
            return entry.strategy_id if entry is not None else None

    def declared_reservation_for_command(self, command_id: str) -> Decimal | None:
        with self._lock:
            entry = self._pending.get(command_id)
            return entry.notional if entry is not None else None

    def match_pending_by_prefix(self, prefix: str) -> str | None:
        with self._lock:
            for command_id in self._pending:
                if command_id.replace('-', '').startswith(prefix):
                    return command_id
        return None

    def record_sent(self, command_id: str, venue_order_id: str) -> None:
        with self._lock:
            pending = self._pending.get(command_id)
            assert pending is not None
            self._controller.send_order(pending.reservation_id, venue_order_id)
            pending.sent = True

    def record_ack_and_fill(self, command_id: str, venue_order_id: str, fill_notional: Decimal, fees: Decimal, *, release_residual: bool=False) -> None:
        with self._lock:
            pending = self._pending.get(command_id)
            assert pending is not None
            if not pending.acked:
                self._controller.order_ack(venue_order_id)
                pending.acked = True
            if fill_notional > 0:
                self._controller.order_fill(venue_order_id, fill_notional, fees)
            if release_residual and fill_notional < pending.notional:
                self._controller.order_cancel(venue_order_id)
            self._pending.pop(command_id, None)

    def record_rejection(self, command_id: str, venue_order_id: str) -> None:
        with self._lock:
            pending = self._pending.pop(command_id, None)
            assert pending is not None
            if pending.sent:
                self._controller.order_reject(venue_order_id)
            else:
                self._controller.release_reservation(pending.reservation_id)

    def record_open_position(self, *, command_id: str, strategy_id: str, cost_basis: Decimal, entry_fees: Decimal, entry_qty: Decimal) -> None:
        with self._lock:
            self._open_positions.append(_OpenPosition(command_id=command_id, strategy_id=strategy_id, cost_basis=cost_basis, entry_fees=entry_fees, entry_qty=entry_qty))

    def record_close_position(self, capital_state: CapitalState, *, sell_command_id: str, sell_qty: Decimal, sell_proceeds: Decimal, sell_fees: Decimal) -> tuple[Decimal, _OpenPosition]:
        with self._lock:
            head = self._open_positions[0]
            ratio = sell_qty / head.entry_qty
            cost_release = head.cost_basis * ratio
            fee_release = head.entry_fees * ratio
            deployed_release = cost_release + fee_release
            realized_pnl = sell_proceeds - cost_release - fee_release - sell_fees
            capital_state.position_notional -= deployed_release
            current_attr = capital_state.per_strategy_deployed.get(head.strategy_id, Decimal('0'))
            new_attr = current_attr - deployed_release
            if new_attr <= Decimal('0'):
                capital_state.per_strategy_deployed.pop(head.strategy_id, None)
            if sell_qty == head.entry_qty:
                self._open_positions.pop(0)
                return (realized_pnl, head)
            head.entry_qty -= sell_qty
            head.cost_basis -= cost_release
            head.entry_fees -= fee_release
            partial_record = _OpenPosition(command_id=head.command_id, strategy_id=head.strategy_id, cost_basis=cost_release, entry_fees=fee_release, entry_qty=sell_qty)
            return (realized_pnl, partial_record)

    @property
    def open_position_count(self) -> int:
        with self._lock:
            return len(self._open_positions)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)
