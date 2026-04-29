"""Slice #28 validator-parity tests.

Every Nexus pipeline stage runs a real `validate_*_stage` call; no
`_allow_stage` stubs remain. Each per-stage denial test triggers the
real upstream behavior end-to-end through the pipeline by passing
config / limits / snapshot providers — the same dials Praxis exposes
for paper-trade validator tuning.
"""
from __future__ import annotations

import ast
from decimal import Decimal
from pathlib import Path
from typing import cast

from nexus.core.domain.capital_state import CapitalState
from nexus.core.domain.enums import OrderSide
from nexus.core.domain.instance_state import InstanceState
from nexus.core.domain.order_types import ExecutionMode, OrderType
from nexus.core.domain.risk_state import RiskState
from nexus.core.validator.health_stage import (
    HealthMetricThresholds,
    HealthStagePolicy,
    HealthStageSnapshot,
)
from nexus.core.validator.pipeline_models import (
    ValidationAction,
    ValidationRequestContext,
    ValidationStage,
)
from nexus.core.validator.platform_limits_stage import (
    PlatformLimitsStageLimits,
)
from nexus.core.validator.price_stage import PriceCheckSnapshot
from nexus.core.validator.risk_stage import RiskStageLimits
from nexus.infrastructure.praxis_connector.praxis_outbound import PraxisOutbound
from nexus.instance_config import InstanceConfig
from nexus.strategy.action import Action, ActionType

from backtest_simulator.honesty import build_validation_pipeline
from backtest_simulator.launcher.action_submitter import (
    SubmitterBindings,
    _check_atr_sanity,
    _check_declared_stop,
    build_action_submitter,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CAPITAL_PY = REPO_ROOT / 'backtest_simulator' / 'honesty' / 'capital.py'

_PARITY_DIVERGENCE_PHRASE = 'supplementing the real Nexus INTAKE stage'


def _ctx(
    *,
    command_id: str,
    notional: Decimal = Decimal('1000'),
    fees: Decimal = Decimal('2'),
    budget: Decimal = Decimal('100000'),
    action: ValidationAction = ValidationAction.ENTER,
    side: OrderSide = OrderSide.BUY,
    nexus_config: InstanceConfig | None = None,
    risk_state: RiskState | None = None,
    capital_state: CapitalState | None = None,
) -> ValidationRequestContext:
    state = InstanceState(
        capital=(
            capital_state if capital_state is not None
            else CapitalState(capital_pool=budget)
        ),
        risk=risk_state if risk_state is not None else RiskState(),
    )
    return ValidationRequestContext(
        strategy_id='bts',
        order_notional=notional,
        estimated_fees=fees,
        strategy_budget=budget,
        state=state,
        config=nexus_config or InstanceConfig(
            account_id='bts-test', venue='binance_spot_simulated',
        ),
        action=action,
        symbol='BTCUSDT',
        order_side=side,
        order_size=Decimal('0.01'),
        trade_id='trade-1',
        command_id=command_id,
        current_order_notional=None,
    )


# ---- core parity proofs ----------------------------------------------------


def test_no_allow_stage_in_source() -> None:
    """`_allow_stage` is gone from `capital.py`. AST + grep-clean."""
    src = CAPITAL_PY.read_text(encoding='utf-8')
    tree = ast.parse(src)
    fn_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert '_allow_stage' not in fn_names, (
        '`_allow_stage` must not be defined in capital.py — its presence '
        'proves the validator wiring regressed back to stubs.'
    )
    assert '_allow_stage' not in src, (
        '`_allow_stage` must not appear anywhere in capital.py source '
        '(definition, reference, comment, or docstring).'
    )


def test_all_stages_run_real_validators() -> None:
    """Every pipeline stage binds to a non-`_allow_stage` callable."""
    cfg = InstanceConfig(account_id='bts-test', venue='binance_spot_simulated')
    pipeline, _ctrl, _state = build_validation_pipeline(
        nexus_config=cfg, capital_pool=Decimal('100000'),
    )
    for stage in pipeline.stage_order:
        validator = pipeline._validators[stage]
        qualname = validator.__qualname__
        assert '_allow_stage' not in qualname, (
            f'{stage.name} validator `{qualname}` is the `_allow_stage` stub'
        )
    # Capture the full set of stages so a future PR that drops one fails
    # both this test and the pipeline's own `_validate_validators` check.
    assert set(pipeline.stage_order) == {
        ValidationStage.INTAKE, ValidationStage.RISK, ValidationStage.PRICE,
        ValidationStage.CAPITAL, ValidationStage.HEALTH,
        ValidationStage.PLATFORM_LIMITS,
    }


# ---- per-stage end-to-end denials ------------------------------------------


def test_intake_max_order_rate_denies() -> None:
    """INTAKE: `max_order_rate=1` denies the second ENTER inside one second."""
    cfg = InstanceConfig(
        account_id='bts-test', venue='binance_spot_simulated',
        max_order_rate=1,
    )
    pipeline, _, _ = build_validation_pipeline(
        nexus_config=cfg, capital_pool=Decimal('100000'),
    )
    d1 = pipeline.validate(_ctx(command_id='cmd-rate-1', nexus_config=cfg))
    d2 = pipeline.validate(_ctx(command_id='cmd-rate-2', nexus_config=cfg))
    assert d1.allowed
    assert not d2.allowed
    assert d2.failed_stage == ValidationStage.INTAKE
    assert d2.reason_code == 'INTAKE_MAX_ORDER_RATE_EXCEEDED'


def test_intake_duplicate_command_id_denies() -> None:
    """INTAKE: same command_id within the duplicate window denies."""
    cfg = InstanceConfig(account_id='bts-test', venue='binance_spot_simulated')
    pipeline, _, _ = build_validation_pipeline(
        nexus_config=cfg, capital_pool=Decimal('100000'),
    )
    ctx_a = _ctx(command_id='cmd-dup-1', nexus_config=cfg)
    ctx_b = _ctx(command_id='cmd-dup-1', nexus_config=cfg)
    d1 = pipeline.validate(ctx_a)
    d2 = pipeline.validate(ctx_b)
    assert d1.allowed
    assert not d2.allowed
    assert d2.failed_stage == ValidationStage.INTAKE
    assert d2.reason_code == 'INTAKE_DUPLICATE_ORDER_WINDOW'


def test_price_spread_denies() -> None:
    """PRICE: snapshot spread above `max_spread_bps` denies."""
    cfg = InstanceConfig(
        account_id='bts-test', venue='binance_spot_simulated',
        max_spread_bps=Decimal('5'),
    )

    def wide_spread() -> PriceCheckSnapshot | None:
        return PriceCheckSnapshot(spread_bps=Decimal('50'))

    pipeline, _, _ = build_validation_pipeline(
        nexus_config=cfg, capital_pool=Decimal('100000'),
        price_snapshot_provider=wide_spread,
    )
    decision = pipeline.validate(_ctx(command_id='cmd-price', nexus_config=cfg))
    assert not decision.allowed
    assert decision.failed_stage == ValidationStage.PRICE
    assert decision.reason_code == 'PRICE_SPREAD_LIMIT'


def test_risk_drawdown_denies() -> None:
    """RISK: total_drawdown above `max_total_drawdown` denies."""
    cfg = InstanceConfig(account_id='bts-test', venue='binance_spot_simulated')
    pipeline, _, _ = build_validation_pipeline(
        nexus_config=cfg, capital_pool=Decimal('100000'),
        risk_limits=RiskStageLimits(max_total_drawdown=Decimal('100')),
    )
    risk = RiskState()
    risk.total_drawdown = Decimal('200')
    ctx = _ctx(command_id='cmd-risk', nexus_config=cfg, risk_state=risk)
    # Re-bind context.state.risk because pipeline reads context.state.risk
    # not its own state. Replace the ctx.state with one that already has
    # the breached drawdown.
    decision = pipeline.validate(ctx)
    assert not decision.allowed
    assert decision.failed_stage == ValidationStage.RISK
    assert decision.reason_code == 'RISK_TOTAL_DRAWDOWN_LIMIT'


def test_health_latency_denies() -> None:
    """HEALTH: snapshot latency above breach threshold denies."""
    cfg = InstanceConfig(account_id='bts-test', venue='binance_spot_simulated')
    policy = HealthStagePolicy(
        latency_ms=HealthMetricThresholds(breach=Decimal('10')),
    )

    def hot_snapshot() -> HealthStageSnapshot:
        return HealthStageSnapshot(
            latency_ms=Decimal('100'),
            consecutive_failures=Decimal('0'),
            failure_rate=Decimal('0'),
            rate_limit_headroom=Decimal('1'),
            clock_drift_ms=Decimal('0'),
        )

    pipeline, _, _ = build_validation_pipeline(
        nexus_config=cfg, capital_pool=Decimal('100000'),
        health_policy=policy,
        health_snapshot_provider=hot_snapshot,
    )
    decision = pipeline.validate(_ctx(command_id='cmd-health', nexus_config=cfg))
    assert not decision.allowed
    assert decision.failed_stage == ValidationStage.HEALTH
    assert decision.reason_code == 'HEALTH_LATENCY_BREACH'


def test_platform_notional_denies() -> None:
    """PLATFORM_LIMITS: order_notional above `max_order_notional` denies."""
    cfg = InstanceConfig(account_id='bts-test', venue='binance_spot_simulated')
    pipeline, _, _ = build_validation_pipeline(
        nexus_config=cfg, capital_pool=Decimal('100000'),
        platform_limits=PlatformLimitsStageLimits(
            max_order_notional=Decimal('100'),
        ),
    )
    ctx = _ctx(
        command_id='cmd-platform', notional=Decimal('1000'), nexus_config=cfg,
    )
    decision = pipeline.validate(ctx)
    assert not decision.allowed
    assert decision.failed_stage == ValidationStage.PLATFORM_LIMITS
    assert decision.reason_code.startswith('PLATFORM_LIMITS_MAX_ORDER_NOTIONAL')


# ---- Pipeline-denied reservation is released, not leaked ------------------


class _OutboundStub:
    """Captures `send_command` / `send_abort` for the action_submitter.

    The release-leak test drives the production `submit` end-to-end;
    we never expect a command to reach the outbound when HEALTH denies,
    but we keep the fields populated to satisfy the protocol.
    """

    def __init__(self) -> None:
        self.commands: list[object] = []
        self.aborts: list[object] = []

    def send_command(self, cmd: object) -> str:
        self.commands.append(cmd)
        import uuid
        return str(uuid.uuid4())

    def send_abort(self, **kwargs: object) -> None:
        self.aborts.append(kwargs)


def test_late_stage_denial_releases_capital_reservation() -> None:
    """HEALTH denial after CAPITAL allows must release the reservation.

    End-to-end through the production action_submitter: build pipeline
    + controller, wire SubmitterBindings exactly the way the launcher
    does, drive a BUY ENTER that CAPITAL allows but HEALTH denies, and
    assert that `capital_state.reservation_notional` is back to zero.

    Deleting the release block at `action_submitter.py::_submit_translated`
    must fail this test. CAPITAL is stage 4 of 6, so a HEALTH or
    PLATFORM_LIMITS denial that fires AFTER CAPITAL carries the
    reservation forward in the denied decision; without explicit
    release, available capital would leak per-denial.
    """
    cfg = InstanceConfig(account_id='bts-test', venue='binance_spot_simulated')
    policy = HealthStagePolicy(
        latency_ms=HealthMetricThresholds(breach=Decimal('10')),
    )

    def hot_snapshot() -> HealthStageSnapshot:
        return HealthStageSnapshot(
            latency_ms=Decimal('100'), consecutive_failures=Decimal('0'),
            failure_rate=Decimal('0'), rate_limit_headroom=Decimal('1'),
            clock_drift_ms=Decimal('0'),
        )

    pipeline, controller, capital_state = build_validation_pipeline(
        nexus_config=cfg, capital_pool=Decimal('100000'),
        health_policy=policy, health_snapshot_provider=hot_snapshot,
    )
    outbound = _OutboundStub()
    submit = build_action_submitter(SubmitterBindings(
        nexus_config=cfg,
        state=InstanceState(capital=capital_state),
        praxis_outbound=cast(PraxisOutbound, outbound),
        validation_pipeline=pipeline,
        capital_controller=controller,
        strategy_budget=Decimal('100000'),
    ))
    action = Action(
        action_type=ActionType.ENTER,
        direction=OrderSide.BUY,
        size=Decimal('0.001'),
        execution_mode=ExecutionMode.SINGLE_SHOT,
        order_type=OrderType.MARKET,
        execution_params={'symbol': 'BTCUSDT', 'stop_price': '49750'},
        deadline=60,
        trade_id=None,
        command_id='cmd-action-submit-release',
        maker_preference=None,
        reference_price=Decimal('50000'),
    )
    submit([action], 'long_on_signal')
    # HEALTH denied → no command reached Praxis.
    assert len(outbound.commands) == 0
    # ...AND the CAPITAL-stage reservation was released.
    # (Removing the release block in _submit_translated would leave
    # reservation_notional > 0 here.)
    assert capital_state.reservation_notional == Decimal('0'), (
        f'reservation leaked: notional={capital_state.reservation_notional} — '
        'the action_submitter did not release the CAPITAL reservation '
        'attached to the HEALTH-denied decision.'
    )


# ---- Fail-loud on release_reservation failure -----------------------------


def test_release_reservation_failure_fails_loud() -> None:
    """`release_reservation` returning success=False raises RuntimeError.

    Bit-mis no-defects-conviction P1: the reservation being released
    was just minted by the same `pipeline.validate()` call on the same
    controller, so any non-success means a wiring/state bug — not an
    expected race. Mirrors `CapitalLifecycleTracker.record_rejection`'s
    raise-on-failure pattern. Letting the bug through with a warning
    would silently re-introduce the leak the branch exists to prevent.
    """
    cfg = InstanceConfig(account_id='bts-test', venue='binance_spot_simulated')
    policy = HealthStagePolicy(
        latency_ms=HealthMetricThresholds(breach=Decimal('10')),
    )

    def hot_snapshot() -> HealthStageSnapshot:
        return HealthStageSnapshot(
            latency_ms=Decimal('100'), consecutive_failures=Decimal('0'),
            failure_rate=Decimal('0'), rate_limit_headroom=Decimal('1'),
            clock_drift_ms=Decimal('0'),
        )

    pipeline, controller, capital_state = build_validation_pipeline(
        nexus_config=cfg, capital_pool=Decimal('100000'),
        health_policy=policy, health_snapshot_provider=hot_snapshot,
    )

    # Wrap the real controller so `release_reservation` returns a
    # success=False LifecycleResult — simulating the wiring bug
    # bit-mis flagged. Every other method delegates to the real one
    # so the pipeline's CAPITAL stage still produces a real reservation.
    class _FailingReleaseController:
        def __init__(self, real: object) -> None:
            self._real = real

        def __getattr__(self, name: str) -> object:
            return getattr(self._real, name)

        def release_reservation(self, _reservation_id: str) -> object:
            from nexus.core.capital_controller.capital_controller import (
                LifecycleResult,
            )
            return LifecycleResult(
                success=False, reason='simulated-wiring-bug',
                category=None,
            )

    failing_controller = _FailingReleaseController(controller)
    outbound = _OutboundStub()
    submit = build_action_submitter(SubmitterBindings(
        nexus_config=cfg,
        state=InstanceState(capital=capital_state),
        praxis_outbound=cast(PraxisOutbound, outbound),
        validation_pipeline=pipeline,
        capital_controller=cast(object, failing_controller),  # type: ignore[arg-type]
        strategy_budget=Decimal('100000'),
    ))
    action = Action(
        action_type=ActionType.ENTER, direction=OrderSide.BUY,
        size=Decimal('0.001'), execution_mode=ExecutionMode.SINGLE_SHOT,
        order_type=OrderType.MARKET,
        execution_params={'symbol': 'BTCUSDT', 'stop_price': '49750'},
        deadline=60, trade_id=None,
        command_id='cmd-fail-loud',
        maker_preference=None, reference_price=Decimal('50000'),
    )
    import pytest
    with pytest.raises(RuntimeError, match='pipeline-denied reservation release failed'):
        submit([action], 'long_on_signal')


# ---- SELL fast-path divergence is documented in source --------------------


def test_sell_fast_path_documented_in_source() -> None:
    """The bts-only SELL fast-path's bypass reasons are visible in source.

    Slice #28 deliberately leaves the SELL close path bypassing
    `validation_pipeline.validate` because (a) the long-only strategy
    template does not propagate `Action.trade_id` from BUY to SELL, so
    Nexus's `make_reference_integrity_hook` would deny every close on
    `INTAKE_TRADE_REFERENCE_INVALID`, and (b) `CapitalController` has
    no `close_position` primitive. Both follow-ups are upstream
    Nexus/strategy work tracked at slice merge. The divergence stays
    explicit in the source comment so a future reviewer sees the
    debt rather than an unexplained `if action.direction == OrderSide.SELL:`.
    """
    submitter_src = (
        Path(__file__).resolve().parents[2]
        / 'backtest_simulator' / 'launcher' / 'action_submitter.py'
    ).read_text(encoding='utf-8')
    assert 'pipeline bypassed' in submitter_src
    for marker in ('trade_id', 'close_position'):
        assert marker in submitter_src


# ---- INTAKE pre-hook divergence is documented, not hidden -----------------


def test_intake_pre_hook_docstring_documents_divergence() -> None:
    """Both pre-hooks cite the parity-divergence phrase in their docstring."""
    declared_doc = _check_declared_stop.__doc__ or ''
    atr_doc = _check_atr_sanity.__doc__ or ''
    assert _PARITY_DIVERGENCE_PHRASE in declared_doc, (
        '_check_declared_stop docstring must cite the divergence phrase'
    )
    assert _PARITY_DIVERGENCE_PHRASE in atr_doc, (
        '_check_atr_sanity docstring must cite the divergence phrase'
    )
