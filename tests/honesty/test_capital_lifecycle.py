"""Part 2 tests for `backtest_simulator.honesty.capital`.

Covers:
  - `build_validation_pipeline` wiring (CAPITAL real, others _allow).
  - `CapitalLifecycleTracker` 4-step lifecycle happy path.
  - Fail-loud on out-of-order lifecycle events.
  - `record_rejection` releases reservations back to the pool.
  - `declared_stop_for_command` / `declared_reservation_for_command`
    accessor correctness (what the launcher's adapter wrapper reads).

Plus 2 mutation tests that inject known bugs (fill-before-ack, skip
record_reservation) and confirm the tracker raises.
"""
from __future__ import annotations

from decimal import Decimal

import pytest
from nexus.core.validator.pipeline_models import (
    ValidationAction,
    ValidationRequestContext,
    ValidationStage,
)

from backtest_simulator.honesty import (
    CapitalLifecycleTracker,
    build_validation_pipeline,
)


def _ctx(
    *,
    strategy_id: str = 'bts',
    notional: Decimal = Decimal('1000'),
    fees: Decimal = Decimal('2'),
    budget: Decimal = Decimal('100000'),
    command_id: str = 'cmd-test-1',
) -> ValidationRequestContext:
    # Mirror `_build_context` shape with just the fields the CAPITAL
    # stage reads; other fields are filled with safe sentinels.
    from nexus.core.domain.capital_state import CapitalState
    from nexus.core.domain.enums import OrderSide
    from nexus.core.validator.pipeline_models import InstanceState
    from nexus.instance_config import InstanceConfig as NexusInstanceConfig
    return ValidationRequestContext(
        strategy_id=strategy_id,
        order_notional=notional,
        estimated_fees=fees,
        strategy_budget=budget,
        state=InstanceState(capital=CapitalState(capital_pool=budget)),
        config=NexusInstanceConfig(
            account_id='bts-acct', venue='binance_spot_simulated',
        ),
        action=ValidationAction.ENTER,
        symbol='BTCUSDT',
        order_side=OrderSide.BUY,
        order_size=Decimal('0.01'),
        trade_id='trade-1',
        command_id=command_id,
        current_order_notional=None,
    )


def test_build_pipeline_has_all_six_stages() -> None:
    pipeline, _controller, _state = build_validation_pipeline(
        capital_pool=Decimal('100000'),
    )
    stages = set(pipeline.stage_order)
    assert stages == {
        ValidationStage.INTAKE,
        ValidationStage.RISK,
        ValidationStage.PRICE,
        ValidationStage.CAPITAL,
        ValidationStage.HEALTH,
        ValidationStage.PLATFORM_LIMITS,
    }


def test_pipeline_accepts_affordable_enter() -> None:
    # Affordable: notional well under budget, CAPITAL approves,
    # other stages are _allow passes. Decision carries a reservation.
    pipeline, _controller, state = build_validation_pipeline(
        capital_pool=Decimal('100000'),
    )
    # Reuse the pipeline's CapitalState — that's what CAPITAL reads.
    ctx = _ctx(notional=Decimal('1000'), fees=Decimal('2'), budget=Decimal('100000'))
    ctx = ValidationRequestContext(
        strategy_id=ctx.strategy_id,
        order_notional=ctx.order_notional,
        estimated_fees=ctx.estimated_fees,
        strategy_budget=ctx.strategy_budget,
        state=_instance_state_sharing(state),
        config=ctx.config,
        action=ctx.action, symbol=ctx.symbol,
        order_side=ctx.order_side, order_size=ctx.order_size,
        trade_id=ctx.trade_id, command_id=ctx.command_id,
        current_order_notional=ctx.current_order_notional,
    )
    decision = pipeline.validate(ctx)
    assert decision.allowed is True
    assert decision.reservation is not None
    assert decision.reservation.notional == Decimal('1000')


def test_pipeline_denies_per_trade_limit_breach() -> None:
    # 15% per-trade cap: a 20k notional against a 100k budget is
    # 20% and must be denied at CAPITAL.
    pipeline, _controller, state = build_validation_pipeline(
        capital_pool=Decimal('100000'),
    )
    ctx = _ctx(notional=Decimal('20000'), fees=Decimal('0'), budget=Decimal('100000'))
    ctx = ValidationRequestContext(
        strategy_id=ctx.strategy_id, order_notional=ctx.order_notional,
        estimated_fees=ctx.estimated_fees, strategy_budget=ctx.strategy_budget,
        state=_instance_state_sharing(state), config=ctx.config,
        action=ctx.action, symbol=ctx.symbol, order_side=ctx.order_side,
        order_size=ctx.order_size, trade_id=ctx.trade_id,
        command_id=ctx.command_id, current_order_notional=None,
    )
    decision = pipeline.validate(ctx)
    assert decision.allowed is False
    assert decision.failed_stage == ValidationStage.CAPITAL


def _instance_state_sharing(capital_state):
    from nexus.core.validator.pipeline_models import InstanceState
    return InstanceState(capital=capital_state)


def test_lifecycle_happy_path() -> None:
    # Walk the 4-step lifecycle end-to-end: reserve → send → ack → fill.
    pipeline, controller, state = build_validation_pipeline(
        capital_pool=Decimal('100000'),
    )
    tracker = CapitalLifecycleTracker(controller)
    # Step 1: reserve via the pipeline so we get a real reservation_id.
    ctx = _ctx(notional=Decimal('1000'), fees=Decimal('2'))
    ctx = ValidationRequestContext(
        strategy_id=ctx.strategy_id, order_notional=ctx.order_notional,
        estimated_fees=ctx.estimated_fees, strategy_budget=ctx.strategy_budget,
        state=_instance_state_sharing(state), config=ctx.config,
        action=ctx.action, symbol=ctx.symbol, order_side=ctx.order_side,
        order_size=ctx.order_size, trade_id=ctx.trade_id,
        command_id=ctx.command_id, current_order_notional=None,
    )
    decision = pipeline.validate(ctx)
    tracker.record_reservation(
        command_id='cmd-happy',
        reservation_id=decision.reservation.reservation_id,
        strategy_id=ctx.strategy_id,
        notional=decision.reservation.notional,
        estimated_fees=decision.reservation.estimated_fees,
    )
    assert tracker.pending_count == 1
    # Step 2+3+4.
    tracker.record_sent('cmd-happy', 'VENUE-ORDER-1')
    tracker.record_ack_and_fill(
        'cmd-happy', 'VENUE-ORDER-1',
        fill_notional=Decimal('1000'), fees=Decimal('2'),
    )
    assert tracker.pending_count == 0


def test_lifecycle_fail_loud_on_skip_reservation() -> None:
    _pipeline, controller, _state = build_validation_pipeline(
        capital_pool=Decimal('100000'),
    )
    tracker = CapitalLifecycleTracker(controller)
    # No record_reservation called — record_sent must raise KeyError.
    with pytest.raises(KeyError, match='unknown command_id'):
        tracker.record_sent('cmd-missing', 'VENUE-1')


def test_lifecycle_fail_loud_on_fill_before_send() -> None:
    pipeline, controller, state = build_validation_pipeline(
        capital_pool=Decimal('100000'),
    )
    tracker = CapitalLifecycleTracker(controller)
    ctx = _ctx()
    ctx = ValidationRequestContext(
        strategy_id=ctx.strategy_id, order_notional=ctx.order_notional,
        estimated_fees=ctx.estimated_fees, strategy_budget=ctx.strategy_budget,
        state=_instance_state_sharing(state), config=ctx.config,
        action=ctx.action, symbol=ctx.symbol, order_side=ctx.order_side,
        order_size=ctx.order_size, trade_id=ctx.trade_id,
        command_id=ctx.command_id, current_order_notional=None,
    )
    decision = pipeline.validate(ctx)
    tracker.record_reservation(
        command_id='cmd-badorder',
        reservation_id=decision.reservation.reservation_id,
        strategy_id=ctx.strategy_id,
        notional=decision.reservation.notional,
        estimated_fees=decision.reservation.estimated_fees,
    )
    # Skip send_order, jump straight to fill — the tracker must raise.
    with pytest.raises(RuntimeError, match='out of order'):
        tracker.record_ack_and_fill(
            'cmd-badorder', 'VENUE-1',
            fill_notional=Decimal('1000'), fees=Decimal('0'),
        )


def test_declared_stop_lookup_roundtrip() -> None:
    _pipeline, controller, _state = build_validation_pipeline(
        capital_pool=Decimal('100000'),
    )
    tracker = CapitalLifecycleTracker(controller)
    tracker.record_reservation(
        command_id='cmd-stop', reservation_id='r-1', strategy_id='s',
        notional=Decimal('1000'), estimated_fees=Decimal('2'),
        declared_stop_price=Decimal('49500'),
    )
    assert tracker.declared_stop_for_command('cmd-stop') == Decimal('49500')
    assert tracker.declared_reservation_for_command('cmd-stop') == Decimal('1000')
    # Unknown commands return None, never raise.
    assert tracker.declared_stop_for_command('unknown') is None
    assert tracker.declared_reservation_for_command('unknown') is None


def test_record_rejection_releases_reservation() -> None:
    _pipeline, controller, state = build_validation_pipeline(
        capital_pool=Decimal('100000'),
    )
    tracker = CapitalLifecycleTracker(controller)
    # Reserve via pipeline so the controller has the reservation.
    _pipeline, _, _ = build_validation_pipeline(capital_pool=Decimal('100000'))
    # Use the same controller for the tracker's pipeline.
    # (build_validation_pipeline creates a fresh controller each call.)
    ctx = _ctx()
    ctx = ValidationRequestContext(
        strategy_id=ctx.strategy_id, order_notional=ctx.order_notional,
        estimated_fees=ctx.estimated_fees, strategy_budget=ctx.strategy_budget,
        state=_instance_state_sharing(state), config=ctx.config,
        action=ctx.action, symbol=ctx.symbol, order_side=ctx.order_side,
        order_size=ctx.order_size, trade_id=ctx.trade_id,
        command_id=ctx.command_id, current_order_notional=None,
    )
    # Reserve against the tracker's controller/state.
    from nexus.core.validator.capital_stage import validate_capital_stage
    decision = validate_capital_stage(ctx, controller, ttl_seconds=60)
    assert decision.allowed is True
    tracker.record_reservation(
        command_id='cmd-reject',
        reservation_id=decision.reservation.reservation_id,
        strategy_id=ctx.strategy_id,
        notional=decision.reservation.notional,
        estimated_fees=decision.reservation.estimated_fees,
    )
    assert state.reservation_notional > 0
    # Reject before send_order — should release the reservation.
    tracker.record_rejection('cmd-reject', 'VENUE-1')
    assert tracker.pending_count == 0
    # Reservation back to zero (cleaned up by CapitalController).
    assert state.reservation_notional == 0


# ---- MUTATION TESTS -------------------------------------------------------


def test_mutation_fill_without_pending_raises() -> None:
    # Injected bug: a hypothetical path calls record_ack_and_fill
    # without a prior record_reservation. Must not silently no-op.
    _pipeline, controller, _state = build_validation_pipeline(
        capital_pool=Decimal('100000'),
    )
    tracker = CapitalLifecycleTracker(controller)
    with pytest.raises(KeyError):
        tracker.record_ack_and_fill(
            'mut-missing', 'VENUE-1',
            fill_notional=Decimal('1000'), fees=Decimal('1'),
        )


def test_mutation_double_fill_raises() -> None:
    # Injected bug: record_ack_and_fill called twice on the same
    # command_id. The second call must fail because pending is popped.
    pipeline, controller, state = build_validation_pipeline(
        capital_pool=Decimal('100000'),
    )
    tracker = CapitalLifecycleTracker(controller)
    ctx = _ctx()
    ctx = ValidationRequestContext(
        strategy_id=ctx.strategy_id, order_notional=ctx.order_notional,
        estimated_fees=ctx.estimated_fees, strategy_budget=ctx.strategy_budget,
        state=_instance_state_sharing(state), config=ctx.config,
        action=ctx.action, symbol=ctx.symbol, order_side=ctx.order_side,
        order_size=ctx.order_size, trade_id=ctx.trade_id,
        command_id=ctx.command_id, current_order_notional=None,
    )
    decision = pipeline.validate(ctx)
    tracker.record_reservation(
        command_id='cmd-dbl', reservation_id=decision.reservation.reservation_id,
        strategy_id=ctx.strategy_id, notional=decision.reservation.notional,
        estimated_fees=decision.reservation.estimated_fees,
    )
    tracker.record_sent('cmd-dbl', 'VENUE-DBL-1')
    tracker.record_ack_and_fill(
        'cmd-dbl', 'VENUE-DBL-1',
        fill_notional=Decimal('1000'), fees=Decimal('2'),
    )
    # Second fill on the same command → pending was popped → raises.
    with pytest.raises(KeyError):
        tracker.record_ack_and_fill(
            'cmd-dbl', 'VENUE-DBL-1',
            fill_notional=Decimal('1000'), fees=Decimal('2'),
        )
