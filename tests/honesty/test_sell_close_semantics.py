"""SELL close semantics — slice #38 EXIT routing.

Long-only convention: BUY opens a long; SELL closes it. After slice
#38, SELL emits with `ActionType.EXIT` and `trade_id` propagated from
the BUY's fill. The action_submitter routes EXITs through
`validation_pipeline.validate`; the Nexus pipeline_executor bypasses
CAPITAL/HEALTH/PLATFORM_LIMITS for `ValidationAction.EXIT`, so:

  - INTAKE/RISK/PRICE run on EXITs (parity with deployed Praxis),
  - the CAPITAL substitute on EXIT lives at fill time via
    `CapitalLifecycleTracker.record_close_position`,
  - `on_submit` still fires so the launcher's drain advances.
"""
from __future__ import annotations

from decimal import Decimal
from typing import cast

from nexus.core.domain.enums import OrderSide
from nexus.core.domain.instance_state import Position
from nexus.core.domain.order_types import ExecutionMode, OrderType
from nexus.core.validator.pipeline_models import InstanceState
from nexus.infrastructure.praxis_connector.praxis_outbound import PraxisOutbound
from nexus.instance_config import InstanceConfig as NexusInstanceConfig
from nexus.strategy.action import Action, ActionType

from backtest_simulator.honesty import build_validation_pipeline
from backtest_simulator.launcher.action_submitter import (
    SubmitterBindings,
    _CommandRegistry,
    build_action_submitter,
)


class _OutboundStub:
    def __init__(self) -> None:
        self.commands: list[object] = []

    def send_command(self, cmd: object) -> str:
        self.commands.append(cmd)
        import uuid
        return str(uuid.uuid4())

    def send_abort(self, **kwargs: object) -> None:
        pass


def _seed_open_position(state: InstanceState, trade_id: str) -> None:
    """Pre-populate `state.positions[trade_id]` so EXIT INTAKE finds it."""
    state.positions[trade_id] = Position(
        trade_id=trade_id, strategy_id='long_on_signal',
        symbol='BTCUSDT', side=OrderSide.BUY,
        size=Decimal('0.001'), entry_price=Decimal('50000'),
    )


def _sell_exit_action(trade_id: str) -> Action:
    return Action(
        action_type=ActionType.EXIT,
        direction=OrderSide.SELL,
        size=Decimal('0.001'),
        execution_mode=ExecutionMode.SINGLE_SHOT,
        order_type=OrderType.MARKET,
        execution_params={'symbol': 'BTCUSDT'},
        deadline=60,
        trade_id=trade_id, command_id=None,
        maker_preference=None, reference_price=Decimal('50000'),
    )


def test_sell_close_does_not_reserve_capital() -> None:
    # EXIT routes through validate() but the pipeline_executor
    # bypasses CAPITAL for ValidationAction.EXIT. So no reservation
    # is created and `capital_state.reservation_notional` stays at 0.
    outbound = _OutboundStub()
    pipeline, controller, capital_state = build_validation_pipeline(
        nexus_config=NexusInstanceConfig(
            account_id='bts-test', venue='binance_spot_simulated',
        ),
        capital_pool=Decimal('100000'),
    )
    state = InstanceState(capital=capital_state)
    _seed_open_position(state, 'open-trade-1')
    reservations_captured: list[tuple[str, object, Action]] = []

    def on_reservation(
        command_id: str, decision: object, context: object, action: Action,
    ) -> None:
        del context
        reservations_captured.append((command_id, decision, action))

    submit = build_action_submitter(
        SubmitterBindings(
            nexus_config=NexusInstanceConfig(
                account_id='bts', venue='binance_spot_simulated',
            ),
            state=state,
            praxis_outbound=cast(PraxisOutbound, outbound),
            validation_pipeline=pipeline,
            capital_controller=controller,
            strategy_budget=Decimal('100000'),
            command_registry=_CommandRegistry(),
        ),
        on_reservation=on_reservation,
    )
    submit([_sell_exit_action('open-trade-1')], 'long_on_signal')
    # Command reached Praxis.
    assert len(outbound.commands) == 1
    # No reservation captured — EXIT bypasses CAPITAL stage entirely.
    assert reservations_captured == []
    # CapitalState unchanged — zero reservation_notional, zero deployed.
    assert capital_state.reservation_notional == Decimal('0')
    assert capital_state.in_flight_order_notional == Decimal('0')
    assert capital_state.position_notional == Decimal('0')


def test_sell_close_still_fires_on_submit() -> None:
    # The drain hook must fire for EXITs too — the launcher's
    # drain counter is how we know the venue fill completed.
    outbound = _OutboundStub()
    pipeline, controller, capital_state = build_validation_pipeline(
        nexus_config=NexusInstanceConfig(
            account_id='bts-test', venue='binance_spot_simulated',
        ),
        capital_pool=Decimal('100000'),
    )
    state = InstanceState(capital=capital_state)
    _seed_open_position(state, 'open-trade-1')
    submitted: list[str] = []
    submit = build_action_submitter(
        SubmitterBindings(
            nexus_config=NexusInstanceConfig(
                account_id='bts', venue='binance_spot_simulated',
            ),
            state=state,
            praxis_outbound=cast(PraxisOutbound, outbound),
            validation_pipeline=pipeline,
            capital_controller=controller,
            strategy_budget=Decimal('100000'),
            command_registry=_CommandRegistry(),
        ),
        on_submit=submitted.append,
    )
    submit([_sell_exit_action('open-trade-1')], 'long_on_signal')
    assert len(submitted) == 1
