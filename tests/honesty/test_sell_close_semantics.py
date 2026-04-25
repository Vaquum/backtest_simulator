"""SELL close semantics — the Part 2 exit leg does NOT reserve capital.

Long-only convention: BUY opens a long (CAPITAL stage reserves notional);
SELL with `ActionType.ENTER` closes the open long. A SELL submitted
through the action-submitter must:

  - bypass the CAPITAL reservation (no double-commit on exit),
  - still flow through `translate_to_trade_command` and Praxis send,
  - still fire the `on_submit` drain hook.

Nexus's `ActionType.EXIT` is the long-run correct tag but requires
`trade_id`-linked lifecycle that exceeds Part 2 scope — documented in
the action_submitter. The test here pins the agreed Part 2 behavior.
"""
from __future__ import annotations

from decimal import Decimal
from typing import cast

from nexus.core.domain.enums import OrderSide
from nexus.core.domain.order_types import ExecutionMode, OrderType
from nexus.core.validator.pipeline_models import InstanceState
from nexus.infrastructure.praxis_connector.praxis_outbound import PraxisOutbound
from nexus.instance_config import InstanceConfig as NexusInstanceConfig
from nexus.strategy.action import Action, ActionType

from backtest_simulator.honesty import build_validation_pipeline
from backtest_simulator.launcher.action_submitter import (
    SubmitterBindings,
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


def _sell_close_action() -> Action:
    return Action(
        action_type=ActionType.ENTER,
        direction=OrderSide.SELL,
        size=Decimal('0.001'),
        execution_mode=ExecutionMode.SINGLE_SHOT,
        order_type=OrderType.MARKET,
        execution_params={'symbol': 'BTCUSDT'},
        deadline=60,
        trade_id=None, command_id=None,
        maker_preference=None, reference_price=Decimal('50000'),
    )


def test_sell_close_does_not_reserve_capital() -> None:
    # Baseline: capital state's reservation_notional starts at zero.
    # After a SELL close, reservation_notional MUST still be zero (no
    # double-commit).
    outbound = _OutboundStub()
    pipeline, _controller, capital_state = build_validation_pipeline(
        capital_pool=Decimal('100000'),
    )
    state = InstanceState(capital=capital_state)
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
            strategy_budget=Decimal('100000'),
        ),
        on_reservation=on_reservation,
    )
    submit([_sell_close_action()], 'long_on_signal')
    # Command reached Praxis.
    assert len(outbound.commands) == 1
    # No reservation captured — `on_reservation` is only fired when
    # the decision has a reservation. The action_submitter short-
    # circuits the CAPITAL stage for SELL and returns a reservation-
    # less decision.
    assert reservations_captured == [] or reservations_captured[0][1].reservation is None
    # CapitalState unchanged — zero reservation_notional, zero deployed.
    assert capital_state.reservation_notional == Decimal('0')
    assert capital_state.in_flight_order_notional == Decimal('0')
    assert capital_state.position_notional == Decimal('0')


def test_sell_close_still_fires_on_submit() -> None:
    # The drain hook must fire for SELL closes too — the launcher's
    # drain counter is how we know the venue fill completed. Skipping
    # the drain for SELL would hang the clock.
    outbound = _OutboundStub()
    pipeline, _controller, capital_state = build_validation_pipeline(
        capital_pool=Decimal('100000'),
    )
    state = InstanceState(capital=capital_state)
    submitted: list[str] = []
    submit = build_action_submitter(
        SubmitterBindings(
            nexus_config=NexusInstanceConfig(
                account_id='bts', venue='binance_spot_simulated',
            ),
            state=state,
            praxis_outbound=cast(PraxisOutbound, outbound),
            validation_pipeline=pipeline,
            strategy_budget=Decimal('100000'),
        ),
        on_submit=submitted.append,
    )
    submit([_sell_close_action()], 'long_on_signal')
    assert len(submitted) == 1
