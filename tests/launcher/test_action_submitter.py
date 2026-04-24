"""ActionSubmitter contract: build_action_submitter returns a correct Callable."""
from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from nexus.core.domain.capital_state import CapitalState
from nexus.core.domain.enums import OrderSide
from nexus.core.domain.order_types import ExecutionMode, OrderType
from nexus.core.validator.pipeline_models import InstanceState
from nexus.instance_config import InstanceConfig as NexusInstanceConfig
from nexus.strategy.action import Action, ActionType

from backtest_simulator.launcher.action_submitter import build_action_submitter


class _OutboundStub:
    """Minimal stand-in for PraxisOutbound that records send_command calls."""

    def __init__(self) -> None:
        self.commands: list[Any] = []
        self.aborts: list[dict[str, Any]] = []

    def send_command(self, cmd: Any) -> str:
        self.commands.append(cmd)
        return cmd.command_id

    def send_abort(self, **kwargs: Any) -> None:
        self.aborts.append(kwargs)


def _nexus_config() -> NexusInstanceConfig:
    return NexusInstanceConfig(account_id='bts-acct', venue='binance_spot_simulated')


def _instance_state() -> InstanceState:
    return InstanceState(capital=CapitalState(capital_pool=Decimal('100000')))


def _enter_action() -> Action:
    return Action(
        action_type=ActionType.ENTER,
        direction=OrderSide.BUY,
        size=Decimal('0.001'),
        execution_mode=ExecutionMode.SINGLE_SHOT,
        order_type=OrderType.MARKET,
        execution_params={'symbol': 'BTCUSDT'},
        deadline=1_700_000_000_000,
        trade_id=None,
        command_id=None,
        maker_preference=None,
        reference_price=Decimal('50000'),
    )


def test_builder_returns_callable() -> None:
    outbound = _OutboundStub()
    submit = build_action_submitter(
        nexus_config=_nexus_config(), state=_instance_state(),
        praxis_outbound=outbound,  # type: ignore[arg-type]
    )
    assert callable(submit)


def test_enter_action_gets_translated_and_sent() -> None:
    outbound = _OutboundStub()
    submit = build_action_submitter(
        nexus_config=_nexus_config(), state=_instance_state(),
        praxis_outbound=outbound,  # type: ignore[arg-type]
    )
    submit([_enter_action()], 'long_on_signal')
    assert len(outbound.commands) == 1
    cmd = outbound.commands[0]
    assert cmd.account_id == 'bts-acct'
    assert cmd.symbol == 'BTCUSDT'
    assert cmd.side == OrderSide.BUY
    assert cmd.size == Decimal('0.001')


def test_abort_action_routed_to_send_abort() -> None:
    outbound = _OutboundStub()
    submit = build_action_submitter(
        nexus_config=_nexus_config(), state=_instance_state(),
        praxis_outbound=outbound,  # type: ignore[arg-type]
    )
    abort = Action(
        action_type=ActionType.ABORT,
        direction=None, size=None, execution_mode=None, order_type=None,
        execution_params=None, deadline=None,
        trade_id=None, command_id='cmd-to-abort',
        maker_preference=None, reference_price=None,
    )
    submit([abort], 'long_on_signal')
    assert len(outbound.commands) == 0
    assert len(outbound.aborts) == 1
    assert outbound.aborts[0]['command_id'] == 'cmd-to-abort'
    assert outbound.aborts[0]['account_id'] == 'bts-acct'


def test_submission_propagates_per_action_errors(monkeypatch: Any) -> None:
    # The backtest action submitter does NOT swallow per-action errors.
    # A failure in `send_command` must surface to `PredictLoop._tick`'s
    # own logging/abort path — silently dropping ENTER/SELL actions
    # would hide bugs and produce a misleading trade summary. The
    # fail-loud contract matches the CLAUDE.md law 4 stance: no silent
    # handlers in the action-submit path.
    class _RaisingOutbound(_OutboundStub):
        def send_command(self, cmd: Any) -> str:
            raise RuntimeError('injected')
    outbound = _RaisingOutbound()
    submit = build_action_submitter(
        nexus_config=_nexus_config(), state=_instance_state(),
        praxis_outbound=outbound,  # type: ignore[arg-type]
    )
    with pytest.raises(RuntimeError, match='injected'):
        submit([_enter_action(), _enter_action()], 'long_on_signal')
    assert len(outbound.commands) == 0


def test_sell_side_passed_through() -> None:
    outbound = _OutboundStub()
    submit = build_action_submitter(
        nexus_config=_nexus_config(), state=_instance_state(),
        praxis_outbound=outbound,  # type: ignore[arg-type]
    )
    sell_action = Action(
        action_type=ActionType.ENTER,
        direction=OrderSide.SELL,
        size=Decimal('0.001'),
        execution_mode=ExecutionMode.SINGLE_SHOT,
        order_type=OrderType.MARKET,
        execution_params={'symbol': 'BTCUSDT'},
        deadline=1_700_000_000_000,
        trade_id=None, command_id=None,
        maker_preference=None, reference_price=Decimal('50000'),
    )
    submit([sell_action], 'long_on_signal')
    assert outbound.commands[0].side == OrderSide.SELL


def test_trade_id_and_command_id_generated_when_missing() -> None:
    outbound = _OutboundStub()
    submit = build_action_submitter(
        nexus_config=_nexus_config(), state=_instance_state(),
        praxis_outbound=outbound,  # type: ignore[arg-type]
    )
    submit([_enter_action()], 'long_on_signal')
    cmd = outbound.commands[0]
    assert cmd.trade_id is not None and cmd.trade_id.startswith('bts-')
    assert cmd.command_id is not None and cmd.command_id.startswith('bts-cmd-')


def test_multiple_actions_processed_in_order() -> None:
    outbound = _OutboundStub()
    submit = build_action_submitter(
        nexus_config=_nexus_config(), state=_instance_state(),
        praxis_outbound=outbound,  # type: ignore[arg-type]
    )
    submit([_enter_action(), _enter_action(), _enter_action()], 'long_on_signal')
    assert len(outbound.commands) == 3


def test_symbol_falls_back_to_btcusdt_when_missing() -> None:
    outbound = _OutboundStub()
    submit = build_action_submitter(
        nexus_config=_nexus_config(), state=_instance_state(),
        praxis_outbound=outbound,  # type: ignore[arg-type]
    )
    action_no_symbol = Action(
        action_type=ActionType.ENTER,
        direction=OrderSide.BUY,
        size=Decimal('0.001'),
        execution_mode=ExecutionMode.SINGLE_SHOT,
        order_type=OrderType.MARKET,
        execution_params={},
        deadline=1_700_000_000_000,
        trade_id=None, command_id=None,
        maker_preference=None,
        reference_price=Decimal('50000'),
    )
    submit([action_no_symbol], 'long_on_signal')
    assert outbound.commands[0].symbol == 'BTCUSDT'
