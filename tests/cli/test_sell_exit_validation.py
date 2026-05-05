"""Slice #38 — SELL exits run through Praxis-equivalent validators.

Each test pins one mechanically-true assertion from the slice spec.
The action_submitter no longer carries a SELL fast-path; ENTER + EXIT
both route through `validation_pipeline.validate`. Nexus's
`pipeline_executor._should_bypass_stage` drops CAPITAL/HEALTH/
PLATFORM_LIMITS for `ValidationAction.EXIT`, so the 6-stage pipeline
runs only INTAKE/RISK/PRICE on EXITs — same as deployed Praxis.
"""
from __future__ import annotations

from decimal import Decimal
from typing import cast

from nexus.core.domain.enums import OrderSide
from nexus.core.domain.instance_state import InstanceState, Position
from nexus.core.domain.order_types import ExecutionMode, OrderType
from nexus.core.validator import ValidationPipeline
from nexus.core.validator.pipeline_models import (
    ValidationAction,
    ValidationDecision,
    ValidationRequestContext,
    ValidationStage,
)
from nexus.infrastructure.praxis_connector.praxis_outbound import PraxisOutbound
from nexus.infrastructure.praxis_connector.trade_outcome import TradeOutcome
from nexus.infrastructure.praxis_connector.trade_outcome_type import (
    TradeOutcomeType,
)
from nexus.instance_config import InstanceConfig as NexusInstanceConfig
from nexus.strategy.action import Action, ActionType

from backtest_simulator.honesty import build_validation_pipeline
from backtest_simulator.launcher.action_submitter import (
    SubmitterBindings,
    _CommandRegistry,
    _SubmittedCommand,
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


def _config() -> NexusInstanceConfig:
    return NexusInstanceConfig(account_id='bts', venue='binance_spot_simulated')


def _pipeline_and_state() -> tuple[
    ValidationPipeline,
    object,  # CapitalController, untyped to keep the import surface small
    InstanceState,
]:
    pipeline, controller, capital_state = build_validation_pipeline(
        nexus_config=_config(), capital_pool=Decimal('100000'),
    )
    state = InstanceState(capital=capital_state)
    return pipeline, controller, state


def _seed_position(
    state: InstanceState, trade_id: str,
    *, size: Decimal = Decimal('0.001'),
    entry_price: Decimal = Decimal('50000'),
) -> None:
    state.positions[trade_id] = Position(
        trade_id=trade_id, strategy_id='long_on_signal',
        symbol='BTCUSDT', side=OrderSide.BUY,
        size=size, entry_price=entry_price,
    )


def _bindings(
    *, outbound: _OutboundStub, pipeline: ValidationPipeline,
    controller: object, state: InstanceState,
    registry: _CommandRegistry | None = None,
) -> SubmitterBindings:
    return SubmitterBindings(
        nexus_config=_config(), state=state,
        praxis_outbound=cast(PraxisOutbound, outbound),
        validation_pipeline=pipeline,
        capital_controller=cast(object, controller),  # type: ignore[arg-type]
        strategy_budget=Decimal('100000'),
        command_registry=registry if registry is not None else _CommandRegistry(),
    )


def _exit_action(trade_id: str | None, *, size: Decimal = Decimal('0.001')) -> Action:
    return Action(
        action_type=ActionType.EXIT,
        direction=OrderSide.SELL,
        size=size,
        execution_mode=ExecutionMode.SINGLE_SHOT,
        order_type=OrderType.MARKET,
        execution_params={'symbol': 'BTCUSDT'},
        deadline=60,
        trade_id=trade_id, command_id=None,
        maker_preference=None, reference_price=Decimal('50000'),
    )


def test_sell_calls_pipeline_validate() -> None:
    """SELL routes through `validation_pipeline.validate` (no fast-path)."""
    outbound = _OutboundStub()
    pipeline, controller, state = _pipeline_and_state()
    _seed_position(state, 'open-1')
    calls: list[ValidationRequestContext] = []
    original_validate = pipeline.validate

    def spy(ctx: ValidationRequestContext) -> ValidationDecision:
        calls.append(ctx)
        return original_validate(ctx)
    pipeline.validate = spy  # type: ignore[method-assign]
    submit = build_action_submitter(_bindings(
        outbound=outbound, pipeline=pipeline, controller=controller, state=state,
    ))
    submit([_exit_action('open-1')], 'long_on_signal')
    assert len(calls) == 1
    assert calls[0].action == ValidationAction.EXIT
    assert calls[0].trade_id == 'open-1'
    assert len(outbound.commands) == 1


def test_sell_with_unknown_trade_id_short_circuits_to_intake_invalid() -> None:
    outbound = _OutboundStub()
    pipeline, controller, state = _pipeline_and_state()
    denied: list[tuple[str, ValidationDecision, Action]] = []

    def on_action_denied(
        cid: str, decision: ValidationDecision, action: Action,
    ) -> None:
        denied.append((cid, decision, action))
    submit = build_action_submitter(
        _bindings(outbound=outbound, pipeline=pipeline, controller=controller, state=state),
        on_action_denied=on_action_denied,
    )
    submit([_exit_action('does-not-exist')], 'long_on_signal')
    assert len(outbound.commands) == 0
    assert len(denied) == 1
    _cid, decision, _action = denied[0]
    assert decision.reason_code == 'INTAKE_TRADE_REFERENCE_INVALID'
    assert decision.failed_stage == ValidationStage.INTAKE


def test_sell_size_exceeds_remaining_denied() -> None:
    outbound = _OutboundStub()
    pipeline, controller, state = _pipeline_and_state()
    _seed_position(state, 'open-1', size=Decimal('0.001'))
    denied: list[tuple[str, ValidationDecision, Action]] = []
    submit = build_action_submitter(
        _bindings(outbound=outbound, pipeline=pipeline, controller=controller, state=state),
        on_action_denied=lambda cid, d, a: denied.append((cid, d, a)),
    )
    # Action size > position remaining — INTAKE denies.
    submit([_exit_action('open-1', size=Decimal('0.01'))], 'long_on_signal')
    assert len(outbound.commands) == 0
    assert len(denied) == 1
    assert denied[0][1].reason_code == 'INTAKE_EXIT_SIZE_EXCEEDS_REMAINING'


def test_sell_runs_risk_stage() -> None:
    outbound = _OutboundStub()
    pipeline, controller, state = _pipeline_and_state()
    _seed_position(state, 'open-1')
    risk_calls: list[ValidationRequestContext] = []
    original = pipeline._validators[ValidationStage.RISK]  # type: ignore[attr-defined]

    def spy(ctx: ValidationRequestContext) -> ValidationDecision:
        risk_calls.append(ctx)
        return original(ctx)
    pipeline._validators[ValidationStage.RISK] = spy  # type: ignore[attr-defined]
    submit = build_action_submitter(_bindings(
        outbound=outbound, pipeline=pipeline, controller=controller, state=state,
    ))
    submit([_exit_action('open-1')], 'long_on_signal')
    assert len(risk_calls) == 1
    assert risk_calls[0].action == ValidationAction.EXIT


def test_sell_runs_price_stage() -> None:
    outbound = _OutboundStub()
    pipeline, controller, state = _pipeline_and_state()
    _seed_position(state, 'open-1')
    price_calls: list[ValidationRequestContext] = []
    original = pipeline._validators[ValidationStage.PRICE]  # type: ignore[attr-defined]

    def spy(ctx: ValidationRequestContext) -> ValidationDecision:
        price_calls.append(ctx)
        return original(ctx)
    pipeline._validators[ValidationStage.PRICE] = spy  # type: ignore[attr-defined]
    submit = build_action_submitter(_bindings(
        outbound=outbound, pipeline=pipeline, controller=controller, state=state,
    ))
    submit([_exit_action('open-1')], 'long_on_signal')
    assert len(price_calls) == 1
    assert price_calls[0].action == ValidationAction.EXIT


def test_sell_skips_capital_stage() -> None:
    """Pipeline executor bypasses CAPITAL on EXIT — `check_and_reserve` not called."""
    outbound = _OutboundStub()
    pipeline, controller, state = _pipeline_and_state()
    _seed_position(state, 'open-1')
    capital_calls: list[ValidationRequestContext] = []
    original = pipeline._validators[ValidationStage.CAPITAL]  # type: ignore[attr-defined]

    def spy(ctx: ValidationRequestContext) -> ValidationDecision:
        capital_calls.append(ctx)
        return original(ctx)
    pipeline._validators[ValidationStage.CAPITAL] = spy  # type: ignore[attr-defined]
    submit = build_action_submitter(_bindings(
        outbound=outbound, pipeline=pipeline, controller=controller, state=state,
    ))
    submit([_exit_action('open-1')], 'long_on_signal')
    assert capital_calls == []


def test_sell_skips_health_stage() -> None:
    outbound = _OutboundStub()
    pipeline, controller, state = _pipeline_and_state()
    _seed_position(state, 'open-1')
    health_calls: list[ValidationRequestContext] = []
    original = pipeline._validators[ValidationStage.HEALTH]  # type: ignore[attr-defined]

    def spy(ctx: ValidationRequestContext) -> ValidationDecision:
        health_calls.append(ctx)
        return original(ctx)
    pipeline._validators[ValidationStage.HEALTH] = spy  # type: ignore[attr-defined]
    submit = build_action_submitter(_bindings(
        outbound=outbound, pipeline=pipeline, controller=controller, state=state,
    ))
    submit([_exit_action('open-1')], 'long_on_signal')
    assert health_calls == []


def test_sell_skips_platform_limits_stage() -> None:
    outbound = _OutboundStub()
    pipeline, controller, state = _pipeline_and_state()
    _seed_position(state, 'open-1')
    platform_calls: list[ValidationRequestContext] = []
    original = pipeline._validators[ValidationStage.PLATFORM_LIMITS]  # type: ignore[attr-defined]

    def spy(ctx: ValidationRequestContext) -> ValidationDecision:
        platform_calls.append(ctx)
        return original(ctx)
    pipeline._validators[ValidationStage.PLATFORM_LIMITS] = spy  # type: ignore[attr-defined]
    submit = build_action_submitter(_bindings(
        outbound=outbound, pipeline=pipeline, controller=controller, state=state,
    ))
    submit([_exit_action('open-1')], 'long_on_signal')
    assert platform_calls == []


def test_validation_denied_sell_dispatches_synthetic_rejected_outcome() -> None:
    """Denied EXIT triggers `on_action_denied` so the launcher can dispatch."""
    outbound = _OutboundStub()
    pipeline, controller, state = _pipeline_and_state()
    denied: list[tuple[str, ValidationDecision, Action]] = []
    submit = build_action_submitter(
        _bindings(outbound=outbound, pipeline=pipeline, controller=controller, state=state),
        on_action_denied=lambda cid, d, a: denied.append((cid, d, a)),
    )
    submit([_exit_action('not-found')], 'long_on_signal')
    assert len(denied) == 1
    cid, decision, action = denied[0]
    assert isinstance(cid, str) and cid
    assert decision.reason_code == 'INTAKE_TRADE_REFERENCE_INVALID'
    assert action.action_type == ActionType.EXIT


def test_strategy_clears_pending_sell_on_synthetic_rejected_outcome() -> None:
    """Strategy template recognises VALIDATION-prefixed REJECTED and clears _pending_sell."""
    # Manually instantiate the strategy class with `_BAKED_CONFIG`
    # already substituted via direct attribute assignment so the
    # template module's literal placeholder doesn't have to be
    # patched. The on_outcome path under test only reads
    # `_pending_sell` and `outcome.*` so a partial setup is enough.
    import importlib
    from datetime import UTC, datetime
    template = importlib.import_module(
        'backtest_simulator.pipeline._strategy_templates.long_on_signal',
    )
    # `Strategy` parses `_BAKED_CONFIG` at module load. The placeholder
    # `__BTS_PARAMS__` was already replaced or it would not import.
    # Substitute config inline by manually constructing a Strategy and
    # patching the `_config` field that on_signal reads.
    strat = template.Strategy.__new__(template.Strategy)
    strat._long = True
    strat._entry_qty = Decimal('0.001')
    strat._pending_buy = False
    strat._pending_sell = True
    strat._open_trade_id = 'open-1'
    outcome = TradeOutcome(
        outcome_id='synthetic-1', command_id='bts-cmd-x',
        outcome_type=TradeOutcomeType.REJECTED,
        timestamp=datetime.now(UTC),
        reject_reason='VALIDATION:INTAKE_EXIT_SIZE_EXCEEDS_REMAINING: too big',
    )
    from nexus.strategy.context import StrategyContext
    from nexus.strategy.params import StrategyParams
    ctx = StrategyContext(
        positions=(), capital_available=Decimal('0'),
        operational_mode=__import__(
            'nexus.core.domain.enums', fromlist=['OperationalMode'],
        ).OperationalMode.ACTIVE,
    )
    actions = strat.on_outcome(outcome, StrategyParams(raw={}), ctx)
    assert actions == []
    assert strat._pending_sell is False
    # _open_trade_id preserved so the next preds=0 can re-emit.
    assert strat._open_trade_id == 'open-1'
    # _long preserved (the position was never closed).
    assert strat._long is True


def test_pending_exit_cleared_on_full_fill() -> None:
    """After a full SELL fill, `state.positions[trade_id]` is popped."""
    from praxis.core.domain.enums import OrderStatus
    from praxis.infrastructure.venue_adapter import SubmitResult

    from backtest_simulator.honesty import CapitalLifecycleTracker
    from backtest_simulator.launcher.launcher import (
        _finalize_sell_close,
        _LifecycleContext,
    )
    _pipeline, controller, state = _pipeline_and_state()
    _seed_position(state, 'open-1', size=Decimal('0.001'))
    # Deploy capital so the close has something to invert without
    # tripping the conservation invariant (position_notional < 0).
    state.capital.position_notional = Decimal('50.05')
    state.capital.per_strategy_deployed['long_on_signal'] = Decimal('50.05')
    tracker = CapitalLifecycleTracker(controller)
    tracker.record_open_position(
        command_id='open-1', strategy_id='long_on_signal',
        cost_basis=Decimal('50'), entry_fees=Decimal('0.05'),
        entry_qty=Decimal('0.001'),
    )
    registry = _CommandRegistry()
    registry.register('sell-cmd-1', _SubmittedCommand(
        action_type=ActionType.EXIT, trade_id='open-1',
        strategy_id='long_on_signal', symbol='BTCUSDT',
        order_size=Decimal('0.001'),
        reservation_id=None, declared_stop_price=None,
    ))
    ctx = _LifecycleContext(
        tracker=tracker, capital_state=state.capital,
        initial_pool=Decimal('100000'),
        state=state, command_registry=registry,
    )

    from praxis.infrastructure.venue_adapter import ImmediateFill
    result = SubmitResult(
        venue_order_id='vo-1', status=OrderStatus.FILLED,
        immediate_fills=(ImmediateFill(
            venue_trade_id='t-1', qty=Decimal('0.001'),
            price=Decimal('51000'), fee=Decimal('0.05'),
            fee_asset='USDT', is_maker=False,
        ),),
    )
    _finalize_sell_close(ctx, result, 'sell-cmd-1')
    assert 'open-1' not in state.positions


def test_pending_exit_cleared_on_partial_terminal_sell() -> None:
    from praxis.core.domain.enums import OrderStatus
    from praxis.infrastructure.venue_adapter import SubmitResult

    from backtest_simulator.honesty import CapitalLifecycleTracker
    from backtest_simulator.launcher.launcher import (
        _finalize_sell_close,
        _LifecycleContext,
    )
    _pipeline, controller, state = _pipeline_and_state()
    _seed_position(state, 'open-1', size=Decimal('0.002'))
    state.positions['open-1'].pending_exit = Decimal('0.002')
    state.capital.position_notional = Decimal('100.1')
    state.capital.per_strategy_deployed['long_on_signal'] = Decimal('100.1')
    tracker = CapitalLifecycleTracker(controller)
    tracker.record_open_position(
        command_id='open-1', strategy_id='long_on_signal',
        cost_basis=Decimal('100'), entry_fees=Decimal('0.1'),
        entry_qty=Decimal('0.002'),
    )
    registry = _CommandRegistry()
    registry.register('sell-cmd-1', _SubmittedCommand(
        action_type=ActionType.EXIT, trade_id='open-1',
        strategy_id='long_on_signal', symbol='BTCUSDT',
        order_size=Decimal('0.002'),
        reservation_id=None, declared_stop_price=None,
    ))
    ctx = _LifecycleContext(
        tracker=tracker, capital_state=state.capital,
        initial_pool=Decimal('100000'),
        state=state, command_registry=registry,
    )

    from praxis.infrastructure.venue_adapter import ImmediateFill
    # Partial-terminal: only 0.001 of 0.002 filled; remainder is dead.
    result = SubmitResult(
        venue_order_id='vo-1', status=OrderStatus.PARTIALLY_FILLED,
        immediate_fills=(ImmediateFill(
            venue_trade_id='t-1', qty=Decimal('0.001'),
            price=Decimal('51000'), fee=Decimal('0.05'),
            fee_asset='USDT', is_maker=False,
        ),),
    )
    _finalize_sell_close(ctx, result, 'sell-cmd-1')
    assert 'open-1' in state.positions
    assert state.positions['open-1'].size == Decimal('0.001')
    assert state.positions['open-1'].pending_exit == Decimal('0')


def test_pending_exit_cleared_on_rejected_sell() -> None:
    from backtest_simulator.launcher.action_submitter import (
        release_sell_pending_exit,
    )
    _pipeline, _controller, state = _pipeline_and_state()
    _seed_position(state, 'open-1', size=Decimal('0.001'))
    state.positions['open-1'].pending_exit = Decimal('0.001')
    registry = _CommandRegistry()
    registry.register('sell-cmd-1', _SubmittedCommand(
        action_type=ActionType.EXIT, trade_id='open-1',
        strategy_id='long_on_signal', symbol='BTCUSDT',
        order_size=Decimal('0.001'),
        reservation_id=None, declared_stop_price=None,
    ))
    release_sell_pending_exit(state, registry, 'sell-cmd-1')
    assert state.positions['open-1'].pending_exit == Decimal('0')
    # size unchanged because no fill landed
    assert state.positions['open-1'].size == Decimal('0.001')


def test_pending_exit_cleared_on_expired_sell() -> None:
    # Same code path as rejected: terminal-no-fill releases pending_exit.
    from backtest_simulator.launcher.action_submitter import (
        release_sell_pending_exit,
    )
    _pipeline, _controller, state = _pipeline_and_state()
    _seed_position(state, 'open-1', size=Decimal('0.001'))
    state.positions['open-1'].pending_exit = Decimal('0.001')
    registry = _CommandRegistry()
    registry.register('sell-cmd-1', _SubmittedCommand(
        action_type=ActionType.EXIT, trade_id='open-1',
        strategy_id='long_on_signal', symbol='BTCUSDT',
        order_size=Decimal('0.001'),
        reservation_id=None, declared_stop_price=None,
    ))
    release_sell_pending_exit(state, registry, 'sell-cmd-1')
    assert state.positions['open-1'].pending_exit == Decimal('0')
    assert state.positions['open-1'].size == Decimal('0.001')


def test_buy_fill_populates_state_positions() -> None:
    """The launcher's adapter wrapper writes `state.positions[command_id]` on BUY fill."""
    from praxis.core.domain.enums import OrderStatus
    from praxis.infrastructure.venue_adapter import ImmediateFill, SubmitResult

    from backtest_simulator.honesty import CapitalLifecycleTracker
    from backtest_simulator.launcher.launcher import (
        _finalize_successful_fill,
        _LifecycleContext,
    )
    _pipeline, controller, state = _pipeline_and_state()
    tracker = CapitalLifecycleTracker(controller)
    # Drive a real reservation through the controller so the
    # subsequent `send_order`/`order_ack`/`order_fill` path doesn't
    # trip the controller's "reservation not found" miss.
    reserve_result = controller.check_and_reserve(
        strategy_id='long_on_signal',
        order_notional=Decimal('100'),
        estimated_fees=Decimal('0.1'),
        strategy_budget=Decimal('100000'),
    )
    assert reserve_result.granted and reserve_result.reservation is not None
    tracker.record_reservation(
        command_id='cmd-buy-1',
        reservation_id=reserve_result.reservation.reservation_id,
        strategy_id='long_on_signal',
        notional=Decimal('100'),
        estimated_fees=Decimal('0.1'),
    )
    registry = _CommandRegistry()
    registry.register('cmd-buy-1', _SubmittedCommand(
        action_type=ActionType.ENTER, trade_id='cmd-buy-1',
        strategy_id='long_on_signal', symbol='BTCUSDT',
        order_size=Decimal('0.001'),
        reservation_id=reserve_result.reservation.reservation_id,
        declared_stop_price=None,
    ))
    ctx = _LifecycleContext(
        tracker=tracker, capital_state=state.capital,
        initial_pool=state.capital.capital_pool,
        state=state, command_registry=registry,
    )
    result = SubmitResult(
        venue_order_id='vo-1', status=OrderStatus.FILLED,
        immediate_fills=(ImmediateFill(
            venue_trade_id='t-1', qty=Decimal('0.001'),
            price=Decimal('50000'), fee=Decimal('0.05'),
            fee_asset='USDT', is_maker=False,
        ),),
    )
    _finalize_successful_fill(ctx, 'cmd-buy-1', result, 'vo-1', 'FILLED')
    assert 'cmd-buy-1' in state.positions
    assert state.positions['cmd-buy-1'].size == Decimal('0.001')
    assert state.positions['cmd-buy-1'].entry_price == Decimal('50000')
    assert state.positions['cmd-buy-1'].side == OrderSide.BUY


def test_strategy_captures_command_id_at_buy_fill_uses_for_sell() -> None:
    """Strategy reads `outcome.command_id` at BUY fill, emits SELL with it as trade_id."""
    import importlib
    from datetime import UTC, datetime
    template = importlib.import_module(
        'backtest_simulator.pipeline._strategy_templates.long_on_signal',
    )
    from nexus.core.domain.enums import OperationalMode
    from nexus.strategy.context import StrategyContext
    from nexus.strategy.params import StrategyParams
    from nexus.strategy.signal import Signal
    strat = template.Strategy.__new__(template.Strategy)
    strat._config = template._Config(
        symbol='BTCUSDT', capital=Decimal('1000'),
        kelly_pct=Decimal('5'), estimated_price=Decimal('50000'),
        stop_bps=Decimal('50'),
        force_flatten_after=None,
    )
    strat._long = False
    strat._entry_qty = Decimal('0')
    strat._pending_buy = True
    strat._pending_sell = False
    strat._open_trade_id = None
    strat._must_close_outstanding = False
    # Simulate BUY fill with a known praxis command_id.
    fill_outcome = TradeOutcome(
        outcome_id='outc-1', command_id='praxis-buy-cmd-id',
        outcome_type=TradeOutcomeType.FILLED,
        timestamp=datetime.now(UTC),
        fill_size=Decimal('0.001'),
        fill_price=Decimal('50000'),
        fill_notional=Decimal('50'),
        actual_fees=Decimal('0'),
    )
    ctx = StrategyContext(
        positions=(), capital_available=Decimal('0'),
        operational_mode=OperationalMode.ACTIVE,
    )
    strat.on_outcome(fill_outcome, StrategyParams(raw={}), ctx)
    assert strat._open_trade_id == 'praxis-buy-cmd-id'
    # Now simulate the next preds=0 — strategy emits a SELL with the
    # captured trade_id stamped on the action.
    sell_actions = strat.on_signal(
        Signal(
            predictor_fn_id='preds',
            values={'_preds': 0, '_probs': 0.4},
            timestamp=datetime.now(UTC),
        ),
        StrategyParams(raw={}), ctx,
    )
    assert len(sell_actions) == 1
    sell = sell_actions[0]
    assert sell.action_type == ActionType.EXIT
    assert sell.direction == OrderSide.SELL
    assert sell.trade_id == 'praxis-buy-cmd-id'


def test_command_registry_written_before_send_command() -> None:
    """Registry has the entry at the time `send_command` is called.

    Closes the codex round 3 P0 race: the prior `on_reservation` hook
    fired AFTER `send_command` returned, leaving a window for the
    account_loop callback to dispatch first. Registering BEFORE
    `send_command` (under the registry's own lock) guarantees the
    write-ahead invariant.
    """
    seen_entries: list[_SubmittedCommand | None] = []
    pipeline, controller, state = _pipeline_and_state()
    _seed_position(state, 'open-1')
    registry = _CommandRegistry()

    class _Spying(_OutboundStub):
        def send_command(self, cmd: object) -> str:
            cmd_id = getattr(cmd, 'command_id', None)
            if isinstance(cmd_id, str):
                seen_entries.append(registry.get(cmd_id))
            return super().send_command(cmd)
    outbound = _Spying()
    submit = build_action_submitter(_bindings(
        outbound=outbound, pipeline=pipeline, controller=controller, state=state,
        registry=registry,
    ))
    submit([_exit_action('open-1')], 'long_on_signal')
    assert len(seen_entries) == 1
    entry = seen_entries[0]
    assert entry is not None, (
        'registry was empty when send_command fired — registration '
        'happened AFTER send_command instead of before; race is open.'
    )
    assert entry.action_type == ActionType.EXIT
    assert entry.trade_id == 'open-1'
