"""ActionSubmitter contract — Part 2.

The submitter runs the real ValidationPipeline, carries the declared
stop on the side, and propagates fail-loud on per-action errors.
"""
from __future__ import annotations

from decimal import Decimal
from typing import cast

import pytest
from nexus.core.capital_controller.capital_controller import CapitalController
from nexus.core.domain.enums import OrderSide
from nexus.core.domain.order_types import ExecutionMode, OrderType
from nexus.core.validator import ValidationPipeline
from nexus.core.validator.pipeline_models import (
    InstanceState,
    ValidationDecision,
)
from nexus.infrastructure.praxis_connector.praxis_outbound import PraxisOutbound
from nexus.instance_config import InstanceConfig as NexusInstanceConfig
from nexus.strategy.action import Action, ActionType

from backtest_simulator.honesty import build_validation_pipeline
from backtest_simulator.launcher.action_submitter import (
    SubmitterBindings,
    build_action_submitter,
)


class _OutboundStub:
    """Minimal stand-in for PraxisOutbound that records send_command calls."""

    def __init__(self) -> None:
        self.commands: list[object] = []
        self.aborts: list[dict[str, object]] = []

    def send_command(self, cmd: object) -> str:
        self.commands.append(cmd)
        # Generate a command_id — Praxis does this server-side but the
        # outbound stub needs to hand one back.
        import uuid
        return str(uuid.uuid4())

    def send_abort(self, **kwargs: object) -> None:
        self.aborts.append(kwargs)


def _nexus_config() -> NexusInstanceConfig:
    return NexusInstanceConfig(account_id='bts-acct', venue='binance_spot_simulated')


def _make_bindings(
    outbound: _OutboundStub,
    pipeline: ValidationPipeline,
    controller: CapitalController,
    state: InstanceState,
) -> SubmitterBindings:
    return SubmitterBindings(
        nexus_config=_nexus_config(), state=state,
        praxis_outbound=cast(PraxisOutbound, outbound),
        validation_pipeline=pipeline,
        capital_controller=controller,
        strategy_budget=Decimal('100000'),
    )


def _pipeline_and_state() -> tuple[ValidationPipeline, CapitalController, InstanceState]:
    # Build a Part 2-shaped pipeline that shares the same CapitalState
    # the InstanceState carries — that's the invariant the real
    # launcher enforces and what the CAPITAL validator reads from.
    pipeline, controller, capital_state = build_validation_pipeline(
        nexus_config=NexusInstanceConfig(account_id='bts-test', venue='binance_spot_simulated'),
        capital_pool=Decimal('100000'),
    )
    state = InstanceState(capital=capital_state)
    return pipeline, controller, state


def _enter_action() -> Action:
    """BUY ENTER with a declared stop — the Part 2 INTAKE hook accepts it."""
    return Action(
        action_type=ActionType.ENTER,
        direction=OrderSide.BUY,
        size=Decimal('0.001'),
        execution_mode=ExecutionMode.SINGLE_SHOT,
        order_type=OrderType.MARKET,
        execution_params={
            'symbol': 'BTCUSDT',
            'stop_bps': '50',
            'stop_price': '49750',
        },
        deadline=60,
        trade_id=None,
        command_id=None,
        maker_preference=None,
        reference_price=Decimal('50000'),
    )


def _sell_action() -> Action:
    """SELL exit — no stop_price, per Part 2 long-only convention."""
    return Action(
        action_type=ActionType.ENTER,
        direction=OrderSide.SELL,
        size=Decimal('0.001'),
        execution_mode=ExecutionMode.SINGLE_SHOT,
        order_type=OrderType.MARKET,
        execution_params={'symbol': 'BTCUSDT'},
        deadline=60,
        trade_id=None,
        command_id=None,
        maker_preference=None,
        reference_price=Decimal('50000'),
    )


def test_builder_returns_callable() -> None:
    outbound = _OutboundStub()
    pipeline, controller, state = _pipeline_and_state()
    submit = build_action_submitter(_make_bindings(outbound, pipeline, controller, state))
    assert callable(submit)


def test_enter_with_declared_stop_passes_validation_and_sends() -> None:
    outbound = _OutboundStub()
    pipeline, controller, state = _pipeline_and_state()
    submit = build_action_submitter(_make_bindings(outbound, pipeline, controller, state))
    submit([_enter_action()], 'long_on_signal')
    assert len(outbound.commands) == 1
    cmd = outbound.commands[0]
    assert cmd.account_id == 'bts-acct'
    assert cmd.symbol == 'BTCUSDT'
    # `_to_praxis_enums` converts the cmd's side to Praxis OrderSide,
    # which is a different class than Nexus OrderSide. Compare by name.
    assert cmd.side.name == OrderSide.BUY.name
    assert cmd.size == Decimal('0.001')


def test_buy_entry_without_declared_stop_is_rejected_by_intake() -> None:
    # Part 2 INTAKE gate — no stop, no entry.
    outbound = _OutboundStub()
    pipeline, controller, state = _pipeline_and_state()
    submit = build_action_submitter(_make_bindings(outbound, pipeline, controller, state))
    action_no_stop = Action(
        action_type=ActionType.ENTER,
        direction=OrderSide.BUY,
        size=Decimal('0.001'),
        execution_mode=ExecutionMode.SINGLE_SHOT,
        order_type=OrderType.MARKET,
        execution_params={'symbol': 'BTCUSDT'},
        deadline=60,
        trade_id=None, command_id=None,
        maker_preference=None, reference_price=Decimal('50000'),
    )
    submit([action_no_stop], 'long_on_signal')
    # INTAKE denies → no command reaches Praxis.
    assert len(outbound.commands) == 0


def test_sell_exit_without_stop_is_accepted() -> None:
    # Long-only convention — SELL closes an existing long and is itself
    # the risk close, so no stop_price is required.
    outbound = _OutboundStub()
    pipeline, controller, state = _pipeline_and_state()
    submit = build_action_submitter(_make_bindings(outbound, pipeline, controller, state))
    submit([_sell_action()], 'long_on_signal')
    assert len(outbound.commands) == 1
    # `_to_praxis_enums` converts side to Praxis OrderSide; compare by name.
    assert outbound.commands[0].side.name == OrderSide.SELL.name


def test_abort_action_routed_to_send_abort() -> None:
    outbound = _OutboundStub()
    pipeline, controller, state = _pipeline_and_state()
    submit = build_action_submitter(_make_bindings(outbound, pipeline, controller, state))
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


def test_submission_propagates_per_action_errors() -> None:
    # Fail-loud contract: a RuntimeError from send_command must NOT
    # be swallowed. The PredictLoop's own `action_submit raised`
    # handler catches at a higher level; silencing here would hide
    # bugs.
    class _RaisingOutbound(_OutboundStub):
        def send_command(self, cmd: object) -> str:
            raise RuntimeError('injected')
    outbound = _RaisingOutbound()
    pipeline, controller, state = _pipeline_and_state()
    submit = build_action_submitter(_make_bindings(outbound, pipeline, controller, state))
    with pytest.raises(RuntimeError, match='injected'):
        submit([_enter_action()], 'long_on_signal')
    # No command got through; the capture happens inside the outbound
    # stub only when send_command returns successfully (which it doesn't).
    assert len(outbound.commands) == 0


def test_on_reservation_hook_receives_decision_with_reservation() -> None:
    # The launcher's CapitalLifecycleTracker feeds off this hook — it's
    # how the `check_and_reserve → send_order → order_ack → order_fill`
    # chain ties command_id to its reservation. Part 2 requires the
    # decision to carry a `reservation` so downstream send_order/ack
    # calls can reference it.
    outbound = _OutboundStub()
    pipeline, controller, state = _pipeline_and_state()
    captured: list[tuple[str, object, object, Action]] = []

    def on_reservation(
        command_id: str, decision: object, context: object, action: Action,
    ) -> None:
        captured.append((command_id, decision, context, action))

    submit = build_action_submitter(
        _make_bindings(outbound, pipeline, controller, state),
        on_reservation=on_reservation,
    )
    submit([_enter_action()], 'long_on_signal')
    assert len(captured) == 1
    _command_id, decision, context, action = captured[0]
    assert decision.allowed is True
    assert decision.reservation is not None
    assert decision.reservation.notional > 0
    assert context.command_id is not None
    assert action.action_type == ActionType.ENTER


def test_on_submit_hook_fires_with_command_id() -> None:
    # The `on_submit` hook is how the launcher bumps its
    # `submitted_commands` counter for the synchronous drain. It
    # must fire exactly once per action that reached Praxis.
    outbound = _OutboundStub()
    pipeline, controller, state = _pipeline_and_state()
    submitted: list[str] = []
    submit = build_action_submitter(
        _make_bindings(outbound, pipeline, controller, state),
        on_submit=submitted.append,
    )
    submit([_enter_action(), _sell_action()], 'long_on_signal')
    assert len(submitted) == 2
    assert all(isinstance(cmd_id, str) for cmd_id in submitted)


def test_maybe_refresh_limit_to_touch_buy_biases_below() -> None:
    """LIMIT BUY action's price gets rewritten to `touch - tick`
    BEFORE the cmd reaches Praxis (codex round 4 P2).

    Mutation proof: capture the cmd that landed at
    `praxis_outbound.send_command`. If `_maybe_refresh_limit_to_touch`
    regresses, the cmd carries the strategy-emitted `price=10000`
    instead of the touch-biased `49999.99`.
    """
    outbound = _OutboundStub()
    pipeline, controller, state = _pipeline_and_state()
    bindings = SubmitterBindings(
        nexus_config=_nexus_config(), state=state,
        praxis_outbound=cast(PraxisOutbound, outbound),
        validation_pipeline=pipeline,
        capital_controller=controller,
        strategy_budget=Decimal('100000'),
        touch_provider=lambda _sym: Decimal('50000'),
        tick_provider=lambda _sym: Decimal('0.01'),
    )
    submit = build_action_submitter(bindings)
    limit_action = Action(
        action_type=ActionType.ENTER,
        direction=OrderSide.BUY,
        size=Decimal('0.001'),
        execution_mode=ExecutionMode.SINGLE_SHOT,
        order_type=OrderType.LIMIT,
        execution_params={
            'symbol': 'BTCUSDT',
            'price': '10000',  # stale, far from market
            'stop_price': '9000',  # declared protective stop
        },
        deadline=60,
        trade_id=None, command_id=None,
        maker_preference=None,
        reference_price=Decimal('10000'),
    )
    submit([limit_action], 'long_on_signal')
    assert len(outbound.commands) == 1, (
        f'expected exactly one cmd to reach send_command, got '
        f'{len(outbound.commands)} — validation may have rejected.'
    )
    cmd = outbound.commands[0]
    # The action's stale price (10000) must NOT survive into the
    # Praxis SingleShotParams. The touch-refreshed price 49999.99
    # (50000 - 0.01) must.
    assert str(cmd.execution_params.price) == '49999.99', (
        f'expected SingleShotParams.price=49999.99 (touch - tick), got '
        f'{cmd.execution_params.price} — touch refresh did not run '
        f'before validation, or wrong bias direction.'
    )


def test_maybe_refresh_limit_to_touch_sell_biases_above() -> None:
    """LIMIT SELL action's price gets rewritten to `touch + tick`."""
    outbound = _OutboundStub()
    pipeline, controller, state = _pipeline_and_state()
    bindings = SubmitterBindings(
        nexus_config=_nexus_config(), state=state,
        praxis_outbound=cast(PraxisOutbound, outbound),
        validation_pipeline=pipeline,
        capital_controller=controller,
        strategy_budget=Decimal('100000'),
        touch_provider=lambda _sym: Decimal('50000'),
        tick_provider=lambda _sym: Decimal('0.01'),
    )
    submit = build_action_submitter(bindings)
    sell_limit = Action(
        action_type=ActionType.ENTER,
        direction=OrderSide.SELL,
        size=Decimal('0.001'),
        execution_mode=ExecutionMode.SINGLE_SHOT,
        order_type=OrderType.LIMIT,
        execution_params={'symbol': 'BTCUSDT', 'price': '10000'},
        deadline=60,
        trade_id=None, command_id=None,
        maker_preference=None,
        reference_price=Decimal('10000'),
    )
    submit([sell_limit], 'long_on_signal')
    assert len(outbound.commands) == 1
    cmd = outbound.commands[0]
    assert str(cmd.execution_params.price) == '50000.01', (
        f'expected SingleShotParams.price=50000.01 (touch + tick) '
        f'for SELL maker post, got {cmd.execution_params.price}.'
    )


def _atr_bindings(
    outbound: _OutboundStub,
    pipeline: ValidationPipeline,
    controller: CapitalController,
    state: InstanceState,
    *,
    atr: Decimal | None = Decimal('300'),
    k: Decimal = Decimal('0.5'),
) -> SubmitterBindings:
    """SubmitterBindings with ATR gate wired (slice #17 Task 29).

    `atr` is the constant the stub provider returns for every
    (symbol, t). `atr=None` simulates an empty pre-decision tape
    so the gate's `ATR_UNCALIBRATED` path fires.
    """
    from backtest_simulator.honesty.atr import AtrSanityGate
    gate = AtrSanityGate(atr_window_seconds=300, k=k)

    def _provider(_symbol: str, _t: object) -> Decimal | None:
        return atr
    return SubmitterBindings(
        nexus_config=_nexus_config(), state=state,
        praxis_outbound=cast(PraxisOutbound, outbound),
        validation_pipeline=pipeline,
        capital_controller=controller,
        strategy_budget=Decimal('100000'),
        atr_gate=gate, atr_provider=_provider,
    )


def test_atr_sanity_rejects_tight_stop_buy_entry() -> None:
    """A 1 bp stop on a $50k symbol is denied before reaching Praxis.

    Slice #17 Task 29 contract: ENTER+BUY whose declared stop is
    closer than `gate.k * ATR(window)` from entry MUST be denied
    at INTAKE. With ATR=300, k=0.5 → min_distance=150. Tight
    stop at 49_995 (distance=5) is well under the floor.

    Mutation proof: removing the `_check_atr_sanity` call would
    let the command flow to outbound and break this test.
    """
    outbound = _OutboundStub()
    pipeline, controller, state = _pipeline_and_state()
    bindings = _atr_bindings(outbound, pipeline, controller, state)
    rejected: list[ValidationDecision] = []
    submit = build_action_submitter(
        bindings,
        on_atr_reject=lambda d, _a: rejected.append(d),
    )
    tight = Action(
        action_type=ActionType.ENTER, direction=OrderSide.BUY,
        size=Decimal('0.001'), execution_mode=ExecutionMode.SINGLE_SHOT,
        order_type=OrderType.MARKET,
        execution_params={
            'symbol': 'BTCUSDT', 'stop_bps': '1',
            'stop_price': '49995',  # 5 USDT distance, ATR floor=150
        },
        deadline=60, trade_id=None, command_id=None,
        maker_preference=None, reference_price=Decimal('50000'),
    )
    submit([tight], 'long_on_signal')
    assert len(outbound.commands) == 0, (
        f'tight-stop ENTER+BUY must NOT reach Praxis; got '
        f'{len(outbound.commands)} command(s) sent.'
    )
    assert len(rejected) == 1
    assert rejected[0].reason_code == 'ATR_STOP_TIGHTER_THAN_MIN_ATR_FRACTION'


def test_atr_sanity_allows_sane_stop_buy_entry() -> None:
    """A 50 bp stop ($250 distance) clears the gate at ATR=300, k=0.5.

    `min_distance = 0.5 * 300 = 150`. Distance=250 > 150 → allow.
    The strategy template's default `stop_bps=50` produces this
    shape on BTCUSDT, so production runs do NOT trigger the gate.
    """
    outbound = _OutboundStub()
    pipeline, controller, state = _pipeline_and_state()
    bindings = _atr_bindings(outbound, pipeline, controller, state)
    rejected: list[ValidationDecision] = []
    submit = build_action_submitter(
        bindings,
        on_atr_reject=lambda d, _a: rejected.append(d),
    )
    submit([_enter_action()], 'long_on_signal')
    assert len(outbound.commands) == 1, (
        f'sane-stop ENTER+BUY should reach Praxis; got '
        f'{len(outbound.commands)} command(s) sent.'
    )
    assert rejected == []


def test_atr_sanity_uncalibrated_denies() -> None:
    """ATR provider returning None → `ATR_UNCALIBRATED` denial.

    Strategy submitting before the pre-decision tape has data
    (e.g., warm-up window) cannot honestly compute R; the gate
    denies loudly with a distinct reason_code so the operator
    can tell warm-up gaps apart from gameability rejections.
    """
    outbound = _OutboundStub()
    pipeline, controller, state = _pipeline_and_state()
    bindings = _atr_bindings(outbound, pipeline, controller, state, atr=None)
    rejected: list[ValidationDecision] = []
    submit = build_action_submitter(
        bindings,
        on_atr_reject=lambda d, _a: rejected.append(d),
    )
    submit([_enter_action()], 'long_on_signal')
    assert len(outbound.commands) == 0
    assert len(rejected) == 1
    assert rejected[0].reason_code == 'ATR_UNCALIBRATED'


def test_atr_sanity_uses_limit_price_over_seed_for_entry() -> None:
    """LIMIT `execution_params['price']` shadows stale `reference_price`.

    Codex round 1 P1: the gate's entry must track the value R̄'s
    denominator will use. For a maker LIMIT, that's the
    `_maybe_refresh_limit_to_touch`-rewritten price, not the
    seed `reference_price` baked at window-start.

    Mutation-proof setup (codex round 2): floor = `0.5 * 300 = 150`.
      seed = 50_000, limit_price = 49_998, stop_price = 49_849.
      seed_distance = 151 (PASS — above floor)
      limit_distance = 149 (REJECT — below floor)

    A regression that drops the priority lookup and falls back to
    `reference_price` would compute distance=151 and ALLOW the
    order; this test would fail.
    """
    outbound = _OutboundStub()
    pipeline, controller, state = _pipeline_and_state()
    bindings = _atr_bindings(outbound, pipeline, controller, state)
    rejected: list[ValidationDecision] = []
    submit = build_action_submitter(
        bindings,
        on_atr_reject=lambda d, _a: rejected.append(d),
    )
    limit = Action(
        action_type=ActionType.ENTER, direction=OrderSide.BUY,
        size=Decimal('0.001'), execution_mode=ExecutionMode.SINGLE_SHOT,
        order_type=OrderType.LIMIT,
        execution_params={
            'symbol': 'BTCUSDT', 'stop_bps': '30',
            'stop_price': '49849',
            'price': '49998',  # refreshed touch price; limit_distance=149
        },
        deadline=60, trade_id=None, command_id=None,
        maker_preference=None, reference_price=Decimal('50000'),
    )
    submit([limit], 'long_on_signal')
    assert len(outbound.commands) == 0, (
        f'gate must use refreshed LIMIT price (distance=149 < 150 floor) '
        f'not seed (distance=151 > floor); a regression that uses seed '
        f'would let this through. Got {len(outbound.commands)} command(s).'
    )
    assert len(rejected) == 1
    assert rejected[0].reason_code == 'ATR_STOP_TIGHTER_THAN_MIN_ATR_FRACTION'


def test_atr_sanity_uses_touch_provider_when_available() -> None:
    """`touch_provider` shadows `reference_price` for MARKET orders.

    Codex round 1 P1: even MARKET orders should use the freshest
    pre-submit price proxy available, not the stale seed.

    Mutation-proof setup (codex round 2): floor = `0.5 * 300 = 150`.
      seed = 50_000, touch = 49_998, stop_price = 49_849.
      seed_distance = 151 (PASS — above floor)
      touch_distance = 149 (REJECT — below floor)

    A regression that drops the priority lookup falls back to
    `reference_price` and ALLOWS the order; this test would fail.
    """
    outbound = _OutboundStub()
    pipeline, controller, state = _pipeline_and_state()
    rejected: list[ValidationDecision] = []
    from backtest_simulator.honesty.atr import AtrSanityGate
    gate = AtrSanityGate(atr_window_seconds=300, k=Decimal('0.5'))

    def _atr(_sym: str, _t: object) -> Decimal:
        return Decimal('300')

    def _touch(_sym: str) -> Decimal:
        return Decimal('49998')
    bindings = SubmitterBindings(
        nexus_config=_nexus_config(), state=state,
        praxis_outbound=cast(PraxisOutbound, outbound),
        validation_pipeline=pipeline,
        capital_controller=controller,
        strategy_budget=Decimal('100000'),
        touch_provider=_touch,
        atr_gate=gate, atr_provider=_atr,
    )
    submit = build_action_submitter(
        bindings,
        on_atr_reject=lambda d, _a: rejected.append(d),
    )
    market_with_tight_stop = Action(
        action_type=ActionType.ENTER, direction=OrderSide.BUY,
        size=Decimal('0.001'), execution_mode=ExecutionMode.SINGLE_SHOT,
        order_type=OrderType.MARKET,
        execution_params={
            'symbol': 'BTCUSDT', 'stop_bps': '30',
            'stop_price': '49849',  # touch_distance=149 < 150 floor
        },
        deadline=60, trade_id=None, command_id=None,
        maker_preference=None, reference_price=Decimal('50000'),
    )
    submit([market_with_tight_stop], 'long_on_signal')
    assert len(rejected) == 1, (
        f'gate must use touch (distance=149 < 150) over seed '
        f'(distance=151 > 150). A regression that uses seed allows '
        f'the order; this test would fail. Got {len(rejected)} rejection(s).'
    )


def test_atr_sanity_k_zero_disables_gate_even_on_uncalibrated() -> None:
    """`--atr-k 0` admits ENTER+BUY even when ATR provider returns None.

    Slice #17 Task 29 / codex round 4 P1: the standalone
    `AtrSanityGate` primitive treats `k=0` as the "operator
    explicitly disabled the gate" knob. The wiring must honor that
    BEFORE calling `atr_provider`, otherwise an empty pre-decision
    tape (provider returns None) routes to `ATR_UNCALIBRATED`
    rejection and a disabled gate still denies orders.

    Mutation proof: removing the `gate.k == Decimal('0')` early
    return makes the `atr_provider=lambda: None` provider trip the
    uncalibrated path and `n_rejected/uncal` rises.
    """
    outbound = _OutboundStub()
    pipeline, controller, state = _pipeline_and_state()
    rejected: list[ValidationDecision] = []
    from backtest_simulator.honesty.atr import AtrSanityGate
    gate = AtrSanityGate(atr_window_seconds=300, k=Decimal('0'))

    def _atr_returns_none(_sym: str, _t: object) -> Decimal | None:
        return None
    bindings = SubmitterBindings(
        nexus_config=_nexus_config(), state=state,
        praxis_outbound=cast(PraxisOutbound, outbound),
        validation_pipeline=pipeline,
        capital_controller=controller,
        strategy_budget=Decimal('100000'),
        atr_gate=gate, atr_provider=_atr_returns_none,
    )
    submit = build_action_submitter(
        bindings,
        on_atr_reject=lambda d, _a: rejected.append(d),
    )
    submit([_enter_action()], 'long_on_signal')
    assert rejected == [], (
        f'k=0 must disable the gate completely, including the '
        f'uncalibrated path; got {len(rejected)} rejection(s).'
    )
    # Order proceeds to the rest of the pipeline; whether it sends
    # or fails downstream is not the gate's concern.


def test_atr_sanity_skips_sell_exit() -> None:
    """SELL exits are not gated — long-only template scope.

    The gate covers ENTER+BUY only (matching the strict-impact
    gate's scoping in slice #17 Task 31's audit follow-up).
    A SELL with no stop is the closing leg of an existing
    long; gating it would leave the strategy holding risk.
    """
    outbound = _OutboundStub()
    pipeline, controller, state = _pipeline_and_state()
    bindings = _atr_bindings(outbound, pipeline, controller, state)
    rejected: list[ValidationDecision] = []
    submit = build_action_submitter(
        bindings,
        on_atr_reject=lambda d, _a: rejected.append(d),
    )
    submit([_sell_action()], 'long_on_signal')
    # SELL exit goes through the dedicated SELL fast-path.
    assert len(outbound.commands) == 1
    assert rejected == [], (
        f'SELL exit must NOT trigger the ATR gate; got '
        f'{len(rejected)} rejection(s).'
    )


def test_maybe_refresh_limit_to_touch_skips_market_orders() -> None:
    """MARKET actions do NOT get the price-refresh treatment.

    The touch refresh must short-circuit when `order_type !=
    LIMIT` so MARKET MARKET / SELL MARKET pass through unchanged
    (they don't carry `execution_params['price']` in the first
    place; rewriting it would either fail or pollute the
    Praxis SingleShotParams).
    """
    outbound = _OutboundStub()
    pipeline, controller, state = _pipeline_and_state()
    refresh_calls: list[str] = []

    def _spying_touch(symbol: str) -> Decimal | None:
        refresh_calls.append(symbol)
        return Decimal('50000')

    bindings = SubmitterBindings(
        nexus_config=_nexus_config(), state=state,
        praxis_outbound=cast(PraxisOutbound, outbound),
        validation_pipeline=pipeline,
        capital_controller=controller,
        strategy_budget=Decimal('100000'),
        touch_provider=_spying_touch,
        tick_provider=lambda _sym: Decimal('0.01'),
    )
    submit = build_action_submitter(bindings)
    submit([_enter_action()], 'long_on_signal')
    assert refresh_calls == [], (
        f'touch_provider was called for a MARKET action — refresh hook '
        f'should short-circuit on non-LIMIT order types. Calls: '
        f'{refresh_calls}'
    )
