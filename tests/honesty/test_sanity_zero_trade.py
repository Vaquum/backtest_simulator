"""Sanity baseline #1: a no-op strategy emits no fills, no fees, no positions.

Pins the slice #17 Task 2 MVC: `bts test -k test_sanity_zero_trade`
returns exit 0 against a real Nexus `Strategy` whose every callback
returns the empty action list. The chain has many places a phantom
trade could leak in — the venue's fill engine, the capital lifecycle's
working_order_notional bookkeeping, the strategy's own on_outcome /
on_timer callbacks — and the floor of the simulation is "if the
strategy never decided to trade, the account stays clean."

Test surface:
  1. Direct callbacks: every `on_signal`, `on_startup`, `on_timer`,
     `on_outcome`, `on_shutdown` returns `[]`. Even when fed every
     signal shape the long-only logreg_binary template would entry
     on, the no-op strategy stays quiet.
  2. State persistence: `on_save()` returns `b''`. `on_load(b'')` does
     not raise. `save -> load -> save` is bytes-identical (the floor
     case for Task 9's broader on_save_purity test).

This test runs unit-level — no ClickHouse, no Praxis runtime — so it
slots into the `pr_checks_honesty` gate's wall-clock budget. The
end-to-end "no fills via the full venue + capital pipeline" assertion
is owned by the integration tests (Task 24).
"""
from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

from nexus.core.domain.operational_mode import OperationalMode
from nexus.instance_config import InstanceConfig as NexusInstanceConfig
from nexus.strategy.context import StrategyContext
from nexus.strategy.params import StrategyParams
from nexus.strategy.signal import Signal

from backtest_simulator.strategies import ZeroTradeStrategy


def _make_signal(preds: int) -> Signal:
    return Signal(
        predictor_fn_id='sanity-zero-trade',
        timestamp=datetime(2026, 4, 20, 8, 0, tzinfo=UTC),
        values={'_preds': preds, '_probs': 0.5},
    )


def _empty_context() -> StrategyContext:
    return StrategyContext(
        positions=(),
        capital_available=Decimal('100000'),
        operational_mode=OperationalMode.ACTIVE,
    )


def test_sanity_zero_trade() -> None:
    strategy = ZeroTradeStrategy('sanity-zero-trade')
    params = StrategyParams(raw={})
    context = _empty_context()

    # Startup contributes no actions.
    assert strategy.on_startup(params, context) == []

    # Every plausible signal — preds=1 (would normally enter), preds=0
    # (would normally exit), unknown preds — produces zero actions.
    for preds in (0, 1, 2, -1):
        actions = strategy.on_signal(_make_signal(preds), params, context)
        assert actions == [], (
            f'ZeroTradeStrategy emitted {len(actions)} action(s) on '
            f'preds={preds}; sanity floor violated.'
        )

    # Outcome / timer / shutdown callbacks all return [].
    fake_outcome = SimpleNamespace(
        trade_id='trade-1',
        command_id='cmd-1',
        fill_price=Decimal('70000'),
        fill_qty=Decimal('0.001'),
        fee_paid=Decimal('0.07'),
    )
    assert strategy.on_outcome(fake_outcome, params, context) == []
    assert strategy.on_timer('any-timer', params, context) == []
    assert strategy.on_shutdown(params, context) == []


def test_sanity_zero_trade_save_load_purity() -> None:
    """`save -> load -> save` is bytes-identical for the no-op strategy."""
    s1 = ZeroTradeStrategy('sanity-zero-trade')
    payload_a = s1.on_save()
    s2 = ZeroTradeStrategy('sanity-zero-trade')
    s2.on_load(payload_a)
    payload_b = s2.on_save()
    assert payload_a == payload_b
    assert payload_a == b''


def test_sanity_zero_trade_through_real_action_submitter() -> None:
    """Drive ZeroTradeStrategy through the production action_submitter chain.

    Builds the real `backtest_simulator.launcher.action_submitter`
    callback (the same one `BacktestLauncher` installs into Nexus's
    PredictLoop), feeds it 24 hours of signals via the strategy, and
    asserts:
      1. Strategy returns no Actions on any signal.
      2. The real submitter callback was invoked once per signal but
         no `TradeCommand` was sent to the outbound stub (zero
         submissions reach Praxis).
      3. `pipeline.validate` was never called (no Actions = no
         validation paths exercised).
      4. `controller.check_and_reserve` was never called (no capital
         was reserved).
      5. Conservation holds and every CapitalState component remains
         at its initial value.

    This is the meaningful pipeline proof: the same submission path
    `BacktestLauncher` uses sees 24 calls and produces zero side
    effects. A regression where the submitter or pipeline gained
    side-effects on empty input would trip the post-state assertions.
    """
    from typing import cast

    from nexus.core.validator.pipeline_models import InstanceState
    from nexus.infrastructure.praxis_connector.praxis_outbound import PraxisOutbound

    from backtest_simulator.honesty import (
        assert_conservation,
        build_validation_pipeline,
    )
    from backtest_simulator.launcher.action_submitter import (
        SubmitterBindings,
        build_action_submitter,
    )

    class _OutboundStub:
        def __init__(self) -> None:
            self.commands: list[object] = []
            self.aborts: list[dict[str, object]] = []

        def send_command(self, cmd: object) -> str:
            self.commands.append(cmd)
            return 'cmd-sentinel'

        def send_abort(self, **kwargs: object) -> None:
            self.aborts.append(kwargs)

    initial_pool = Decimal('100000')
    pipeline, controller, capital_state = build_validation_pipeline(
        nexus_config=NexusInstanceConfig(account_id='bts-test', venue='binance_spot_simulated'),
        capital_pool=initial_pool,
    )

    validate_calls: list[object] = []
    reserve_calls: list[object] = []
    original_validate = pipeline.validate
    original_reserve = controller.check_and_reserve

    def _counting_validate(ctx):
        validate_calls.append(ctx)
        return original_validate(ctx)

    def _counting_reserve(*args, **kwargs):
        reserve_calls.append((args, kwargs))
        return original_reserve(*args, **kwargs)

    pipeline.validate = _counting_validate  # type: ignore[method-assign]
    controller.check_and_reserve = _counting_reserve  # type: ignore[method-assign]

    outbound = _OutboundStub()
    bindings = SubmitterBindings(
        nexus_config=NexusInstanceConfig(
            account_id='bts-acct',
            venue='binance_spot_simulated',
        ),
        state=InstanceState(capital=capital_state),
        praxis_outbound=cast(PraxisOutbound, outbound),
        validation_pipeline=pipeline,
        strategy_budget=Decimal('100000'),
    )
    submit = build_action_submitter(bindings)

    strategy = ZeroTradeStrategy('sanity-zero-trade-pipeline')
    strategy_params = StrategyParams(raw={})
    context = _empty_context()

    base_ts = datetime(2026, 4, 20, 0, 0, tzinfo=UTC)
    submit_calls = 0
    for hour in range(24):
        sig = Signal(
            predictor_fn_id='sanity-zero-trade-pipeline',
            timestamp=base_ts.replace(hour=hour),
            values={'_preds': hour % 2, '_probs': 0.5},
        )
        actions = strategy.on_signal(sig, strategy_params, context)
        # Drive the production submitter exactly as BacktestLauncher would.
        submit(actions, 'sanity-zero-trade-pipeline')
        submit_calls += 1
    # Submitter was invoked 24 times but produced zero outbound commands.
    assert submit_calls == 24
    assert outbound.commands == [], (
        f'submitter forwarded {len(outbound.commands)} command(s) to '
        f'Praxis despite ZeroTradeStrategy returning no actions.'
    )
    assert outbound.aborts == [], (
        f'submitter sent {len(outbound.aborts)} abort(s); no-op strategy must not.'
    )
    assert validate_calls == [], (
        f'pipeline.validate called {len(validate_calls)} time(s) on a '
        f'no-op strategy run; the chain has a phantom validator path.'
    )
    assert reserve_calls == [], (
        f'controller.check_and_reserve called {len(reserve_calls)} '
        f'time(s); no-op strategy must not perturb capital reservation.'
    )
    assert_conservation(capital_state, initial_pool, context='zero_trade_floor')
    assert capital_state.capital_pool == initial_pool
    assert capital_state.position_notional == Decimal('0')
    assert capital_state.reservation_notional == Decimal('0')
    assert capital_state.working_order_notional == Decimal('0')
    assert capital_state.in_flight_order_notional == Decimal('0')
    assert capital_state.fee_reserve == Decimal('0')
