"""Slice #38 — `record_close_position` is the EXIT capital substitute seam.

Nexus's `CapitalController` has no `close_position` primitive, so the
backtest's SELL-fill path substitutes `record_close_position` to invert
the BUY's deployment exactly. The pipeline_executor's bypass of the
CAPITAL stage on `ValidationAction.EXIT` is the upstream half; this
seam is the matching closure on the bts side.
"""
from __future__ import annotations

from decimal import Decimal

from nexus.core.capital_controller.capital_controller import CapitalController
from nexus.core.domain.capital_state import CapitalState

from backtest_simulator.honesty.capital import CapitalLifecycleTracker


def test_record_close_position_runs_after_sell_fill() -> None:
    """Realized PnL inverts the BUY's deployment exactly.

    BUY deployed `cost_basis + entry_fees = 50.05` into
    `position_notional`. SELL fill at +1% returns
    `sell_proceeds - cost_basis - entry_fees - sell_fees =
    50.5 - 50 - 0.05 - 0.05 = 0.4`. The seam mutates capital_state
    so `position_notional -= 50.05` and the per-strategy attribution
    decrements by the same amount.
    """
    capital_state = CapitalState(capital_pool=Decimal('100000'))
    controller = CapitalController(capital_state)
    tracker = CapitalLifecycleTracker(controller)
    # Simulate a deployed BUY directly on the capital state so the
    # close has something to invert.
    capital_state.position_notional = Decimal('50.05')
    capital_state.per_strategy_deployed['long_on_signal'] = Decimal('50.05')
    tracker.record_open_position(
        command_id='buy-cmd-1', strategy_id='long_on_signal',
        cost_basis=Decimal('50'), entry_fees=Decimal('0.05'),
        entry_qty=Decimal('0.001'),
    )
    realized_pnl, closed = tracker.record_close_position(
        capital_state,
        sell_command_id='sell-cmd-1',
        sell_qty=Decimal('0.001'),
        sell_proceeds=Decimal('50.5'),
        sell_fees=Decimal('0.05'),
    )
    assert realized_pnl == Decimal('0.40')
    assert closed.command_id == 'buy-cmd-1'
    assert capital_state.position_notional == Decimal('0')
    assert capital_state.per_strategy_deployed.get('long_on_signal') is None


def test_capital_controller_check_and_reserve_not_called_for_exit() -> None:
    """EXIT validation never invokes `CapitalController.check_and_reserve`.

    The pipeline_executor short-circuits the CAPITAL stage for
    `ValidationAction.EXIT`, so calling `validate(EXIT_context)`
    never reaches `validate_capital_stage` (which is what would
    call `check_and_reserve`). Mutation proof: a counter wrapped
    around the controller method stays at zero.
    """
    from nexus.core.domain.enums import OrderSide
    from nexus.core.domain.instance_state import InstanceState, Position
    from nexus.core.validator.pipeline_models import (
        ValidationAction,
        ValidationRequestContext,
    )
    from nexus.instance_config import InstanceConfig

    from backtest_simulator.honesty import build_validation_pipeline
    config = InstanceConfig(account_id='bts', venue='binance_spot_simulated')
    pipeline, controller, capital_state = build_validation_pipeline(
        nexus_config=config, capital_pool=Decimal('100000'),
    )
    state = InstanceState(capital=capital_state)
    state.positions['open-1'] = Position(
        trade_id='open-1', strategy_id='long_on_signal', symbol='BTCUSDT',
        side=OrderSide.BUY, size=Decimal('0.001'),
        entry_price=Decimal('50000'),
    )
    calls: list[tuple[object, ...]] = []
    original = controller.check_and_reserve

    def spy(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return original(*args, **kwargs)  # type: ignore[arg-type]
    controller.check_and_reserve = spy  # type: ignore[method-assign]
    ctx = ValidationRequestContext(
        strategy_id='long_on_signal', order_notional=Decimal('50'),
        estimated_fees=Decimal('0'), strategy_budget=Decimal('0'),
        state=state, config=config,
        action=ValidationAction.EXIT,
        symbol='BTCUSDT', order_side=OrderSide.SELL,
        order_size=Decimal('0.001'), trade_id='open-1',
        command_id='bts-cmd-1',
    )
    decision = pipeline.validate(ctx)
    assert decision.allowed
    assert calls == [], (
        f'check_and_reserve must NOT be called on EXIT; got '
        f'{len(calls)} call(s).'
    )


def test_record_close_position_docstring_names_exit_capital_substitute_seam() -> None:
    """The docstring explicitly names `EXIT capital substitute seam`."""
    doc = CapitalLifecycleTracker.record_close_position.__doc__ or ''
    assert 'EXIT capital substitute seam' in doc
