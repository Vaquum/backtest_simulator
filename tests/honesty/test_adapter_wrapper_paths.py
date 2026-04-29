"""Tests for the fail-loud paths in the launcher's capital-adapter wrapper.

Covers:
  - `CapitalOvershootError`: actual fill notional exceeds the reservation.
  - `PARTIALLY_FILLED` (strict-live-reality): the wrapper drives
    `order_fill(partial) + order_cancel(residual)` to release the
    unfilled reservation; ledger stays conservation-green.
  - Unmatched BUY client_order_id raises `RuntimeError` (honesty: every BUY
    MUST clear CAPITAL, a missing tracker match means a gate was bypassed).
  - `record_rejection` raises when the CapitalController's underlying
    release/reject call fails (mutation: simulated controller failure).
"""
from __future__ import annotations

import asyncio
from decimal import Decimal
from types import SimpleNamespace

import pytest
from nexus.core.domain.capital_state import CapitalState
from nexus.core.domain.enums import OrderSide
from nexus.instance_config import InstanceConfig as NexusInstanceConfig

from backtest_simulator.honesty import (
    CapitalLifecycleTracker,
    build_validation_pipeline,
)
from backtest_simulator.launcher.launcher import (
    CapitalOvershootError,
    _install_capital_adapter_wrapper,
)


class _FakeAdapter:
    """Stand-in for SimulatedVenueAdapter that returns a scripted
    SubmitResult. The wrapper installs itself on this adapter's
    `submit_order` attribute.
    """

    def __init__(self, result: SimpleNamespace) -> None:
        self._result = result

    async def submit_order(self, *args, **kwargs):
        return self._result


def _tracker_with_pending(
    capital_state: CapitalState,
    *,
    command_id: str = 'cmd-1',
    notional: Decimal = Decimal('1000'),
) -> CapitalLifecycleTracker:
    # Bind the tracker's internal `CapitalController` to the SAME
    # `capital_state` the test asserts on. The earlier helper built
    # its own state inside `build_validation_pipeline`, leaving the
    # test's passed-in state untouched — so assertions on
    # `capital_state.position_notional` were checking a different
    # object from the one the wrapper was mutating.
    from nexus.core.validator.capital_stage import CapitalController
    controller = CapitalController(capital_state)
    # Short-circuit: directly call check_and_reserve so we get a real
    # reservation_id without going through the full ValidationPipeline
    # (which would also require InstanceState etc.).
    result = controller.check_and_reserve(
        strategy_id='bts', order_notional=notional,
        estimated_fees=Decimal('2'), strategy_budget=Decimal('100000'),
        ttl_seconds=3600,
    )
    tracker = CapitalLifecycleTracker(controller)
    tracker.record_reservation(
        command_id=command_id,
        reservation_id=result.reservation.reservation_id,
        strategy_id='bts',
        notional=notional,
        estimated_fees=Decimal('2'),
    )
    return tracker


def test_overshoot_raises_capital_overshoot_error() -> None:
    _pipeline, _controller, capital_state = build_validation_pipeline(
        nexus_config=NexusInstanceConfig(account_id='bts-test', venue='binance_spot_simulated'),
        capital_pool=Decimal('100000'),
    )
    tracker = _tracker_with_pending(capital_state, notional=Decimal('1000'))
    # SubmitResult with fill_notional = 1100 (exceeds reservation 1000).
    fills = [
        SimpleNamespace(qty=Decimal('1'), price=Decimal('1100'), fee=Decimal('2'))
    ]
    result = SimpleNamespace(
        status=SimpleNamespace(name='FILLED'),
        venue_order_id='VENUE-1',
        immediate_fills=fills,
    )
    adapter = _FakeAdapter(result)
    _install_capital_adapter_wrapper(
        adapter=adapter, tracker=tracker, capital_state=capital_state,
        initial_pool=Decimal('100000'), declared_stops={},
    )
    # client_order_id must include a prefix that matches the pending
    # command_id after dash-strip. The first 16 chars of dash-stripped
    # 'cmd-1' is just 'cmd1' — short. Pad to match `_match_command_id`
    # logic by using a hex-style id instead.
    with pytest.raises(CapitalOvershootError, match='fill_notional'):
        asyncio.run(adapter.submit_order(
            'bts-acct', 'BTCUSDT', OrderSide.BUY,
            SimpleNamespace(name='MARKET'), Decimal('1'),
            client_order_id='SS-cmd1-000',
        ))


def test_partially_filled_releases_residual_reservation() -> None:
    """PARTIALLY_FILLED drives order_fill(partial) + order_cancel(residual).

    Under the strict-live-reality fill model, `_walk_market` halts
    on stop breach and returns a partial fill. The wrapper must:
      1. Record the ack.
      2. Drive order_fill for the filled notional.
      3. Cancel the residual so working_order_notional doesn't
         permanently hold the unfilled reservation.

    Mutation proof: if the wrapper skipped the order_cancel step,
    `capital_state.working_order_notional` would remain at
    (reserved - filled) > 0 after the submit, and
    `capital_state.available` would be under-reported by that amount.
    This test pins the expected-post-state invariants.
    """
    _pipeline, _controller, capital_state = build_validation_pipeline(
        nexus_config=NexusInstanceConfig(account_id='bts-test', venue='binance_spot_simulated'),
        capital_pool=Decimal('100000'),
    )
    tracker = _tracker_with_pending(capital_state, notional=Decimal('1000'))
    # SubmitResult with status=PARTIALLY_FILLED — filled qty 0.5 at 1000
    # = 500 notional, reservation was 1000, so residual = 500 must
    # release back to available.
    fills = [
        SimpleNamespace(qty=Decimal('0.5'), price=Decimal('1000'), fee=Decimal('1'))
    ]
    result = SimpleNamespace(
        status=SimpleNamespace(name='PARTIALLY_FILLED'),
        venue_order_id='VENUE-1',
        immediate_fills=fills,
    )
    adapter = _FakeAdapter(result)
    _install_capital_adapter_wrapper(
        adapter=adapter, tracker=tracker, capital_state=capital_state,
        initial_pool=Decimal('100000'), declared_stops={},
    )
    # No exception should raise; the wrapper handles the partial as a
    # terminal state with residual release.
    asyncio.run(adapter.submit_order(
        'bts-acct', 'BTCUSDT', OrderSide.BUY,
        SimpleNamespace(name='MARKET'), Decimal('1'),
        client_order_id='SS-cmd1-000',
    ))
    # Post-state invariants:
    # - Tracker is clean: partial fills are terminal, no lingering lifecycle.
    assert tracker.pending_count == 0, (
        f'expected tracker pending_count=0 after PARTIALLY_FILLED, '
        f'got {tracker.pending_count} — release path did not fire.'
    )
    # - working_order_notional is zero: no residual leak.
    assert capital_state.working_order_notional == Decimal('0'), (
        f'expected working_order_notional=0 after terminal partial '
        f'(residual released via order_cancel), got '
        f'{capital_state.working_order_notional}. Ledger under-count bug.'
    )
    # - position_notional reflects the filled portion (plus fees).
    # The controller books fill_notional + actual_fees = 500 + 1 = 501.
    assert capital_state.position_notional == Decimal('501'), (
        f'expected position_notional=501 (500 fill + 1 fee), got '
        f'{capital_state.position_notional}.'
    )


def test_expired_zero_fill_releases_reservation() -> None:
    """EXPIRED (validated, no liquidity in window) must release the reservation.

    Pre-fix the wrapper only treated REJECTED as a release-and-return
    terminal; EXPIRED fell through to `_finalize_successful_fill` with
    `fill_notional=0`. That popped the lifecycle but left the
    CapitalController's `working_order_notional` locked at the original
    reservation, under-counting available capital on every subsequent
    validation. Post-fix EXPIRED routes through `record_rejection`
    (releases reservation cleanly).

    Mutation proof: if the launcher's status branch reverts to
    `if status_name == 'REJECTED'` only, this test fires by either
    (a) tracker.pending_count != 0 (lifecycle still hanging), or
    (b) capital_state.reservation_notional != 0 (lock leaked).
    """
    _pipeline, _controller, capital_state = build_validation_pipeline(
        nexus_config=NexusInstanceConfig(account_id='bts-test', venue='binance_spot_simulated'),
        capital_pool=Decimal('100000'),
    )
    tracker = _tracker_with_pending(capital_state, notional=Decimal('1000'))
    # Pre-state assertion: the reservation IS holding 1000 (with fees) in
    # `reservation_notional`. The test verifies it's released after the
    # EXPIRED outcome.
    assert capital_state.reservation_notional > 0
    result = SimpleNamespace(
        status=SimpleNamespace(name='EXPIRED'),
        venue_order_id='VENUE-1',
        immediate_fills=[],
    )
    adapter = _FakeAdapter(result)
    _install_capital_adapter_wrapper(
        adapter=adapter, tracker=tracker, capital_state=capital_state,
        initial_pool=Decimal('100000'), declared_stops={},
    )
    asyncio.run(adapter.submit_order(
        'bts-acct', 'BTCUSDT', OrderSide.BUY,
        SimpleNamespace(name='MARKET'), Decimal('1'),
        client_order_id='SS-cmd1-000',
    ))
    # Post-state invariants:
    assert tracker.pending_count == 0, (
        f'expected tracker pending_count=0 after EXPIRED, '
        f'got {tracker.pending_count} — release path did not fire.'
    )
    assert capital_state.reservation_notional == Decimal('0'), (
        f'expected reservation_notional=0 after EXPIRED release, got '
        f'{capital_state.reservation_notional}. Capital lock leaked.'
    )
    # Critical regression catch: if the wrapper falls back to
    # `_finalize_successful_fill` on EXPIRED, `record_ack_and_fill(
    # fill_notional=0)` moves the reservation through `send_order` →
    # `order_ack` → `order_fill(0)` and parks the un-filled notional in
    # `working_order_notional`. The reservation_notional check above
    # would still pass under that regression. This invariant is the
    # one that fails loud if the EXPIRED status branch is removed.
    assert capital_state.working_order_notional == Decimal('0'), (
        f'expected working_order_notional=0 after EXPIRED release, got '
        f'{capital_state.working_order_notional}. The EXPIRED status was '
        f'routed through the fill path instead of the release path.'
    )
    assert capital_state.position_notional == Decimal('0'), (
        f'expected position_notional=0 (no fill happened), got '
        f'{capital_state.position_notional}.'
    )


def test_unmatched_buy_raises_runtime_error() -> None:
    # No pending reservation → a BUY that reaches the wrapper has
    # bypassed the CAPITAL gate. Must fail loud.
    _pipeline, _controller, capital_state = build_validation_pipeline(
        nexus_config=NexusInstanceConfig(account_id='bts-test', venue='binance_spot_simulated'),
        capital_pool=Decimal('100000'),
    )
    tracker = CapitalLifecycleTracker(_controller)  # empty tracker
    result = SimpleNamespace(
        status=SimpleNamespace(name='FILLED'),
        venue_order_id='VENUE-1',
        immediate_fills=[],
    )
    adapter = _FakeAdapter(result)
    _install_capital_adapter_wrapper(
        adapter=adapter, tracker=tracker, capital_state=capital_state,
        initial_pool=Decimal('100000'), declared_stops={},
    )
    with pytest.raises(RuntimeError, match='no matching reservation'):
        asyncio.run(adapter.submit_order(
            'bts-acct', 'BTCUSDT', OrderSide.BUY,
            SimpleNamespace(name='MARKET'), Decimal('1'),
            client_order_id='SS-ghost1234567890-000',
        ))


def test_unmatched_sell_is_accepted() -> None:
    # SELL-as-close has no pending reservation by design; the wrapper
    # must NOT raise for SELL.
    _pipeline, _controller, capital_state = build_validation_pipeline(
        nexus_config=NexusInstanceConfig(account_id='bts-test', venue='binance_spot_simulated'),
        capital_pool=Decimal('100000'),
    )
    tracker = CapitalLifecycleTracker(_controller)
    result = SimpleNamespace(
        status=SimpleNamespace(name='FILLED'),
        venue_order_id='VENUE-1',
        immediate_fills=[],
    )
    adapter = _FakeAdapter(result)
    _install_capital_adapter_wrapper(
        adapter=adapter, tracker=tracker, capital_state=capital_state,
        initial_pool=Decimal('100000'), declared_stops={},
    )
    # Must NOT raise.
    out = asyncio.run(adapter.submit_order(
        'bts-acct', 'BTCUSDT', OrderSide.SELL,
        SimpleNamespace(name='MARKET'), Decimal('1'),
        client_order_id='SS-ghost1234567890-000',
    ))
    assert out is result


def test_record_rejection_raises_on_controller_failure() -> None:
    # Mutation: simulate a CapitalController whose `order_reject`
    # returns success=False. The tracker MUST raise rather than
    # report "clean" while the controller still holds the capital.
    class _FailingController:
        def order_reject(self, order_id: str):
            return SimpleNamespace(
                success=False,
                reason='simulated_failure',
                category=SimpleNamespace(name='INVARIANT_BREACH'),
            )

        def release_reservation(self, reservation_id: str):
            return SimpleNamespace(success=True)

    tracker = CapitalLifecycleTracker(_FailingController())
    tracker.record_reservation(
        command_id='cmd-fail', reservation_id='res-1',
        strategy_id='bts', notional=Decimal('1000'),
        estimated_fees=Decimal('2'),
    )
    # Mark as sent so record_rejection calls order_reject (not
    # release_reservation).
    tracker._pending['cmd-fail'].sent = True
    with pytest.raises(RuntimeError, match='order_reject failed'):
        tracker.record_rejection('cmd-fail', 'VENUE-1')


def test_buy_fill_records_open_position_through_wrapper() -> None:
    """BUY fill through the wrapper appends to the open-positions ledger.

    Pre-state: tracker has zero open positions. After a FILLED
    BUY through the wrapper, the open-position count is 1 with
    cost_basis = fill_notional and entry_fees = actual fees.
    capital_state.position_notional reflects the controller's
    `fill_notional + actual_fees` deployment.

    Mutation proof: if `_finalize_successful_fill` regresses
    and stops calling `record_open_position`, the count stays
    at 0 — and the matching SELL close test below would raise
    `record_close_position: SELL ... has no matching open
    position`. Pin the BUY half here so a regression on the
    BUY side surfaces with a precise message.
    """
    _pipeline, _controller, capital_state = build_validation_pipeline(
        nexus_config=NexusInstanceConfig(account_id='bts-test', venue='binance_spot_simulated'),
        capital_pool=Decimal('100000'),
    )
    tracker = _tracker_with_pending(capital_state, notional=Decimal('1000'))
    fills = [
        SimpleNamespace(qty=Decimal('1'), price=Decimal('1000'), fee=Decimal('1')),
    ]
    result = SimpleNamespace(
        status=SimpleNamespace(name='FILLED'),
        venue_order_id='VENUE-1',
        immediate_fills=fills,
    )
    adapter = _FakeAdapter(result)
    _install_capital_adapter_wrapper(
        adapter=adapter, tracker=tracker, capital_state=capital_state,
        initial_pool=Decimal('100000'), declared_stops={},
    )
    asyncio.run(adapter.submit_order(
        'bts-acct', 'BTCUSDT', OrderSide.BUY,
        SimpleNamespace(name='MARKET'), Decimal('1'),
        client_order_id='SS-cmd1-000',
    ))
    assert tracker.open_position_count == 1, (
        f'expected 1 open position after BUY FILLED, got '
        f'{tracker.open_position_count} — record_open_position '
        f'was not called from _finalize_successful_fill.'
    )
    # position_notional carries fill_notional + entry_fees per controller's
    # order_fill semantics.
    assert capital_state.position_notional == Decimal('1001'), (
        f'expected position_notional=1001 (1000 fill + 1 fee), got '
        f'{capital_state.position_notional}.'
    )


def test_buy_then_sell_close_releases_position_and_attribution() -> None:
    """Full BUY→SELL round trip leaves position_notional and
    per_strategy_deployed at zero, capital_pool untouched.

    Post-fix conservation invariants — if any regression bypasses
    `record_close_position`, this test fires:
      - tracker.open_position_count == 0 after SELL close
      - capital_state.position_notional == 0 (released cost_basis + entry_fees)
      - capital_state.per_strategy_deployed has no `bts` entry (released to zero)
      - capital_state.capital_pool unchanged (immutable budget)

    Mutation proof: if `_finalize_sell_close` regresses to a
    log-and-return (the codex round 4 P0 shape), the BUY's
    deployed amount stays committed in `position_notional`
    forever and the next BUY's CAPITAL stage either denies on
    over-deployment or silently double-books.
    """
    _pipeline, _controller, capital_state = build_validation_pipeline(
        nexus_config=NexusInstanceConfig(account_id='bts-test', venue='binance_spot_simulated'),
        capital_pool=Decimal('100000'),
    )
    tracker = _tracker_with_pending(capital_state, notional=Decimal('1000'))
    initial_capital_pool = capital_state.capital_pool
    # BUY fill: 1 unit at 1000 with 1 fee.
    buy_result = SimpleNamespace(
        status=SimpleNamespace(name='FILLED'),
        venue_order_id='VENUE-1',
        immediate_fills=[
            SimpleNamespace(qty=Decimal('1'), price=Decimal('1000'), fee=Decimal('1')),
        ],
    )
    buy_adapter = _FakeAdapter(buy_result)
    _install_capital_adapter_wrapper(
        adapter=buy_adapter, tracker=tracker, capital_state=capital_state,
        initial_pool=initial_capital_pool, declared_stops={},
    )
    asyncio.run(buy_adapter.submit_order(
        'bts-acct', 'BTCUSDT', OrderSide.BUY,
        SimpleNamespace(name='MARKET'), Decimal('1'),
        client_order_id='SS-cmd1-000',
    ))
    # Snapshot post-BUY state.
    pool_after_buy = capital_state.capital_pool
    assert tracker.open_position_count == 1
    assert capital_state.position_notional == Decimal('1001')
    assert capital_state.per_strategy_deployed.get('bts') == Decimal('1001')
    # SELL close: 1 unit at 1100 with 1.1 fee — profitable round trip.
    sell_result = SimpleNamespace(
        status=SimpleNamespace(name='FILLED'),
        venue_order_id='VENUE-2',
        immediate_fills=[
            SimpleNamespace(qty=Decimal('1'), price=Decimal('1100'), fee=Decimal('1.1')),
        ],
    )
    sell_adapter = _FakeAdapter(sell_result)
    _install_capital_adapter_wrapper(
        adapter=sell_adapter, tracker=tracker, capital_state=capital_state,
        initial_pool=initial_capital_pool, declared_stops={},
    )
    asyncio.run(sell_adapter.submit_order(
        'bts-acct', 'BTCUSDT', OrderSide.SELL,
        SimpleNamespace(name='MARKET'), Decimal('1'),
        client_order_id='SS-cmdSELL-000',
    ))
    # Post-SELL invariants:
    assert tracker.open_position_count == 0, (
        f'expected open_position_count=0 after SELL close, got '
        f'{tracker.open_position_count} — _finalize_sell_close did not pop.'
    )
    assert capital_state.position_notional == Decimal('0'), (
        f'expected position_notional=0 after SELL close, got '
        f'{capital_state.position_notional} — cost_basis + entry_fees '
        f'not released.'
    )
    assert 'bts' not in capital_state.per_strategy_deployed, (
        f'expected per_strategy_deployed[bts] popped to zero after close, '
        f'got entries: {capital_state.per_strategy_deployed}.'
    )
    # capital_pool MUST NOT change on close — codex round 5 P1 caught the
    # prior shape that credited `sell_proceeds - sell_fees` to the pool.
    assert capital_state.capital_pool == pool_after_buy, (
        f'capital_pool changed across SELL close: pre={pool_after_buy} '
        f'post={capital_state.capital_pool}. capital_pool is immutable '
        f'budget; SELL proceeds are realized PnL.'
    )


def test_partial_sell_close_shrinks_head_keeps_residual() -> None:
    """Partial SELL releases only the sold portion; residual stays open.

    Codex round 5 P2: prior implementation popped the entire FIFO
    head on PARTIALLY_FILLED, collapsing the residual to zero
    cost_basis. Post-fix the head's `entry_qty`, `cost_basis`,
    and `entry_fees` shrink proportionally and the entry stays
    in the ledger for the next SELL to finish.
    """
    _pipeline, _controller, capital_state = build_validation_pipeline(
        nexus_config=NexusInstanceConfig(account_id='bts-test', venue='binance_spot_simulated'),
        capital_pool=Decimal('100000'),
    )
    tracker = _tracker_with_pending(capital_state, notional=Decimal('1000'))
    initial_pool = capital_state.capital_pool
    # BUY 1.0 unit at 1000, fee 1.
    buy_result = SimpleNamespace(
        status=SimpleNamespace(name='FILLED'),
        venue_order_id='V-BUY',
        immediate_fills=[
            SimpleNamespace(qty=Decimal('1'), price=Decimal('1000'), fee=Decimal('1')),
        ],
    )
    buy_adapter = _FakeAdapter(buy_result)
    _install_capital_adapter_wrapper(
        adapter=buy_adapter, tracker=tracker, capital_state=capital_state,
        initial_pool=initial_pool, declared_stops={},
    )
    asyncio.run(buy_adapter.submit_order(
        'bts-acct', 'BTCUSDT', OrderSide.BUY,
        SimpleNamespace(name='MARKET'), Decimal('1'),
        client_order_id='SS-cmd1-000',
    ))
    # Partial SELL 0.4 unit at 1050, fee 0.42.
    partial_sell = SimpleNamespace(
        status=SimpleNamespace(name='PARTIALLY_FILLED'),
        venue_order_id='V-SELL-1',
        immediate_fills=[
            SimpleNamespace(qty=Decimal('0.4'), price=Decimal('1050'), fee=Decimal('0.42')),
        ],
    )
    partial_adapter = _FakeAdapter(partial_sell)
    _install_capital_adapter_wrapper(
        adapter=partial_adapter, tracker=tracker, capital_state=capital_state,
        initial_pool=initial_pool, declared_stops={},
    )
    asyncio.run(partial_adapter.submit_order(
        'bts-acct', 'BTCUSDT', OrderSide.SELL,
        SimpleNamespace(name='MARKET'), Decimal('0.4'),
        client_order_id='SS-cmdSELL1-000',
    ))
    # Residual stays open — entry_qty 0.6, cost_basis 0.6 * 1000 = 600,
    # entry_fees 0.6 * 1 = 0.6.
    assert tracker.open_position_count == 1, (
        f'partial SELL collapsed the residual: open_position_count='
        f'{tracker.open_position_count}, expected 1.'
    )
    head = tracker._open_positions[0]
    assert head.entry_qty == Decimal('0.6'), (
        f'entry_qty after partial SELL: {head.entry_qty}, expected 0.6'
    )
    assert head.cost_basis == Decimal('600.0'), (
        f'cost_basis after partial SELL: {head.cost_basis}, expected 600.0'
    )
    # position_notional released: original 1001 - (0.4/1.0 * 1001) = 600.6.
    assert capital_state.position_notional == Decimal('600.6'), (
        f'position_notional after partial SELL: '
        f'{capital_state.position_notional}, expected 600.6.'
    )


def test_sell_close_without_open_position_raises() -> None:
    """SELL fill with no matching open position raises RuntimeError.

    The strategy's `_long` gate should prevent this in practice;
    raising loudly when it slips through surfaces the state
    machine bug rather than silently corrupting capital.
    """
    _pipeline, _controller, capital_state = build_validation_pipeline(
        nexus_config=NexusInstanceConfig(account_id='bts-test', venue='binance_spot_simulated'),
        capital_pool=Decimal('100000'),
    )
    from nexus.core.validator.capital_stage import CapitalController
    controller = CapitalController(capital_state)
    tracker = CapitalLifecycleTracker(controller)
    # No BUY fill, no open positions.
    sell_result = SimpleNamespace(
        status=SimpleNamespace(name='FILLED'),
        venue_order_id='V-SELL',
        immediate_fills=[
            SimpleNamespace(qty=Decimal('1'), price=Decimal('1100'), fee=Decimal('1.1')),
        ],
    )
    sell_adapter = _FakeAdapter(sell_result)
    _install_capital_adapter_wrapper(
        adapter=sell_adapter, tracker=tracker, capital_state=capital_state,
        initial_pool=Decimal('100000'), declared_stops={},
    )
    with pytest.raises(RuntimeError, match='no matching open position'):
        asyncio.run(sell_adapter.submit_order(
            'bts-acct', 'BTCUSDT', OrderSide.SELL,
            SimpleNamespace(name='MARKET'), Decimal('1'),
            client_order_id='SS-orphanSELL-000',
        ))
