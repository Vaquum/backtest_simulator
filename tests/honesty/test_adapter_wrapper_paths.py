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
