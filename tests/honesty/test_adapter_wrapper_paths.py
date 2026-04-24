"""Tests for the fail-loud paths in the launcher's capital-adapter wrapper.

Covers:
  - `CapitalOvershootError`: actual fill notional exceeds the reservation.
  - `CapitalPartialFillError`: SubmitResult.status is PARTIALLY_FILLED.
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
    CapitalPartialFillError,
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
    # Build a real pipeline so the controller has a live reservation.
    _pipeline, controller, _state = build_validation_pipeline(
        capital_pool=Decimal('100000'),
    )
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


def test_partially_filled_raises_capital_partial_fill_error() -> None:
    _pipeline, _controller, capital_state = build_validation_pipeline(
        capital_pool=Decimal('100000'),
    )
    tracker = _tracker_with_pending(capital_state, notional=Decimal('1000'))
    # SubmitResult with status=PARTIALLY_FILLED — must raise.
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
    with pytest.raises(CapitalPartialFillError, match='PARTIALLY_FILLED'):
        asyncio.run(adapter.submit_order(
            'bts-acct', 'BTCUSDT', OrderSide.BUY,
            SimpleNamespace(name='MARKET'), Decimal('1'),
            client_order_id='SS-cmd1-000',
        ))


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
