"""Part 2 capital invariants (INV-1a/1b/2/3) + 3 mutation tests.

Complements the existing `test_conservation.py` (which tests
fill-closure/fee-nonnegative at the ledger level) by pinning the
`backtest_simulator.honesty.conservation.assert_conservation`
Part 2 invariants on Nexus's `CapitalState` ledger model.
"""
from __future__ import annotations

from decimal import Decimal

import pytest
from nexus.core.domain.capital_state import CapitalState

from backtest_simulator.exceptions import ConservationViolation
from backtest_simulator.honesty import (
    assert_conservation,
    capital_totals,
)
from backtest_simulator.honesty.conservation import _PrevPoolTracker


def _fresh_state(pool: Decimal = Decimal('100000')) -> CapitalState:
    state = CapitalState(capital_pool=pool)
    # Reset per-state prev-snapshot so INV-1b starts from a known
    # baseline rather than seeing leftover state from another test.
    _PrevPoolTracker._prev.pop(id(state), None)
    return state


def test_pass_on_fresh_state() -> None:
    state = _fresh_state()
    assert_conservation(state, Decimal('100000'), context='fresh')


def test_inv1a_fails_on_pool_growing_past_initial() -> None:
    state = _fresh_state()
    state.capital_pool = Decimal('100001')
    with pytest.raises(ConservationViolation, match='INV-1'):
        assert_conservation(state, Decimal('100000'), context='inv1a')


def test_inv1b_fails_on_event_to_event_pool_growth() -> None:
    # 100000 → 99900 (drop, OK) → 99950 (grow, FAIL INV-1b).
    state = _fresh_state()
    state.capital_pool = Decimal('99900')
    assert_conservation(state, Decimal('100000'), context='step_1')
    state.capital_pool = Decimal('99950')
    with pytest.raises(ConservationViolation, match='INV-1'):
        assert_conservation(state, Decimal('100000'), context='step_2')


def test_inv2_fails_on_negative_component() -> None:
    state = _fresh_state()
    state.position_notional = Decimal('-10')
    with pytest.raises(ConservationViolation, match='INV-2'):
        assert_conservation(state, Decimal('100000'), context='inv2')


def test_inv3_fails_on_overcommitment() -> None:
    state = _fresh_state()
    state.position_notional = Decimal('50000')
    state.working_order_notional = Decimal('60000')
    with pytest.raises(ConservationViolation, match='INV-3'):
        assert_conservation(state, Decimal('100000'), context='inv3')


def test_tolerance_covers_cent_level_rounding() -> None:
    state = _fresh_state()
    state.capital_pool = Decimal('100000.001')
    # 0.001 drift is within the default 0.01 tolerance — must not raise.
    assert_conservation(state, Decimal('100000'), context='tolerance')


def test_capital_totals_matches_state() -> None:
    state = _fresh_state()
    state.position_notional = Decimal('100')
    state.fee_reserve = Decimal('5')
    totals = capital_totals(state)
    assert totals.capital_pool == Decimal('100000')
    assert totals.position_notional == Decimal('100')
    assert totals.fee_reserve == Decimal('5')
    assert totals.total_deployed == Decimal('100')


# ---- MUTATION TESTS -------------------------------------------------------


def test_mutation_pool_doubled() -> None:
    # Injected bug: capital_pool doubled somewhere post-init. INV-1a catches.
    state = _fresh_state()
    state.capital_pool = Decimal('200000')
    with pytest.raises(ConservationViolation):
        assert_conservation(state, Decimal('100000'), context='mut_pool_doubled')


def test_mutation_overcommit_via_stuck_reservation() -> None:
    # Injected bug: reservation never released on cancelled order +
    # new order committed; total_deployed now exceeds capital_pool.
    state = _fresh_state()
    state.reservation_notional = Decimal('80000')
    state.position_notional = Decimal('30000')
    with pytest.raises(ConservationViolation):
        assert_conservation(
            state, Decimal('100000'), context='mut_stuck_reservation',
        )


def test_mutation_negative_fee_reserve_underflow() -> None:
    # Injected bug: actual_fees > fee_reserve drew fee_reserve
    # negative. INV-2 catches.
    state = _fresh_state()
    state.fee_reserve = Decimal('-0.50')
    with pytest.raises(ConservationViolation):
        assert_conservation(
            state, Decimal('100000'), context='mut_fee_underflow',
        )
