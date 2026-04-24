"""Conservation laws for Nexus's CapitalController ledger model.

Nexus's `CapitalState.capital_pool` is a STATIC BUDGET, not a balance
that gets debited and credited by each transition. The other components
(`reservation_notional`, `in_flight_order_notional`,
`working_order_notional`, `position_notional`) represent DEPLOYED
claims against that budget; the CapitalController's `check_and_reserve`
simply increments `reservation_notional`, leaving `capital_pool`
untouched. Fees paid at fill time decrement `fee_reserve` (and may
draw from `capital_pool` depending on adapter accounting).

The Part 2 honesty invariants this module enforces after every
CapitalController transition:

  INV-1  capital_pool is monotonically non-increasing event-to-event
         (no free capital appearing anywhere on the timeline — the
         launcher tracks the PREVIOUS snapshot per account and this
         function compares new_pool to prev_pool, not only to
         initial_pool).
  INV-2  every component is non-negative (no negative balances).
  INV-3  total_deployed = position + working + in_flight + reservation
         never exceeds capital_pool (no overcommitment).

A violation raises `ConservationViolation` so the backtest aborts at
the first offending event boundary with enough context for the
operator to find the bug.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from threading import Lock

from nexus.core.domain.capital_state import CapitalState

from backtest_simulator.exceptions import ConservationViolation


@dataclass(frozen=True)
class CapitalTotals:
    """A snapshot of the capital components at one event boundary."""

    capital_pool: Decimal
    position_notional: Decimal
    working_order_notional: Decimal
    in_flight_order_notional: Decimal
    reservation_notional: Decimal
    fee_reserve: Decimal

    @property
    def total_deployed(self) -> Decimal:
        """Claims against `capital_pool` — all active, committed, and reserved notional."""
        return (
            self.position_notional
            + self.working_order_notional
            + self.in_flight_order_notional
            + self.reservation_notional
        )

    @property
    def total(self) -> Decimal:
        """Legacy sum-of-all for display; NOT an invariant target.

        `capital_pool + total_deployed + fee_reserve` is NOT constant in
        Nexus's ledger model because `capital_pool` includes the budget
        the deployed components already claim against. Use
        `total_deployed` and `capital_pool` separately for invariants.
        """
        return self.capital_pool + self.total_deployed + self.fee_reserve


def capital_totals(state: CapitalState) -> CapitalTotals:
    """Snapshot the current state's component totals for comparison."""
    return CapitalTotals(
        capital_pool=state.capital_pool,
        position_notional=state.position_notional,
        working_order_notional=state.working_order_notional,
        in_flight_order_notional=state.in_flight_order_notional,
        reservation_notional=state.reservation_notional,
        fee_reserve=state.fee_reserve,
    )


class _PrevPoolTracker:
    """Per-process record of the most recent `capital_pool` snapshot,
    keyed by the `id()` of the CapitalState object.

    INV-1 ("capital_pool is monotonically non-increasing") requires
    comparing new pool to PREVIOUS pool, not only to initial pool.
    Checking only initial would miss a sequence like
    `100_000 -> 99_900 -> 99_950` — the final value is still below
    initial but the middle transition grew the pool, which is a
    violation. Storing the previous value per CapitalState id
    catches that.
    """

    _lock = Lock()
    _prev: dict[int, Decimal] = {}

    @classmethod
    def snapshot_and_record(cls, state: CapitalState) -> Decimal:
        with cls._lock:
            prev = cls._prev.get(id(state), state.capital_pool)
            cls._prev[id(state)] = state.capital_pool
            return prev


def assert_conservation(
    state: CapitalState,
    initial_pool: Decimal,
    *,
    context: str,
    tolerance: Decimal = Decimal('0.01'),
) -> None:
    """Raise `ConservationViolation` on any Part 2 invariant breach.

    Checks:
      INV-1  `state.capital_pool` has not increased relative to either
             `initial_pool` OR the most recent snapshot of this state.
             Both bounds matter — a pool that grows and then shrinks
             back would pass the initial-only check but still violate
             event-to-event monotonicity.
      INV-2  No component is negative beyond `tolerance`.
      INV-3  `total_deployed <= capital_pool + tolerance` — the
             CapitalController's gating logic uses this; if it's ever
             violated we've bypassed the controller somewhere.

    `tolerance` covers rounding noise from Decimal fee arithmetic at
    the cent level. Anything bigger signals a real ledger bug.
    """
    totals = capital_totals(state)
    prev_pool = _PrevPoolTracker.snapshot_and_record(state)

    # INV-1a: capital_pool must not exceed initial_pool.
    pool_vs_initial = totals.capital_pool - initial_pool
    if pool_vs_initial > tolerance:
        msg = (
            f'conservation INV-1 violated after {context}: '
            f'capital_pool={totals.capital_pool} > initial_pool={initial_pool} '
            f'(growth={pool_vs_initial}); capital cannot appear from nothing.'
        )
        raise ConservationViolation(msg)
    # INV-1b: capital_pool must not grow from one event to the next.
    pool_vs_prev = totals.capital_pool - prev_pool
    if pool_vs_prev > tolerance:
        msg = (
            f'conservation INV-1 violated after {context}: '
            f'capital_pool={totals.capital_pool} > previous={prev_pool} '
            f'(event-to-event growth={pool_vs_prev}); capital_pool '
            f'must be monotonically non-increasing.'
        )
        raise ConservationViolation(msg)

    # INV-2: non-negative components.
    negatives: list[str] = []
    for name in (
        'capital_pool', 'position_notional', 'working_order_notional',
        'in_flight_order_notional', 'reservation_notional', 'fee_reserve',
    ):
        value = getattr(totals, name)
        if value < -tolerance:
            negatives.append(f'{name}={value}')
    if negatives:
        msg = (
            f'conservation INV-2 violated after {context}: '
            f'negative components: {", ".join(negatives)}'
        )
        raise ConservationViolation(msg)

    # INV-3: total deployed does not exceed capital_pool.
    overcommit = totals.total_deployed - totals.capital_pool
    if overcommit > tolerance:
        msg = (
            f'conservation INV-3 violated after {context}: '
            f'total_deployed={totals.total_deployed} > capital_pool={totals.capital_pool} '
            f'(overcommit={overcommit}); capital gate bypassed. '
            f'components={totals}'
        )
        raise ConservationViolation(msg)
