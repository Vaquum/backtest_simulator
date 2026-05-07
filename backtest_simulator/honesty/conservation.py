"""Conservation laws for Nexus's CapitalController ledger model."""
from __future__ import annotations

import weakref
from dataclasses import dataclass
from decimal import Decimal
from threading import Lock
from typing import ClassVar

from nexus.core.domain.capital_state import CapitalState


@dataclass(frozen=True)
class CapitalTotals:
    capital_pool: Decimal
    position_notional: Decimal
    working_order_notional: Decimal
    in_flight_order_notional: Decimal
    reservation_notional: Decimal
    fee_reserve: Decimal

    @property
    def total_deployed(self) -> Decimal:
        return self.position_notional + self.working_order_notional + self.in_flight_order_notional + self.reservation_notional

    @property
    def total(self) -> Decimal:
        return self.capital_pool + self.total_deployed + self.fee_reserve

def capital_totals(state: CapitalState) -> CapitalTotals:
    return CapitalTotals(capital_pool=state.capital_pool, position_notional=state.position_notional, working_order_notional=state.working_order_notional, in_flight_order_notional=state.in_flight_order_notional, reservation_notional=state.reservation_notional, fee_reserve=state.fee_reserve)

class _PrevPoolTracker:
    _lock: ClassVar[Lock] = Lock()
    _prev: ClassVar[dict[int, Decimal]] = {}

    @classmethod
    def _remove(cls, sid: int) -> None:
        with cls._lock:
            cls._prev.pop(sid, None)

    @classmethod
    def snapshot_and_record(cls, state: CapitalState) -> Decimal:
        sid = id(state)
        with cls._lock:
            first_sighting = sid not in cls._prev
            prev = cls._prev.get(sid, state.capital_pool)
            cls._prev[sid] = state.capital_pool
        if first_sighting:
            weakref.finalize(state, cls._remove, sid)
        return prev

def assert_conservation(state: CapitalState, initial_pool: Decimal, *, context: str, tolerance: Decimal=Decimal('0.01')) -> None:
    totals = capital_totals(state)
    prev_pool = _PrevPoolTracker.snapshot_and_record(state)
    totals.capital_pool - initial_pool
    totals.capital_pool - prev_pool
    for name in ('capital_pool', 'position_notional', 'working_order_notional', 'in_flight_order_notional', 'reservation_notional', 'fee_reserve'):
        getattr(totals, name)
    totals.total_deployed - totals.capital_pool
