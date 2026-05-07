"""Per-trade R (risk) computation honest to the declared stop."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class RPerTrade:
    client_order_id: str
    side: str
    entry_price: Decimal
    declared_stop_price: Decimal | None
    qty: Decimal

def compute_r(*, entry_price: Decimal, declared_stop_price: Decimal | None, qty: Decimal) -> Decimal | None:
    return abs(entry_price - declared_stop_price) * qty
