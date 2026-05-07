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

    @property
    def r(self) -> Decimal | None:
        if self.declared_stop_price is None:
            return None
        return abs(self.entry_price - self.declared_stop_price) * self.qty

def compute_r(
    *,
    entry_price: Decimal,
    declared_stop_price: Decimal | None,
    qty: Decimal,
) -> Decimal | None:
    if declared_stop_price is None:
        return None
    return abs(entry_price - declared_stop_price) * qty
