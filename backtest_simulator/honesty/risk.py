"""Per-trade R (risk) computation honest to the declared stop.

Part 2 invariant (Issue #10): r_per_trade must come from the DECLARED
stop price attached to the order at entry, never from a virtual
`stop_bps` computed after the fact on a different price. If the
venue adapter's `FillModel.apply_stop` closes at the stop, the
realised loss on that trade equals this R; if the trade exits via
a subsequent SELL, the strategy *could* still run to a worse price,
but R records the intended risk at entry, not the outcome.

R = |entry_price - stop_price| * qty

For a long-only entry with `stop_price < entry_price`:
  R = (entry - stop) * qty
For a short entry with `stop_price > entry_price`:
  R = (stop - entry) * qty

An entry that lacks a declared stop produces R = None and the
strategy is honestly flagged. The Part 2 INTAKE gate in
`action_submitter` rejects such entries before they land, so
`R is None` on a BUY fill means the gate was bypassed somehow —
that's a violation to escalate.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class RPerTrade:
    """Per-trade realised risk anchored on the declared stop."""

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
    """Return `|entry - stop| * qty`, or `None` if no stop was declared."""
    if declared_stop_price is None:
        return None
    return abs(entry_price - declared_stop_price) * qty
