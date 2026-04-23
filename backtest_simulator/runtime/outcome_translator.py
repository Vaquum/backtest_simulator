"""Praxis TradeOutcome -> Nexus TradeOutcome translator."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True)
class NexusOutcomeShape:
    """Minimal Nexus TradeOutcome shape used by backtest_simulator.

    Kept as a local dataclass so the translator contract is unit-testable
    without importing `nexus.infrastructure.praxis_connector.trade_outcome`;
    the NexusRuntime shim constructs the real nexus type at dispatch time
    from these fields.
    """

    outcome_id: str
    command_id: str
    outcome_type: str
    timestamp: datetime
    fill_size: Decimal
    fill_price: Decimal
    fill_notional: Decimal
    actual_fees: Decimal
    remaining_size: Decimal
    reject_reason: str | None = None
    cancel_reason: str | None = None


def translate(  # noqa: PLR0913 - schema-mapping function; one kwarg per Praxis field
    *,
    command_id: str,
    outcome_id: str,
    timestamp: datetime,
    status: str,
    filled_qty: Decimal,
    avg_fill_price: Decimal,
    actual_fees: Decimal,
    target_qty: Decimal,
    reason: str | None = None,
) -> NexusOutcomeShape:
    """Translate a Praxis-shaped TradeOutcome into the Nexus shape.

    The Nexus schema adds `outcome_id`, `fill_notional`, and
    `remaining_size`; the Praxis side ships `target_qty` + `filled_qty`
    + `avg_fill_price` + `status`. Fees are carried through from the
    submit-time estimate (see `NexusRuntime`); Praxis `TradeOutcome`
    itself does not name `actual_fees`.
    """
    outcome_type = _map_status(status)
    return NexusOutcomeShape(
        outcome_id=outcome_id,
        command_id=command_id,
        outcome_type=outcome_type,
        timestamp=timestamp,
        fill_size=filled_qty,
        fill_price=avg_fill_price,
        fill_notional=filled_qty * avg_fill_price,
        actual_fees=actual_fees,
        remaining_size=target_qty - filled_qty,
        reject_reason=reason if outcome_type == 'REJECTED' else None,
        cancel_reason=reason if outcome_type == 'CANCELLED' else None,
    )


def _map_status(praxis_status: str) -> str:
    # Simple bijection; anything we don't know about is treated as REJECTED.
    known = {
        'FILLED': 'FILL',
        'PARTIAL_FILL': 'PARTIAL_FILL',
        'CANCELLED': 'CANCELLED',
        'REJECTED': 'REJECTED',
        'ACK': 'ACK',
    }
    return known.get(praxis_status, 'REJECTED')
