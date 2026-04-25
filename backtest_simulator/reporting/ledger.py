"""Conservation invariants over the Nexus TradeOutcome stream."""
from __future__ import annotations

from decimal import Decimal

from nexus.infrastructure.praxis_connector.trade_outcome import TradeOutcome

from backtest_simulator.exceptions import ConservationViolation


def assert_fill_closure(outcomes: list[TradeOutcome]) -> None:
    """Σ(fill_size) per command_id must equal the target (non-zero remaining_size => mismatch)."""
    by_command: dict[str, list[TradeOutcome]] = {}
    for o in outcomes:
        by_command.setdefault(o.command_id, []).append(o)
    for cmd_id, legs in by_command.items():
        last = legs[-1]
        remaining = last.remaining_size
        if remaining is not None and remaining != Decimal('0'):
            msg = f'fill closure: command {cmd_id} has remaining_size={remaining} after {len(legs)} outcome(s)'
            raise ConservationViolation(msg)


def assert_fee_nonnegative(outcomes: list[TradeOutcome]) -> None:
    """Every per-fill `actual_fees` must be >= 0; strategies can't earn rebates without a rebate model."""
    for o in outcomes:
        fees = o.actual_fees
        if fees is not None and fees < Decimal('0'):
            msg = f'fee accounting: command {o.command_id} produced negative fee {fees}'
            raise ConservationViolation(msg)


def sum_fees(outcomes: list[TradeOutcome]) -> Decimal:
    return sum(
        (o.actual_fees for o in outcomes if o.actual_fees is not None),
        Decimal('0'),
    )
