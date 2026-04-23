"""Conservation invariants over the outcome stream (MVP: fill-closure + fee-accounting)."""
from __future__ import annotations

from decimal import Decimal

from backtest_simulator.exceptions import ConservationViolation
from backtest_simulator.runtime.outcome_translator import NexusOutcomeShape


def assert_fill_closure(outcomes: list[NexusOutcomeShape]) -> None:
    """Σ(fill_size) per command_id must equal the target (non-zero remaining_size => mismatch)."""
    by_command: dict[str, list[NexusOutcomeShape]] = {}
    for o in outcomes:
        by_command.setdefault(o.command_id, []).append(o)
    for cmd_id, legs in by_command.items():
        last = legs[-1]
        if last.remaining_size != 0:
            msg = f'fill closure: command {cmd_id} has remaining_size={last.remaining_size} after {len(legs)} outcome(s)'
            raise ConservationViolation(msg)


def assert_fee_nonnegative(outcomes: list[NexusOutcomeShape]) -> None:
    """Every per-fill `actual_fees` must be >= 0; strategies can't earn rebates without a rebate model."""
    for o in outcomes:
        if o.actual_fees < 0:
            msg = f'fee accounting: command {o.command_id} produced negative fee {o.actual_fees}'
            raise ConservationViolation(msg)


def sum_fees(outcomes: list[NexusOutcomeShape]) -> Decimal:
    return sum((o.actual_fees for o in outcomes), Decimal('0'))
