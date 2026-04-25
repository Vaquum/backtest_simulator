"""Conservation: fill-closure and fee-nonnegative invariants fire on violation."""
from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from nexus.infrastructure.praxis_connector.trade_outcome import TradeOutcome
from nexus.infrastructure.praxis_connector.trade_outcome_type import TradeOutcomeType

from backtest_simulator.exceptions import ConservationViolation
from backtest_simulator.reporting.ledger import (
    assert_fee_nonnegative,
    assert_fill_closure,
    sum_fees,
)


def _outcome(command_id: str, remaining: Decimal, fees: Decimal = Decimal('1')) -> TradeOutcome:
    return TradeOutcome(
        outcome_id=f'O-{command_id}', command_id=command_id,
        outcome_type=TradeOutcomeType.FILLED,
        timestamp=datetime(2020, 4, 1, tzinfo=UTC),
        fill_size=Decimal('0.1'), fill_price=Decimal('7000'),
        fill_notional=Decimal('700'), actual_fees=fees,
        remaining_size=remaining,
    )


def test_fill_closure_passes_on_complete_fill() -> None:
    assert_fill_closure([_outcome('C1', remaining=Decimal('0'))])


def test_fill_closure_fires_on_incomplete_fill() -> None:
    with pytest.raises(ConservationViolation):
        assert_fill_closure([_outcome('C1', remaining=Decimal('0.05'))])


def test_fill_closure_uses_last_leg_per_command() -> None:
    legs = [
        _outcome('C2', remaining=Decimal('0.05')),
        _outcome('C2', remaining=Decimal('0')),
    ]
    assert_fill_closure(legs)  # final leg has 0 remaining -> closed


def test_fee_nonnegative_passes_on_zero_or_positive() -> None:
    assert_fee_nonnegative([
        _outcome('C1', Decimal('0'), fees=Decimal('0')),
        _outcome('C2', Decimal('0'), fees=Decimal('2.5')),
    ])


def test_fee_nonnegative_fires_on_negative_fee() -> None:
    # TradeOutcome's own __post_init__ validates actual_fees >= 0 for fill
    # outcomes, so we can't construct a negative-fee fill via the real type.
    # Build an ACK outcome (non-fill, so fees are allowed absent) and
    # monkey-patch `actual_fees` post-construction via object.__setattr__
    # to land a negative that the ledger pass must still catch.
    ack = TradeOutcome(
        outcome_id='O-C1', command_id='C1',
        outcome_type=TradeOutcomeType.ACK,
        timestamp=datetime(2020, 4, 1, tzinfo=UTC),
    )
    object.__setattr__(ack, 'actual_fees', Decimal('-0.5'))
    with pytest.raises(ConservationViolation):
        assert_fee_nonnegative([ack])


def test_sum_fees_sums_fees_across_outcomes() -> None:
    total = sum_fees([
        _outcome('C1', Decimal('0'), fees=Decimal('1')),
        _outcome('C2', Decimal('0'), fees=Decimal('2')),
        _outcome('C3', Decimal('0'), fees=Decimal('3')),
    ])
    assert total == Decimal('6')
