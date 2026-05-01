from decimal import Decimal

import pytest

from backtest_simulator.cli._run_window import (
    _effective_kelly_pct_for_allocation,
)
from backtest_simulator.launcher.action_submitter import (
    NOTIONAL_RESERVATION_BUFFER,
)


def test_effective_kelly_unchanged_without_allocation_cap() -> None:
    assert _effective_kelly_pct_for_allocation(
        Decimal('20.71'), None,
    ) == Decimal('20.71')


def test_effective_kelly_unchanged_below_allocation_cap() -> None:
    assert _effective_kelly_pct_for_allocation(
        Decimal('10'), Decimal('0.4'),
    ) == Decimal('10')


def test_effective_kelly_caps_reserved_notional_at_allocation_cap() -> None:
    capped = _effective_kelly_pct_for_allocation(
        Decimal('50'), Decimal('0.4'),
    )

    assert capped < Decimal('40')
    assert (
        capped
        * (Decimal('1') + NOTIONAL_RESERVATION_BUFFER)
        / Decimal('100')
    ) <= Decimal('0.4')


def test_effective_kelly_rejects_non_positive_allocation_cap() -> None:
    with pytest.raises(ValueError, match='must be positive'):
        _effective_kelly_pct_for_allocation(
            Decimal('20.71'), Decimal('0'),
        )
