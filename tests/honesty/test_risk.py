"""Part 2 tests for `backtest_simulator.honesty.risk.compute_r`.

R is the honest |entry - stop| * qty metric. The test pins:
  - R is positive and deterministic for a BUY with stop below entry.
  - R is positive and symmetric for a SELL-style with stop above entry.
  - R is None when no stop was declared.
  - `RPerTrade.r` uses the same formula (identity property).
"""
from __future__ import annotations

from decimal import Decimal

from backtest_simulator.honesty.risk import RPerTrade, compute_r


def test_compute_r_long_entry() -> None:
    r = compute_r(
        entry_price=Decimal('100.0'),
        declared_stop_price=Decimal('99.0'),
        qty=Decimal('10'),
    )
    assert r == Decimal('10.0')  # |100-99| * 10


def test_compute_r_symmetric_for_short_style_stop() -> None:
    # For a short-style entry, stop_price is above entry — R still
    # uses |entry - stop| so the sign doesn't matter.
    r = compute_r(
        entry_price=Decimal('100.0'),
        declared_stop_price=Decimal('102.0'),
        qty=Decimal('5'),
    )
    assert r == Decimal('10.0')  # |100-102| * 5


def test_compute_r_none_when_no_stop() -> None:
    assert compute_r(
        entry_price=Decimal('100'),
        declared_stop_price=None,
        qty=Decimal('1'),
    ) is None


def test_compute_r_zero_qty_is_zero_not_none() -> None:
    # A zero-qty order with a declared stop has zero R, not None —
    # the distinction "no stop declared" vs "zero size" matters.
    r = compute_r(
        entry_price=Decimal('100'),
        declared_stop_price=Decimal('99'),
        qty=Decimal('0'),
    )
    assert r == Decimal('0')


def test_rpertrade_r_property_matches_compute_r() -> None:
    rec = RPerTrade(
        client_order_id='SS-xxx-000',
        side='BUY',
        entry_price=Decimal('100'),
        declared_stop_price=Decimal('99'),
        qty=Decimal('2'),
    )
    assert rec.r == Decimal('2.0')  # same formula
    assert rec.r == compute_r(
        entry_price=rec.entry_price,
        declared_stop_price=rec.declared_stop_price,
        qty=rec.qty,
    )


def test_rpertrade_r_none_without_stop() -> None:
    rec = RPerTrade(
        client_order_id='SS-xxx-000',
        side='SELL',
        entry_price=Decimal('100'),
        declared_stop_price=None,
        qty=Decimal('2'),
    )
    assert rec.r is None
