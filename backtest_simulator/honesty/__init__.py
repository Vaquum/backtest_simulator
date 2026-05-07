"""Honesty gates — the mechanical invariants of Part 2."""
from __future__ import annotations

from backtest_simulator.honesty.capital import (
    CapitalLifecycleTracker,
    build_validation_pipeline,
)
from backtest_simulator.honesty.conservation import (
    assert_conservation,
    capital_totals,
)
from backtest_simulator.honesty.risk import (
    RPerTrade,
    compute_r,
)

__all__ = [
    'CapitalLifecycleTracker',
    'RPerTrade',
    'assert_conservation',
    'build_validation_pipeline',
    'capital_totals',
    'compute_r',
]
