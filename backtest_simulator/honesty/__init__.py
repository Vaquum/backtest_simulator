"""Honesty gates — the mechanical invariants of Part 2."""
from __future__ import annotations

# `capital` owns the real-CAPITAL ValidationPipeline and a backtest-
# driven CapitalController whose 4-step lifecycle
# (`check_and_reserve → send_order → order_ack → order_fill`) is called
# by the `BacktestLauncher` as orders flow through Praxis and return
# as fills. `conservation` owns the per-event invariant checks; any
# drift between `capital_pool`, `position_notional`, `working_order_notional`,
# `in_flight_order_notional`, and `reservation_notional` raises
# `ConservationViolation` so the backtest aborts the instant the
# capital books stop balancing.
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
