"""backtest_simulator — honesty-gated backtester for Nexus strategies."""
from __future__ import annotations

from backtest_simulator.driver import SimulationDriver
from backtest_simulator.environment import BacktestEnvironment
from backtest_simulator.exceptions import (
    ConservationViolation,
    DeterminismViolation,
    HonestyViolation,
    LookAheadViolation,
    ParityViolation,
    PerformanceViolation,
    SanityViolation,
    StopContractViolation,
)
from backtest_simulator.launcher import BacktestLauncher

__all__ = [
    'BacktestEnvironment',
    'BacktestLauncher',
    'ConservationViolation',
    'DeterminismViolation',
    'HonestyViolation',
    'LookAheadViolation',
    'ParityViolation',
    'PerformanceViolation',
    'SanityViolation',
    'SimulationDriver',
    'StopContractViolation',
]
