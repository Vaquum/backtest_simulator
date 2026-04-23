"""backtest_simulator — honesty-gated backtester for unmodified Nexus strategies."""
from __future__ import annotations

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
from backtest_simulator.feed.clickhouse import ClickHouseConfig, ClickHouseFeed
from backtest_simulator.launcher import BacktestLauncher, BacktestMarketDataPoller
from backtest_simulator.venue.fees import FeeSchedule
from backtest_simulator.venue.filters import BinanceSpotFilters
from backtest_simulator.venue.simulated import SimulatedVenueAdapter
from backtest_simulator.venue.types import FillModelConfig

__all__ = [
    'BacktestLauncher',
    'BacktestMarketDataPoller',
    'BinanceSpotFilters',
    'ClickHouseConfig',
    'ClickHouseFeed',
    'ConservationViolation',
    'DeterminismViolation',
    'FeeSchedule',
    'FillModelConfig',
    'HonestyViolation',
    'LookAheadViolation',
    'ParityViolation',
    'PerformanceViolation',
    'SanityViolation',
    'SimulatedVenueAdapter',
    'StopContractViolation',
]
