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
from backtest_simulator.venue.fees import FeeSchedule
from backtest_simulator.venue.filters import BinanceSpotFilters
from backtest_simulator.venue.types import FillModelConfig

_LAZY_NAMES = ('install_cache', 'BacktestLauncher', 'BacktestMarketDataPoller', 'SimulatedVenueAdapter')
try:
    from backtest_simulator._limen_cache import install_cache
    from backtest_simulator.launcher import BacktestLauncher, BacktestMarketDataPoller
    from backtest_simulator.venue.simulated import SimulatedVenueAdapter
except ImportError as _e:
    _integration_error: ImportError | None = _e
    for _n in _LAZY_NAMES:
        globals().pop(_n, None)
else:
    _integration_error = None
__all__ = ['BacktestLauncher', 'BacktestMarketDataPoller', 'BinanceSpotFilters', 'ClickHouseConfig', 'ClickHouseFeed', 'ConservationViolation', 'DeterminismViolation', 'FeeSchedule', 'FillModelConfig', 'HonestyViolation', 'LookAheadViolation', 'ParityViolation', 'PerformanceViolation', 'SanityViolation', 'SimulatedVenueAdapter', 'StopContractViolation', 'install_cache']
