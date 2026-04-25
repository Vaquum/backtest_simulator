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

# Integration re-exports require the `[integration]` extra (Praxis +
# Nexus + Limen). Try eager so `_limen_cache`'s side-effect (installing
# the Limen klines patch) fires on package import. On a slim install
# the imports fail and accessing one of the integration names raises
# `ImportError` with install guidance via `__getattr__` below — the
# pure-Python surface above stays available either way.
_LAZY_NAMES = (
    'install_cache', 'BacktestLauncher',
    'BacktestMarketDataPoller', 'SimulatedVenueAdapter',
)
try:
    from backtest_simulator._limen_cache import install_cache
    from backtest_simulator.launcher import BacktestLauncher, BacktestMarketDataPoller
    from backtest_simulator.venue.simulated import SimulatedVenueAdapter
except ImportError as _e:
    _INTEGRATION_ERROR: ImportError | None = _e
    # Drop any names that succeeded before the failed import. Without
    # this cleanup a partial-integration env (e.g. limen installed but
    # praxis absent) leaves the earlier name in globals and bypasses
    # `__getattr__`'s ImportError path — `bs.install_cache` would
    # return the live function while `bs.BacktestLauncher` correctly
    # raises, an inconsistent surface. `dict.pop(name, None)` is
    # idempotent for names that were never assigned.
    for _n in _LAZY_NAMES:
        globals().pop(_n, None)
    del _n
else:
    _INTEGRATION_ERROR = None


def __getattr__(name: str) -> object:
    if _INTEGRATION_ERROR is not None and name in _LAZY_NAMES:
        msg = (
            f'backtest_simulator.{name} requires the [integration] extra. '
            f'Install: pip install backtest_simulator[integration]. '
            f'Underlying ImportError: {_INTEGRATION_ERROR}'
        )
        raise ImportError(msg) from _INTEGRATION_ERROR
    raise AttributeError(f'module backtest_simulator has no attribute {name!r}')


__all__ = [
    'BacktestLauncher', 'BacktestMarketDataPoller', 'BinanceSpotFilters',
    'ClickHouseConfig', 'ClickHouseFeed', 'ConservationViolation',
    'DeterminismViolation', 'FeeSchedule', 'FillModelConfig',
    'HonestyViolation', 'LookAheadViolation', 'ParityViolation',
    'PerformanceViolation', 'SanityViolation', 'SimulatedVenueAdapter',
    'StopContractViolation', 'install_cache',
]
