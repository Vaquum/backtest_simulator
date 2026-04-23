"""BacktestEnvironment — bundles freezegun + feed + venue + runtime + driver."""
from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final

from freezegun import freeze_time

from backtest_simulator.driver import SimulationDriver
from backtest_simulator.feed.protocol import HistoricalFeed
from backtest_simulator.runtime.nexus_runtime import NexusRuntime
from backtest_simulator.runtime.orjson_patch import apply as apply_orjson_patch
from backtest_simulator.venue.simulated import SimulatedVenueAdapter

BTS_PERF_GATE_MULTIPLIER_ENV: Final[str] = 'BTS_PERF_GATE_MULTIPLIER'
BTS_PERF_GATE_DEFAULT_MULTIPLIER: Final[float] = 1.0


def perf_gate_multiplier() -> float:
    """Reads BTS_PERF_GATE_MULTIPLIER.

    Default 1.0 on reference hardware. CI workflows on non-reference
    runners may override. MUST NOT be relaxed on merges to main — any
    value that widens the gate has to show up in the PR diff.
    """
    raw = os.environ.get(BTS_PERF_GATE_MULTIPLIER_ENV)
    if raw is None:
        return BTS_PERF_GATE_DEFAULT_MULTIPLIER
    try:
        return float(raw)
    except ValueError:
        return BTS_PERF_GATE_DEFAULT_MULTIPLIER


@dataclass
class BacktestEnvironment:
    """Context manager that boots freezegun + patches + driver for one run.

    - Enters freezegun with `real_asyncio=True` (SPEC §5.1.2 #9).
      Without it, `asyncio.sleep` becomes a no-op and any account-loop
      drain deadlocks.
    - Applies the orjson/FakeDatetime monkey-patch so event-spine
      serialization doesn't trip on freezegun subclasses.
    - Exits cleanly on any exception — freezegun is always released.
    """

    feed: HistoricalFeed
    venue: SimulatedVenueAdapter
    runtime: NexusRuntime
    driver: SimulationDriver
    start: datetime
    _tick_hooks: list[Callable[[datetime], Awaitable[None]]] = field(default_factory=list)

    @contextmanager
    def frozen(self) -> Iterator[Any]:
        apply_orjson_patch()
        with freeze_time(self.start, real_asyncio=True) as freezer:
            yield freezer
