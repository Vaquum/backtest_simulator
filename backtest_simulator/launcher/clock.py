"""accelerated_clock: freezegun wrapper for backtest replay."""
# Frozen time is owned by `backtest_simulator.launcher.replay_clock.ReplayClock`,
# which calls `freezer.move_to(boundary)` synchronously from a single thread
# at every pre-computed kline boundary. No `threading.Timer` polling, no
# `asyncio.sleep` patch — strategy `produce_signal` calls fire when (and
# only when) `ReplayClock` invokes `PredictLoop.tick_once(wired)` (the
# public single-shot entry exposed by vaquum-nexus >= 0.41.0).
#
# The legacy `_frozen_aware_timer_run` monkey-patch on `threading.Timer.run`
# is gone. The race it tried to absorb (multiple threads writing the
# shared frozen clock) is gone with it: only `ReplayClock`'s thread ever
# moves frozen time forward.
from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

from freezegun import freeze_time
from freezegun.api import (
    FrozenDateTimeFactory,
    StepTickTimeFactory,
    TickingDateTimeFactory,
)

_log = logging.getLogger(__name__)

# `freeze_time()` yields a Union of three factory classes depending on
# tick / auto_tick_seconds. We construct it with defaults (tick=False,
# auto_tick_seconds=0) so the runtime value is always
# FrozenDateTimeFactory — but pyright reads the declared Union from the
# library signature, so the variable type must accept all three.
_FreezerFactory = FrozenDateTimeFactory | StepTickTimeFactory | TickingDateTimeFactory

_active_freezer: _FreezerFactory | None = None


@contextmanager
def accelerated_clock(start: datetime) -> Iterator[_FreezerFactory]:
    """Freeze `datetime.now(UTC)` at `start` for the duration of the block.

    `real_asyncio=True` keeps asyncio's own sleep machinery using real
    monotonic time (i.e. event-loop callback scheduling fires in real
    wall time regardless of freezegun's `time.monotonic` patch).
    `BacktestLauncher` additionally overrides the running event loop's
    `.time` method with the captured real `time.monotonic` — a
    belt-and-suspenders measure because `real_asyncio=True` is not
    reliable across every freezegun version.

    Only `datetime.now`, `time.time`, etc. are frozen (via freezegun).
    Strategy signal timestamps and feed window bounds read
    `datetime.now(UTC)` and so see frozen time. The asyncio event loop's
    internal scheduling reads `loop.time()` which we force to real
    monotonic, so `asyncio.sleep` callbacks fire in real wall time and
    the account loop polls its submit queue at the real cadence it was
    designed for.

    The `_active_freezer` module-global guards against re-entrancy:
    nesting two `accelerated_clock` blocks would interleave two
    `freeze_time` factories on the same datetime read, which is never
    a backtest shape we want.
    """
    global _active_freezer
    if _active_freezer is not None:
        msg = 'accelerated_clock is not re-entrant'
        raise RuntimeError(msg)
    with freeze_time(start, real_asyncio=True) as freezer:
        _active_freezer = freezer
        _log.info('accelerated clock engaged; freezer pinned at %s', start)
        try:
            yield freezer
        finally:
            _active_freezer = None
            _log.info('accelerated clock released')


def tick_frozen_time(freezer: FrozenDateTimeFactory, seconds: float) -> None:
    """Tick the frozen clock forward; pair with a brief real sleep to yield.

    Retained for callers outside the replay path that want a small
    relative tick rather than the absolute `freezer.move_to(target)`
    that `ReplayClock` uses.
    """
    freezer.tick(timedelta(seconds=seconds))
    time.sleep(0.001)


# `UTC` is re-exported because callers that import `accelerated_clock`
# from this module also import `UTC` from here in a few places.
__all__ = ['UTC', 'accelerated_clock', 'tick_frozen_time']
