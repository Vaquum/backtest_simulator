"""Accelerated-clock context manager — freezegun + threading.Timer frozen-aware wait.

Design: the main thread in `BacktestLauncher._advance_clock_until` is the
SINGLE authoritative driver of frozen time. Everyone else (asyncio
coroutines, PredictLoop Timers) reads frozen time via `datetime.now(UTC)`
but never ticks it themselves. `asyncio.sleep` is left alone; the event
loop's scheduler uses `loop.time = real_monotonic` (rebound in
`BacktestLauncher._install_real_loop_time`) so its callback firings
happen in real time regardless of freezegun's `time.monotonic` patch.
Only `threading.Timer.run` is patched, because its default
`threading.Event.wait(timeout)` is a blocking real-time wait that would
otherwise make a `kline_size=3600` Timer take 3600 real seconds; the
replacement polls `datetime.now(UTC)` (frozen) and fires when enough
frozen time has elapsed, yielding control to the main driver in between.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

from freezegun import freeze_time
from freezegun.api import FrozenDateTimeFactory

_log = logging.getLogger(__name__)

_active_freezer: FrozenDateTimeFactory | None = None

# ---- threading.Timer patch -------------------------------------------------
#
# `nexus.strategy.predict_loop.PredictLoop` schedules ticks via
# `threading.Timer(interval_seconds, tick_fn)` — NOT asyncio.
# `threading.Timer` inherits from `threading.Thread` and its `run` waits
# on a `threading.Event.wait(timeout)` which uses the real monotonic
# clock regardless of freezegun. Without intervention, a `kline_size=3600`
# Timer would wait 3600 REAL seconds before firing the strategy's tick.
#
# This patch replaces `Timer.run` with a version that polls
# `datetime.now(UTC)` (which IS frozen under freezegun) until the
# Timer's interval has elapsed in frozen time, then fires. The poll
# loop does NOT tick the frozen clock — it waits for the main driver
# (`BacktestLauncher._advance_clock_until`) to advance it.

_TIMER_POLL_INTERVAL_SECONDS: float = 0.005


def _frozen_aware_timer_run(self: threading.Timer) -> None:
    # Accumulate elapsed time from `datetime.now(UTC)` deltas rather
    # than comparing against a fixed target timestamp. That matters for
    # Timers scheduled BEFORE `accelerated_clock` engages: a target
    # computed under real time (e.g. 2026-04-24 + 1h) would be
    # unreachable once the frozen clock takes over (jumps to 2026-04-07).
    # Delta-accumulation is robust to the forward-jump and resets
    # cleanly on any clock discontinuity.
    #
    # Outside `accelerated_clock`, `datetime.now(UTC)` is real, so
    # elapsed tracks real wall time with ~5ms polling overhead.
    interval = timedelta(seconds=self.interval)
    elapsed = timedelta(0)
    last = datetime.now(UTC)
    while elapsed < interval:
        if self.finished.wait(_TIMER_POLL_INTERVAL_SECONDS):
            return
        now = datetime.now(UTC)
        if now < last:
            # Clock jumped backwards (freeze_time transition); reset
            # baseline and keep accumulating from there.
            last = now
        else:
            elapsed += now - last
            last = now
    if not self.finished.is_set():
        self.function(*self.args, **self.kwargs)  # type: ignore[attr-defined]
    self.finished.set()


threading.Timer.run = _frozen_aware_timer_run  # type: ignore[method-assign]


@contextmanager
def accelerated_clock(start: datetime) -> Iterator[FrozenDateTimeFactory]:
    """Freeze `datetime.now(UTC)` at `start`; engage the Timer-frozen wait.

    `real_asyncio=True` keeps asyncio's own sleep machinery using real
    monotonic time (i.e. event-loop callback scheduling fires in real
    wall time regardless of freezegun's `time.monotonic` patch).
    `BacktestLauncher` additionally overrides the running event loop's
    `.time` method with the captured real `time.monotonic` — a
    belt-and-suspenders measure because `real_asyncio=True` is not
    reliable across every freezegun version.

    Only `datetime.now`, `time.time`, etc. are frozen (via freezegun).
    Strategy signal timestamps, feed window bounds, `PredictLoop` Timer
    firings — all of these read `datetime.now(UTC)` and so see frozen
    time. The asyncio event loop's internal scheduling reads
    `loop.time()` which we force to real monotonic, so `asyncio.sleep`
    callbacks fire in real wall time and the account loop polls its
    submit queue at the real cadence it was designed for.
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


# ---- runtime helper ---------------------------------------------------------


def tick_frozen_time(freezer: FrozenDateTimeFactory, seconds: float) -> None:
    """Tick the frozen clock forward; pair with a brief real sleep to yield."""
    freezer.tick(timedelta(seconds=seconds))
    time.sleep(0.001)
