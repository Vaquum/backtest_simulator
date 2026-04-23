"""Accelerated-clock context manager — freezegun + asyncio.sleep + threading.Timer hijack."""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

from freezegun import freeze_time
from freezegun.api import FrozenDateTimeFactory

_log = logging.getLogger(__name__)

# Permanent sleep patch installed at module import. The wrapper falls
# back to real-time sleeping when no freezer is active; only inside
# `accelerated_clock` does it start ticking the frozen clock and
# returning early. Installing permanently (rather than only during the
# block) catches sleeps that were scheduled BEFORE the block engaged —
# PredictLoop's interval sleep is scheduled at the moment
# `predict_loop.start()` runs, which may be well before the test code
# reaches its `with accelerated_clock(...)` line.
_original_sleep = asyncio.sleep
_active_freezer: FrozenDateTimeFactory | None = None

_POLL_INTERVAL_SECONDS: float = 0.05


async def _conditional_sleep(delay: float, result: Any = None) -> Any:  # noqa: ANN401 - asyncio.sleep signature
    if delay <= 0:
        return await _original_sleep(delay, result=result)
    # Fast path: freezer already active when the sleep is first awaited —
    # tick the full delay at once, yield via a zero-second real sleep,
    # return. This is what `PredictLoop` experiences when the backtest
    # outer loop has entered `accelerated_clock` before the tick.
    if _active_freezer is not None:
        _active_freezer.tick(timedelta(seconds=delay))
        await _original_sleep(0)
        return result
    # Slow path: freezer not yet active. Poll every
    # `_POLL_INTERVAL_SECONDS` real seconds until either a freezer
    # engages (tick the remainder, return) or the full delay elapses in
    # real time (normal behavior). The poll granularity bounds the
    # latency between a freezer engaging and a pre-scheduled sleep
    # catching it — critical for PredictLoop sleeps that were scheduled
    # during Nexus boot under real time and need to pick up the freezer
    # once the outer loop transitions into `accelerated_clock`.
    waited = 0.0
    while waited < delay:
        step = min(_POLL_INTERVAL_SECONDS, delay - waited)
        await _original_sleep(step)
        waited += step
        if _active_freezer is not None:
            remaining = delay - waited
            if remaining > 0:
                _active_freezer.tick(timedelta(seconds=remaining))
            return result
    return result


asyncio.sleep = _conditional_sleep  # type: ignore[assignment]


# ---- threading.Timer patch ---------------------------------------------------
#
# `nexus.strategy.predict_loop.PredictLoop` schedules ticks via
# `threading.Timer(interval_seconds, tick_fn)` — NOT asyncio. threading.Timer
# waits on a `threading.Event.wait(timeout)` which uses the real monotonic
# clock regardless of freezegun. Without intervention, a kline_size=3600
# Timer waits 3600 REAL seconds before firing the strategy's tick.
#
# The original Timer.run is saved at module import; during
# `accelerated_clock`, Timer.run is replaced with a version that polls
# `datetime.now(UTC)` (frozen) until enough frozen time has elapsed, then
# fires the function. Outside the block, original real-time behavior is
# restored.

_TIMER_POLL_INTERVAL_SECONDS: float = 0.005


def _frozen_aware_timer_run(self: threading.Timer) -> None:
    # Permanent patch. Accumulates elapsed time from `datetime.now(UTC)`
    # deltas rather than comparing against a fixed target timestamp.
    # That matters for Timers scheduled BEFORE `accelerated_clock`
    # engages: a target computed under real time (e.g. 2026-04-23 + 1h)
    # would be unreachable once the frozen clock takes over (jumps to
    # 2021). Delta-accumulation is robust to the forward-jump and
    # resets cleanly on any clock discontinuity.
    #
    # Outside `accelerated_clock`, `datetime.now(UTC)` is real, so
    # elapsed tracks real wall time with 5ms polling overhead.
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
    """Freeze time at `start`; engage the conditional sleep patch.

    Inside the block, `asyncio.sleep(N)` ticks the frozen clock by N
    seconds (less whatever real time was already polled) and returns.
    Outside the block, the permanent sleep wrapper falls through to
    real-time `asyncio.sleep`. Because the wrapper itself is installed
    at module import — before any Nexus coroutine runs — sleeps that
    were scheduled before the block engaged still see the freezer
    activate on their next poll iteration and catch up.
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
