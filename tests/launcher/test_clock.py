"""accelerated_clock: main-thread freezer drive + frozen-aware threading.Timer.

Design note: `clock.py` deliberately DOES NOT patch `asyncio.sleep`.
Main is the sole driver of frozen time; asyncio and Timer threads are
readers that wait for frozen time to reach their targets. The previous
`_conditional_sleep` patch compounded frozen-time advancement with
main's ticks and produced huge drift, which made the e2e flaky. Only
`threading.Timer.run` is patched — the backtest's sensor PredictLoop
schedules ticks via `threading.Timer`, so that's the seam that needs
frozen-aware waiting.
"""
from __future__ import annotations

import threading
import time
from datetime import UTC, datetime

import pytest

from backtest_simulator.launcher.clock import (
    _frozen_aware_timer_run,
    accelerated_clock,
)


def test_accelerated_clock_pins_datetime() -> None:
    start = datetime(2021, 1, 1, tzinfo=UTC)
    with accelerated_clock(start):
        assert datetime.now(UTC) == start


def test_accelerated_clock_is_not_reentrant() -> None:
    start = datetime(2021, 1, 1, tzinfo=UTC)
    with accelerated_clock(start), pytest.raises(RuntimeError, match='not re-entrant'):
        with accelerated_clock(start):
            pass


def test_exception_inside_block_releases_freezer() -> None:
    start = datetime(2021, 1, 1, tzinfo=UTC)
    with pytest.raises(RuntimeError, match='injected'):
        with accelerated_clock(start):
            raise RuntimeError('injected')
    # If the freezer was released, the next entry succeeds.
    with accelerated_clock(start):
        assert datetime.now(UTC) == start


def test_threading_timer_is_permanently_patched() -> None:
    # The Timer.run patch is installed at module import (not just inside
    # accelerated_clock) so Timers scheduled BEFORE a freezer engages
    # still fire based on frozen-time once the block opens. PredictLoop
    # schedules its tick Timer during boot under real time; that Timer
    # has to keep polling frozen-time once the backtest transitions to
    # accelerated_clock, or no tick ever fires.
    assert threading.Timer.run is _frozen_aware_timer_run


def test_timer_cancel_stops_wait() -> None:
    start = datetime(2021, 1, 1, tzinfo=UTC)
    ran = threading.Event()

    def on_fire() -> None:
        ran.set()

    with accelerated_clock(start):
        timer = threading.Timer(3600.0, on_fire)
        timer.start()
        timer.cancel()
        # Give the wait loop a chance to observe the cancel.
        time.sleep(0.05)

    assert not ran.is_set(), 'Timer fired despite being cancelled'
