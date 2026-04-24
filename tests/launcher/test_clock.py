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
from datetime import UTC, datetime, timedelta

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


def test_timer_fires_at_epoch_aligned_boundary() -> None:
    # Under `accelerated_clock`, `datetime.now(UTC)` is frozen. The
    # patched Timer waits for the NEXT epoch-aligned interval boundary
    # and fires there, not at `start + interval`. At start=02:23:15
    # with interval=3600s, the first boundary is 03:00:00. Epoch
    # alignment removes per-run drift and makes strategy tick times
    # deterministic across runs.
    #
    # Deterministic driver: we advance exactly two frozen-clock steps —
    # first to `target - 1 second` (Timer must not fire yet), then
    # across the boundary. Between steps we sleep well past the Timer
    # poll interval so the Timer thread observes each frozen state at
    # most one poll late. `timer.join()` then deterministically waits
    # for the firing callback to record its timestamp.
    start = datetime(2021, 1, 1, 2, 23, 15, tzinfo=UTC)
    fired_at: list[datetime] = []

    def on_fire() -> None:
        fired_at.append(datetime.now(UTC))

    boundary = datetime(2021, 1, 1, 3, 0, 0, tzinfo=UTC)
    poll_slack = 0.05  # >> _TIMER_POLL_INTERVAL_SECONDS (5ms)
    with accelerated_clock(start) as freezer:
        timer = threading.Timer(3600.0, on_fire)
        timer.start()
        # Step 1: park just before the boundary. Timer must still be
        # waiting; give it a full poll-slack to confirm it has NOT fired.
        freezer.tick(boundary - timedelta(seconds=1) - start)
        time.sleep(poll_slack)
        assert not fired_at, f'Timer fired before boundary: {fired_at}'
        # Step 2: cross the boundary in a single 1-second tick. Within
        # one poll the Timer sees `now >= target` and fires.
        freezer.tick(timedelta(seconds=1))
        timer.join(timeout=2.0)

    assert fired_at, 'Timer.on_fire never ran'
    # Fires AT 03:00:00 UTC — the exact epoch-aligned 3600s boundary.
    # No overshoot budget is needed because step 2's single second
    # lands `datetime.now(UTC)` precisely at the boundary.
    assert fired_at[0] == boundary, fired_at[0]


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
