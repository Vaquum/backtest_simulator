"""accelerated_clock: freezegun wrapper used by the schedule-driven replay path.

Frozen-time advancement is owned by
`backtest_simulator.launcher.replay_clock.ReplayClock`. The legacy
`threading.Timer.run` monkey-patch is gone, so the two tests that pinned
that monkey-patch (`test_threading_timer_is_permanently_patched`,
`test_timer_cancel_stops_wait`) are gone with it; the remaining tests
exercise the freezegun wrapper itself.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from backtest_simulator.launcher.clock import accelerated_clock


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
