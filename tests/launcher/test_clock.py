"""accelerated_clock: asyncio.sleep(N) ticks the frozen clock by N seconds."""
from __future__ import annotations

import asyncio
import itertools
import time
from datetime import UTC, datetime, timedelta

import pytest

from backtest_simulator.launcher.clock import accelerated_clock


def test_asyncio_sleep_advances_frozen_clock() -> None:
    start = datetime(2021, 1, 1, tzinfo=UTC)

    async def tick() -> datetime:
        await asyncio.sleep(3600)
        return datetime.now(UTC)

    real_start = time.monotonic()
    with accelerated_clock(start):
        after_one_tick = asyncio.run(tick())
    real_elapsed = time.monotonic() - real_start

    assert after_one_tick == start + timedelta(hours=1), after_one_tick
    # 3600 frozen seconds must compress into well under 1 real second.
    assert real_elapsed < 1.0, f'took {real_elapsed:.2f}s real time'


def test_multiple_sleeps_compose() -> None:
    start = datetime(2021, 6, 15, 12, 0, tzinfo=UTC)

    async def drive() -> list[datetime]:
        out = [datetime.now(UTC)]
        for _ in range(24):
            await asyncio.sleep(3600)
            out.append(datetime.now(UTC))
        return out

    with accelerated_clock(start):
        timestamps = asyncio.run(drive())

    assert timestamps[0] == start
    assert timestamps[-1] == start + timedelta(hours=24)
    # Monotone strictly increasing — one hour per tick.
    for prev, nxt in itertools.pairwise(timestamps):
        assert nxt - prev == timedelta(hours=1)


def test_zero_sleep_does_not_tick() -> None:
    start = datetime(2021, 1, 1, tzinfo=UTC)

    async def tick() -> datetime:
        await asyncio.sleep(0)
        return datetime.now(UTC)

    with accelerated_clock(start):
        now = asyncio.run(tick())

    assert now == start, f'zero-sleep moved clock to {now}'


def test_asyncio_sleep_is_permanently_wrapped() -> None:
    # The conditional wrapper is installed at module import — not restored
    # on context-manager exit — so sleeps scheduled BEFORE a freezer
    # engages still pick it up once it does. The wrapper falls through
    # to real-time behavior when no freezer is active.
    from backtest_simulator.launcher.clock import _conditional_sleep
    assert asyncio.sleep is _conditional_sleep


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


def test_sleep_returns_configured_result() -> None:
    start = datetime(2021, 1, 1, tzinfo=UTC)

    async def take_result() -> str:
        return await asyncio.sleep(5, result='sentinel')

    with accelerated_clock(start):
        value = asyncio.run(take_result())

    assert value == 'sentinel'


def test_threading_timer_is_permanently_patched() -> None:
    # The Timer.run patch is installed at module import (not just inside
    # accelerated_clock) so Timers scheduled BEFORE a freezer engages
    # still fire based on frozen-time delta accumulation once the block
    # opens. PredictLoop schedules its tick Timer during boot under real
    # time; that Timer has to keep polling frozen-time once the backtest
    # transitions to accelerated_clock, or no tick ever fires.
    import threading

    from backtest_simulator.launcher.clock import _frozen_aware_timer_run
    assert threading.Timer.run is _frozen_aware_timer_run


def test_pre_scheduled_sleep_catches_late_freezer() -> None:
    # The load-bearing guarantee for BacktestLauncher: a sleep scheduled
    # BEFORE `accelerated_clock` engages must still be picked up by the
    # freezer on its next poll once the block opens. This is what lets
    # PredictLoop start under real time (avoiding the Trainer SSL issue)
    # and then have its pending interval sleep catch the freezer when
    # the outer loop transitions to accelerated time.
    start = datetime(2021, 1, 1, tzinfo=UTC)

    async def driver() -> datetime:
        # Start a long sleep BEFORE engaging the freezer. The task is
        # already awaiting `_conditional_sleep(60)` in its poll loop
        # when the freezer below engages; the next poll iteration picks
        # it up and ticks the remainder.
        async def sleep_task() -> datetime:
            await asyncio.sleep(60)
            return datetime.now(UTC)

        task = asyncio.create_task(sleep_task())
        # Yield enough that the task hits its first poll step (50ms).
        await asyncio.sleep(0)
        with accelerated_clock(start):
            return await task

    real_start = time.monotonic()
    observed = asyncio.run(driver())
    real_elapsed = time.monotonic() - real_start

    # The frozen clock should have advanced by ~60s, not by the real
    # seconds waited.
    elapsed_frozen = observed - start
    assert elapsed_frozen >= timedelta(seconds=59)
    assert elapsed_frozen <= timedelta(seconds=61)
    # Real wall clock should be well under the 60-second sleep.
    assert real_elapsed < 5.0, f'real elapsed {real_elapsed:.2f}s > 5s'
