"""accelerated_clock: asyncio.sleep(N) ticks the frozen clock by N seconds."""
from __future__ import annotations

import asyncio
import itertools
import time
from datetime import UTC, datetime, timedelta

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


def test_sleep_restoration_after_block() -> None:
    original = asyncio.sleep
    start = datetime(2021, 1, 1, tzinfo=UTC)

    with accelerated_clock(start):
        assert asyncio.sleep is not original

    assert asyncio.sleep is original


def test_exception_inside_block_still_restores_sleep() -> None:
    original = asyncio.sleep
    start = datetime(2021, 1, 1, tzinfo=UTC)

    try:
        with accelerated_clock(start):
            raise RuntimeError('injected')
    except RuntimeError:
        pass

    assert asyncio.sleep is original


def test_sleep_returns_configured_result() -> None:
    start = datetime(2021, 1, 1, tzinfo=UTC)

    async def take_result() -> str:
        return await asyncio.sleep(5, result='sentinel')

    with accelerated_clock(start):
        value = asyncio.run(take_result())

    assert value == 'sentinel'
