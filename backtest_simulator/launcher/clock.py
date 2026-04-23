"""Accelerated-clock context manager — freezegun + asyncio.sleep hijack for backtests."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any

from freezegun import freeze_time
from freezegun.api import FrozenDateTimeFactory

_log = logging.getLogger(__name__)


@contextmanager
def accelerated_clock(start: datetime) -> Iterator[FrozenDateTimeFactory]:
    """Freeze time at `start`; every `asyncio.sleep(N)` ticks the frozen clock by N.

    The default freezegun behavior is: frozen clock is frozen, real
    wall-clock advances. With `real_asyncio=True` asyncio's event-loop
    timing uses real time, so `asyncio.sleep(3600)` sleeps 3600 real
    seconds regardless of the frozen clock — useless for a backtest.

    This context manager monkey-patches `asyncio.sleep` so the intended
    "interval has elapsed" semantics that PredictLoop / TimerLoop /
    Trading all rely on advances the FROZEN clock instead. A call to
    `asyncio.sleep(N)` inside the block:

      1. ticks the freezer forward by N seconds,
      2. yields to the event loop via `original_asyncio.sleep(0)`,
      3. returns.

    The net effect: one year of historical time compresses into
    however long it takes to actually do the per-tick work in
    PredictLoop / TimerLoop / strategy callbacks — typically seconds
    of real wall clock per year of backtest at hourly kline_size.

    The patch is process-global for the duration of the block. Every
    `asyncio.sleep` in every thread inside the block follows the
    accelerated semantics. This is deliberate: in a backtest, ALL
    time is simulated, not just the PredictLoop's interval timer.
    Trading heartbeats, event-spine flushes, and any other sleep-based
    coroutine should advance the frozen clock too.
    """
    with freeze_time(start, real_asyncio=True) as freezer:
        original_sleep = asyncio.sleep

        async def fast_sleep(delay: float, result: Any = None) -> Any:  # noqa: ANN401 - matches asyncio.sleep's Optional[T]
            if delay > 0:
                freezer.tick(timedelta(seconds=delay))
            # Yield one event-loop cycle so other scheduled coroutines
            # (handlers, callbacks) get their turn. A zero-second
            # original_sleep is the documented "yield" primitive.
            await original_sleep(0)
            return result

        asyncio.sleep = fast_sleep  # type: ignore[assignment]
        _log.info('asyncio.sleep accelerated; freezer pinned at %s', start)
        try:
            yield freezer
        finally:
            asyncio.sleep = original_sleep
            _log.info('asyncio.sleep restored to real-time semantics')
