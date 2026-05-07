"""accelerated_clock: freezegun wrapper for backtest replay."""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

from freezegun import freeze_time
from freezegun.api import FrozenDateTimeFactory, StepTickTimeFactory, TickingDateTimeFactory

_log = logging.getLogger(__name__)
_FreezerFactory = FrozenDateTimeFactory | StepTickTimeFactory | TickingDateTimeFactory
_active_freezer: _FreezerFactory | None = None

@contextmanager
def accelerated_clock(start: datetime) -> Iterator[_FreezerFactory]:
    global _active_freezer
    with freeze_time(start, real_asyncio=True) as freezer:
        _active_freezer = freezer
        _log.info('accelerated clock engaged; freezer pinned at %s', start)
        try:
            yield freezer
        finally:
            _active_freezer = None
            _log.info('accelerated clock released')

def tick_frozen_time(freezer: FrozenDateTimeFactory, seconds: float) -> None:
    freezer.tick(timedelta(seconds=seconds))
    time.sleep(0.001)
__all__ = ['UTC', 'accelerated_clock', 'tick_frozen_time']
