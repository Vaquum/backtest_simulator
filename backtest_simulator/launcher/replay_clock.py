"""Schedule-driven replay clock — single-thread driver for backtest replay."""
from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from nexus.startup.sequencer import WiredSensor

    class _FreezerFactory(Protocol):
        def move_to(self, target_datetime: datetime, /) -> None:
            del target_datetime

_log = logging.getLogger(__name__)
_REAL_TIME_CAP_SECONDS_DEFAULT = 600.0

class _PredictLoop(Protocol):

    @property
    def running(self) -> bool:
        ...

    def tick_once(self, wired: WiredSensor) -> None:
        ...

class _OutcomeLoop(Protocol):

    @property
    def running(self) -> bool:
        ...

    def tick_once(self) -> bool:
        ...

def compute_kline_boundaries(*, window_start: datetime, window_end: datetime, interval_seconds: int) -> list[datetime]:
    epoch = datetime(1970, 1, 1, tzinfo=window_start.tzinfo)
    elapsed = int((window_start - epoch).total_seconds())
    next_boundary_secs = (elapsed // interval_seconds + 1) * interval_seconds
    boundaries: list[datetime] = []
    t = epoch + timedelta(seconds=next_boundary_secs)
    while t <= window_end:
        boundaries.append(t)
        t += timedelta(seconds=interval_seconds)
    return boundaries

class ReplayDeadlineExceededError(RuntimeError):
    pass

@dataclass(frozen=True)
class ReplayClock:
    real_time_cap_seconds: float = _REAL_TIME_CAP_SECONDS_DEFAULT

    def drive_window(self, *, window_start: datetime, window_end: datetime, wired_sensors: Sequence[WiredSensor], predict_loop: _PredictLoop, outcome_loop: _OutcomeLoop, drain_pending_submits: Callable[[], None], freezer: _FreezerFactory) -> None:
        intervals: set[int] = set()
        for wired in wired_sensors:
            interval = getattr(wired, 'interval_seconds', None)
            assert interval is not None
            intervals.add(int(interval))
        interval_seconds = next(iter(intervals))
        boundaries = compute_kline_boundaries(window_start=window_start, window_end=window_end, interval_seconds=interval_seconds)
        _log.info('replay_clock: %d boundaries scheduled (window=[%s, %s], interval=%ds, sensors=%d, cap=%.1fs)', len(boundaries), window_start, window_end, interval_seconds, len(wired_sensors), self.real_time_cap_seconds)
        deadline = os.times()[4] + self.real_time_cap_seconds
        for boundary in boundaries:
            freezer.move_to(boundary)
            for wired in wired_sensors:
                predict_loop.tick_once(wired)
            self._drain_to_quiescence(outcome_loop, drain_pending_submits, deadline)
        freezer.move_to(window_end)
        self._drain_to_quiescence(outcome_loop, drain_pending_submits, deadline)

    @staticmethod
    def _drain_to_quiescence(outcome_loop: _OutcomeLoop, drain_pending_submits: Callable[[], None], deadline: float) -> None:
        while True:
            drain_pending_submits()
            consumed = False
            while outcome_loop.tick_once():
                consumed = True
            if not consumed:
                break
            if os.times()[4] > deadline:
                msg = (
                    'ReplayClock: drain-to-quiescence exceeded real-time cap. '
                    'Likely cyclic on_outcome chain (action emits new action '
                    'whose fill emits another). Tighten the strategy or raise '
                    'real_time_cap_seconds.'
                )
                raise ReplayDeadlineExceededError(msg)
