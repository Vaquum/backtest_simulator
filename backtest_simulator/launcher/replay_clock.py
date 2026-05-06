"""Schedule-driven replay clock — single-thread driver for backtest replay."""
# Replaces the racy `threading.Timer` + `freezer.tick(...)` collision in
# the legacy `_advance_clock_until` path with a single-thread driver:
#   - boundaries are pre-computed from `(window_start, window_end]`
#   - frozen time jumps directly to each boundary (`freezer.move_to`),
#     no polling, no real-time sleep
#   - per-sensor predict cadence is invoked synchronously via Nexus's
#     public `PredictLoop.tick_once(wired)` (vaquum-nexus >= 0.41.0)
#   - submits and outcomes drain to quiescence between boundaries; the
#     drain loop repeats until both queues are empty in the same pass,
#     so an `on_outcome` that emits new actions cannot leave residue.
#
# The Timer-driven `PredictLoop` and worker-thread `OutcomeLoop` are
# not started in the synchronous replay path; `drive_window` fails loud
# on entry if either is running, and on heterogeneous-cadence wired
# sensors. Strategy-authored timers (`TimerLoop`) are not supported in
# this path; the launcher fails loud upstream when the sequencer
# declares `timer_specs`.
#
# Production Praxis is unchanged. The simulator stops faking real-time
# scheduling; everything else (strategy logic, venue adapter, account
# loop, fill chain) is the same code Praxis runs.

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    # `WiredSensor` is imported under TYPE_CHECKING so the Protocol's
    # `tick_once(wired: WiredSensor)` matches Nexus's actual
    # `PredictLoop.tick_once` signature without a runtime import (the
    # Nexus circular import — `nexus.startup.shutdown_sequencer` <->
    # `nexus.strategy.predict_loop` — would otherwise propagate to any
    # caller that imports `replay_clock` from a clean interpreter).
    from nexus.startup.sequencer import WiredSensor

_log = logging.getLogger(__name__)


class _Freezer(Protocol):
    """Minimal protocol for the freezegun factory `drive_window` uses.

    `freezegun.FrozenDateTimeFactory.move_to(target)` jumps the frozen
    clock to `target`. We only need that one method, so the parameter
    is typed via this Protocol rather than the union of three concrete
    freezegun classes (which complicates type narrowing without buying
    anything for callers).
    """

    def move_to(self, target_datetime: datetime) -> None: ...


class _PredictLoop(Protocol):
    """Minimal protocol for the Nexus `PredictLoop` surface used here."""

    @property
    def running(self) -> bool: ...

    def tick_once(self, wired: WiredSensor) -> None: ...


class _OutcomeLoop(Protocol):
    """Minimal protocol for the Nexus `OutcomeLoop` surface used here."""

    @property
    def running(self) -> bool: ...

    def tick_once(self) -> bool: ...


def compute_kline_boundaries(
    *,
    window_start: datetime,
    window_end: datetime,
    interval_seconds: int,
) -> list[datetime]:
    """Return every epoch-aligned boundary in the half-open replay interval.

    Returns every `interval_seconds`-aligned datetime that falls in
    `(window_start, window_end]`.

    The interval is half-open at the start: when `window_start` lies
    exactly on an epoch-aligned boundary, the first returned tick is
    `window_start + interval_seconds`. This matches the runtime
    contract that `PredictLoop`'s first tick after `start` fires one
    interval after the boundary at which the loop began.

    Args:
        window_start: Inclusive lower bound of the trading window.
            Must be timezone-aware.
        window_end: Inclusive upper bound of the trading window.
        interval_seconds: Positive integer kline cadence in seconds.

    Returns:
        Boundaries in monotonic ascending order. Empty when no
        epoch-aligned boundary falls in the half-open interval.

    Raises:
        ValueError: When `interval_seconds <= 0`, when
            `window_start.tzinfo is None`, or when
            `window_end < window_start`.
    """

    if interval_seconds <= 0:
        msg = (
            f'compute_kline_boundaries: interval_seconds must be positive, '
            f'got {interval_seconds}'
        )
        raise ValueError(msg)
    if window_start.tzinfo is None:
        msg = (
            'compute_kline_boundaries: window_start must be timezone-aware'
        )
        raise ValueError(msg)
    if window_end.tzinfo is None:
        msg = (
            'compute_kline_boundaries: window_end must be timezone-aware'
        )
        raise ValueError(msg)
    if window_start.tzinfo != window_end.tzinfo:
        msg = (
            f'compute_kline_boundaries: window_start.tzinfo ({window_start.tzinfo}) '
            f'and window_end.tzinfo ({window_end.tzinfo}) must match. The '
            f'epoch alignment is anchored to window_start.tzinfo; mismatched '
            f'tzinfo would produce off-by-N-hour boundaries.'
        )
        raise ValueError(msg)
    if window_end < window_start:
        msg = (
            f'compute_kline_boundaries: window_end ({window_end}) must be '
            f'>= window_start ({window_start})'
        )
        raise ValueError(msg)

    epoch = datetime(1970, 1, 1, tzinfo=window_start.tzinfo)
    elapsed = int((window_start - epoch).total_seconds())
    next_boundary_secs = (elapsed // interval_seconds + 1) * interval_seconds
    boundaries: list[datetime] = []
    t = epoch + timedelta(seconds=next_boundary_secs)
    while t <= window_end:
        boundaries.append(t)
        t += timedelta(seconds=interval_seconds)
    return boundaries


@dataclass(frozen=True)
class ReplayClock:
    """Single-thread schedule-driven replay clock for one backtest window.

    Drives one `(window_start, window_end]` window through:

      1. boundaries = `compute_kline_boundaries(...)` (per-sensor
         cadence is the single shared `interval_seconds`).
      2. for each boundary:
         - `freezer.move_to(boundary)`
         - for each wired sensor: `predict_loop.tick_once(wired)`
         - drain to quiescence (REPEAT UNTIL FIXED POINT):
             while True:
                 drain_pending_submits()
                 consumed = False
                 while outcome_loop.tick_once():
                     consumed = True
                 if not consumed:
                     break
           — `on_outcome` may emit new actions, which produce new
           submits, which in turn produce new outcomes; the loop only
           exits when both queues are empty in the same pass.
      3. `freezer.move_to(window_end)` and one final drain to
         quiescence so any tick that emitted late actions still
         flushes through the venue → outcome → on_outcome chain.

    Fail-loud invariants enforced inside `drive_window`:

      - All wired sensors share one `interval_seconds` (else
        `ValueError`). The legacy sweep already enforces homogeneous
        cadence at the pick layer; this is a defence-in-depth check.
      - `predict_loop.running` is False on entry (else `RuntimeError`).
        The synchronous path must not interleave with the Timer-driven
        loop.
      - `outcome_loop.running` is False on entry (else `RuntimeError`).
        Symmetric guard for the outcome worker.

    The class carries no state. It exists as a class (rather than a
    free function) to keep the call site explicit at the launcher and
    to give the drain helper a natural method home.

    Drain contract for `drain_pending_submits`: the caller-provided
    callable MUST yield to the running asyncio loop until every
    command delivered to the venue adapter has also been routed
    through the outcome router into the per-account outcome queue
    (delivered == routed). A simple "delivered count >= submitted
    count" check is insufficient because Praxis may have processed
    the submit before the router's awaitable resolved. The launcher's
    bound method satisfies that contract; tests pass a mock that
    increments a "drained" counter so the loop can be observed.
    """

    def drive_window(
        self,
        *,
        window_start: datetime,
        window_end: datetime,
        wired_sensors: Sequence[object],
        predict_loop: _PredictLoop,
        outcome_loop: _OutcomeLoop,
        drain_pending_submits: Callable[[], None],
        freezer: _Freezer,
    ) -> None:
        """Drive one backtest window synchronously to completion.

        Caller owns the freezegun context manager; `drive_window`
        only `move_to`'s the freezer, never `start`s or `stop`s it.

        Args:
            window_start: Trading-window start (inclusive bound on
                schedule generation).
            window_end: Trading-window end (inclusive bound on
                schedule generation).
            wired_sensors: Non-empty sequence of Nexus `WiredSensor`s.
                All must share the same `interval_seconds`.
            predict_loop: A constructed but un-started Nexus
                `PredictLoop`. `tick_once` is invoked per sensor per
                boundary.
            outcome_loop: A constructed but un-started Nexus
                `OutcomeLoop`. `tick_once` is drained between
                boundaries.
            drain_pending_submits: Callable that blocks until every
                submitted command has been delivered AND routed.
            freezer: Active freezegun factory; `move_to` is the only
                method called.
        """

        if not wired_sensors:
            msg = 'drive_window: wired_sensors must be a non-empty sequence'
            raise ValueError(msg)

        intervals: set[int] = set()
        for wired in wired_sensors:
            interval = getattr(wired, 'interval_seconds', None)
            if interval is None:
                msg = (
                    f'drive_window: wired sensor {wired!r} has no '
                    f'interval_seconds attribute'
                )
                raise ValueError(msg)
            intervals.add(int(interval))
        if len(intervals) != 1:
            msg = (
                f'drive_window: wired_sensors declare heterogeneous '
                f'interval_seconds: {sorted(intervals)}. ReplayClock '
                f'requires one cadence per window (one bundle / one '
                f'kline_size).'
            )
            raise ValueError(msg)
        interval_seconds = next(iter(intervals))

        if predict_loop.running:
            msg = (
                'drive_window: predict_loop.running is True. The '
                'synchronous replay path owns the predict cadence and '
                'must not interleave with the Timer-driven loop.'
            )
            raise RuntimeError(msg)
        if outcome_loop.running:
            msg = (
                'drive_window: outcome_loop.running is True. The '
                'synchronous replay path drives outcomes via '
                'tick_once and must not interleave with the '
                'worker-thread loop.'
            )
            raise RuntimeError(msg)

        boundaries = compute_kline_boundaries(
            window_start=window_start,
            window_end=window_end,
            interval_seconds=interval_seconds,
        )
        _log.info(
            'replay_clock: %d boundaries scheduled '
            '(window=[%s, %s], interval=%ds, sensors=%d)',
            len(boundaries), window_start, window_end,
            interval_seconds, len(wired_sensors),
        )

        for boundary in boundaries:
            freezer.move_to(boundary)
            for wired in wired_sensors:
                predict_loop.tick_once(wired)
            self._drain_to_quiescence(outcome_loop, drain_pending_submits)

        freezer.move_to(window_end)
        self._drain_to_quiescence(outcome_loop, drain_pending_submits)

    @staticmethod
    def _drain_to_quiescence(
        outcome_loop: _OutcomeLoop,
        drain_pending_submits: Callable[[], None],
    ) -> None:
        """Repeat-until-fixed-point drain.

        Each pass: settle pending submits, then consume every
        currently-routed outcome. If consuming an outcome triggered
        `on_outcome` to emit new actions, those produce new submits
        in the next pass; the loop only exits when both queues are
        empty in the same pass.
        """

        while True:
            drain_pending_submits()
            consumed = False
            while outcome_loop.tick_once():
                consumed = True
            if not consumed:
                break
