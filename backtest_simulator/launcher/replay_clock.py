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
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    # Imported under TYPE_CHECKING so neither runtime import propagates
    # to clean-interpreter callers of `replay_clock` (the Nexus
    # circular import — `nexus.startup.shutdown_sequencer` <->
    # `nexus.strategy.predict_loop` — would otherwise fire on cold
    # import; `freezegun.api` adds no runtime cost but stays
    # consistent here). The Protocols below use these types to match
    # Nexus's actual `PredictLoop.tick_once(wired: WiredSensor)` and
    # freezegun's `FrozenDateTimeFactory.move_to(target_datetime)`
    # signatures byte-for-byte; renaming the parameter on either side
    # would break Protocol contravariance at the launcher's call site.
    from freezegun.api import FrozenDateTimeFactory
    from nexus.startup.sequencer import WiredSensor

_log = logging.getLogger(__name__)

# Default wall-clock cap on a single `drive_window` call. Aborts loud
# rather than letting a cyclic `on_outcome` chain (action emits new
# action whose fill emits another action ...) spin the
# repeat-until-fixed-point drain forever. 600s is the same budget the
# legacy `_advance_clock_until` enforced; constructor-injectable so
# tests can use a tiny cap without wall-clock waiting.
_REAL_TIME_CAP_SECONDS_DEFAULT = 600.0


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


class ReplayDeadlineExceededError(RuntimeError):
    """Raised when `ReplayClock.drive_window` exceeds its wall-clock cap.

    The most likely causes are a cyclic `on_outcome` chain (action
    emits new action whose fill emits another action) or a strategy
    that never quiesces. The replay path aborts loudly rather than
    silently spinning so an operator (or CI) sees the failure.
    """


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

    The class carries one field: `real_time_cap_seconds`, the
    wall-clock budget enforced by every drain pass and every boundary
    crossing. Defaults to 600s (matches the legacy
    `_advance_clock_until` cap); tests pass a tiny value to exercise
    the deadline path without wall-clock waiting. Exceeding the cap
    raises `ReplayDeadlineExceededError`.

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

    real_time_cap_seconds: float = _REAL_TIME_CAP_SECONDS_DEFAULT

    def drive_window(
        self,
        *,
        window_start: datetime,
        window_end: datetime,
        wired_sensors: Sequence[WiredSensor],
        predict_loop: _PredictLoop,
        outcome_loop: _OutcomeLoop,
        drain_pending_submits: Callable[[], None],
        freezer: FrozenDateTimeFactory,
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
            '(window=[%s, %s], interval=%ds, sensors=%d, cap=%.1fs)',
            len(boundaries), window_start, window_end,
            interval_seconds, len(wired_sensors), self.real_time_cap_seconds,
        )

        deadline = os.times()[4] + self.real_time_cap_seconds
        for boundary in boundaries:
            freezer.move_to(boundary)
            for wired in wired_sensors:
                predict_loop.tick_once(wired)
            self._drain_to_quiescence(outcome_loop, drain_pending_submits, deadline)

        freezer.move_to(window_end)
        self._drain_to_quiescence(outcome_loop, drain_pending_submits, deadline)

    @staticmethod
    def _drain_to_quiescence(
        outcome_loop: _OutcomeLoop,
        drain_pending_submits: Callable[[], None],
        deadline: float,
    ) -> None:
        """Repeat-until-fixed-point drain, bounded by `deadline`.

        Each pass: settle pending submits, then consume every
        currently-routed outcome. If consuming an outcome triggered
        `on_outcome` to emit new actions, those produce new submits
        in the next pass; the loop only exits when both queues are
        empty in the same pass.

        `deadline` is a `os.times()[4]` wall-time budget; if a pass
        finishes after the deadline the loop raises
        `ReplayDeadlineExceededError` rather than spinning further.
        """

        while True:
            drain_pending_submits()
            consumed = False
            while outcome_loop.tick_once():
                consumed = True
            if not consumed:
                break
            if os.times()[4] > deadline:
                raise ReplayDeadlineExceededError('ReplayClock: drain-to-quiescence exceeded real-time cap. Likely cyclic on_outcome chain (action emits new action whose fill emits another). Tighten the strategy or raise real_time_cap_seconds.')
