"""wall_clock: CLOCK_MONOTONIC reader with hard-failure fallback path."""
from __future__ import annotations

from backtest_simulator import wall_clock


def test_monotonic_seconds_returns_float() -> None:
    # Baseline: on any supported platform the primary clock path
    # returns a finite non-zero float.
    ts = wall_clock.monotonic_seconds()
    assert isinstance(ts, float)
    assert ts > 0.0


def test_monotonic_seconds_is_monotonic_across_calls() -> None:
    # Two consecutive calls return non-decreasing values. This is the
    # load-bearing property: if the function silently returned an
    # uninitialised timespec (the pre-fix behaviour on clock_gettime
    # failure), the second call could easily return < the first and
    # every wall-time gate in §9 would miss the regression.
    first = wall_clock.monotonic_seconds()
    second = wall_clock.monotonic_seconds()
    assert second >= first


def test_monotonic_seconds_falls_back_when_libc_missing() -> None:
    # The libc-missing branch (`_libc is None`) must route to
    # time.perf_counter without raising. Simulate by patching the
    # module-level `_libc` to None and re-importing through the
    # module handle so the runtime reads the mutated value.
    import time

    saved = wall_clock._libc
    wall_clock._libc = None
    try:
        ts = wall_clock.monotonic_seconds()
        # perf_counter is monotonic and non-negative; the fallback
        # path keeps the same type contract as the primary path.
        assert isinstance(ts, float)
        assert ts >= 0.0
        # Sanity: the fallback should be within reasonable range of
        # perf_counter (same underlying clock source on most platforms).
        delta = abs(ts - time.perf_counter())
        assert delta < 1.0
    finally:
        wall_clock._libc = saved
