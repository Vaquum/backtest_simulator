"""Book-gap instrumentation produces honest per-run latency metrics.

Pins slice #17 Task 11 / SPEC §9.5 venue-fidelity sub-rule.

The simulator's venue adapter walks historical trades to fill orders.
A stop-loss trigger should land at the first tape tick that crosses
the declared stop — not later. The book-gap instrument records the
delay between stop-cross and the trade that actually filled against
the stop, so the operator can see whether the venue is giving free
price improvements (zero gap, every time) or whether the simulator
is honestly pricing slippage (non-trivial gap distribution). A run
with a wide max gap is the smoke signal for a venue regression.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from backtest_simulator.honesty.book_gap import (
    BookGapInstrument,
    BookGapMetric,
)

_T0 = datetime(2026, 4, 20, 0, 0, tzinfo=UTC)


def test_book_gap_max_in_report() -> None:
    """Max and p95 of stop-cross-to-trade gaps surface honestly.

    Drives 20 stop-cross events with linear gaps from 0 to 19s. The
    nearest-rank p95 with n=20 is index ceil(0.95 * 20) - 1 = 18 →
    value 18s. Max is 19s. The test pins both to catch a regression
    that swaps p95 for p99 or mean, or drops the max.
    """
    instr = BookGapInstrument()
    for i in range(20):
        instr.record_stop_cross(
            t_cross=_T0,
            t_first_trade=_T0 + timedelta(seconds=i),
        )
    snap = instr.snapshot()
    assert isinstance(snap, BookGapMetric), (
        f'snapshot must return BookGapMetric, got {type(snap)}'
    )
    assert snap.n_stops_observed == 20, (
        f'n_stops_observed should be 20, got {snap.n_stops_observed}'
    )
    assert snap.max_stop_cross_to_trade_seconds == 19.0, (
        f'max should be 19s, got {snap.max_stop_cross_to_trade_seconds}'
    )
    assert snap.p95_stop_cross_to_trade_seconds == 18.0, (
        f'p95 (nearest-rank) should be 18s for n=20 sequence 0..19; '
        f'got {snap.p95_stop_cross_to_trade_seconds}'
    )


def test_book_gap_max_in_report_unsorted_input() -> None:
    """Max and p95 are robust to insertion order — caller doesn't sort.

    The venue adapter records gaps in the order trades arrive on the
    tape, not sorted. The instrument must compute percentiles over
    the full sample regardless of input ordering. A regression that
    accidentally took max-of-last-N or relied on chronological order
    would silently under-report on a backwards-arriving sample. Pin
    by inserting the same 0..19 sequence in reverse and asserting
    identical metrics.
    """
    instr = BookGapInstrument()
    for i in range(19, -1, -1):
        instr.record_stop_cross(
            t_cross=_T0,
            t_first_trade=_T0 + timedelta(seconds=i),
        )
    snap = instr.snapshot()
    assert snap.max_stop_cross_to_trade_seconds == 19.0
    assert snap.p95_stop_cross_to_trade_seconds == 18.0
    assert snap.n_stops_observed == 20


def test_book_gap_p95_responds_to_distribution() -> None:
    """A different 20-sample distribution flips the p95 expectation.

    The earlier tests use values 0..19 (n=20), pinning p95=18. An
    implementation that hard-codes `p95=18 when n==20` would slip
    through. This test feeds 20 samples drawn from `[0]*19 + [100]`
    where the nearest-rank p95 lands on the 19th sorted value
    (index 18 → 0). Codex Task 11 round 1 pinned the
    fixed-distribution gap.
    """
    instr = BookGapInstrument()
    samples_seconds = [0.0] * 19 + [100.0]
    for s in samples_seconds:
        instr.record_stop_cross(
            t_cross=_T0,
            t_first_trade=_T0 + timedelta(seconds=s),
        )
    snap = instr.snapshot()
    assert snap.n_stops_observed == 20
    assert snap.max_stop_cross_to_trade_seconds == 100.0, (
        f'max should be 100s (single outlier); got '
        f'{snap.max_stop_cross_to_trade_seconds}'
    )
    assert snap.p95_stop_cross_to_trade_seconds == 0.0, (
        f'p95 (nearest-rank index 18 of [0]*19+[100]) should be 0s; '
        f'got {snap.p95_stop_cross_to_trade_seconds}. The instrument '
        f'is probably hard-coding p95=18 instead of computing it.'
    )


def test_book_gap_empty_snapshot() -> None:
    """A run with zero stop crosses produces zero metrics — not None.

    The downstream JSON report (`bts run --output-format json`) needs
    a numeric value in every field; a `None` would force the consumer
    to handle the missing case in addition to the zero-gap case. The
    instrument promises a numeric zero on empty.
    """
    snap = BookGapInstrument().snapshot()
    assert snap.n_stops_observed == 0
    assert snap.max_stop_cross_to_trade_seconds == 0.0
    assert snap.p95_stop_cross_to_trade_seconds == 0.0


def test_book_gap_rejects_inverted_timestamps() -> None:
    """t_first_trade < t_cross is non-causal and raises loudly.

    A tape that fills before the stop is crossed is either a clock
    glitch or a venue bug — either way the operator must know. The
    instrument refuses to compute a negative gap (which would
    silently shrink max / p95 toward zero) and raises ValueError.
    """
    instr = BookGapInstrument()
    with pytest.raises(ValueError, match='precedes t_cross'):
        instr.record_stop_cross(
            t_cross=_T0,
            t_first_trade=_T0 - timedelta(seconds=1),
        )


def test_book_gap_zero_gap_is_recorded() -> None:
    """t_first_trade == t_cross is the ideal case and counts as a sample.

    A zero-gap fill is what perfect price impact would look like.
    The instrument must still record the sample (so n_stops_observed
    increments) — silently dropping zero-gap events would let a
    pathological venue model that always fills at-the-stop look
    like it never observed any stops. Codex anti-pattern hunt.
    """
    instr = BookGapInstrument()
    instr.record_stop_cross(t_cross=_T0, t_first_trade=_T0)
    snap = instr.snapshot()
    assert snap.n_stops_observed == 1
    assert snap.max_stop_cross_to_trade_seconds == 0.0
    assert snap.p95_stop_cross_to_trade_seconds == 0.0
