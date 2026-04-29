"""Mutation-proof tests for `assert_signals_parity`.

Auditor (post-v2.0.2) "make it real": SignalsTable was being built
every sweep without being consumed by any decision metric (CPCV
moved to deployed-strategy daily returns in v2.0.2). The fix
turned it into the sweep-time PARITY REFERENCE — runtime
`Sensor.predict` outputs (captured per-tick via the
`produce_signal` hook in `_run_window`) MUST match the
SignalsTable for the same decoder; mismatches raise
`ParityViolation`.

These tests pin the comparison contract:
  - Match → no raise + correct count of comparisons.
  - Mismatch → ParityViolation.
  - Out-of-coverage ticks → silent skip (correct — table doesn't
    claim coverage there).
  - Empty captures → 0 comparisons, no raise.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path  # noqa: F401 - tmp_path fixture type

import numpy as np
import pytest

from backtest_simulator.cli._signals_builder import assert_signals_parity
from backtest_simulator.exceptions import ParityViolation
from backtest_simulator.sensors.precompute import (
    PredictionsInput,
    SignalsTable,
)


def _build_table(
    *, decoder_id: str = 'd1', n_ticks: int = 8, bar_seconds: int = 60,
    preds: list[int] | None = None,
) -> tuple[SignalsTable, list[datetime]]:
    """Build a SignalsTable + parallel tick list for tests."""
    base = datetime(2026, 4, 20, 0, 0, tzinfo=UTC)
    ticks = [
        base + timedelta(seconds=i * bar_seconds) for i in range(n_ticks)
    ]
    if preds is None:
        preds = [1, 0, 1, 1, 0, 1, 0, 1]
    table = SignalsTable.from_predictions(
        decoder_id=decoder_id,
        split_config=(70, 15, 15),
        predictions=PredictionsInput(
            timestamps=ticks,
            probs=np.array([0.5 + 0.01 * i for i in range(n_ticks)]),
            preds=np.array(preds, dtype=np.int64),
            label_horizon_bars=1,
            bar_seconds=bar_seconds,
        ),
    )
    return table, ticks


def test_signals_parity_match_returns_count() -> None:
    """Runtime preds matching SignalsTable → no raise + comparison count.

    Mutation proof: changing `row.pred != runtime_pred` to `row.pred
    == runtime_pred` makes every match raise; this test catches that.
    """
    preds = [1, 0, 1, 1, 0, 1, 0, 1]
    table, ticks = _build_table(preds=preds)
    runtime_predictions = [
        {'sensor_id': 'd1', 'timestamp': t.isoformat(), 'pred': p}
        for t, p in zip(ticks, preds, strict=True)
    ]
    n = assert_signals_parity(
        decoder_id='d1', table=table,
        runtime_predictions=runtime_predictions,
        expected_ticks=ticks,
        interval_seconds=60,
    )
    assert n == len(ticks), (
        f'expected {len(ticks)} successful comparisons; got {n}'
    )


def test_signals_parity_single_mismatch_raises() -> None:
    """Even one tick where runtime != table preds raises ParityViolation.

    Mutation proof: removing the raise makes this test pass with no
    exception (assertion `pytest.raises(ParityViolation)` fires).
    """
    preds = [1, 0, 1, 1, 0, 1, 0, 1]
    table, ticks = _build_table(preds=preds)
    # Flip the prediction at tick index 3 — one mismatch.
    runtime_predictions = [
        {'sensor_id': 'd1', 'timestamp': t.isoformat(), 'pred': p}
        for t, p in zip(ticks, preds, strict=True)
    ]
    target_pred = runtime_predictions[3]['pred']
    assert isinstance(target_pred, int)
    runtime_predictions[3] = dict(runtime_predictions[3])
    runtime_predictions[3]['pred'] = 1 - target_pred  # flip 0↔1

    with pytest.raises(ParityViolation, match='SignalsTable parity violation'):
        assert_signals_parity(
            decoder_id='d1', table=table,
            runtime_predictions=runtime_predictions,
            expected_ticks=ticks,
            interval_seconds=60,
        )


def test_signals_parity_empty_tick_set_raises() -> None:
    """Empty `tick_timestamps` allowlist → ParityViolation (auditor post-v2.0.3).

    Edge case: empty grid means EVERY runtime entry is out-of-grid.
    Codex post-auditor-4 P1 made out-of-grid raise per-entry, so
    the FIRST runtime entry trips the `OUTSIDE the scheduled` raise
    rather than the post-loop `0 comparisons made` raise. Either
    is a ParityViolation — the test pins that the empty-grid case
    raises SOME violation.

    Mutation proof: relaxing either the per-entry out-of-grid
    raise OR the post-loop 0-comparisons raise lets one of these
    paths through silently and `pytest.raises` fires.
    """
    preds = [1, 0, 1, 1]
    table, ticks = _build_table(preds=preds, n_ticks=4)
    runtime_predictions = [
        {'sensor_id': 'd1', 'timestamp': t.isoformat(), 'pred': p}
        for t, p in zip(ticks, preds, strict=True)
    ]
    with pytest.raises(ParityViolation):
        assert_signals_parity(
            decoder_id='d1', table=table,
            runtime_predictions=runtime_predictions,
            expected_ticks=[],
            interval_seconds=60,
        )


def test_signals_parity_only_out_of_grid_ticks_raises() -> None:
    """All-out-of-grid runtime predictions → ParityViolation.

    Codex (post-auditor-4) P1: out-of-grid ticks now raise
    per-entry. The FIRST entry in the list hits the
    `OUTSIDE the scheduled` raise — does not reach the post-loop.

    Mutation proof: removing the `if ts not in covered: raise`
    guard lets the post-window ticks compare against the
    forward-filled last table row, producing a different mismatch
    raise (still a ParityViolation, but a different message). The
    operator gets the precise message naming the offending tick.
    """
    preds = [1, 0, 1, 1, 0, 1, 0, 1]
    table, ticks = _build_table(preds=preds)
    pre_warmup = ticks[0] - timedelta(hours=1)
    post_window_1 = ticks[-1] + timedelta(hours=1)
    post_window_2 = ticks[-1] + timedelta(hours=2)
    out_of_grid = [
        {'sensor_id': 'd1', 'timestamp': pre_warmup.isoformat(), 'pred': 99},
        {'sensor_id': 'd1', 'timestamp': post_window_1.isoformat(), 'pred': 99},
        {'sensor_id': 'd1', 'timestamp': post_window_2.isoformat(), 'pred': 99},
    ]
    with pytest.raises(ParityViolation, match='OUTSIDE the scheduled'):
        assert_signals_parity(
            decoder_id='d1', table=table,
            runtime_predictions=out_of_grid,
            expected_ticks=ticks,
            interval_seconds=60,
        )


def test_signals_parity_empty_runtime_predictions_raises() -> None:
    """No runtime predictions → ParityViolation (auditor post-v2.0.3).

    Empty captures means the `produce_signal` hook produced nothing.
    The auditor's post-v2.0.3 mandate forbids treating that as
    success — a broken capture hook or missing subprocess payload
    must NOT silently leave the SignalsTable path unvalidated. Now
    raises `ParityViolation` on 0 comparisons.

    Mutation proof: removing the post-loop `n_compared == 0` raise
    makes the helper return 0 silently and `pytest.raises` fires.
    """
    table, ticks = _build_table()
    with pytest.raises(
        ParityViolation, match=r'expected tick.* NOT captured',
    ):
        assert_signals_parity(
            decoder_id='d1', table=table, runtime_predictions=[],
            expected_ticks=ticks,
            interval_seconds=60,
        )


def test_signals_parity_malformed_entry_raises() -> None:
    """Non-str timestamp → ParityViolation (codex post-auditor-4 P1).

    Prior round silently skipped malformed entries when at least
    one comparison succeeded — codex's repro
    `runtime=[valid match, out-of-grid pred=99]` returned 1 OK
    despite the partial capture-failure. Now ANY skip raises.

    Mutation proof: re-introducing the silent `continue` makes
    the helper return 1 and `pytest.raises` fires.
    """
    preds = [1, 0, 1, 1, 0, 1, 0, 1]
    table, ticks = _build_table(preds=preds)
    runtime_predictions: list[dict[str, object]] = [
        {'sensor_id': 'd1', 'timestamp': ticks[0].isoformat(), 'pred': 1},
        {'sensor_id': 'd1', 'timestamp': None, 'pred': 0},  # malformed ts
    ]
    with pytest.raises(ParityViolation, match='non-string `timestamp`'):
        assert_signals_parity(
            decoder_id='d1', table=table,
            runtime_predictions=runtime_predictions,
            expected_ticks=ticks,
            interval_seconds=60,
        )


def test_signals_parity_non_int_pred_raises() -> None:
    """Non-int pred → ParityViolation (codex post-auditor-4 P1).

    A float pred (e.g. from a regression head emitted via the wrong
    `_extract_values` codepath) is unverifiable against the int
    `pred` column on the table. Loud, not silent.
    """
    preds = [1, 0, 1, 1, 0, 1, 0, 1]
    table, ticks = _build_table(preds=preds)
    runtime_predictions: list[dict[str, object]] = [
        {'sensor_id': 'd1', 'timestamp': ticks[0].isoformat(), 'pred': 1},
        {'sensor_id': 'd1', 'timestamp': ticks[1].isoformat(), 'pred': 0.5},
    ]
    with pytest.raises(ParityViolation, match='non-int `pred`'):
        assert_signals_parity(
            decoder_id='d1', table=table,
            runtime_predictions=runtime_predictions,
            expected_ticks=ticks,
            interval_seconds=60,
        )


def test_signals_parity_partial_out_of_grid_raises() -> None:
    """One in-grid match + one out-of-grid → ParityViolation.

    Codex's exact repro: `runtime=[valid match, out-of-grid
    pred=99]`. Prior round returned 1 OK; the partial capture
    failure was silent. Now: any out-of-grid tick raises.
    """
    preds = [1, 0, 1, 1, 0, 1, 0, 1]
    table, ticks = _build_table(preds=preds)
    rogue = ticks[-1] + timedelta(hours=10)
    runtime_predictions = [
        {'sensor_id': 'd1', 'timestamp': ticks[0].isoformat(), 'pred': preds[0]},
        {'sensor_id': 'd1', 'timestamp': rogue.isoformat(), 'pred': 99},
    ]
    with pytest.raises(ParityViolation, match='OUTSIDE the scheduled'):
        assert_signals_parity(
            decoder_id='d1', table=table,
            runtime_predictions=runtime_predictions,
            expected_ticks=ticks,
            interval_seconds=60,
        )


def test_signals_parity_partial_capture_missing_tick_raises() -> None:
    """Codex (post-auditor-4 round-2) P1: missing-from-runtime → raise.

    The prior one-way check accepted runtime=[2 valid in-grid] vs
    expected=[4 ticks] as success because every CAPTURED entry
    matched. The auditor's "make it real" mandate requires two-way
    multiset equality: every expected tick MUST appear in runtime
    exactly once.

    Mutation proof: removing the post-loop `if missing: raise`
    block makes this test pass with no exception (silent partial
    capture); `pytest.raises` fires.
    """
    preds = [1, 0, 1, 1]
    table, ticks = _build_table(preds=preds, n_ticks=4)
    # Runtime captured ONLY the first 2 ticks; the back half is
    # missing (capture hook truncated, subprocess output truncated,
    # PredictLoop didn't fire near window end, etc.)
    partial = [
        {'sensor_id': 'd1', 'timestamp': ticks[0].isoformat(), 'pred': preds[0]},
        {'sensor_id': 'd1', 'timestamp': ticks[1].isoformat(), 'pred': preds[1]},
    ]
    with pytest.raises(
        ParityViolation, match=r'expected tick.* NOT captured',
    ):
        assert_signals_parity(
            decoder_id='d1', table=table,
            runtime_predictions=partial,
            expected_ticks=ticks,
            interval_seconds=60,
        )


def test_signals_parity_duplicate_runtime_tick_raises() -> None:
    """Codex (post-auditor-4 round-2) P1: duplicate runtime tick → raise.

    PredictLoop double-firing or capture-side double-emission
    indicates a real-runtime bug. Even when both entries match
    the table's pred at that tick (so per-entry comparisons all
    pass), the duplicate itself is a parity violation.
    """
    preds = [1, 0, 1, 1]
    table, ticks = _build_table(preds=preds, n_ticks=4)
    # Same timestamp twice with the same pred — both per-entry
    # comparisons pass, but the multiset doesn't match.
    duped = [
        {'sensor_id': 'd1', 'timestamp': ticks[0].isoformat(), 'pred': preds[0]},
        {'sensor_id': 'd1', 'timestamp': ticks[0].isoformat(), 'pred': preds[0]},
        {'sensor_id': 'd1', 'timestamp': ticks[1].isoformat(), 'pred': preds[1]},
        {'sensor_id': 'd1', 'timestamp': ticks[2].isoformat(), 'pred': preds[2]},
        {'sensor_id': 'd1', 'timestamp': ticks[3].isoformat(), 'pred': preds[3]},
    ]
    with pytest.raises(ParityViolation, match='duplicate ticks'):
        assert_signals_parity(
            decoder_id='d1', table=table,
            runtime_predictions=duped,
            expected_ticks=ticks,
            interval_seconds=60,
        )


def test_signals_parity_cross_window_in_sweep_grid_raises() -> None:
    """Codex (post-auditor-4 round-2) P1: cross-window leak → raise.

    The first-window check used to receive the full sweep
    `tick_timestamps`, so a captured tick from window 2 (still
    in-grid for the SWEEP, but NOT for window 1) was accepted as
    valid. The fix passes per-window expected ticks. This test
    pins the contract: a runtime entry whose timestamp belongs
    to a different window's grid raises `OUTSIDE the scheduled`.
    """
    base = datetime(2026, 4, 20, 0, 0, tzinfo=UTC)
    window1_ticks = [base + timedelta(seconds=i * 60) for i in range(4)]
    # Window 2 ticks are in the sweep grid but NOT in window 1's
    # per-window slice.
    window2_first = base + timedelta(hours=4)
    table, _ = _build_table(preds=[1, 0, 1, 1], n_ticks=4)
    runtime_for_window1 = [
        {
            'sensor_id': 'd1', 'timestamp': window1_ticks[0].isoformat(),
            'pred': 1,
        },
        # Cross-window leak — captured tick from window 2:
        {
            'sensor_id': 'd1', 'timestamp': window2_first.isoformat(),
            'pred': 1,
        },
    ]
    with pytest.raises(ParityViolation, match='OUTSIDE the scheduled'):
        assert_signals_parity(
            decoder_id='d1', table=table,
            runtime_predictions=runtime_for_window1,
            # Per-window expected ticks (window 1 only):
            expected_ticks=window1_ticks,
            interval_seconds=60,
        )


def test_signals_parity_snaps_runtime_drift_to_expected_tick() -> None:
    """Runtime tick lands AFTER its planned `expected_tick` boundary by
    less than `interval_seconds` (clock-callback drift in the launcher).
    The parity check must SNAP each such tick to its anchoring
    expected_tick — same row, same comparison.

    Mutation proof: removing the snap and reverting to strict equality
    makes this test raise `OUTSIDE the scheduled` on every drifted
    entry. Hardcoding interval_seconds=0 makes every drift unmatched.

    The bundle path on `LogReg-Placeholder__r0007.zip` (`kline_size
    =7200`) hits this: the launcher's main-loop tick advances frozen
    time in 120s steps, and a Timer callback that runs while the
    main loop is mid-tick observes frozen time several minutes past
    the planned target. Without the snap, every fire raises.
    """
    preds = [1, 0, 1, 1, 0, 1, 0, 1]
    table, ticks = _build_table(preds=preds)
    # 60-second bars; simulate a 17-second drift (well within one bar).
    runtime_predictions = [
        {
            'sensor_id': 'd1',
            'timestamp': (t + timedelta(seconds=17)).isoformat(),
            'pred': p,
        }
        for t, p in zip(ticks, preds, strict=True)
    ]
    n = assert_signals_parity(
        decoder_id='d1', table=table,
        runtime_predictions=runtime_predictions,
        expected_ticks=ticks,
        interval_seconds=60,
    )
    assert n == len(ticks), (
        f'expected {len(ticks)} successful comparisons after snap; got {n}'
    )


def test_signals_parity_snap_drift_beyond_grid_raises() -> None:
    """A runtime tick whose timestamp falls past the LAST expected_tick
    + interval_seconds does not anchor to any expected_tick — that's a
    real cadence divergence, not callback jitter. Raise OUTSIDE.

    Mutation proof: a snap implementation that expanded the tolerance
    to `+ 2 * interval_seconds` would silently match this entry to
    the last tick and skip the divergence.
    """
    preds = [1, 0, 1, 1, 0, 1, 0, 1]
    table, ticks = _build_table(preds=preds)
    # ticks[-1] + 60s is the boundary; +120s lands strictly past it.
    far_future = ticks[-1] + timedelta(seconds=120)
    runtime_predictions = [
        {
            'sensor_id': 'd1',
            'timestamp': far_future.isoformat(),
            'pred': preds[-1],
        },
    ]
    with pytest.raises(ParityViolation, match='OUTSIDE the scheduled'):
        assert_signals_parity(
            decoder_id='d1', table=table,
            runtime_predictions=runtime_predictions,
            expected_ticks=ticks,
            interval_seconds=60,
        )


def test_signals_parity_kline_size_7200_with_drift() -> None:
    """End-to-end at the bundle's actual cadence: kline_size=7200
    (2-hour klines) with a 10-minute drift on every fire. The fix
    must accept these as in-grid via snap.
    """
    base = datetime(2026, 4, 15, 0, 0, tzinfo=UTC)
    ticks = [base + timedelta(seconds=i * 7200) for i in range(12)]
    preds = [1, 0] * 6
    table = SignalsTable.from_predictions(
        decoder_id='bundle-snap',
        split_config=(70, 15, 15),
        predictions=PredictionsInput(
            timestamps=ticks,
            probs=np.array([0.55] * 12),
            preds=np.array(preds, dtype=np.int64),
            label_horizon_bars=1,
            bar_seconds=7200,
        ),
    )
    runtime_predictions = [
        {
            'sensor_id': 'bundle-snap',
            'timestamp': (t + timedelta(minutes=10)).isoformat(),
            'pred': p,
        }
        for t, p in zip(ticks, preds, strict=True)
    ]
    n = assert_signals_parity(
        decoder_id='bundle-snap', table=table,
        runtime_predictions=runtime_predictions,
        expected_ticks=ticks,
        interval_seconds=7200,
    )
    assert n == len(ticks)
