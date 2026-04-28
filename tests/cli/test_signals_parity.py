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
        tick_timestamps=ticks,
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
            tick_timestamps=ticks,
        )


def test_signals_parity_empty_tick_set_skips_all() -> None:
    """Empty `tick_timestamps` allowlist → 0 comparisons.

    Edge case: if the sweep's tick_timestamps list is empty (e.g.
    interval larger than window), no runtime tick is in coverage.
    Mutation proof: removing the `if ts not in covered: continue`
    guard would let the lookup forward-fill semantics take over,
    making the test fail with a count > 0.
    """
    preds = [1, 0, 1, 1]
    table, ticks = _build_table(preds=preds, n_ticks=4)
    runtime_predictions = [
        {'sensor_id': 'd1', 'timestamp': t.isoformat(), 'pred': p}
        for t, p in zip(ticks, preds, strict=True)
    ]
    n = assert_signals_parity(
        decoder_id='d1', table=table,
        runtime_predictions=runtime_predictions,
        tick_timestamps=[],
    )
    assert n == 0


def test_signals_parity_out_of_grid_ticks_silent() -> None:
    """Ticks NOT in `tick_timestamps` are silently skipped.

    Runtime PredictLoop ticks that fall OUTSIDE the sweep's
    scheduled `tick_timestamps` grid (warmup pre-window ticks the
    table never covered, or any post-window tick) are NOT
    compared. `SignalsTable.lookup` forward-fills past the last
    covered row by contract, so an `lookup is None` check alone
    would let post-window ticks slip through and spuriously match
    against the LAST table row. The explicit allowlist via
    `tick_timestamps` is the precise gate.

    Mutation proof: removing the `if ts not in covered: continue`
    guard makes the post-window ticks compare against the last
    table row (pred=1 vs runtime pred=99), tripping
    ParityViolation.
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
    n = assert_signals_parity(
        decoder_id='d1', table=table,
        runtime_predictions=out_of_grid,
        tick_timestamps=ticks,
    )
    assert n == 0, (
        f'out-of-grid ticks must not be counted; got n_compared={n}'
    )


def test_signals_parity_empty_runtime_predictions_returns_zero() -> None:
    """No runtime predictions → 0 comparisons, no raise.

    Sweep summary uses this to surface "no comparisons made" so the
    operator distinguishes "ran + matched" from "didn't run". Mutation
    proof: a default-raise-on-empty would fire here.
    """
    table, ticks = _build_table()
    n = assert_signals_parity(
        decoder_id='d1', table=table, runtime_predictions=[],
        tick_timestamps=ticks,
    )
    assert n == 0


def test_signals_parity_skips_malformed_entries_silently() -> None:
    """Entries with non-string timestamp or non-int pred are skipped.

    Defense against a JSON serialization edge case (operator's
    serialiser sends a float for `pred`, or omits the field). The
    function must not crash on these — it just doesn't count them.
    Healthy entries still get verified.
    """
    preds = [1, 0, 1, 1, 0, 1, 0, 1]
    table, ticks = _build_table(preds=preds)
    runtime_predictions: list[dict[str, object]] = [
        {'sensor_id': 'd1', 'timestamp': ticks[0].isoformat(), 'pred': 1},
        {'sensor_id': 'd1', 'timestamp': None, 'pred': 0},  # ts None
        {'sensor_id': 'd1', 'timestamp': ticks[1].isoformat()},  # pred missing
        {'sensor_id': 'd1', 'timestamp': ticks[2].isoformat(), 'pred': 0.5},  # pred float
    ]
    n = assert_signals_parity(
        decoder_id='d1', table=table,
        runtime_predictions=runtime_predictions,
        tick_timestamps=ticks,
    )
    assert n == 1, (
        f'only the well-formed entry at tick[0] should count; got n={n}'
    )
