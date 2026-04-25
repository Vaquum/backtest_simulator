"""Honesty gate: SignalsTable.lookup accepts path_id, purge, embargo.

Pins slice #17 Task 16 / SPEC §9.5 lookahead sub-rule.

CPCV (Combinatorially Purged Cross-Validation) splits the data into
`n_groups` and tests on `n_test_groups` at a time, training on the
remainder. To honestly evaluate, signals must be:
  - looked up against the path's training/test partition
    (`path_id`).
  - purged: any signal whose label horizon overlaps the test window
    boundary leaks the future into training; drop it.
  - embargoed: signals after the test window's end need a
    cool-down period before being usable for training.

`SignalsTable.lookup(t, *, path_id=None, purge_seconds=0,
embargo_seconds=0)` is the per-decoder accessor. Task 16 adds the
signature; Task 17 wires the per-path semantics through CPCV.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from freezegun import freeze_time

from backtest_simulator.sensors.precompute import (
    PredictionsInput,
    SignalsTable,
)

_T0 = datetime(2026, 4, 20, 0, 0, tzinfo=UTC)


def _build_signals_table(*, n_signals: int = 10) -> SignalsTable:
    timestamps = [_T0 + timedelta(minutes=i) for i in range(n_signals)]
    probs = np.array([0.5 + 0.01 * i for i in range(n_signals)])
    preds = np.array([i % 2 for i in range(n_signals)])
    return SignalsTable.from_predictions(
        decoder_id='task-16-fixture',
        split_config=(70, 15, 15),
        predictions=PredictionsInput(
            timestamps=timestamps,
            probs=probs,
            preds=preds,
            label_horizon_bars=2,
            bar_seconds=60,
        ),
    )


def test_signals_table_lookup_accepts_path_id_and_purge_embargo() -> None:
    """The new kwargs are accepted; defaults preserve existing behavior.

    A baseline lookup with no kwargs returns the same row as a
    lookup with explicit defaults — no silent semantic shift.
    """
    table = _build_signals_table()
    query_t = _T0 + timedelta(minutes=5)
    with freeze_time(_T0 + timedelta(minutes=10)):
        baseline = table.lookup(query_t)
        with_kwargs = table.lookup(
            query_t, path_id=None, purge_seconds=0, embargo_seconds=0,
        )
    assert baseline is not None and with_kwargs is not None
    assert baseline.timestamp == with_kwargs.timestamp
    assert baseline.pred == with_kwargs.pred
    assert baseline.prob == with_kwargs.prob


def test_signals_table_lookup_embargo_shifts_cutoff_back() -> None:
    """`embargo_seconds=N` returns the row with `timestamp <= t - N`.

    Without embargo, lookup at minute 5 returns the row at minute 5.
    With embargo_seconds=120, the cutoff shifts back to minute 3 →
    lookup returns the row at minute 3.
    """
    table = _build_signals_table()
    query_t = _T0 + timedelta(minutes=5)
    with freeze_time(_T0 + timedelta(minutes=10)):
        no_embargo = table.lookup(query_t)
        with_embargo = table.lookup(query_t, embargo_seconds=120)
    assert no_embargo is not None and with_embargo is not None
    assert no_embargo.timestamp == _T0 + timedelta(minutes=5)
    assert with_embargo.timestamp == _T0 + timedelta(minutes=3), (
        f'embargo_seconds=120 should shift cutoff back to minute 3; '
        f'got {with_embargo.timestamp}'
    )


def test_signals_table_lookup_purge_drops_overlapping_labels() -> None:
    """Rows whose label horizon extends past `t - purge_seconds` are dropped.

    The fixture has label_horizon_bars=2 + bar_seconds=60 → each
    label_t1 = timestamp + 120s. At query t = minute 5:
      - row at minute 5 has label_t1 = minute 7. With purge_seconds
        = 60, purge_floor = minute 4. label_t1 (7) > purge_floor
        (4) → drop.
      - row at minute 4 has label_t1 = minute 6. Still > minute 4
        → drop.
      - row at minute 3 has label_t1 = minute 5. Still > minute 4
        → drop.
      - row at minute 2 has label_t1 = minute 4. Not > minute 4 →
        keep. This is the row returned.
    """
    table = _build_signals_table()
    query_t = _T0 + timedelta(minutes=5)
    with freeze_time(_T0 + timedelta(minutes=10)):
        no_purge = table.lookup(query_t)
        with_purge = table.lookup(query_t, purge_seconds=60)
    assert no_purge is not None and with_purge is not None
    assert no_purge.timestamp == _T0 + timedelta(minutes=5)
    assert with_purge.timestamp == _T0 + timedelta(minutes=2), (
        f'purge_seconds=60 with label horizon=120s should drop rows '
        f'at minutes 3-5 and return minute 2; got {with_purge.timestamp}'
    )


def test_signals_table_lookup_rejects_negative_purge() -> None:
    """`purge_seconds < 0` is a misconfiguration; raise loud."""
    table = _build_signals_table()
    with freeze_time(_T0 + timedelta(minutes=10)):
        with pytest.raises(ValueError, match='purge_seconds must be non-negative'):
            table.lookup(_T0 + timedelta(minutes=5), purge_seconds=-1)


def test_signals_table_lookup_rejects_negative_embargo() -> None:
    """`embargo_seconds < 0` is a misconfiguration; raise loud."""
    table = _build_signals_table()
    with freeze_time(_T0 + timedelta(minutes=10)):
        with pytest.raises(
            ValueError, match='embargo_seconds must be non-negative',
        ):
            table.lookup(_T0 + timedelta(minutes=5), embargo_seconds=-1)
