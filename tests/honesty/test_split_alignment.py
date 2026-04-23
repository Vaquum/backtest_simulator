"""Split-alignment gate: decoder split mismatch -> LookAheadViolation."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from backtest_simulator.exceptions import LookAheadViolation
from backtest_simulator.sensors.precompute import SignalsTable


def _build(split: tuple[int, int, int]) -> SignalsTable:
    stamps = [datetime(2020, 4, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(10)]
    return SignalsTable.from_predictions(
        decoder_id='d-0', split_config=split,
        timestamps=stamps,
        probs=np.full(10, 0.7),
        preds=np.ones(10, dtype=int),
        label_horizon_bars=3, bar_seconds=3600,
    )


def test_split_alignment_passes_on_match() -> None:
    table = _build((8, 1, 2))
    table.assert_split_alignment((8, 1, 2))


def test_split_alignment_fires_on_mismatch() -> None:
    table = _build((8, 1, 2))
    with pytest.raises(LookAheadViolation):
        table.assert_split_alignment((5, 1, 4))


def test_lookup_returns_greatest_preceding_row() -> None:
    table = _build((8, 1, 2))
    query = datetime(2020, 4, 1, 5, 30, tzinfo=UTC)
    row = table.lookup(query)
    assert row is not None
    assert row.timestamp == datetime(2020, 4, 1, 5, tzinfo=UTC)
    assert row.prob == pytest.approx(0.7)


def test_lookup_returns_none_before_first_row() -> None:
    table = _build((8, 1, 2))
    before = datetime(2020, 3, 31, tzinfo=UTC)
    assert table.lookup(before) is None
