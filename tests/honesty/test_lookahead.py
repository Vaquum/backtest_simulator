"""Look-ahead gate: feed refuses to return rows past frozen_now()."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from freezegun import freeze_time

from backtest_simulator.exceptions import LookAheadViolation
from backtest_simulator.feed.lookahead import (
    assert_trades_causal,
    assert_window_causal,
    frozen_now,
)
from backtest_simulator.feed.parquet_fixture import ParquetFixtureFeed

FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures' / 'market' / 'btcusdt_1h_fixture.parquet'


def test_frozen_now_respects_freezegun() -> None:
    with freeze_time('2020-04-01T00:00:00+00:00'):
        assert frozen_now() == datetime(2020, 4, 1, tzinfo=UTC)


def test_assert_window_causal_passes_when_all_rows_precede_now() -> None:
    data = pl.DataFrame({
        'open_time': [datetime(2020, 3, 30, 0, tzinfo=UTC),
                      datetime(2020, 3, 31, 12, tzinfo=UTC)],
    })
    with freeze_time('2020-04-01T00:00:00+00:00'):
        assert_window_causal(data, symbol='BTCUSDT')


def test_assert_window_causal_raises_on_future_row() -> None:
    data = pl.DataFrame({
        'open_time': [datetime(2020, 4, 2, tzinfo=UTC)],
    })
    with freeze_time('2020-04-01T00:00:00+00:00'), pytest.raises(LookAheadViolation):
        assert_window_causal(data, symbol='BTCUSDT')


def test_assert_trades_causal_raises_on_future_end() -> None:
    with freeze_time('2020-04-01T00:00:00+00:00'), pytest.raises(LookAheadViolation):
        assert_trades_causal(datetime(2020, 4, 2, tzinfo=UTC), symbol='BTCUSDT')


def test_parquet_fixture_window_stops_at_frozen_now() -> None:
    feed = ParquetFixtureFeed(FIXTURE)
    with freeze_time('2020-04-01T00:00:00+00:00'):
        window = feed.get_window('BTCUSDT', kline_size=3600, n_rows=200)
    assert not window.is_empty()
    latest = window['open_time'].max()
    assert latest <= datetime(2020, 4, 1, tzinfo=UTC)


def test_parquet_fixture_get_trades_raises_on_future_end() -> None:
    feed = ParquetFixtureFeed(FIXTURE)
    with freeze_time('2020-04-01T00:00:00+00:00'), pytest.raises(LookAheadViolation):
        feed.get_trades(
            'BTCUSDT',
            start=datetime(2020, 3, 31, tzinfo=UTC),
            end=datetime(2020, 4, 3, tzinfo=UTC),
        )
