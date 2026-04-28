"""Parquet-backed HistoricalFeed for tests and local runs."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from backtest_simulator.feed.lookahead import (
    assert_trades_causal,
    assert_window_causal,
    frozen_now,
)


class ParquetFixtureFeed:
    """In-memory, parquet-backed HistoricalFeed.

    One parquet file per (symbol, kline_size). Load once at construct,
    slice by timestamp on every read. The LookAhead gate runs on every
    return path — test fixtures and production reads share the same
    invariant.
    """

    def __init__(self, klines_path: Path, trades_path: Path | None = None) -> None:
        self._klines = pl.read_parquet(klines_path).sort('open_time')
        if trades_path is not None and trades_path.is_file():
            self._trades = pl.read_parquet(trades_path).sort('time')
        else:
            self._trades = pl.DataFrame(schema={'time': pl.Datetime, 'price': pl.Float64, 'qty': pl.Float64})

    def get_window(self, symbol: str, kline_size: int, n_rows: int) -> pl.DataFrame:
        del kline_size  # fixture holds one (symbol, kline_size) frame; stored at construct
        now = frozen_now()
        sliced = self._klines.filter(pl.col('open_time') <= now)
        result = sliced.tail(n_rows) if n_rows > 0 else sliced
        assert_window_causal(result, symbol=symbol, column='open_time')
        return result

    def get_trades(self, symbol: str, start: datetime, end: datetime) -> pl.DataFrame:
        """Strategy-facing strict path: `end <= frozen_now()` always."""
        assert_trades_causal(end, symbol=symbol, venue_lookahead_seconds=0)
        return self._trades.filter(
            (pl.col('time') >= start) & (pl.col('time') <= end),
        )

    def get_trades_for_venue(
        self, symbol: str, start: datetime, end: datetime,
        *, venue_lookahead_seconds: int,
    ) -> pl.DataFrame:
        """Venue-only carve-out: `end <= frozen_now() + venue_lookahead_seconds`.

        Underscore-prefixed and not on `HistoricalFeed`; strategies have
        no public path to this method. The simulated venue's adapter
        passes its declared `trade_window_seconds` for the realistic
        submit/fill-window peek.
        """
        assert_trades_causal(
            end, symbol=symbol, venue_lookahead_seconds=venue_lookahead_seconds,
        )
        return self._trades.filter(
            (pl.col('time') >= start) & (pl.col('time') <= end),
        )
