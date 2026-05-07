"""HistoricalFeed Protocol — the single no-look-ahead gate for market data."""
from __future__ import annotations

from datetime import datetime
from typing import Protocol

import polars as pl


class HistoricalFeed(Protocol):

    def get_window(self, symbol: str, kline_size: int, n_rows: int) -> pl.DataFrame:
        del symbol, kline_size, n_rows
        raise NotImplementedError

    def get_trades(
        self, symbol: str, start: datetime, end: datetime,
    ) -> pl.DataFrame:
        del symbol, start, end
        raise NotImplementedError

class VenueFeed(HistoricalFeed, Protocol):

    def get_trades_for_venue(
        self, symbol: str, start: datetime, end: datetime,
        *, venue_lookahead_seconds: int,
    ) -> pl.DataFrame:
        del symbol, start, end, venue_lookahead_seconds
        raise NotImplementedError
