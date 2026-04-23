"""HistoricalFeed Protocol — the single no-look-ahead gate for market data."""
from __future__ import annotations

from datetime import datetime
from typing import Protocol

import polars as pl


class HistoricalFeed(Protocol):
    """Reads historical bars and trades subject to the no-look-ahead invariant."""

    def get_window(self, symbol: str, kline_size: int, n_rows: int) -> pl.DataFrame:
        """Return the latest `n_rows` rows with `timestamp <= frozen_now()`.

        Raises LookAheadViolation if any returned row has a timestamp
        greater than the frozen clock's current time.
        """

    def get_trades(self, symbol: str, start: datetime, end: datetime) -> pl.DataFrame:
        """Return trades in [start, end]. Raises LookAheadViolation if end > now."""
