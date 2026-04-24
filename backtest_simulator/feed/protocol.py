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
        # Protocol stub — implementations supply the body. `del` keeps
        # vulture from flagging these Protocol-required parameters as unused.
        del symbol, kline_size, n_rows
        raise NotImplementedError

    def get_trades(
        self, symbol: str, start: datetime, end: datetime,
        *, venue_lookahead_seconds: int = 0,
    ) -> pl.DataFrame:
        """Return trades in [start, end].

        Strategy-facing callers (the default) MUST have `end <= frozen_now()`
        or LookAheadViolation is raised. The simulated venue may pass
        `venue_lookahead_seconds > 0` to consult trades within a bounded
        fill-simulation window past the frozen clock; the ceiling is the
        adapter's declared `trade_window_seconds`.
        """
        del symbol, start, end, venue_lookahead_seconds
        raise NotImplementedError
