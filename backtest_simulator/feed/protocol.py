"""HistoricalFeed Protocol — the single no-look-ahead gate for market data."""
from __future__ import annotations

from datetime import datetime
from typing import Protocol

import polars as pl


class HistoricalFeed(Protocol):
    """Strategy-facing market data reader, strict no-look-ahead.

    The strategy-facing surface has *no* venue carve-out kwarg: a
    cheating strategy with a feed reference cannot pass `venue_lookahead_seconds`
    because the parameter is not on the Protocol and not on the
    strategy-side reader at all. Venue-only access goes through
    `_get_trades_for_venue` on the implementation, which is
    deliberately not part of this Protocol.
    """

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
    ) -> pl.DataFrame:
        """Return trades in [start, end] with `end <= frozen_now()`.

        Strict strategy-facing path: there is no venue carve-out kwarg
        and the implementation MUST raise `LookAheadViolation` if `end`
        is past the frozen clock. The simulated venue uses a separate
        bounded-carve-out method (`_get_trades_for_venue`) that is not
        part of this Protocol — strategies cannot reach it through the
        public Protocol surface.
        """
        del symbol, start, end
        raise NotImplementedError


class VenueFeed(HistoricalFeed, Protocol):
    """Adapter-facing feed: `HistoricalFeed` + the bounded-carve-out method.

    The simulated venue adapter needs to peek up to its declared
    `trade_window_seconds` past `frozen_now()` to simulate a realistic
    submit/fill window. That carve-out lives on `_get_trades_for_venue`,
    NOT on `HistoricalFeed`. Adapters type their feed parameter as
    `VenueFeed` so a strategy-only feed (one that satisfies only the
    `HistoricalFeed` Protocol) cannot accidentally be plugged in and
    crash on submit. Codex pinned this contract in slice #17 Task 6.
    """

    def _get_trades_for_venue(
        self, symbol: str, start: datetime, end: datetime,
        *, venue_lookahead_seconds: int,
    ) -> pl.DataFrame:
        """Return trades in [start, end] with `end <= now + carve-out`."""
        del symbol, start, end, venue_lookahead_seconds
        raise NotImplementedError
