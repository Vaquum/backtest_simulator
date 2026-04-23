"""ClickHouseFeed — prefetch a (symbol, kline_size) window into Polars once."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import polars as pl

from backtest_simulator.feed.lookahead import (
    assert_trades_causal,
    assert_window_causal,
    frozen_now,
)


class ClickHouseFeed:
    """Pull hourly klines from the Limen HistoricalData endpoint once at boot.

    Designed to sit behind Limen's `get_spot_klines(kline_size, n_rows)`
    so the notebook gets real BTCUSDT hourly bars. Trades retrieval is a
    stub for M1 (returns an empty frame) — the simulated venue falls
    back to a bar-close fill when no trades are available, but strict
    stop enforcement still uses the declared stop price.
    """

    def __init__(self, klines: pl.DataFrame, trades: pl.DataFrame | None = None) -> None:
        self._klines = klines.sort('open_time') if not klines.is_empty() else klines
        self._trades = trades if trades is not None else pl.DataFrame(
            schema={'time': pl.Datetime, 'price': pl.Float64, 'qty': pl.Float64},
        )

    @classmethod
    def from_limen(
        cls,
        historical_data: Any,  # noqa: ANN401 - Limen HistoricalData is a dynamically-typed boundary
        *,
        kline_size: int,
        n_rows: int,
    ) -> ClickHouseFeed:
        """Build via Limen's `HistoricalData.get_spot_klines`."""
        klines = historical_data.get_spot_klines(kline_size=kline_size, n_rows=n_rows)
        return cls(klines=klines)

    def get_window(self, symbol: str, kline_size: int, n_rows: int) -> pl.DataFrame:
        now = frozen_now()
        sliced = self._klines.filter(pl.col('open_time') <= now)
        result = sliced.tail(n_rows) if n_rows > 0 else sliced
        assert_window_causal(result, symbol=symbol, column='open_time')
        return result

    def get_trades(self, symbol: str, start: datetime, end: datetime) -> pl.DataFrame:
        assert_trades_causal(end, symbol=symbol)
        if self._trades.is_empty():
            return self._trades
        return self._trades.filter(
            (pl.col('time') >= start) & (pl.col('time') <= end),
        )

    def all_bars(self) -> pl.DataFrame:
        return self._klines
