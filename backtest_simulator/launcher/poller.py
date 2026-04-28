"""BacktestMarketDataPoller — klines from Limen HistoricalData instead of Binance REST."""
from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime

import polars as pl
from limen import HistoricalData

_log = logging.getLogger(__name__)

# Public (no leading underscore) so callers that mirror the poller's
# fetch semantics — slice #17 Task 16's `_signals_builder.py`, which
# rebuilds at sweep time exactly the slice the poller serves at
# runtime — can pin to the same constants without reaching into
# private state. Changing either default here shifts both runtime
# and sweep-replay together, preserving "strategy tested, strategy
# deployed".
DEFAULT_START_DATE_LIMIT = '2019-01-01 00:00:00'
DEFAULT_N_ROWS = 5000


class BacktestMarketDataPoller:
    """Drop-in replacement for `praxis.market_data_poller.MarketDataPoller`.

    Same public surface (`start`/`stop`/`running`/`get_market_data`/
    `add_kline_size`/`remove_kline_size`) so `praxis.Launcher` consumes
    it interchangeably. The difference is the data source: klines come
    from the same Limen HistoricalData path Praxis uses for offline
    analysis, not from live Binance REST.

    freezegun compatibility: `get_market_data` filters the cached frame
    by `pl.col('datetime') <= datetime.now(utc)`. Under freezegun that
    yields the strategy's frozen view of history; under real time it
    yields everything up to now.
    """

    def __init__(
        self,
        kline_intervals: dict[int, int] | None = None,
        n_rows: int = DEFAULT_N_ROWS,
        historical_data: HistoricalData | None = None,
        start_date_limit: str = DEFAULT_START_DATE_LIMIT,
    ) -> None:
        self._n_rows = n_rows
        self._historical_data = historical_data or HistoricalData()
        self._start_date_limit = start_date_limit
        self._lock = threading.Lock()
        self._kline_sizes: set[int] = set((kline_intervals or {}).keys())
        self._klines: dict[int, pl.DataFrame] = {}
        self._started = False

    @property
    def running(self) -> bool:
        return self._started

    def start(self) -> None:
        if self._started:
            return
        with self._lock:
            for kline_size in self._kline_sizes:
                self._klines[kline_size] = self._fetch(kline_size)
            self._started = True
        _log.info(
            'backtest market data poller started',
            extra={'kline_sizes': sorted(self._kline_sizes)},
        )

    def stop(self) -> None:
        with self._lock:
            self._started = False

    def add_kline_size(self, kline_size: int, interval: int) -> None:
        del interval  # poll interval is meaningless when data is historical
        with self._lock:
            self._kline_sizes.add(kline_size)
            if self._started and kline_size not in self._klines:
                self._klines[kline_size] = self._fetch(kline_size)

    def remove_kline_size(self, kline_size: int) -> None:
        with self._lock:
            self._kline_sizes.discard(kline_size)
            self._klines.pop(kline_size, None)

    def get_market_data(self, kline_size: int) -> pl.DataFrame:
        with self._lock:
            frame = self._klines.get(kline_size)
        if frame is None or frame.is_empty():
            return pl.DataFrame()
        # freezegun patches `datetime.now`; `now()` is the strategy's view
        # of history under a `freeze_time` block and real wall time otherwise.
        now = datetime.now(UTC)
        causal = frame.filter(pl.col('datetime') <= now)
        return causal.tail(self._n_rows)

    def _fetch(self, kline_size: int) -> pl.DataFrame:
        return self._historical_data.get_spot_klines(
            kline_size=kline_size,
            start_date_limit=self._start_date_limit,
        )
