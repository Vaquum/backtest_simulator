"""BacktestMarketDataPoller — klines from Limen HistoricalData instead of Binance REST."""
from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime

import polars as pl
from limen import HistoricalData

_log = logging.getLogger(__name__)
DEFAULT_START_DATE_LIMIT = '2019-01-01 00:00:00'
DEFAULT_N_ROWS = 5000

class BacktestMarketDataPoller:

    def __init__(self, kline_intervals: dict[int, int] | None=None, n_rows: int=DEFAULT_N_ROWS, historical_data: HistoricalData | None=None, start_date_limit: str=DEFAULT_START_DATE_LIMIT, params_by_kline_size: dict[int, dict[str, object]] | None=None) -> None:
        self._n_rows = n_rows
        self._historical_data = historical_data or HistoricalData()
        self._start_date_limit = start_date_limit
        self._params_by_kline_size: dict[int, dict[str, object]] = params_by_kline_size or {}
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
        _log.info('backtest market data poller started', extra={'kline_sizes': sorted(self._kline_sizes)})

    def stop(self) -> None:
        with self._lock:
            self._started = False

    def get_market_data(self, kline_size: int) -> pl.DataFrame:
        with self._lock:
            frame = self._klines.get(kline_size)
        if frame is None or frame.is_empty():
            return pl.DataFrame()
        now = datetime.now(UTC)
        causal = frame.filter(pl.col('datetime') <= now)
        params = self._params_by_kline_size.get(kline_size, {})
        n_rows_obj = params.get('n_rows', self._n_rows)
        if not isinstance(n_rows_obj, int):
            msg = f'BacktestMarketDataPoller.get_market_data: bundle n_rows for kline_size={kline_size} must be int, got {type(n_rows_obj).__name__}={n_rows_obj!r}'
            raise TypeError(msg)
        return causal.tail(n_rows_obj)

    def _fetch(self, kline_size: int) -> pl.DataFrame:
        params = dict(self._params_by_kline_size.get(kline_size, {}))
        params.pop('kline_size', None)
        n_rows_obj = params.pop('n_rows', self._n_rows)
        start_date_limit_obj = params.pop('start_date_limit', self._start_date_limit)
        if params:
            msg = f'BacktestMarketDataPoller._fetch: kline_size={kline_size} has unsupported data_source.params keys: {sorted(params)}. Limen.HistoricalData.get_spot_klines accepts only n_rows / kline_size / start_date_limit.'
            raise ValueError(msg)
        if not isinstance(n_rows_obj, int):
            msg = f'BacktestMarketDataPoller._fetch: kline_size={kline_size} n_rows must be int, got {type(n_rows_obj).__name__}={n_rows_obj!r}'
            raise TypeError(msg)
        if not isinstance(start_date_limit_obj, str):
            msg = f'BacktestMarketDataPoller._fetch: kline_size={kline_size} start_date_limit must be str, got {type(start_date_limit_obj).__name__}={start_date_limit_obj!r}'
            raise TypeError(msg)
        return self._historical_data.get_spot_klines(n_rows=n_rows_obj, kline_size=kline_size, start_date_limit=start_date_limit_obj)
