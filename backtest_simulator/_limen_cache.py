"""Opt-in parquet cache for `limen.data.HistoricalData.get_spot_klines`."""
from __future__ import annotations

import functools
import logging
from pathlib import Path

import polars as pl
from limen.data import HistoricalData

_log = logging.getLogger(__name__)
_CACHE_ROOT = Path.home() / '.cache' / 'backtest_simulator' / 'limen_klines'
_INSTALLED_ATTR = '_bts_cache_installed'
__all__ = ['install_cache']

def _cache_path(kline_size: int) -> Path:
    return _CACHE_ROOT / f'btcusdt_{kline_size}.parquet'

def install_cache() -> None:
    original = HistoricalData.get_spot_klines

    @functools.wraps(original)
    def wrapper(self: HistoricalData, n_rows: int | None=None, kline_size: int=60, start_date_limit: str | None=None) -> pl.DataFrame:
        path = _cache_path(kline_size)
        if path.is_file():
            cached = pl.read_parquet(path)
            _log.info('limen_cache: hit %s (rows=%d)', path.name, cached.height)
            sliced = cached.head(n_rows) if n_rows is not None else cached
            self.data = sliced
            self.data_columns = sliced.columns
            return sliced
        _log.info('limen_cache: miss %s (kline_size=%d), fetching from HuggingFace', path.name, kline_size)
        frame = original(self, n_rows=None, kline_size=kline_size, start_date_limit=start_date_limit)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.tmp')
        frame.write_parquet(tmp)
        tmp.replace(path)
        _log.info('limen_cache: wrote %s (rows=%d)', path.name, frame.height)
        sliced = frame.head(n_rows) if n_rows is not None else frame
        self.data = sliced
        self.data_columns = sliced.columns
        return sliced
    setattr(HistoricalData, 'get_spot_klines', wrapper)
    setattr(HistoricalData, _INSTALLED_ATTR, True)
