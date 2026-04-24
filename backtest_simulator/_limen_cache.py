"""Local parquet cache for `limen.data.HistoricalData.get_spot_klines`."""
from __future__ import annotations

# Limen has no on-disk cache — every call re-streams BTCUSDT klines from
# HuggingFace. On a Trainer boot the Sensor reconstruction call pays
# ~15-20s real wall time to refetch the same data Limen already pulled at
# experiment-training time. This module installs a monkey-patch at module
# import: the first call per `kline_size` writes the full frame to
# `~/.cache/backtest_simulator/limen_klines/btcusdt_{kline_size}.parquet`;
# every subsequent call in ANY process reads that parquet directly.
#
# Invariants:
#   - Cache key is `kline_size` only. `n_rows` is sliced client-side so
#     one cache file serves every row-count request.
#   - Cache write is atomic (`*.tmp` → `os.replace`).
#   - `functools.wraps` preserves `__qualname__` — Limen's
#     `DataSourceResolver` introspects it.
#   - Idempotent install guarded by `HistoricalData._bts_cache_installed`.
#   - Failures are not silenced: parquet read/write errors raise.
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
    if getattr(HistoricalData, _INSTALLED_ATTR, False):
        return
    original = HistoricalData.get_spot_klines

    @functools.wraps(original)
    def wrapper(
        self: HistoricalData,
        n_rows: int | None = None,
        kline_size: int = 60,
        start_date_limit: str | None = None,
    ) -> pl.DataFrame:
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


install_cache()
