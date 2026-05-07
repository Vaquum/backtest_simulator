"""ClickHouseFeed — real trades from `origo.binance_daily_spot_trades`; klines via Limen."""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl
import pyarrow as pa
from clickhouse_connect import get_client as _ch_get_client
from clickhouse_connect.driver.client import Client

from backtest_simulator.feed.lookahead import assert_trades_causal, assert_window_causal

_TRADES_DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'

@dataclass(frozen=True)
class ClickHouseConfig:
    host: str
    port: int
    user: str
    password: str
    database: str = 'origo'
    trades_table: str = 'binance_daily_spot_trades'

    @classmethod
    def from_env(cls) -> ClickHouseConfig:
        missing: list[str] = []
        host = os.environ.get('CLICKHOUSE_HOST', '')
        port_raw = os.environ.get('CLICKHOUSE_PORT', '')
        user = os.environ.get('CLICKHOUSE_USER', '')
        password = os.environ.get('CLICKHOUSE_PASSWORD', '')
        database = os.environ.get('CLICKHOUSE_DATABASE', 'origo')
        trades_table = os.environ.get('CLICKHOUSE_TRADES_TABLE', 'binance_daily_spot_trades')
        if not host:
            missing.append('CLICKHOUSE_HOST')
        if not port_raw:
            missing.append('CLICKHOUSE_PORT')
        if not user:
            missing.append('CLICKHOUSE_USER')
        if not password:
            missing.append('CLICKHOUSE_PASSWORD')
        if missing:
            msg = f"ClickHouseConfig.from_env: missing env vars: {', '.join(missing)}"
            raise RuntimeError(msg)
        return cls(host=host, port=int(port_raw), user=user, password=password, database=database, trades_table=trades_table)

class ClickHouseFeed:

    def _connect(self) -> Client:
        if self._client is not None:
            return self._client
        self._client = _make_client(host=self._config.host, port=self._config.port, username=self._config.user, password=self._config.password, database=self._config.database)
        return self._client

    def get_trades(self, symbol: str, start: datetime, end: datetime) -> pl.DataFrame:
        return self._get_trades_impl(symbol, start, end, venue_lookahead_seconds=0)

    def get_trades_for_venue(self, symbol: str, start: datetime, end: datetime, *, venue_lookahead_seconds: int) -> pl.DataFrame:
        return self._get_trades_impl(symbol, start, end, venue_lookahead_seconds=venue_lookahead_seconds)

    def _get_trades_impl(self, symbol: str, start: datetime, end: datetime, *, venue_lookahead_seconds: int) -> pl.DataFrame:
        assert_trades_causal(end, symbol=symbol, venue_lookahead_seconds=venue_lookahead_seconds)
        if symbol != self._symbol:
            msg = f'ClickHouseFeed configured for {self._symbol}; received {symbol}'
            raise ValueError(msg)
        client = self._connect()
        query = f'SELECT datetime, price, quantity, is_buyer_maker, trade_id FROM {self._config.database}.{self._config.trades_table} WHERE datetime BETWEEN %(start)s AND %(end)s ORDER BY datetime, trade_id'
        start_str = _format_datetime64(start)
        end_str = _format_datetime64(end)
        arrow = _query_arrow(client, query, parameters={'start': start_str, 'end': end_str})
        frame = pl.from_arrow(arrow)
        if not isinstance(frame, pl.DataFrame):
            msg = f'expected DataFrame from Arrow conversion, got {type(frame).__name__}'
            raise TypeError(msg)
        if frame.is_empty():
            return pl.DataFrame(schema={'time': pl.Datetime(time_zone='UTC'), 'price': pl.Float64, 'qty': pl.Float64, 'is_buyer_maker': pl.Boolean, 'trade_id': pl.UInt64})
        frame = frame.rename({'datetime': 'time', 'quantity': 'qty'}).with_columns(pl.col('time').cast(pl.Datetime('us', 'UTC')), pl.col('is_buyer_maker').cast(pl.Boolean), pl.col('trade_id').cast(pl.UInt64), pl.col('price').cast(pl.Float64), pl.col('qty').cast(pl.Float64))
        assert_window_causal(frame, symbol=symbol, column='time', venue_lookahead_seconds=venue_lookahead_seconds)
        return frame

def _format_datetime64(value: datetime) -> str:
    if value.tzinfo is not None:
        from datetime import UTC
        value = value.astimezone(UTC).replace(tzinfo=None)
    return value.strftime(_TRADES_DATETIME_FORMAT)

def _make_client(*, host: str, port: int, username: str, password: str, database: str) -> Client:
    return _ch_get_client(host=host, port=port, username=username, password=password, database=database, compress='lz4')

class InMemoryTradesFeed:

    def __init__(self, frame: pl.DataFrame, symbol: str='BTCUSDT') -> None:
        self._frame = frame
        self._symbol = symbol

    def get_trades(self, symbol: str, start: datetime, end: datetime) -> pl.DataFrame:
        return self._slice(symbol, start, end, venue_lookahead_seconds=0)

    def get_trades_for_venue(self, symbol: str, start: datetime, end: datetime, *, venue_lookahead_seconds: int) -> pl.DataFrame:
        return self._slice(symbol, start, end, venue_lookahead_seconds=venue_lookahead_seconds)

    def _slice(self, symbol: str, start: datetime, end: datetime, *, venue_lookahead_seconds: int) -> pl.DataFrame:
        assert_trades_causal(end, symbol=symbol, venue_lookahead_seconds=venue_lookahead_seconds)
        if symbol != self._symbol:
            msg = f'InMemoryTradesFeed configured for {self._symbol}; received {symbol}'
            raise ValueError(msg)
        sliced = self._frame.filter((pl.col('time') >= start) & (pl.col('time') <= end))
        assert_window_causal(sliced, symbol=symbol, column='time', venue_lookahead_seconds=venue_lookahead_seconds)
        return sliced

def prefetch_sweep_trades(*, config: ClickHouseConfig, symbol: str, start: datetime, end: datetime, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{symbol.lower()}_{start.strftime('%Y%m%dT%H%M%S')}_{end.strftime('%Y%m%dT%H%M%S')}.parquet"
    path = cache_dir / fname
    if path.is_file():
        return path
    feed = ClickHouseFeed(config=config, symbol=symbol)
    frame = feed.get_trades_for_venue(symbol, start, end, venue_lookahead_seconds=int((end - start).total_seconds()) + 86400)
    frame = frame.sort(['time', 'trade_id']).set_sorted('time')
    tmp = path.with_suffix(path.suffix + '.tmp')
    frame.write_parquet(tmp)
    tmp.replace(path)
    return path

def _query_arrow(client: Client, query: str, *, parameters: Mapping[str, str]) -> pa.Table:
    raw_result = client.query_arrow(query, parameters=dict(parameters))
    if not isinstance(raw_result, pa.Table):
        msg = f'ClickHouseFeed._query_arrow: expected pyarrow.Table from query_arrow, got {type(raw_result).__name__}'
        raise TypeError(msg)
    return raw_result
