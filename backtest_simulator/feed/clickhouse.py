"""ClickHouseFeed — real trades from tdw.binance_trades_complete; klines via Limen."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import clickhouse_connect
import polars as pl

from backtest_simulator.feed.lookahead import (
    assert_trades_causal,
    assert_window_causal,
)

# Eager import of `clickhouse_connect` is deliberate. Its `driver.tzutil`
# submodule calls `dateutil.tz.tzlocal().tzname(None)` at import time, and
# that path blows up under `freezegun.freeze_time`. Loading the module once,
# at program start, caches it in `sys.modules` before any `freeze_time`
# block can patch `datetime`.


@dataclass(frozen=True)
class ClickHouseConfig:
    """Connection params. Use `from_env` for the standard deployment."""

    host: str
    port: int
    user: str
    password: str
    database: str = 'tdw'

    @classmethod
    def from_env(cls) -> ClickHouseConfig:
        """Read from `CLICKHOUSE_HOST/PORT/USER/PASSWORD/DATABASE` env vars.

        Missing values raise; we refuse to silently default a credential
        and then emit results "from ClickHouse" against a localhost that
        is actually empty. Fail loud.
        """
        missing: list[str] = []
        host = os.environ.get('CLICKHOUSE_HOST', '')
        port_raw = os.environ.get('CLICKHOUSE_PORT', '')
        user = os.environ.get('CLICKHOUSE_USER', '')
        password = os.environ.get('CLICKHOUSE_PASSWORD', '')
        database = os.environ.get('CLICKHOUSE_DATABASE', 'tdw')
        if not host:
            missing.append('CLICKHOUSE_HOST')
        if not port_raw:
            missing.append('CLICKHOUSE_PORT')
        if not user:
            missing.append('CLICKHOUSE_USER')
        if not password:
            missing.append('CLICKHOUSE_PASSWORD')
        if missing:
            msg = f'ClickHouseConfig.from_env: missing env vars: {", ".join(missing)}'
            raise RuntimeError(msg)
        return cls(host=host, port=int(port_raw), user=user, password=password, database=database)


class ClickHouseFeed:
    """Trades feed backed by `tdw.binance_trades_complete`.

    Every `get_trades` call hits the view (archived + recent UNION) and
    returns rows with `datetime BETWEEN start AND end`. Look-ahead guard
    enforces `end <= frozen_now()` on every call.

    Klines are NOT served here — use `limen.HistoricalData().get_spot_klines()`
    per the experiment pattern. Limen aggregates from the same Binance trade
    source, so the two feeds are consistent.
    """

    def __init__(self, config: ClickHouseConfig, symbol: str = 'BTCUSDT') -> None:
        self._config = config
        self._symbol = symbol
        self._client: Any = None

    def _connect(self) -> Any:
        # Lazy connect so construction doesn't require the DB to be reachable.
        if self._client is not None:
            return self._client
        self._client = clickhouse_connect.get_client(
            host=self._config.host,
            port=self._config.port,
            username=self._config.user,
            password=self._config.password,
            database=self._config.database,
        )
        return self._client

    def get_window(self, symbol: str, kline_size: int, n_rows: int) -> pl.DataFrame:
        # Klines path intentionally delegated. Callers should use
        # `limen.HistoricalData().get_spot_klines(...)` and feed the
        # resulting frame to whatever consumer needs bars. This feed
        # owns the fill-path (trades), not the feature-path (klines).
        msg = (
            'ClickHouseFeed.get_window: klines are served by '
            'limen.HistoricalData().get_spot_klines(); this feed only '
            'provides trades via get_trades().'
        )
        raise NotImplementedError(msg)

    def get_trades(self, symbol: str, start: datetime, end: datetime) -> pl.DataFrame:
        assert_trades_causal(end, symbol=symbol)
        if symbol != self._symbol:
            msg = f'ClickHouseFeed configured for {self._symbol}; received {symbol}'
            raise ValueError(msg)
        client = self._connect()
        # query_arrow pulls a columnar batch -> Polars in bulk. Row-by-row
        # conversion is O(N × Python-dispatch) and hits ~30s for an hour of
        # ticks (~50K rows). Arrow is the bulk path.
        query = (
            'SELECT datetime, price, quantity, is_buyer_maker, trade_id '
            f'FROM {self._config.database}.binance_trades_complete '
            'WHERE datetime BETWEEN %(start)s AND %(end)s '
            'ORDER BY datetime, trade_id'
        )
        arrow = client.query_arrow(query, parameters={'start': start, 'end': end})
        frame = pl.from_arrow(arrow)
        if not isinstance(frame, pl.DataFrame):
            msg = f'expected DataFrame from Arrow conversion, got {type(frame).__name__}'
            raise TypeError(msg)
        if frame.is_empty():
            return pl.DataFrame(schema={
                'time': pl.Datetime(time_zone='UTC'),
                'price': pl.Float64, 'qty': pl.Float64,
                'is_buyer_maker': pl.Boolean, 'trade_id': pl.UInt64,
            })
        # ClickHouse `DateTime` returns as UInt32 seconds via Arrow. Convert
        # to tz-aware Datetime[μs, UTC] without `pl.from_epoch` because that
        # pathway calls `datetime.fromtimestamp` internally, which freezegun
        # patches — the patched call then stalls under a `freeze_time` block.
        # Multiplying into microseconds and casting directly to the typed
        # Datetime is fully numeric, touches no Python datetime object, and
        # therefore the freezegun patch cannot interpose.
        frame = frame.rename({'datetime': 'time', 'quantity': 'qty'}).with_columns(
            (pl.col('time').cast(pl.Int64) * 1_000_000)
                .cast(pl.Datetime('us', 'UTC'))
                .alias('time'),
            pl.col('is_buyer_maker').cast(pl.Boolean),
            pl.col('trade_id').cast(pl.UInt64),
            pl.col('price').cast(pl.Float64),
            pl.col('qty').cast(pl.Float64),
        )
        assert_window_causal(frame, symbol=symbol, column='time')
        return frame
