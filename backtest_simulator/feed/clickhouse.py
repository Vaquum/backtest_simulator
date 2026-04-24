"""ClickHouseFeed — real trades from `origo.binance_daily_spot_trades`; klines via Limen."""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

import polars as pl
import pyarrow as pa
from clickhouse_connect import get_client as _ch_get_client
from clickhouse_connect.driver.client import Client

from backtest_simulator.feed.lookahead import (
    assert_trades_causal,
    assert_window_causal,
)

# Eager import of `clickhouse_connect` is deliberate. Its `driver.tzutil`
# submodule calls `dateutil.tz.tzlocal().tzname(None)` at import time, and
# that path blows up under `freezegun.freeze_time`. Loading the module once,
# at program start, caches it in `sys.modules` before any `freeze_time`
# block can patch `datetime`.

# `origo.binance_daily_spot_trades` stores BTCUSDT only — no `symbol`
# column exists. ClickHouse's `DateTime64(6)` parameter binder rejects
# ISO-T timestamps with `+00:00` tz suffix; we format explicitly as
# `'%Y-%m-%d %H:%M:%S.%f'` (naive, microsecond precision) and the server
# interprets it in UTC because the column is UTC-stored.
_TRADES_DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'


@dataclass(frozen=True)
class ClickHouseConfig:
    """Connection params. Use `from_env` for the standard deployment."""

    host: str
    port: int
    user: str
    password: str
    database: str = 'origo'
    trades_table: str = 'binance_daily_spot_trades'

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
            msg = f'ClickHouseConfig.from_env: missing env vars: {", ".join(missing)}'
            raise RuntimeError(msg)
        return cls(
            host=host, port=int(port_raw), user=user, password=password,
            database=database, trades_table=trades_table,
        )


class ClickHouseFeed:
    """Trades feed backed by `origo.binance_daily_spot_trades`.

    Every `get_trades` call hits the table and returns rows with
    `datetime BETWEEN start AND end`. Look-ahead guard enforces
    `end <= frozen_now() + venue_lookahead_seconds` on every call.

    The `origo.binance_daily_spot_trades` table stores BTCUSDT only, so
    there is no `symbol` column to filter on — the `symbol` argument is
    still accepted for provenance/assertions.

    Klines are NOT served here — use `limen.HistoricalData().get_spot_klines()`
    per the experiment pattern. Limen aggregates from the same Binance trade
    source, so the two feeds are consistent.
    """

    def __init__(self, config: ClickHouseConfig, symbol: str = 'BTCUSDT') -> None:
        self._config = config
        self._symbol = symbol
        self._client: Client | None = None

    def _connect(self) -> Client:
        # Lazy connect so construction doesn't require the DB to be reachable.
        if self._client is not None:
            return self._client
        self._client = _make_client(
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
        del kline_size, n_rows, symbol
        msg = (
            'ClickHouseFeed.get_window: klines are served by '
            'limen.HistoricalData().get_spot_klines(); this feed only '
            'provides trades via get_trades().'
        )
        raise NotImplementedError(msg)

    def get_trades(
        self, symbol: str, start: datetime, end: datetime,
        *, venue_lookahead_seconds: int = 0,
    ) -> pl.DataFrame:
        assert_trades_causal(end, symbol=symbol, venue_lookahead_seconds=venue_lookahead_seconds)
        if symbol != self._symbol:
            msg = f'ClickHouseFeed configured for {self._symbol}; received {symbol}'
            raise ValueError(msg)
        client = self._connect()
        # query_arrow pulls a columnar batch -> Polars in bulk. Row-by-row
        # conversion is O(N x Python-dispatch) and hits ~30s for an hour of
        # ticks (~50K rows). Arrow is the bulk path. `datetime` is
        # DateTime64(6); clickhouse-connect's native datetime binder
        # rejects timezone-aware inputs for DateTime64 — format the two
        # bounds ourselves and bind them as strings.
        query = (
            'SELECT datetime, price, quantity, is_buyer_maker, trade_id '
            f'FROM {self._config.database}.{self._config.trades_table} '
            'WHERE datetime BETWEEN %(start)s AND %(end)s '
            'ORDER BY datetime, trade_id'
        )
        start_str = _format_datetime64(start)
        end_str = _format_datetime64(end)
        arrow = _query_arrow(client, query, parameters={'start': start_str, 'end': end_str})
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
        # `datetime` comes back as DateTime64(6) — Arrow timestamp[us]. We
        # rename/retype via Polars without going through `pl.from_epoch`
        # (that path calls `datetime.fromtimestamp` internally and stalls
        # under an active `freeze_time` block).
        frame = frame.rename({'datetime': 'time', 'quantity': 'qty'}).with_columns(
            pl.col('time').cast(pl.Datetime('us', 'UTC')),
            pl.col('is_buyer_maker').cast(pl.Boolean),
            pl.col('trade_id').cast(pl.UInt64),
            pl.col('price').cast(pl.Float64),
            pl.col('qty').cast(pl.Float64),
        )
        assert_window_causal(
            frame, symbol=symbol, column='time',
            venue_lookahead_seconds=venue_lookahead_seconds,
        )
        return frame


def _format_datetime64(value: datetime) -> str:
    """Format a datetime for ClickHouse DateTime64(6) parameter binding.

    `DateTime64(6)` rejects ISO-T with `+00:00`; we emit
    `'%Y-%m-%d %H:%M:%S.%f'` which the server interprets as UTC.
    Timezone-aware inputs are normalised to UTC first; naive inputs are
    assumed UTC (caller's contract).
    """
    if value.tzinfo is not None:
        from datetime import UTC
        value = value.astimezone(UTC).replace(tzinfo=None)
    return value.strftime(_TRADES_DATETIME_FORMAT)


# clickhouse_connect exposes `get_client` and `Client.query_arrow`
# via signatures that include `**kwargs: Any`. Reading those as
# member-access expressions flags `reportUnknownMemberType`. The
# fixes used here:
#   - `get_client` is imported as a top-level symbol via
#     `from clickhouse_connect import get_client as _ch_get_client`.
#     The import name resolves at module load (typed), so the call
#     site reads clean — no member access on the package object.
#   - `Client.query_arrow` is necessarily a method on the client
#     instance; we wrap the call once and check the return at the
#     boundary.

def _make_client(
    *, host: str, port: int, username: str, password: str, database: str,
) -> Client:
    return _ch_get_client(
        host=host, port=port, username=username,
        password=password, database=database,
    )


def _query_arrow(
    client: Client, query: str, *, parameters: Mapping[str, str],
) -> pa.Table:
    raw_result = client.query_arrow(query, parameters=dict(parameters))
    if not isinstance(raw_result, pa.Table):
        msg = (
            f'ClickHouseFeed._query_arrow: expected pyarrow.Table '
            f'from query_arrow, got {type(raw_result).__name__}'
        )
        raise TypeError(msg)
    return raw_result
