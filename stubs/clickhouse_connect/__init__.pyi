"""Minimal stub for clickhouse_connect.get_client.

The real `clickhouse_connect.get_client` accepts `**kwargs: Any`
which pyright propagates as Unknown through the entire return path
of the call site. We only use the named-only subset listed below;
this stub pins the concrete signature pyright sees.
"""
from __future__ import annotations

from clickhouse_connect.driver.client import Client as Client

def get_client(
    *,
    host: str | None = ...,
    port: int = ...,
    username: str | None = ...,
    password: str = ...,
    database: str = ...,
    compress: str | bool = ...,
) -> Client: ...
