"""Minimal stub for clickhouse_connect.driver.client.Client."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import pyarrow as pa


class QueryResult:
    result_rows: Sequence[Sequence[object]]


class Client:
    def query_arrow(
        self,
        query: str,
        *,
        parameters: Mapping[str, str] | None = ...,
    ) -> pa.Table: ...
    def query(
        self,
        query: str,
        *,
        parameters: Mapping[str, str] | None = ...,
    ) -> QueryResult: ...
    def close(self) -> None: ...
