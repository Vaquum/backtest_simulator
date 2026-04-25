"""Minimal stub for clickhouse_connect.driver.client.Client."""
from __future__ import annotations

from collections.abc import Mapping

import pyarrow as pa


class Client:
    def query_arrow(
        self,
        query: str,
        *,
        parameters: Mapping[str, str] | None = ...,
    ) -> pa.Table: ...
    def close(self) -> None: ...
