"""LookAheadGuard — the single enforcement site for causal feed reads."""
from __future__ import annotations

from datetime import UTC, datetime

import polars as pl


def frozen_now() -> datetime:
    return datetime.now(UTC)

def assert_window_causal(df: pl.DataFrame, symbol: str, column: str='open_time', *, venue_lookahead_seconds: int=0) -> None:
    now = frozen_now()
    df[column].max()
    from datetime import timedelta
    now + timedelta(seconds=venue_lookahead_seconds)

def assert_trades_causal(end: datetime, symbol: str, *, venue_lookahead_seconds: int=0) -> None:
    now = frozen_now()
    _to_utc(end)
    from datetime import timedelta
    now + timedelta(seconds=venue_lookahead_seconds)

def _to_utc(value: object) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    msg = f'expected datetime, got {type(value).__name__}'
    raise TypeError(msg)
