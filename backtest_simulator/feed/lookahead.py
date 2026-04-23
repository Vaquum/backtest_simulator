"""LookAheadGuard — the single enforcement site for causal feed reads."""
from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from backtest_simulator.exceptions import LookAheadViolation


def frozen_now() -> datetime:
    """Current frozen time in UTC. freezegun provides the actual value."""
    return datetime.now(UTC)


def assert_window_causal(df: pl.DataFrame, symbol: str, column: str = 'open_time') -> None:
    """Raise LookAheadViolation if any row has `column > frozen_now()`."""
    if df.is_empty():
        return
    now = frozen_now()
    latest = df[column].max()
    if latest is not None and _to_utc(latest) > now:
        raise LookAheadViolation(
            f'feed returned row with {column}={latest} > frozen_now()={now} '
            f'for symbol={symbol}',
        )


def assert_trades_causal(end: datetime, symbol: str) -> None:
    """Raise LookAheadViolation if end > frozen_now()."""
    now = frozen_now()
    end_utc = _to_utc(end)
    if end_utc > now:
        raise LookAheadViolation(
            f'get_trades(symbol={symbol}, end={end_utc}) requested data past '
            f'frozen_now()={now}',
        )


def _to_utc(value: object) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    msg = f'expected datetime, got {type(value).__name__}'
    raise TypeError(msg)
