"""LookAheadGuard — the single enforcement site for causal feed reads."""
from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from backtest_simulator.exceptions import LookAheadViolation


def frozen_now() -> datetime:
    """Current frozen time in UTC. freezegun provides the actual value."""
    return datetime.now(UTC)


def assert_window_causal(
    df: pl.DataFrame, symbol: str, column: str = 'open_time',
    *, venue_lookahead_seconds: int = 0,
) -> None:
    """Raise LookAheadViolation if any row has `column > frozen_now() + venue_lookahead_seconds`.

    See `assert_trades_causal` for the rationale of `venue_lookahead_seconds`.
    """
    if df.is_empty():
        return
    now = frozen_now()
    latest = df[column].max()
    if latest is None:
        return
    from datetime import timedelta
    ceiling = now + timedelta(seconds=venue_lookahead_seconds)
    if _to_utc(latest) > ceiling:
        raise LookAheadViolation(
            f'feed returned row with {column}={latest} > frozen_now()={now} '
            f'+ venue_lookahead_seconds={venue_lookahead_seconds} for symbol={symbol}',
        )


def assert_trades_causal(
    end: datetime, symbol: str, *, venue_lookahead_seconds: int = 0,
) -> None:
    """Raise LookAheadViolation if end > frozen_now() + venue_lookahead_seconds.

    `venue_lookahead_seconds=0` is the strategy-facing default: no peek
    beyond the frozen wall clock. `venue_lookahead_seconds>0` carves out
    a bounded window for the simulated venue to consult future trades
    while it simulates fills — real venues see trades arrive over a
    realistic submit/fill latency window, and refusing that makes the
    adapter unable to fill any order. The fill-window bound is the
    adapter's declared `trade_window_seconds`, so this escape hatch is
    also a contract: venues may look ahead by at most the declared
    window and not a millisecond more.
    """
    now = frozen_now()
    end_utc = _to_utc(end)
    from datetime import timedelta
    ceiling = now + timedelta(seconds=venue_lookahead_seconds)
    if end_utc > ceiling:
        raise LookAheadViolation(
            f'get_trades(symbol={symbol}, end={end_utc}) requested data past '
            f'frozen_now()={now} + venue_lookahead_seconds={venue_lookahead_seconds}',
        )


def _to_utc(value: object) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    msg = f'expected datetime, got {type(value).__name__}'
    raise TypeError(msg)
