"""Book-gap instrumentation — record stop-cross-to-first-trade latency."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class BookGapMetric:

    max_stop_cross_to_trade_seconds: float
    n_stops_observed: int
    p95_stop_cross_to_trade_seconds: float

class BookGapInstrument:

    def __init__(self) -> None:
        self._gaps_seconds: list[float] = []

    def record_stop_cross(
        self, *, t_cross: datetime, t_first_trade: datetime,
    ) -> None:
        if t_first_trade < t_cross:
            msg = (
                f'record_stop_cross: t_first_trade {t_first_trade!r} '
                f'precedes t_cross {t_cross!r}; the venue cannot fill '
                f'before the stop is crossed. This is a non-causal '
                f'event and breaks the lookahead contract.'
            )
            raise ValueError(msg)
        gap = (t_first_trade - t_cross).total_seconds()
        self._gaps_seconds.append(gap)

    def snapshot(self) -> BookGapMetric:
        if not self._gaps_seconds:
            return BookGapMetric(
                max_stop_cross_to_trade_seconds=0.0,
                n_stops_observed=0,
                p95_stop_cross_to_trade_seconds=0.0,
            )
        ordered = sorted(self._gaps_seconds)
        n = len(ordered)
        p95_index = max(0, min(n - 1, int(0.95 * n + 0.999999) - 1))
        return BookGapMetric(
            max_stop_cross_to_trade_seconds=ordered[-1],
            n_stops_observed=n,
            p95_stop_cross_to_trade_seconds=ordered[p95_index],
        )
