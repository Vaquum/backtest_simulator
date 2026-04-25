"""Book-gap instrumentation — record stop-cross-to-first-trade latency."""
from __future__ import annotations

# Slice #17 Task 11: instrument the gap between when the tape price
# first crosses a declared stop and when a trade actually lands at
# that stop level. Praxis publishes the full order book snapshot;
# Binance re-publishes ticks at the head of the book. A wide gap
# means the venue gave the strategy a price improvement that real
# books would not have provided — and the simulator silently issued
# free fills.
#
# `record_stop_cross(t_cross, t_first_trade)` is called by the venue
# adapter every time a STOP_*  walk steps over the declared stop. The
# instrument keeps a running list of gaps in seconds. `snapshot()`
# returns a `BookGapMetric` with max, p95, and total count — the
# fields surfaced into `bts run --output-format json`.
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class BookGapMetric:
    """Per-run summary of stop-cross-to-first-trade gaps in seconds.

    `max_stop_cross_to_trade_seconds` and
    `p95_stop_cross_to_trade_seconds` are the operator-visible
    health metrics. `n_stops_observed` is the denominator — both
    percentiles are meaningless on a tiny sample, so the report
    surfaces the count alongside.
    """

    max_stop_cross_to_trade_seconds: float
    n_stops_observed: int
    p95_stop_cross_to_trade_seconds: float


class BookGapInstrument:
    """Accumulate stop-cross-to-first-trade gap samples across a run.

    The instrument is stateful; one per run. Thread-safe access is
    not required because the venue adapter is single-threaded
    inside `walk_trades`. The clock comes in via the t_cross /
    t_first_trade arguments — the instrument never reads time
    itself, so freezegun-frozen tests get deterministic results.
    """

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
        # p95 via nearest-rank: ceil(0.95 * n) - 1, clamped to [0, n-1].
        p95_index = max(0, min(n - 1, int(0.95 * n + 0.999999) - 1))
        return BookGapMetric(
            max_stop_cross_to_trade_seconds=ordered[-1],
            n_stops_observed=n,
            p95_stop_cross_to_trade_seconds=ordered[p95_index],
        )
