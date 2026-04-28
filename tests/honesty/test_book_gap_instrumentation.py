"""Book-gap instrumentation produces honest per-run latency metrics.

Pins slice #17 Task 11 / SPEC §9.5 venue-fidelity sub-rule.

The simulator's venue adapter walks historical trades to fill orders.
A stop-loss trigger should land at the first tape tick that crosses
the declared stop — not later. The book-gap instrument records the
delay between stop-cross and the trade that actually filled against
the stop, so the operator can see whether the venue is giving free
price improvements (zero gap, every time) or whether the simulator
is honestly pricing slippage (non-trivial gap distribution). A run
with a wide max gap is the smoke signal for a venue regression.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import polars as pl
import pytest

from backtest_simulator.honesty.book_gap import (
    BookGapInstrument,
    BookGapMetric,
)
from backtest_simulator.venue.types import PendingOrder

_T0 = datetime(2026, 4, 20, 0, 0, tzinfo=UTC)


def test_book_gap_max_in_report() -> None:
    """Max and p95 of stop-cross-to-trade gaps surface honestly.

    Drives 20 stop-cross events with linear gaps from 0 to 19s. The
    nearest-rank p95 with n=20 is index ceil(0.95 * 20) - 1 = 18 →
    value 18s. Max is 19s. The test pins both to catch a regression
    that swaps p95 for p99 or mean, or drops the max.
    """
    instr = BookGapInstrument()
    for i in range(20):
        instr.record_stop_cross(
            t_cross=_T0,
            t_first_trade=_T0 + timedelta(seconds=i),
        )
    snap = instr.snapshot()
    assert isinstance(snap, BookGapMetric), (
        f'snapshot must return BookGapMetric, got {type(snap)}'
    )
    assert snap.n_stops_observed == 20, (
        f'n_stops_observed should be 20, got {snap.n_stops_observed}'
    )
    assert snap.max_stop_cross_to_trade_seconds == 19.0, (
        f'max should be 19s, got {snap.max_stop_cross_to_trade_seconds}'
    )
    assert snap.p95_stop_cross_to_trade_seconds == 18.0, (
        f'p95 (nearest-rank) should be 18s for n=20 sequence 0..19; '
        f'got {snap.p95_stop_cross_to_trade_seconds}'
    )


def test_book_gap_max_in_report_unsorted_input() -> None:
    """Max and p95 are robust to insertion order — caller doesn't sort.

    The venue adapter records gaps in the order trades arrive on the
    tape, not sorted. The instrument must compute percentiles over
    the full sample regardless of input ordering. A regression that
    accidentally took max-of-last-N or relied on chronological order
    would silently under-report on a backwards-arriving sample. Pin
    by inserting the same 0..19 sequence in reverse and asserting
    identical metrics.
    """
    instr = BookGapInstrument()
    for i in range(19, -1, -1):
        instr.record_stop_cross(
            t_cross=_T0,
            t_first_trade=_T0 + timedelta(seconds=i),
        )
    snap = instr.snapshot()
    assert snap.max_stop_cross_to_trade_seconds == 19.0
    assert snap.p95_stop_cross_to_trade_seconds == 18.0
    assert snap.n_stops_observed == 20


def test_book_gap_p95_responds_to_distribution() -> None:
    """A different 20-sample distribution flips the p95 expectation.

    The earlier tests use values 0..19 (n=20), pinning p95=18. An
    implementation that hard-codes `p95=18 when n==20` would slip
    through. This test feeds 20 samples drawn from `[0]*19 + [100]`
    where the nearest-rank p95 lands on the 19th sorted value
    (index 18 → 0). Codex Task 11 round 1 pinned the
    fixed-distribution gap.
    """
    instr = BookGapInstrument()
    samples_seconds = [0.0] * 19 + [100.0]
    for s in samples_seconds:
        instr.record_stop_cross(
            t_cross=_T0,
            t_first_trade=_T0 + timedelta(seconds=s),
        )
    snap = instr.snapshot()
    assert snap.n_stops_observed == 20
    assert snap.max_stop_cross_to_trade_seconds == 100.0, (
        f'max should be 100s (single outlier); got '
        f'{snap.max_stop_cross_to_trade_seconds}'
    )
    assert snap.p95_stop_cross_to_trade_seconds == 0.0, (
        f'p95 (nearest-rank index 18 of [0]*19+[100]) should be 0s; '
        f'got {snap.p95_stop_cross_to_trade_seconds}. The instrument '
        f'is probably hard-coding p95=18 instead of computing it.'
    )


def test_book_gap_empty_snapshot() -> None:
    """A run with zero stop crosses produces zero metrics — not None.

    The downstream JSON report (`bts run --output-format json`) needs
    a numeric value in every field; a `None` would force the consumer
    to handle the missing case in addition to the zero-gap case. The
    instrument promises a numeric zero on empty.
    """
    snap = BookGapInstrument().snapshot()
    assert snap.n_stops_observed == 0
    assert snap.max_stop_cross_to_trade_seconds == 0.0
    assert snap.p95_stop_cross_to_trade_seconds == 0.0


def test_book_gap_rejects_inverted_timestamps() -> None:
    """t_first_trade < t_cross is non-causal and raises loudly.

    A tape that fills before the stop is crossed is either a clock
    glitch or a venue bug — either way the operator must know. The
    instrument refuses to compute a negative gap (which would
    silently shrink max / p95 toward zero) and raises ValueError.
    """
    instr = BookGapInstrument()
    with pytest.raises(ValueError, match='precedes t_cross'):
        instr.record_stop_cross(
            t_cross=_T0,
            t_first_trade=_T0 - timedelta(seconds=1),
        )


def test_book_gap_zero_gap_is_recorded() -> None:
    """t_first_trade == t_cross is the ideal case and counts as a sample.

    A zero-gap fill is what perfect price impact would look like.
    The instrument must still record the sample (so n_stops_observed
    increments) — silently dropping zero-gap events would let a
    pathological venue model that always fills at-the-stop look
    like it never observed any stops. Codex anti-pattern hunt.
    """
    instr = BookGapInstrument()
    instr.record_stop_cross(t_cross=_T0, t_first_trade=_T0)
    snap = instr.snapshot()
    assert snap.n_stops_observed == 1
    assert snap.max_stop_cross_to_trade_seconds == 0.0
    assert snap.p95_stop_cross_to_trade_seconds == 0.0


def _stop_walk_window(
    times_seconds: list[float], prices: list[str],
) -> pl.DataFrame:
    """Build a tiny polars DataFrame for `_walk_stop` testing."""
    rows = list(zip(times_seconds, prices, strict=True))
    return pl.DataFrame({
        'time': [_T0 + timedelta(seconds=t) for t, _ in rows],
        'price': [p for _, p in rows],
        'qty': ['1.0'] * len(rows),
    }).with_columns(pl.col('time').dt.replace_time_zone('UTC'))


def _make_stop_order(*, side: str, stop_price: str) -> PendingOrder:
    return PendingOrder(
        order_id='test-id',
        side=side, order_type='STOP_LOSS',
        qty=Decimal('1.0'),
        limit_price=None, stop_price=Decimal(stop_price),
        time_in_force='GTC',
        submit_time=_T0, symbol='BTCUSDT',
    )


def test_walk_stop_records_gap_first_row_trigger() -> None:
    """First-row trigger -> gap=0 but n_observed counts (codex round 1 P1).

    When the very first post-submit tape tick already crosses the
    stop, there is no prior sub-stop tick — `t_cross = t_first_trade`
    by convention, and the recorded gap is 0. The primitive's
    `n_stops_observed` MUST still increment to 1 so downstream
    aggregates (sweep total_stops) reflect the trigger.

    Mutation proof: dropping the zero-gap record would leave
    n_observed=0 and the assert below fires.
    """
    from backtest_simulator.venue.fills import _walk_stop
    from backtest_simulator.venue.filters import BinanceSpotFilters
    instrument = BookGapInstrument()
    # Single row at exactly stop price -> SELL stop triggers immediately.
    window = _stop_walk_window([0.0], ['100.0'])
    filters = BinanceSpotFilters.binance_spot('BTCUSDT')
    fills = _walk_stop(
        _make_stop_order(side='SELL', stop_price='100.0'),
        window, filters, instrument,
    )
    assert len(fills) == 1, 'SELL stop at 100 should trigger on row at 100'
    snap = instrument.snapshot()
    assert snap.n_stops_observed == 1
    assert snap.max_stop_cross_to_trade_seconds == 0.0


def test_walk_stop_records_gap_multi_row_trigger() -> None:
    """Gap = trigger_row.time - last_sub_stop_row.time (codex round 1 design).

    With a SELL stop at 100, a tape sequence (101, 99) triggers on
    the second row. The recorded gap equals (row[1].time - row[0].time)
    in seconds — the time during which price could have crossed the
    stop but no trade landed yet.

    Mutation proof: if `_walk_stop` records `t_cross = row[1].time`
    (instead of `row[0].time`), the gap would be 0 and this assert
    fires. If `_walk_stop` skips the record entirely, n_observed
    would be 0.
    """
    from backtest_simulator.venue.fills import _walk_stop
    from backtest_simulator.venue.filters import BinanceSpotFilters
    instrument = BookGapInstrument()
    # row 0 at t=0 price 101 (above 100, no trigger);
    # row 1 at t=5 price 99 (crosses 100 SELL stop).
    window = _stop_walk_window([0.0, 5.0], ['101.0', '99.0'])
    filters = BinanceSpotFilters.binance_spot('BTCUSDT')
    fills = _walk_stop(
        _make_stop_order(side='SELL', stop_price='100.0'),
        window, filters, instrument,
    )
    assert len(fills) == 1
    snap = instrument.snapshot()
    assert snap.n_stops_observed == 1
    assert snap.max_stop_cross_to_trade_seconds == 5.0, (
        f'gap should be 5s (row[1]-row[0]); got '
        f'{snap.max_stop_cross_to_trade_seconds}'
    )


def test_walk_stop_records_nothing_when_instrument_none() -> None:
    """`book_gap_instrument=None` skips recording — backward compat."""
    from backtest_simulator.venue.fills import _walk_stop
    from backtest_simulator.venue.filters import BinanceSpotFilters
    window = _stop_walk_window([0.0], ['100.0'])
    filters = BinanceSpotFilters.binance_spot('BTCUSDT')
    fills = _walk_stop(
        _make_stop_order(side='SELL', stop_price='100.0'),
        window, filters, None,
    )
    assert len(fills) == 1
    # No instrument -> no record, no aggregate observable.
