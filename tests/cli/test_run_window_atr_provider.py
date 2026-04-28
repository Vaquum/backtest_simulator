"""Mutation-proof tests for `_build_atr_gate_and_provider`'s strict-causal seam.

Auditor: the BTS seam where ATR is computed for the action-submitter
gate fetches `[t - window, t)` from the feed and then filters
`pl.col('time') < t` BEFORE handing the slice to
`compute_atr_from_tape`. The pure ATR math (`compute_atr_from_tape`)
is tested in `tests/honesty/test_r_denominator_gameability.py`; the
gate hook is tested in `tests/launcher/test_action_submitter.py`.
This file pins the GLUE in `_run_window._build_atr_gate_and_provider`
that runtime depends on to keep ATR strict-causal — without it, the
gate would silently consume the post-decision tick.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import polars as pl

from backtest_simulator.cli import _run_window


class _CapturingFeed:
    """Feed stub that records every `_get_trades_for_venue` call.

    Returns a deterministic 4-tick slice spanning `[t - window, t]`
    (note: INCLUDES `t`, so the post-`<` filter inside the provider
    is observable: any tick at `time == t` must be dropped before
    `compute_atr_from_tape` sees the slice).
    """

    def __init__(self, ts: datetime, window_seconds: int) -> None:
        self.ts = ts
        self.window = window_seconds
        self.calls: list[dict[str, object]] = []
        # Deterministic 4-tick fixture. The SECOND-to-last is at
        # `ts - 1s` (kept), the LAST is at `ts` exactly (must be
        # dropped by the strict-causal filter).
        self.tape = pl.DataFrame({
            'time': [
                ts - timedelta(seconds=window_seconds),
                ts - timedelta(seconds=window_seconds // 2),
                ts - timedelta(seconds=1),
                ts,  # at-decision tick — must NOT reach compute_atr
            ],
            'price': [70_000.0, 70_050.0, 70_100.0, 70_500.0],
        }).with_columns(pl.col('time').cast(pl.Datetime('us', 'UTC')))

    def _get_trades_for_venue(
        self, symbol: str, start: datetime, end: datetime,
        *, venue_lookahead_seconds: int,
    ) -> pl.DataFrame:
        self.calls.append({
            'symbol': symbol,
            'start': start,
            'end': end,
            'venue_lookahead_seconds': venue_lookahead_seconds,
        })
        return self.tape


def test_atr_provider_fetches_strict_causal_window() -> None:
    """Provider must request `[t - window, t)` with no venue lookahead.

    The fetched range AND the `venue_lookahead_seconds=0` kwarg
    are the two knobs that make the slice strict-causal at the
    feed boundary. Mutation proof:
      - Changing `t - window_seconds` to `t - 2 * window_seconds`
        flips the recorded `start`; assert fires.
      - Dropping `venue_lookahead_seconds=0` (defaulting to non-
        zero) flips the recorded kwarg; assert fires.
    """
    ts = datetime(2024, 12, 1, 12, 5, 0, tzinfo=timezone.utc)
    feed = _CapturingFeed(ts, window_seconds=900)
    _gate, atr_provider = _run_window._build_atr_gate_and_provider(
        feed, k=Decimal('0.5'), window_seconds=900,
    )

    atr_provider('BTCUSDT', ts)

    assert len(feed.calls) == 1, (
        f'provider must call feed exactly once per atr_provider '
        f'invocation; got {len(feed.calls)} calls'
    )
    call = feed.calls[0]
    assert call['symbol'] == 'BTCUSDT'
    assert call['start'] == ts - timedelta(seconds=900), (
        f"provider must request `[t - window_seconds, t)`; got "
        f"start={call['start']} (expected {ts - timedelta(seconds=900)})"
    )
    assert call['end'] == ts, (
        f"provider's end must be t exactly (t - window, t); got "
        f"end={call['end']} (expected {ts})"
    )
    assert call['venue_lookahead_seconds'] == 0, (
        f'provider must pass `venue_lookahead_seconds=0`; got '
        f'{call["venue_lookahead_seconds"]} (any non-zero would '
        f'let post-decision ticks bleed into the slice).'
    )


def test_atr_provider_filters_out_at_decision_tick(
    monkeypatch,
) -> None:
    """Provider must filter `pl.col('time') < t` before computing ATR.

    The feed boundary `[t - window, t)` is half-open in INTENT,
    but `_get_trades_for_venue`'s exact end-inclusive vs
    end-exclusive semantics is venue-implementation-defined.
    The `pl.col('time') < t` filter inside the provider is the
    bts-side BACKSTOP that guarantees no tick AT or AFTER t
    leaks into ATR — any leak would let the gate "see" the
    decision-time price when judging volatility before the
    decision.

    Mutation proof: replacing the filter with
    `pl.col('time') <= t` lets the at-decision tick through;
    `compute_atr_from_tape` then receives 4 rows instead of 3,
    and the assert fires.
    """
    ts = datetime(2024, 12, 1, 12, 5, 0, tzinfo=timezone.utc)
    feed = _CapturingFeed(ts, window_seconds=900)

    captured: dict[str, object] = {}

    def _capture(
        *, trades_pre_decision: pl.DataFrame, period_seconds: int,
    ) -> Decimal | None:
        captured['n_rows'] = trades_pre_decision.height
        captured['max_time'] = trades_pre_decision['time'].max()
        captured['period_seconds'] = period_seconds
        # Real implementation returns a Decimal; the seam test
        # doesn't care about the value, only the input.
        return Decimal('100')

    # Patch `compute_atr_from_tape` on the import path the
    # provider uses (its closure imports it locally inside the
    # provider via `from backtest_simulator.honesty.atr import
    # compute_atr_from_tape`). Patch at the module level so the
    # local import hits the patched version.
    from backtest_simulator.honesty import atr as atr_module
    monkeypatch.setattr(
        atr_module, 'compute_atr_from_tape', _capture,
    )

    _gate, atr_provider = _run_window._build_atr_gate_and_provider(
        feed, k=Decimal('0.5'), window_seconds=900,
    )
    atr_provider('BTCUSDT', ts)

    # The fixture has 4 rows: one each at `ts - 900`, `ts - 450`,
    # `ts - 1`, and `ts` itself. The strict-causal filter must
    # drop the `ts` row, leaving 3.
    assert captured['n_rows'] == 3, (
        f'provider must filter `time < t`; expected 3 rows after '
        f'filter, got {captured["n_rows"]}. The 4th tick at '
        f't={ts} is post-decision and must NOT reach '
        f'compute_atr_from_tape.'
    )
    max_time = captured['max_time']
    assert max_time is not None
    # Confirm the kept-tick is `ts - 1s`, NOT `ts`.
    assert max_time < ts, (
        f'max time in pre-decision slice must be strictly less '
        f'than t={ts}; got max={max_time}'
    )
    # And the period_seconds=60 contract.
    assert captured['period_seconds'] == 60, (
        f'provider must call compute_atr_from_tape with '
        f'period_seconds=60 (per-1-min bucket); got '
        f'{captured["period_seconds"]}'
    )
