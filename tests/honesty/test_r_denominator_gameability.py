"""Honesty gate: stops too tight relative to local ATR are rejected.

Pins slice #17 Task 10 / SPEC §9.5 R-denominator gameability sub-rule.

R per trade is `|entry - declared_stop| * qty`. A strategy that
declares `stop = entry * 0.9999` (1 bp distance) makes R artificially
small — every realised loss looks like a tiny fraction of risk, every
win looks enormous in R units. The R-distribution becomes meaningless
for ranking strategies, and the operator sees a 10R win that is
really a 0.1% return.

`AtrSanityGate.evaluate` compares the declared stop distance against
`k * ATR(window)`: a stop closer than `k` of one local ATR is
rejected. ATR is supplied externally (the gate is side-effect free);
production wiring computes ATR from the feed at action-submit time
and hands it in.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import polars as pl

from backtest_simulator.honesty.atr import (
    AtrSanityDecision,
    AtrSanityGate,
    compute_atr_from_tape,
)


def test_r_denominator_gameability() -> None:
    """1 bp stop is rejected; a sane stop is allowed.

    Setup: entry = 70_000 USDT (BTC-like), local ATR = 350 USDT
    (50 bp of entry). Threshold: k=0.5, so any declared stop closer
    than 0.25 of one ATR (175 USDT) is rejected.

      Tight: stop = entry * 0.9999 → distance = 7.    7 < 175 → reject.
      Sane:  stop = entry * 0.995  → distance = 350.  350 ≥ 175 → allow.
    """
    entry = Decimal('70000')
    atr = Decimal('350')  # 50 bp of entry
    k = Decimal('0.5')
    gate = AtrSanityGate(atr_window_seconds=300, k=k)

    tight_stop = entry * Decimal('0.9999')  # 7 USDT below entry
    decision_tight = gate.evaluate(
        entry_price=entry, stop_price=tight_stop, atr=atr,
    )
    assert isinstance(decision_tight, AtrSanityDecision), (
        f'evaluate must return AtrSanityDecision, got {type(decision_tight)}'
    )
    assert decision_tight.allowed is False, (
        f'1 bp stop ({entry} → {tight_stop}, distance=7) must be rejected '
        f'against ATR={atr} k={k}; got allowed=True. R denominator is '
        f'gameable to ~zero.'
    )
    assert decision_tight.reason == 'stop_tighter_than_min_atr_fraction', (
        f'rejection reason must identify the tight-stop case; '
        f'got {decision_tight.reason!r}'
    )
    expected_distance = abs(entry - tight_stop)
    assert decision_tight.stop_distance == expected_distance, (
        f'stop_distance should be |entry-stop|={expected_distance}; '
        f'got {decision_tight.stop_distance}'
    )
    expected_min = k * atr
    assert decision_tight.min_required_distance == expected_min, (
        f'min_required_distance should be k*atr={expected_min}; '
        f'got {decision_tight.min_required_distance}'
    )

    sane_stop = entry * Decimal('0.995')  # 350 USDT below entry
    decision_sane = gate.evaluate(
        entry_price=entry, stop_price=sane_stop, atr=atr,
    )
    assert decision_sane.allowed is True, (
        f'50 bp stop ({entry} → {sane_stop}, distance=350) must be allowed '
        f'against ATR={atr} k={k} (min required = 175); '
        f'got allowed=False reason={decision_sane.reason!r}'
    )
    assert decision_sane.reason is None, (
        f'allowed decision should have reason=None; '
        f'got {decision_sane.reason!r}'
    )


def test_r_denominator_gameability_zero_atr_rejects() -> None:
    """ATR=0 is suspicious (flat tape / empty window); reject loudly.

    A strategy that submits during a halt or before any trades have
    landed cannot honestly compute R. The gate must not silently
    pass `min_required_distance = 0` and accept every stop — that
    would be the same gameability bug back via a degenerate
    threshold. Loud rejection is the honest answer; the caller
    should wait for ATR > 0 before submitting.
    """
    gate = AtrSanityGate(atr_window_seconds=300, k=Decimal('0.5'))
    entry = Decimal('70000')
    stop = entry * Decimal('0.999')
    decision = gate.evaluate(
        entry_price=entry, stop_price=stop, atr=Decimal('0'),
    )
    assert decision.allowed is False, (
        f'ATR=0 must reject all entries; got allowed=True '
        f'reason={decision.reason!r}'
    )
    assert decision.reason == 'atr_zero', (
        f'ATR=0 rejection must be tagged `atr_zero`, not the generic '
        f'tightness reason; got {decision.reason!r}'
    )


def test_r_denominator_gameability_negative_atr_rejects() -> None:
    """ATR<0 is a bad upstream feed; reject loudly.

    `min_required_distance = k * atr` with `atr<0` and `k>0`
    produces a negative threshold, which `stop_distance >= negative`
    trivially satisfies. The gate would silently disable itself
    against any garbage input. Loud rejection forces the caller to
    fix the ATR pipeline rather than continue with an inverted
    contract. Codex Task 10 round 1 pinned this gap.
    """
    gate = AtrSanityGate(atr_window_seconds=300, k=Decimal('0.5'))
    entry = Decimal('70000')
    tight_stop = entry * Decimal('0.9999')
    decision = gate.evaluate(
        entry_price=entry, stop_price=tight_stop, atr=Decimal('-100'),
    )
    assert decision.allowed is False, (
        f'ATR<0 must reject all entries; got allowed=True '
        f'reason={decision.reason!r}'
    )
    assert decision.reason == 'atr_negative', (
        f'ATR<0 rejection must be tagged `atr_negative` (distinct from '
        f'`atr_zero`) so the operator can tell whether the tape is flat '
        f'or the feed is broken; got {decision.reason!r}'
    )


def test_r_denominator_gameability_threshold_responds_to_k_and_atr() -> None:
    """A fixed stop distance flips allowed/rejected as `k * atr` varies.

    The earlier tests pinned a single `(k, atr)` pair → threshold
    175. An implementation could hard-code 175 and still pass. This
    test holds `stop_distance` fixed at 100 and varies the
    threshold across the boundary by changing `k` and `atr`
    independently, forcing any honest implementation to actually
    compute `k * atr`. Codex Task 10 round 1 pinned this.
    """
    entry = Decimal('70000')
    # Stop distance = 100 (entry minus stop exactly 100).
    stop = entry - Decimal('100')

    # Vary `atr` (k held at 0.5):
    #   atr=300 → threshold=150 → distance(100) < 150 → reject.
    #   atr=100 → threshold= 50 → distance(100) >  50 → allow.
    gate_k_half = AtrSanityGate(atr_window_seconds=300, k=Decimal('0.5'))
    decision_high_atr = gate_k_half.evaluate(
        entry_price=entry, stop_price=stop, atr=Decimal('300'),
    )
    decision_low_atr = gate_k_half.evaluate(
        entry_price=entry, stop_price=stop, atr=Decimal('100'),
    )
    assert decision_high_atr.allowed is False, (
        'distance=100 vs threshold=150 (k=0.5, atr=300) must reject; '
        'got allowed=True. The gate is not actually computing k*atr.'
    )
    assert decision_low_atr.allowed is True, (
        f'distance=100 vs threshold=50 (k=0.5, atr=100) must allow; '
        f'got allowed=False reason={decision_low_atr.reason!r}'
    )

    # Vary `k` (atr held at 200):
    #   k=1.0 → threshold=200 → distance(100) < 200 → reject.
    #   k=0.25 → threshold=50 → distance(100) >  50 → allow.
    gate_k_full = AtrSanityGate(atr_window_seconds=300, k=Decimal('1.0'))
    gate_k_quarter = AtrSanityGate(atr_window_seconds=300, k=Decimal('0.25'))
    decision_high_k = gate_k_full.evaluate(
        entry_price=entry, stop_price=stop, atr=Decimal('200'),
    )
    decision_low_k = gate_k_quarter.evaluate(
        entry_price=entry, stop_price=stop, atr=Decimal('200'),
    )
    assert decision_high_k.allowed is False, (
        'distance=100 vs threshold=200 (k=1.0, atr=200) must reject; '
        'got allowed=True. The gate is not actually computing k*atr.'
    )
    assert decision_low_k.allowed is True, (
        f'distance=100 vs threshold=50 (k=0.25, atr=200) must allow; '
        f'got allowed=False reason={decision_low_k.reason!r}'
    )


def test_r_denominator_gameability_short_side_symmetric() -> None:
    """A short with stop above entry uses |distance| — gate is symmetric.

    The R-per-trade formula already takes `abs(entry - stop)`, so
    the gate must be symmetric in side too. A 1 bp stop above
    entry must be rejected for the same reason the long-side 1 bp
    stop is.
    """
    gate = AtrSanityGate(atr_window_seconds=300, k=Decimal('0.5'))
    entry = Decimal('70000')
    atr = Decimal('350')
    tight_stop_above = entry * Decimal('1.0001')  # 7 USDT above entry
    decision = gate.evaluate(
        entry_price=entry, stop_price=tight_stop_above, atr=atr,
    )
    assert decision.allowed is False, (
        f'short-side 1 bp stop ({entry} → {tight_stop_above}) must be '
        f'rejected; the gate is side-symmetric. Got allowed=True.'
    )


def test_r_denominator_gameability_k_zero_disables_gate() -> None:
    """k=0 admits any non-negative stop distance, including 0.

    The gate stores k as the operator-controlled honesty knob. k=0
    is the "off" position: every stop_distance >= 0 satisfies
    `>= 0 * atr = 0`. Tests pin this so a future refactor that
    introduces a hidden floor cannot quietly tighten the contract
    on operators who explicitly disabled the gate.
    """
    gate = AtrSanityGate(atr_window_seconds=300, k=Decimal('0'))
    entry = Decimal('70000')
    atr = Decimal('350')
    # 1 bp stop — would be rejected at k=0.5; must pass at k=0.
    tight_stop = entry * Decimal('0.9999')
    decision = gate.evaluate(
        entry_price=entry, stop_price=tight_stop, atr=atr,
    )
    assert decision.allowed is True, (
        f'k=0 must disable the gate; got allowed=False '
        f'reason={decision.reason!r}'
    )


def test_r_denominator_gameability_k_zero_does_not_deny_on_bad_atr() -> None:
    """k=0 (gate disabled) admits orders even when ATR is missing/flat.

    The k=0 contract is "the operator turned the gate off" — the
    gate's only honest move is to step out of the way. Round-2's
    atr<=0 rejection must NOT fire when k=0, otherwise a disabled
    gate still denies on a flat tape (e.g. start-of-day, halt) or
    when the ATR feed is briefly absent. Codex Task 10 round 3
    pinned this contradiction.
    """
    gate = AtrSanityGate(atr_window_seconds=300, k=Decimal('0'))
    entry = Decimal('70000')
    stop = entry * Decimal('0.9999')

    decision_zero_atr = gate.evaluate(
        entry_price=entry, stop_price=stop, atr=Decimal('0'),
    )
    assert decision_zero_atr.allowed is True, (
        f'k=0 gate must admit on atr=0; got allowed=False '
        f'reason={decision_zero_atr.reason!r}'
    )

    decision_neg_atr = gate.evaluate(
        entry_price=entry, stop_price=stop, atr=Decimal('-100'),
    )
    assert decision_neg_atr.allowed is True, (
        f'k=0 gate must admit on atr<0 too; got allowed=False '
        f'reason={decision_neg_atr.reason!r}'
    )

    # Signalling-NaN and Infinity ATR sentinels: a disabled gate
    # must NOT raise InvalidOperation when computing `k * atr`
    # (round-4 codex finding). The short-circuit at k==0 has to
    # come BEFORE the multiplication.
    decision_nan_atr = gate.evaluate(
        entry_price=entry, stop_price=stop, atr=Decimal('NaN'),
    )
    assert decision_nan_atr.allowed is True, (
        f'k=0 gate must admit on NaN ATR; got allowed=False '
        f'reason={decision_nan_atr.reason!r}'
    )

    decision_inf_atr = gate.evaluate(
        entry_price=entry, stop_price=stop, atr=Decimal('Infinity'),
    )
    assert decision_inf_atr.allowed is True, (
        f'k=0 gate must admit on Infinity ATR; got allowed=False '
        f'reason={decision_inf_atr.reason!r}'
    )


def test_r_denominator_gameability_negative_k_rejected_at_construction() -> None:
    """Constructing the gate with k<0 raises loudly.

    A negative `k` makes `min_required = k * atr` negative for any
    positive ATR, so `stop_distance >= negative` admits every stop
    and the gate silently disables itself. Reject at construction
    so the misconfiguration surfaces at the wiring site instead of
    leaking through every evaluate call. Codex Task 10 round 2
    pinned this gap, symmetric to the `atr_negative` reason vector.
    """
    import pytest
    with pytest.raises(ValueError, match='must be non-negative'):
        AtrSanityGate(atr_window_seconds=300, k=Decimal('-0.5'))


def test_atr_sanity_gate_exposes_constructor_args() -> None:
    """`atr_window_seconds` and `k` are operator-visible via properties.

    Run reports / honesty diagnostics need to surface the gate's
    configured threshold so the operator can correlate
    `min_required_distance` against the policy. Properties are the
    minimal surface; getter methods would invite call-site
    boilerplate.
    """
    gate = AtrSanityGate(atr_window_seconds=300, k=Decimal('0.5'))
    assert gate.atr_window_seconds == 300
    assert gate.k == Decimal('0.5')


# `compute_atr_from_tape` — the data-source helper that the bts
# venue/launcher path uses to build the `atr_provider` callable
# threaded into action_submitter. ATR is mean of true range per
# period over the supplied strict-causal tape — TR_i = max(H_i -
# L_i, |H_i - C_{i-1}|, |L_i - C_{i-1}|), Wilder's formula. Tests
# pin the formula so the gate rejection counter `n_atr_rejected`
# you read in `bts run --output-format json` reflects a real
# volatility floor, not a hard-coded one or a range-only
# understatement.

def _trades(rows: list[tuple[datetime, float]]) -> pl.DataFrame:
    return pl.DataFrame({
        'time': [r[0] for r in rows],
        'price': [r[1] for r in rows],
    }).with_columns(pl.col('time').cast(pl.Datetime('us', 'UTC')))


def test_compute_atr_from_tape_basic() -> None:
    """Two 1-min periods → ATR = mean of true-range per bucket.

    True range includes gap-vs-previous-close (Wilder's ATR).
    Bucket 1 (12:00-12:01): high=70_100 low=70_000 → TR_1 = H-L = 100
    (no prev_close).
    Bucket 2 (12:01-12:02): high=70_080 low=70_050 close=70_080.
    Last trade in bucket 1 is 70_100 (the higher of the two), so
    prev_close=70_100. TR_2 = max(80-50, |80-100|, |50-100|) =
    max(30, 20, 50) = 50.
    Mean(TR) = (100 + 50) / 2 = 75.
    """
    t0 = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    rows = [
        (t0 + timedelta(seconds=10), 70_000.0),
        (t0 + timedelta(seconds=30), 70_100.0),
        (t0 + timedelta(seconds=70), 70_050.0),
        (t0 + timedelta(seconds=90), 70_080.0),
    ]
    atr = compute_atr_from_tape(
        trades_pre_decision=_trades(rows), period_seconds=60,
    )
    assert atr is not None
    assert atr == Decimal('75')


def test_compute_atr_from_tape_gap_between_buckets() -> None:
    """Bucket gaps materially increase ATR vs intra-bucket range alone.

    Auditor round 2 P1: range-only ATR understates volatility on
    tapes that gap BETWEEN buckets but stay tight WITHIN each
    bucket. True range catches the gap via |H - prev_close| /
    |L - prev_close|.

    Setup:
      Bucket 1 (12:00-12:01): prices 100, 101 → H=101 L=100 close=101
      Bucket 2 (12:01-12:02): prices 110, 111 → H=111 L=110 close=111

    Range-only impl:
      TR_1 = 101 - 100 = 1
      TR_2 = 111 - 110 = 1
      mean = 1   ← would let very tight stops through

    True-range impl:
      TR_1 = 1 (no prev_close)
      TR_2 = max(1, |111-101|, |110-101|) = max(1, 10, 9) = 10
      mean = 5.5   ← honest floor

    Mutation proof: a regression to range-only would compute
    Decimal('1') here and the assertion fails.
    """
    t0 = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    rows = [
        (t0 + timedelta(seconds=10), 100.0),
        (t0 + timedelta(seconds=30), 101.0),
        (t0 + timedelta(seconds=70), 110.0),
        (t0 + timedelta(seconds=90), 111.0),
    ]
    atr = compute_atr_from_tape(
        trades_pre_decision=_trades(rows), period_seconds=60,
    )
    assert atr is not None
    assert atr == Decimal('5.5'), (
        f'expected true-range ATR=5.5 (TR_1=1, TR_2=10, mean=5.5); '
        f'got {atr}. A range-only impl would return 1.'
    )


def test_compute_atr_from_tape_empty_returns_none() -> None:
    """Empty tape → None (the gate's `ATR_UNCALIBRATED` rejection signal)."""
    empty = pl.DataFrame({
        'time': pl.Series('time', [], dtype=pl.Datetime('us', 'UTC')),
        'price': pl.Series('price', [], dtype=pl.Float64),
    })
    assert compute_atr_from_tape(
        trades_pre_decision=empty, period_seconds=60,
    ) is None


def test_compute_atr_from_tape_flat_returns_zero() -> None:
    """All same price → ATR=0 (the gate's `atr_zero` reject path).

    The honest answer is "no volatility", which the gate itself
    reads as a flat tape and rejects loudly. Don't return None
    here — that would conflate "no data" with "data shows zero
    movement" and an operator could not tell which case fired.
    """
    t0 = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    rows = [(t0 + timedelta(seconds=i * 10), 70_000.0) for i in range(6)]
    atr = compute_atr_from_tape(
        trades_pre_decision=_trades(rows), period_seconds=60,
    )
    assert atr == Decimal('0')


def test_compute_atr_from_tape_period_changes_bucket_count() -> None:
    """Halving period_seconds doubles the bucket count → different mean.

    Pins that period_seconds is actually applied. A volume-blind
    impl that ignored the parameter would produce identical ATR
    for both calls.
    """
    t0 = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    rows = [
        (t0 + timedelta(seconds=15), 70_000.0),
        (t0 + timedelta(seconds=45), 70_100.0),
        (t0 + timedelta(seconds=75), 70_050.0),
        (t0 + timedelta(seconds=105), 70_080.0),
    ]
    df = _trades(rows)
    atr_60 = compute_atr_from_tape(
        trades_pre_decision=df, period_seconds=60,
    )
    atr_30 = compute_atr_from_tape(
        trades_pre_decision=df, period_seconds=30,
    )
    assert atr_60 != atr_30, (
        f'period_seconds must change the bucket boundaries; '
        f'period=60 gave {atr_60}, period=30 gave {atr_30}'
    )
