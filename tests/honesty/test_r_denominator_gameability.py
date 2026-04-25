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

from decimal import Decimal

from backtest_simulator.honesty.atr import AtrSanityDecision, AtrSanityGate


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
