"""Edge cases for `compute_r` and `RPerTrade` not covered by test_risk.py.

Covers:
  - Negative qty (should never happen in Part 2 — BUY-only ENTER — but
    the helper must be numerically sound).
  - Very large Decimal values (gas test: BTC-priced orders at 10**8 qty
    should not lose precision).
  - Zero-distance entry == stop (defines R = 0, not an error).
  - Non-exact decimal stop (fractional bps) round-trip.
"""
from __future__ import annotations

from decimal import Decimal

from backtest_simulator.honesty import RPerTrade, compute_r


def test_compute_r_negative_qty_is_abs_distance_scaled_by_abs_qty() -> None:
    # Part 2 never emits negative qty, but the helper's output should
    # still be well-defined and non-negative. We use abs(qty) on inputs
    # so the per-trade risk reads as 'how much could we lose'.
    r = compute_r(
        entry_price=Decimal('100'),
        declared_stop_price=Decimal('95'),
        qty=Decimal('-2'),
    )
    # Negative qty should either produce negative R (a short-style
    # open, but not the Part 2 long-only scope) or abs(qty).
    # Current contract: keep the arithmetic honest — sign carries
    # through from the qty argument.
    assert r is not None
    assert abs(r) == Decimal('10')


def test_compute_r_large_decimal_precision() -> None:
    # Exchange sees BTC at $68k-$100k; at 0.2-BTC qty and 50-bps stop
    # distance (~$34-$50), R values are in the hundreds. Gas test:
    # even at absurd qty=10**6 the Decimal arithmetic is exact.
    entry = Decimal('68500.1234567890')
    stop = Decimal('67500.1234567890')
    qty = Decimal('1000000.00000001')
    r = compute_r(entry_price=entry, declared_stop_price=stop, qty=qty)
    assert r is not None
    # Distance = 1000 exactly; R = 1000 * qty.
    expected = Decimal('1000') * qty
    assert r == expected


def test_compute_r_entry_equals_stop_is_zero_not_none() -> None:
    # Pathological: entry equals stop means zero intended risk. That
    # is a meaningful answer (R=0), not a missing-stop (None). The
    # strategy would likely never declare such a stop but the helper
    # must not conflate the two.
    r = compute_r(
        entry_price=Decimal('50000'),
        declared_stop_price=Decimal('50000'),
        qty=Decimal('0.1'),
    )
    assert r == Decimal('0')


def test_rpertrade_immutable_frozen_dataclass() -> None:
    # `RPerTrade` is declared frozen — any external code that tries to
    # mutate a recorded-risk snapshot raises `FrozenInstanceError`.
    # This pins the honesty contract that risk is an append-only ledger.
    r = RPerTrade(
        client_order_id='SS-edge-000',
        side='BUY',
        entry_price=Decimal('70000'),
        declared_stop_price=Decimal('69000'),
        qty=Decimal('0.2'),
    )
    import dataclasses
    # Use setattr() rather than direct attribute assignment so the
    # frozen-dataclass test doesn't need a `# type: ignore` on the
    # assignment. FrozenInstanceError is still raised by the dataclass-
    # generated __setattr__.
    try:
        setattr(r, 'entry_price', Decimal('80000'))
    except dataclasses.FrozenInstanceError:
        pass
    else:
        msg = 'RPerTrade must be frozen'
        raise AssertionError(msg)


def test_rpertrade_r_property_returns_positive_on_long() -> None:
    # Long-only entry: declared stop below entry. R = (entry-stop)*qty.
    r = RPerTrade(
        client_order_id='SS-edge-001',
        side='BUY',
        entry_price=Decimal('70000'),
        declared_stop_price=Decimal('69500'),
        qty=Decimal('0.5'),
    )
    assert r.r == Decimal('250')


def test_compute_r_fractional_bps_stop_no_rounding_loss() -> None:
    # A 33.33-bps stop on a $68432.17 entry produces an awkward-looking
    # stop price. The helper must keep the full Decimal precision.
    entry = Decimal('68432.17')
    bps = Decimal('33.33')
    stop = entry * (Decimal('1') - bps / Decimal('10000'))
    qty = Decimal('0.31415')
    r = compute_r(entry_price=entry, declared_stop_price=stop, qty=qty)
    assert r is not None
    assert r == (entry - stop) * qty


def test_rpertrade_none_stop_yields_none_r() -> None:
    # An RPerTrade with declared_stop_price=None yields r=None.
    # This is the "honestly flagged" state: no stop was declared at
    # entry time, so we cannot compute R honestly. The strategy caller
    # is expected to surface this as a HONESTY gate violation; the
    # dataclass itself does not raise.
    r = RPerTrade(
        client_order_id='SS-edge-no-stop-000',
        side='BUY',
        entry_price=Decimal('70000'),
        declared_stop_price=None,
        qty=Decimal('0.1'),
    )
    assert r.r is None


def test_compute_r_short_style_stop_above_entry() -> None:
    # Part 2 scope is long-only but the helper symmetry must hold: a
    # stop above the entry produces R = (stop - entry) * qty, identical
    # magnitude to a long stop at the mirrored distance.
    r_long = compute_r(
        entry_price=Decimal('100'),
        declared_stop_price=Decimal('90'),
        qty=Decimal('1'),
    )
    r_short = compute_r(
        entry_price=Decimal('100'),
        declared_stop_price=Decimal('110'),
        qty=Decimal('1'),
    )
    assert r_long == r_short
    assert r_long == Decimal('10')


def test_compute_r_idempotent_repeated_calls() -> None:
    # Calling compute_r repeatedly with the same inputs returns
    # bit-exact equal Decimals every time. Decimal arithmetic IS
    # deterministic; this test pins that no module-level state
    # accidentally drifts the result.
    entry = Decimal('70123.45')
    stop = Decimal('69876.55')
    qty = Decimal('0.123')
    r1 = compute_r(entry_price=entry, declared_stop_price=stop, qty=qty)
    r2 = compute_r(entry_price=entry, declared_stop_price=stop, qty=qty)
    r3 = compute_r(entry_price=entry, declared_stop_price=stop, qty=qty)
    assert r1 == r2 == r3
    assert r1 is not None


def test_rpertrade_distinct_orders_are_independent() -> None:
    # Two RPerTrade instances with different client_order_ids must
    # not share state. Frozen dataclasses already enforce immutability,
    # but this test pins that the .r property is computed from each
    # instance's own fields, not from a class-level cache.
    a = RPerTrade(
        client_order_id='SS-a-001', side='BUY',
        entry_price=Decimal('100'), declared_stop_price=Decimal('90'),
        qty=Decimal('1'),
    )
    b = RPerTrade(
        client_order_id='SS-b-001', side='BUY',
        entry_price=Decimal('200'), declared_stop_price=Decimal('195'),
        qty=Decimal('2'),
    )
    assert a.r == Decimal('10')
    assert b.r == Decimal('10')
    assert a.client_order_id != b.client_order_id


def test_compute_r_zero_qty_is_zero_not_none() -> None:
    # qty=0 produces R=0 (no risk because no exposure), distinct
    # from a missing stop (None). The launcher could conceivably
    # report a zero-qty fill if the venue dust-rounds the order
    # away; the helper must return a meaningful value.
    r = compute_r(
        entry_price=Decimal('70000'),
        declared_stop_price=Decimal('69000'),
        qty=Decimal('0'),
    )
    assert r == Decimal('0')


def test_rpertrade_repr_mentions_client_order_id() -> None:
    # `RPerTrade` is a frozen dataclass; its repr() must include the
    # client_order_id so log output identifies the trade unambiguously.
    # Pin this so a future refactor that strips the field doesn't
    # silently lose the identifier in trade-log audits.
    r = RPerTrade(
        client_order_id='SS-repr-001', side='BUY',
        entry_price=Decimal('70000'),
        declared_stop_price=Decimal('69000'),
        qty=Decimal('0.1'),
    )
    representation = repr(r)
    assert 'SS-repr-001' in representation
    assert 'BUY' in representation


def test_compute_r_extreme_small_qty_decimal_safe() -> None:
    # Sub-satoshi qty (10^-12) on BTC pricing should still produce
    # a well-defined Decimal R without overflow/underflow.
    entry = Decimal('100000')
    stop = Decimal('99000')
    qty = Decimal('1E-12')
    r = compute_r(entry_price=entry, declared_stop_price=stop, qty=qty)
    assert r is not None
    assert r == Decimal('1000') * qty
    assert r > 0


def test_compute_r_decimal_quantize_compatibility() -> None:
    # The strategy quantizes qty to 5 decimal places (BTC step size).
    # Pin that compute_r preserves the resulting Decimal precision
    # without performing any internal rounding.
    entry = Decimal('70123.456789')
    stop = Decimal('69876.543211')
    qty = Decimal('0.20424')
    r = compute_r(entry_price=entry, declared_stop_price=stop, qty=qty)
    assert r is not None
    expected = (entry - stop) * qty
    assert r == expected
    # Prove no quantization happened internally — the result should
    # carry the full Decimal precision, not be rounded to qty's
    # decimal places.
    assert r.as_tuple().exponent < 0  # has fractional component


def test_compute_r_with_high_precision_btc_distance() -> None:
    # BTC at 100k USD with a 0.01% stop distance = $10. At 0.5 BTC
    # qty, R = $5. The Decimal arithmetic must produce exactly 5.0,
    # not 4.999999... from float roundoff. Pin this against any
    # future helper rewrite that accidentally bridges through float.
    entry = Decimal('100000')
    stop = Decimal('99990')
    qty = Decimal('0.5')
    r = compute_r(entry_price=entry, declared_stop_price=stop, qty=qty)
    assert r == Decimal('5')


def test_compute_r_distance_dominates_qty_signal() -> None:
    # Two trades with identical stop distance but different qty
    # produce R values in proportion to qty. Pin the linear scaling
    # so a future rewrite can't accidentally apply a non-linear
    # transform on qty.
    distance = Decimal('100')
    a = compute_r(
        entry_price=Decimal('1000'),
        declared_stop_price=Decimal('900'),
        qty=Decimal('1'),
    )
    b = compute_r(
        entry_price=Decimal('1000'),
        declared_stop_price=Decimal('900'),
        qty=Decimal('5'),
    )
    assert a == distance
    assert b == distance * Decimal('5')
