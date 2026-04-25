"""Pin BinanceSpotFilters.validate against tick_size + step_size enforcement.

The previous implementation claimed in its docstring to enforce these
filters but only checked min/max qty and min_notional — orders with
bad qty/price increments passed validation and got silently rounded
elsewhere. Real Binance Spot rejects them. This test pins the new
behaviour: bad increments are rejected with a clear reason, MARKET
orders (`price=None`) skip the price-side checks but still enforce the
qty-side step_size, and well-formed orders pass.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from backtest_simulator.venue.filters import BinanceSpotFilters


@pytest.fixture
def f() -> BinanceSpotFilters:
    return BinanceSpotFilters.binance_spot('BTCUSDT')


def test_well_formed_order_passes(f: BinanceSpotFilters) -> None:
    # qty=0.001 is a multiple of step_size 0.00001; price=70000 is a
    # multiple of tick_size 0.01; notional 70 >= min_notional 10.
    assert f.validate(Decimal('0.001'), Decimal('70000')) is None


def test_qty_below_min_rejects(f: BinanceSpotFilters) -> None:
    reason = f.validate(Decimal('0.000001'), Decimal('70000'))
    assert reason is not None
    assert 'LOT_SIZE' in reason
    assert 'min_qty' in reason


def test_qty_not_step_multiple_rejects(f: BinanceSpotFilters) -> None:
    # 0.000123 is not a multiple of step_size 0.00001 — 6 decimals,
    # last digit (3) makes it land between 0.00012 and 0.00013.
    reason = f.validate(Decimal('0.000123'), Decimal('70000'))
    assert reason is not None
    assert 'LOT_SIZE' in reason
    assert 'step_size' in reason


def test_price_not_tick_multiple_rejects(f: BinanceSpotFilters) -> None:
    # 70000.005 is not a multiple of tick_size 0.01 — 3 decimals, last
    # digit forces it off the tick boundary.
    reason = f.validate(Decimal('0.001'), Decimal('70000.005'))
    assert reason is not None
    assert 'PRICE_FILTER' in reason
    assert 'tick_size' in reason


def test_market_order_no_price_skips_tick_check(f: BinanceSpotFilters) -> None:
    # MARKET orders pass price=None — tick_size check is irrelevant
    # (no declared limit). Step_size still enforces on the qty side.
    assert f.validate(Decimal('0.001'), price=None) is None


def test_market_order_with_bad_step_still_rejects(f: BinanceSpotFilters) -> None:
    # Even without a price reference, qty must be a step_size multiple.
    # Pre-fix this branch only checked min/max, missing the step rule.
    reason = f.validate(Decimal('0.000123'), price=None)
    assert reason is not None
    assert 'step_size' in reason


def test_min_notional_rejects(f: BinanceSpotFilters) -> None:
    # 0.0001 * 50 = 0.005 < min_notional 10
    reason = f.validate(Decimal('0.0001'), Decimal('50'))
    assert reason is not None
    assert 'MIN_NOTIONAL' in reason


def test_market_with_subtick_stop_does_not_reject_via_filter() -> None:
    """MARKET orders carry `stop_price` as a risk anchor, not a venue limit.

    Pre-fix `reject_reason` substituted `stop_price` for `price` when no
    limit price was given and pushed it through `validate`'s tick_size
    check — so a kelly-derived stop like 69649.9876 (sub-tick) would
    reject the order. This pins the new behaviour: `reject_reason`
    keeps the tick check off the stop reference but still enforces the
    min-notional sanity check via `stop_price` when no limit price is
    provided.
    """
    from types import SimpleNamespace

    from backtest_simulator.venue._adapter_internals import reject_reason

    f = BinanceSpotFilters.binance_spot('BTCUSDT')
    # Sub-tick stop_price: 69649.9876 is not a multiple of tick_size 0.01.
    # qty = 0.001 (step-aligned). Notional vs stop = ~69.65 >> min 10.
    # `order_type='MARKET'` is what flags this as a risk-anchor stop.
    order = SimpleNamespace(
        qty=Decimal('0.001'),
        stop_price=Decimal('69649.9876'),
        order_type='MARKET',
    )
    assert reject_reason(order, f, price=None) is None


def test_market_stop_still_min_notional_checked() -> None:
    """The stop reference still drives the min-notional sanity check."""
    from types import SimpleNamespace

    from backtest_simulator.venue._adapter_internals import reject_reason

    f = BinanceSpotFilters.binance_spot('BTCUSDT')
    # qty * stop = 0.0001 * 50 = 0.005, well under min_notional 10.
    order = SimpleNamespace(
        qty=Decimal('0.0001'),
        stop_price=Decimal('50'),
        order_type='MARKET',
    )
    reason = reject_reason(order, f, price=None)
    assert reason is not None
    assert 'MIN_NOTIONAL' in reason


def test_real_stop_order_validates_subtick_stop_price() -> None:
    """STOP_LOSS / STOP_LOSS_LIMIT / TAKE_PROFIT must enforce tick on stop_price.

    For these venue-quoted stop orders, `stop_price` IS the trigger the
    venue tracks and Binance's PRICE_FILTER applies. A sub-tick
    stop_price like 69649.9876 must be rejected here too — otherwise
    backtest accepts what live rejects (paper/live divergence).
    """
    from types import SimpleNamespace

    from backtest_simulator.venue._adapter_internals import reject_reason

    f = BinanceSpotFilters.binance_spot('BTCUSDT')
    for venue_type in ('STOP_LOSS', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT'):
        order = SimpleNamespace(
            qty=Decimal('0.001'),
            stop_price=Decimal('69649.9876'),  # sub-tick
            order_type=venue_type,
        )
        reason = reject_reason(order, f, price=None)
        assert reason is not None, (
            f'{venue_type} with sub-tick stop_price must reject; got pass'
        )
        assert 'tick_size' in reason, (
            f'{venue_type} sub-tick must reject via PRICE_FILTER; got {reason!r}'
        )


def test_stop_limit_with_valid_price_and_subtick_stop_rejects() -> None:
    """STOP_LOSS_LIMIT carries BOTH a limit price AND a stop trigger.

    Pre-fix the stop_price tick check only ran when `price=None`, so a
    STOP_LOSS_LIMIT with a tick-aligned limit price and a sub-tick
    stop_price slipped through — the limit-side validate(qty, price)
    passed, then the stop-side branch was guarded by `price is None`
    and was skipped. Live Binance rejects on the sub-tick stopPrice
    regardless of the limit price; backtest must too.
    """
    from types import SimpleNamespace

    from backtest_simulator.venue._adapter_internals import reject_reason

    f = BinanceSpotFilters.binance_spot('BTCUSDT')
    order = SimpleNamespace(
        qty=Decimal('0.001'),
        stop_price=Decimal('69649.9876'),  # sub-tick — invalid trigger
        order_type='STOP_LOSS_LIMIT',
    )
    # Limit price is tick-aligned and notional-clean.
    reason = reject_reason(order, f, price=Decimal('70000.00'))
    assert reason is not None, (
        'STOP_LOSS_LIMIT with tick-clean price but sub-tick stop_price '
        'must reject; the stop-side tick check was guarded by price-is-None.'
    )
    assert 'tick_size' in reason


def test_stop_limit_with_both_clean_passes() -> None:
    """Sanity check: STOP_LOSS_LIMIT with both prices tick-clean must pass."""
    from types import SimpleNamespace

    from backtest_simulator.venue._adapter_internals import reject_reason

    f = BinanceSpotFilters.binance_spot('BTCUSDT')
    order = SimpleNamespace(
        qty=Decimal('0.001'),
        stop_price=Decimal('69500.00'),
        order_type='STOP_LOSS_LIMIT',
    )
    assert reject_reason(order, f, price=Decimal('69400.00')) is None
