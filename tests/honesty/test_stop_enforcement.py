"""Stop enforcement: stop fills land at the actual tape tick price (gap-slippage realistic).

Principle from PR #15 body: backtest ≡ paper ≡ live. The declared stop
is the R denominator (definitional), not a promise about fill price.
On a gapping tape the fill lands at the tape tick that breached stop
— which may be *worse* than the declared stop — matching live
execution. R numerator reflects real market move, so gapping stop-outs
can produce R < -1 (which is correct, not a bug).
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import polars as pl

from backtest_simulator.venue.fills import walk_trades
from backtest_simulator.venue.filters import BinanceSpotFilters
from backtest_simulator.venue.types import FillModelConfig, PendingOrder


def test_stop_triggers_at_actual_tape_tick_price_sell_side() -> None:
    """SELL stop fires at first tick <= stop_price, at the TICK'S price (not stop)."""
    now = datetime(2020, 4, 1, tzinfo=UTC)
    filters = BinanceSpotFilters.binance_spot('BTCUSDT')
    config = FillModelConfig(submit_latency_ms=50)
    declared_stop = Decimal('6500.00')

    # Tape crosses downward through 6500: the first breach tick is 6499.
    # Fill must land at 6499 (actual tape price, 1-tick gap slippage), NOT
    # at 6500. Live-reality: a SELL stop triggered by a breach tick fills
    # at that tick's price.
    trades = pl.DataFrame({
        'time': [now + timedelta(seconds=i) for i in (1, 2, 3, 4, 5)],
        'price': [Decimal('7000'), Decimal('6800'), Decimal('6600'),
                  Decimal('6499'), Decimal('6400')],
        'qty':   [Decimal('1'), Decimal('1'), Decimal('1'), Decimal('1'), Decimal('1')],
    })
    order = PendingOrder(
        order_id='O1', side='SELL', order_type='STOP_LOSS',
        qty=Decimal('0.5'), limit_price=None, stop_price=declared_stop,
        time_in_force='GTC', submit_time=now, symbol='BTCUSDT',
    )
    fills = walk_trades(order, trades, config, filters)
    assert len(fills) == 1
    # Fill at the breach tick (6499), NOT at declared stop (6500).
    assert fills[0].fill_price == Decimal('6499.00')
    assert fills[0].reason == 'stop_trigger'
    # The slippage is (declared_stop - fill) = 6500 - 6499 = 1, realised
    # worse than declared -- exactly what live execution would produce on
    # this tape.


def test_stop_fills_at_stop_price_when_tape_touches_exactly() -> None:
    """Edge: if the first breach tick is exactly at stop_price, fill = stop (no slippage)."""
    now = datetime(2020, 4, 1, tzinfo=UTC)
    filters = BinanceSpotFilters.binance_spot('BTCUSDT')
    config = FillModelConfig(submit_latency_ms=50)
    trades = pl.DataFrame({
        'time': [now + timedelta(seconds=i) for i in (1, 2, 3)],
        'price': [Decimal('6700'), Decimal('6600'), Decimal('6500')],
        'qty':   [Decimal('1'), Decimal('1'), Decimal('1')],
    })
    order = PendingOrder(
        order_id='OE', side='SELL', order_type='STOP_LOSS',
        qty=Decimal('0.3'), limit_price=None, stop_price=Decimal('6500'),
        time_in_force='GTC', submit_time=now, symbol='BTCUSDT',
    )
    fills = walk_trades(order, trades, config, filters)
    assert len(fills) == 1
    # Breach tick is exactly 6500; fill = 6500. Zero gap slippage here —
    # the generic "fill at breach tick" rule naturally reduces to stop
    # price when the tape touches stop exactly.
    assert fills[0].fill_price == Decimal('6500.00')
    assert fills[0].reason == 'stop_trigger'


def test_buy_stop_triggers_at_actual_tape_tick_price() -> None:
    """BUY stop fires at first tick >= stop_price; fill = tick price (can gap up worse)."""
    now = datetime(2020, 4, 1, tzinfo=UTC)
    filters = BinanceSpotFilters.binance_spot('BTCUSDT')
    config = FillModelConfig(submit_latency_ms=50)
    # Tape gaps upward: 6400 -> 6600 (skipping 6500). The BUY stop at
    # 6500 triggers on the 6600 tick at price 6600 — worse than declared.
    trades = pl.DataFrame({
        'time': [now + timedelta(seconds=i) for i in (1, 2, 3)],
        'price': [Decimal('6400'), Decimal('6600'), Decimal('6700')],
        'qty':   [Decimal('1'), Decimal('1'), Decimal('1')],
    })
    order = PendingOrder(
        order_id='O2', side='BUY', order_type='STOP_LOSS',
        qty=Decimal('0.1'), limit_price=None, stop_price=Decimal('6500'),
        time_in_force='GTC', submit_time=now, symbol='BTCUSDT',
    )
    fills = walk_trades(order, trades, config, filters)
    assert len(fills) == 1
    # Fill at 6600 (the breach tick), NOT at 6500 (declared stop).
    # 100 quote units of gap slippage realised — matches live execution
    # on a tape that gapped through the stop.
    assert fills[0].fill_price == Decimal('6600.00')
    assert fills[0].reason == 'stop_trigger'


def test_gapping_sell_stop_produces_r_worse_than_minus_one() -> None:
    """Mutation proof: on a gapping tape, the realised stop slippage is NOT zero.

    If the fill model reverted to "fill at stop_price on breach" (the
    previous optimistic behaviour), the exit price would equal the
    declared stop and the round-trip R would be exactly -1. The
    realistic model must show the exit landing WORSE than stop on a
    gapping tape, producing R < -1. This test is the mechanical proof
    that gap risk is not silently hidden.
    """
    now = datetime(2020, 4, 1, tzinfo=UTC)
    filters = BinanceSpotFilters.binance_spot('BTCUSDT')
    config = FillModelConfig(submit_latency_ms=50)
    declared_stop = Decimal('6500.00')
    # Hard gap: tape jumps from 6800 down to 6400 (never touches 6500+).
    trades = pl.DataFrame({
        'time': [now + timedelta(seconds=i) for i in (1, 2, 3)],
        'price': [Decimal('6800'), Decimal('6400'), Decimal('6300')],
        'qty':   [Decimal('1'), Decimal('1'), Decimal('1')],
    })
    order = PendingOrder(
        order_id='OG', side='SELL', order_type='STOP_LOSS',
        qty=Decimal('0.2'), limit_price=None, stop_price=declared_stop,
        time_in_force='GTC', submit_time=now, symbol='BTCUSDT',
    )
    fills = walk_trades(order, trades, config, filters)
    assert len(fills) == 1
    assert fills[0].fill_price == Decimal('6400.00')
    # Compare with the phantom-fill-at-stop model: that would give
    # fills[0].fill_price == 6500.00 and mask the 100-unit gap loss.
    # The realised gap is the difference — 100 quote units here.
    gap_slippage = declared_stop - fills[0].fill_price
    assert gap_slippage == Decimal('100.00'), (
        f'expected 100 units of gap slippage, got {gap_slippage}; '
        f'if this is 0 the stop filled at declared_stop — the old '
        f'optimistic model regressed.'
    )


def test_stop_does_not_fire_when_stream_never_crosses() -> None:
    now = datetime(2020, 4, 1, tzinfo=UTC)
    filters = BinanceSpotFilters.binance_spot('BTCUSDT')
    config = FillModelConfig(submit_latency_ms=50)
    trades = pl.DataFrame({
        'time': [now + timedelta(seconds=i) for i in (1, 2, 3)],
        'price': [Decimal('7000'), Decimal('7100'), Decimal('7200')],
        'qty':   [Decimal('1'), Decimal('1'), Decimal('1')],
    })
    order = PendingOrder(
        order_id='O3', side='SELL', order_type='STOP_LOSS',
        qty=Decimal('0.1'), limit_price=None, stop_price=Decimal('6500'),
        time_in_force='GTC', submit_time=now, symbol='BTCUSDT',
    )
    fills = walk_trades(order, trades, config, filters)
    assert fills == []


def test_market_entry_halts_walk_on_stop_breach_mid_fill() -> None:
    """MARKET BUY with attached stop: walk HALTS on breach, no residual at stop.

    Live reality: once the tape has already moved past the declared
    stop during the entry window, the strategy's R assumption is
    already violated. Booking the unfilled residual at the declared
    stop price would be a phantom fill at a price the tape did not
    offer — which live execution cannot reproduce. The walk halts;
    the partial fill is returned at pre-breach tape prices only.
    """
    now = datetime(2020, 4, 1, tzinfo=UTC)
    filters = BinanceSpotFilters.binance_spot('BTCUSDT')
    config = FillModelConfig(submit_latency_ms=50)
    # Order wants 1.0 BTC; tape provides 0.3 at 70000, 0.3 at 69800,
    # then breaches the declared stop at 69500. Under the old model
    # the residual (0.4) would have been filled at 69500, producing
    # full 1.0 BTC FILLED. Under strict-live-reality the walk halts
    # and only 0.6 BTC fills at the pre-breach tape VWAP.
    trades = pl.DataFrame({
        'time': [now + timedelta(seconds=i) for i in (1, 2, 3, 4)],
        'price': [Decimal('70000'), Decimal('69800'),
                  Decimal('69400'), Decimal('69300')],
        'qty':   [Decimal('0.3'), Decimal('0.3'),
                  Decimal('1.0'), Decimal('1.0')],
    })
    order = PendingOrder(
        order_id='OM', side='BUY', order_type='MARKET',
        qty=Decimal('1.0'), limit_price=None, stop_price=Decimal('69500'),
        time_in_force='GTC', submit_time=now, symbol='BTCUSDT',
    )
    fills = walk_trades(order, trades, config, filters)
    assert len(fills) == 1
    # Filled qty = 0.3 + 0.3 = 0.6 (partial; the remainder 0.4 released).
    assert fills[0].fill_qty == Decimal('0.6')
    # VWAP over pre-breach ticks = (70000*0.3 + 69800*0.3) / 0.6 = 69900.
    assert fills[0].fill_price == Decimal('69900.00')
    assert fills[0].reason == 'market_stop_halted'


def test_r_per_trade_requires_declared_stop() -> None:
    """The metrics layer raises on trades without a declared stop (SPEC §19 #1 option a)."""
    import pytest

    from backtest_simulator.exceptions import StopContractViolation
    from backtest_simulator.reporting.metrics import TradeRecord, r_per_trade

    trade = TradeRecord(
        trade_id='T1', side='BUY',
        entry_price=Decimal('7000'), exit_price=Decimal('7100'),
        declared_stop=Decimal('0'),  # invalid: entry_price == declared_stop -> risk == 0
        qty=Decimal('0.1'),
        entry_fees=Decimal('1'), exit_fees=Decimal('1'),
    )
    with pytest.raises(StopContractViolation):
        r_per_trade(trade)
