"""Stop enforcement: declared stop fires at declared price in the trade stream."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import polars as pl

from backtest_simulator.venue.fills import walk_trades
from backtest_simulator.venue.filters import BinanceSpotFilters
from backtest_simulator.venue.types import FillModelConfig, PendingOrder


def test_stop_triggers_at_declared_price() -> None:
    """The invariant: when the trade stream crosses declared stop, we fill AT that stop."""
    now = datetime(2020, 4, 1, tzinfo=UTC)
    filters = BinanceSpotFilters.binance_spot('BTCUSDT')
    config = FillModelConfig(submit_latency_ms=50)
    declared_stop = Decimal('6500.00')

    # A SELL stop at 6500 on a BUY-side position (we held BUY, exit via SELL stop).
    # Trade stream crosses down through 6500.
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
    # The fill price equals the declared stop rounded to tick_size (no slippage substitution).
    assert fills[0].fill_price == filters.round_price(declared_stop)
    assert fills[0].reason == 'stop_trigger'
    # Proving honesty of R = |entry - stop| * qty: fill didn't slide to 6499 or 6400.
    assert fills[0].fill_price == Decimal('6500.00')


def test_buy_stop_triggers_when_stream_crosses_upward() -> None:
    now = datetime(2020, 4, 1, tzinfo=UTC)
    filters = BinanceSpotFilters.binance_spot('BTCUSDT')
    config = FillModelConfig(submit_latency_ms=50)
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
    assert fills[0].fill_price == Decimal('6500.00')
    assert fills[0].reason == 'stop_trigger'


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
