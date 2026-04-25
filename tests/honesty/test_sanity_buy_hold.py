"""Sanity baseline #2: buy-and-hold returns (close-open)/open - fees, +/- 5 bps.

Pins slice #17 Task 3 MVC. The test:

  1. Constructs a synthetic tape of trades whose `open` price (first
     trade) and `close` price (last trade) are known precisely.
  2. Drives BuyAndHoldStrategy through the production action_submitter
     callback (same one BacktestLauncher installs) so an ENTER BUY
     emits at the open and an ENTER SELL emits at the close.
  3. Walks the trades through the venue's fill engine for each emitted
     action, computes realised return and total fees, and asserts:

         |realised_return - ((close - open)/open - fees_bps/10000)| <= 5 bps

The 5 bps tolerance accounts for step_size rounding (qty quantised to
0.00001 BTC), tick_size rounding on the stop, and fee Decimal precision.
Anything wider indicates a fill / accounting drift in the simulator
chain — slippage that the backtest is masking, fees the venue forgot
to book, or bps math that disagrees with the reference computation.
The test is the closest-possible-market-simulation pin on the
"buy-and-hold reproduces (close-open)/open" identity.

Tests run unit-level (no ClickHouse) via a hand-built `pl.DataFrame`
of trades; the integration-level "buy-and-hold against real BTCUSDT
ClickHouse trades" assertion is owned by Task 24's integration tests.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import cast

import polars as pl
from nexus.core.domain.enums import OrderSide
from nexus.core.domain.operational_mode import OperationalMode
from nexus.core.domain.order_types import OrderType
from nexus.core.validator.pipeline_models import InstanceState
from nexus.infrastructure.praxis_connector.praxis_outbound import PraxisOutbound
from nexus.instance_config import InstanceConfig as NexusInstanceConfig
from nexus.strategy.context import StrategyContext
from nexus.strategy.params import StrategyParams
from nexus.strategy.signal import Signal

from backtest_simulator.honesty import build_validation_pipeline
from backtest_simulator.launcher.action_submitter import (
    SubmitterBindings,
    build_action_submitter,
)
from backtest_simulator.strategies import BuyAndHoldStrategy
from backtest_simulator.venue.fees import FeeSchedule
from backtest_simulator.venue.fills import walk_trades
from backtest_simulator.venue.filters import BinanceSpotFilters
from backtest_simulator.venue.types import FillModelConfig, PendingOrder

_OPEN_PRICE = Decimal('70000.00')
_CLOSE_PRICE = Decimal('71400.00')  # +2% over the window
_BTCUSDT = 'BTCUSDT'


def _trades_dataframe() -> pl.DataFrame:
    """24h tape: open at 70000.00, close at 71400.00, dense ticks at both ends."""
    base = datetime(2026, 4, 20, 0, 0, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    # Open burst — many ticks at exactly _OPEN_PRICE so the BUY walk
    # consumes its qty entirely at that price (no slippage in the test).
    for i in range(20):
        rows.append({
            'time': base + timedelta(seconds=i),
            'price': float(_OPEN_PRICE),
            'qty': 0.5,
            'trade_id': i,
        })
    # Close burst — same shape at _CLOSE_PRICE near end of window.
    end = base + timedelta(hours=23, minutes=59)
    for i in range(20):
        rows.append({
            'time': end + timedelta(seconds=i),
            'price': float(_CLOSE_PRICE),
            'qty': 0.5,
            'trade_id': 1000 + i,
        })
    return pl.DataFrame(rows).with_columns(
        pl.col('time').dt.replace_time_zone('UTC'),
    )


class _OutboundCapture:
    def __init__(self) -> None:
        self.commands: list[object] = []

    def send_command(self, cmd: object) -> str:
        self.commands.append(cmd)
        return f'cmd-{len(self.commands)}'

    def send_abort(self, **kwargs: object) -> None:
        del kwargs


def _empty_context() -> StrategyContext:
    return StrategyContext(
        positions=(),
        capital_available=Decimal('100000'),
        operational_mode=OperationalMode.ACTIVE,
    )


def _extract_side(cmd: object) -> str:
    """Return the order side from a Praxis TradeCommand-shaped object."""
    side = getattr(cmd, 'side', None)
    if side is None:
        msg = f'TradeCommand has no `side` attribute: {cmd!r}'
        raise AssertionError(msg)
    return getattr(side, 'name', str(side))


def _walk_one(
    side: OrderSide, qty: Decimal, submit_time: datetime,
    trades: pl.DataFrame,
) -> tuple[Decimal, Decimal, Decimal]:
    """Walk one ENTER through walk_trades; return (filled_qty, fill_price, fee)."""
    filters = BinanceSpotFilters.binance_spot(_BTCUSDT)
    fees = FeeSchedule()
    order = PendingOrder(
        order_id=f'order-{side.name}',
        side=side.name,
        order_type='MARKET',
        qty=qty,
        limit_price=None,
        stop_price=None,
        time_in_force='IOC',
        submit_time=submit_time,
        symbol=_BTCUSDT,
    )
    fills = walk_trades(order, trades, FillModelConfig(), filters)
    assert fills, (
        f'walk_trades produced no fills for {side.name} qty={qty} at '
        f'{submit_time}; the tape fixture is too thin for the order size.'
    )
    total_qty = Decimal('0')
    total_notional = Decimal('0')
    total_fee = Decimal('0')
    for fill in fills:
        total_qty += fill.fill_qty
        notional = fill.fill_qty * fill.fill_price
        total_notional += notional
        total_fee += fees.fee(_BTCUSDT, notional, is_maker=fill.is_maker)
    avg_price = total_notional / total_qty
    return total_qty, avg_price, total_fee


def test_sanity_buy_hold() -> None:
    """Buy at first signal, hold, sell at last; return within ±5 bps of reference."""
    trades = _trades_dataframe()
    initial_pool = Decimal('100000')
    pipeline, _controller, capital_state = build_validation_pipeline(
        capital_pool=initial_pool,
    )
    outbound = _OutboundCapture()
    bindings = SubmitterBindings(
        nexus_config=NexusInstanceConfig(
            account_id='bts-acct',
            venue='binance_spot_simulated',
        ),
        state=InstanceState(capital=capital_state),
        praxis_outbound=cast(PraxisOutbound, outbound),
        validation_pipeline=pipeline,
        strategy_budget=Decimal('100000'),
    )
    submit = build_action_submitter(bindings)

    strategy = BuyAndHoldStrategy(
        'sanity-buy-hold',
        symbol=_BTCUSDT,
        capital=initial_pool,
        kelly_pct=Decimal('1'),
        estimated_price=_OPEN_PRICE,
        stop_bps=Decimal('500'),
    )
    params = StrategyParams(raw={})
    context = _empty_context()

    open_ts = datetime(2026, 4, 20, 0, 0, tzinfo=UTC)
    close_ts = datetime(2026, 4, 20, 23, 59, tzinfo=UTC)

    # 1. Open — strategy emits ENTER BUY.
    open_actions = strategy.on_signal(
        Signal(
            predictor_fn_id='sanity-buy-hold',
            timestamp=open_ts,
            values={},
        ),
        params,
        context,
    )
    assert len(open_actions) == 1
    assert open_actions[0].direction == OrderSide.BUY
    assert open_actions[0].order_type == OrderType.MARKET
    submit(open_actions, 'sanity-buy-hold')
    # The action submitter must have forwarded one TradeCommand to Praxis.
    assert len(outbound.commands) == 1, (
        f'BuyAndHoldStrategy ENTER BUY did not reach the submitter; '
        f'outbound.commands={outbound.commands}'
    )
    buy_action = open_actions[0]
    buy_qty_filled, buy_fill_px, buy_fee = _walk_one(
        OrderSide.BUY, buy_action.size, open_ts, trades,
    )

    # 2. Close signal — strategy emits ENTER SELL.
    close_actions = strategy.on_signal(
        Signal(
            predictor_fn_id='sanity-buy-hold',
            timestamp=close_ts,
            values={'_close_position': True},
        ),
        params,
        context,
    )
    assert len(close_actions) == 1
    assert close_actions[0].direction == OrderSide.SELL
    sell_action = close_actions[0]
    submit(close_actions, 'sanity-buy-hold')
    # Both legs must reach the production submitter, in the right
    # order, with the right sides. A regression in the SELL-close
    # path (which exists separately because SELL-as-EXIT bypasses
    # the CAPITAL stage in the launcher's action submitter wrapper)
    # or one that double-submits BUY would land here as the wrong
    # outbound shape.
    assert len(outbound.commands) == 2, (
        f'BuyAndHoldStrategy ENTER SELL did not reach the submitter; '
        f'outbound.commands={outbound.commands}'
    )
    sides = [_extract_side(cmd) for cmd in outbound.commands]
    assert sides == ['BUY', 'SELL'], (
        f'expected outbound side sequence [BUY, SELL], got {sides}; '
        f'submitter is sending the wrong side or duplicating one leg.'
    )
    sell_qty_filled, sell_fill_px, sell_fee = _walk_one(
        OrderSide.SELL, sell_action.size, close_ts, trades,
    )

    # Realised return on the full round-trip (gross of fees).
    assert buy_qty_filled == sell_qty_filled, (
        f'qty mismatch buy={buy_qty_filled} sell={sell_qty_filled}; '
        f'BuyAndHoldStrategy entry/exit qty must agree.'
    )
    qty = buy_qty_filled
    # The BUY filled exactly at _OPEN_PRICE and the SELL exactly at
    # _CLOSE_PRICE — the fixture's tick bursts are dense at both
    # boundaries, so walk_trades consumes the full qty at the boundary
    # tick price with no slippage. Pin those exact prices so a fill-
    # engine drift (e.g. a regression that consumes ticks past the
    # boundary and produces VWAP slippage) trips this test.
    assert buy_fill_px == _OPEN_PRICE, (
        f'BUY filled at {buy_fill_px}, expected {_OPEN_PRICE}; the fill '
        f'engine consumed beyond the open burst.'
    )
    assert sell_fill_px == _CLOSE_PRICE, (
        f'SELL filled at {sell_fill_px}, expected {_CLOSE_PRICE}.'
    )

    gross_pnl = (sell_fill_px - buy_fill_px) * qty
    total_fees = buy_fee + sell_fee
    net_pnl = gross_pnl - total_fees
    initial_notional = buy_fill_px * qty
    realised_return_bps = (net_pnl / initial_notional) * Decimal('10000')

    # Reference: (close-open)/open - round_trip_fees_bps. FeeSchedule's
    # default taker fee is 10 bps per side, so a round-trip is exactly
    # 20.2 bps (taker on both legs, against the BUY-side notional).
    expected_gross_bps = (
        (_CLOSE_PRICE - _OPEN_PRICE) / _OPEN_PRICE * Decimal('10000')
    )
    fees_bps = (total_fees / initial_notional) * Decimal('10000')

    # Explicit fee invariant — guards against a regression where the
    # realised / reference computation share a bug that cancels out
    # (both using the same wrong `total_fees`). Round-trip fees should
    # land in the [20.0, 20.4] bps band:
    #   - 10 bps taker on BUY notional (fee = qty * open * 0.001)
    #   - 10 bps taker on SELL notional (fee = qty * close * 0.001)
    #   - SELL notional > BUY notional because price moved up, so the
    #     fee-vs-BUY-notional ratio is slightly above 20 bps.
    assert Decimal('20.0') <= fees_bps <= Decimal('20.4'), (
        f'fees_bps={fees_bps:.4f} outside expected [20.0, 20.4] band for '
        f'round-trip taker fees; FeeSchedule may be charging the wrong '
        f'rate or against the wrong notional.'
    )

    expected_net_bps = expected_gross_bps - fees_bps
    delta_bps = abs(realised_return_bps - expected_net_bps)
    # 0.5 bps — fixture has no actual slippage / rounding noise: BUY
    # and SELL fill at exact open/close prices, qty is step-aligned.
    # Tolerance is just for Decimal precision in the bps math.
    assert delta_bps <= Decimal('0.5'), (
        f'buy-and-hold realised={realised_return_bps:.4f} bps vs '
        f'reference={expected_net_bps:.4f} bps; delta={delta_bps:.4f} bps '
        f'exceeds 0.5 bps tolerance. gross={expected_gross_bps:.4f} bps, '
        f'fees={fees_bps:.4f} bps, qty={qty}, '
        f'buy_px={buy_fill_px}, sell_px={sell_fill_px}.'
    )
