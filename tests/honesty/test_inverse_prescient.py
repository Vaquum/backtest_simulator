"""Honesty gate: sign-flipped peeking strategy must lose catastrophically.

Pins slice #17 Task 7 MVC and SPEC §9.3 — "fill model is causal."

`InversePrescientStrategy` receives a perfect next-bar direction label
via `signal.values['_future_pred']` and DELIBERATELY trades the
opposite direction. On a tape whose true direction the label captures
exactly, the strategy's policy is the worst possible: BUY at every
peak that is about to fall, SELL before every rise.

The expected outcome — and the test's pass condition — is catastrophic
loss. If `walk_trades` somehow lets this strategy profit (or even break
even), the fill model is non-causal: a fill is being awarded at a
price that respects the strategy's *intent* rather than the order's
submit time. That is the regression Task 7 pins SPEC §9.3's "fill
model is causal" guarantee against.

Tape design — strict alternating high/low:
  * 20 bars at minute boundaries, prices alternating 75_000 / 65_000.
  * Bar i price: HIGH (75_000) if i % 2 == 0 else LOW (65_000).
  * `_future_pred` at bar i = 1 if bar[i+1] > bar[i] else 0
    (the perfect causal label of the *next-bar* direction).
  * Inverse strategy reads the label, acts on `1 - label`:
      label = 0 (next bar low):  inverse = 1 → BUY  (entry at HIGH)
      label = 1 (next bar high): inverse = 0 → SELL (exit at LOW)
    Every BUY lands at 75_000, every matching SELL lands at 65_000.
    Per-pair gross PnL = (65_000 - 75_000) * qty = -10_000 * qty.

Test asserts:
  1. Each strategy-emitted side reaches the production submitter
     (`build_action_submitter` round-trips with side+qty intact).
  2. Per-pair gross PnL is strictly negative — every closed pair
     loses on a tape where the inverse policy is consistently wrong.
  3. Total return as a fraction of initial capital is below
     `_CATASTROPHIC_THRESHOLD` (well below zero). A simulator with
     a non-causal fill model would shift this above the threshold;
     pinning a strict number catches the regression.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import cast

import polars as pl
from nexus.core.domain.enums import OrderSide
from nexus.core.domain.operational_mode import OperationalMode
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
from backtest_simulator.strategies import InversePrescientStrategy
from backtest_simulator.venue.fills import walk_trades
from backtest_simulator.venue.filters import BinanceSpotFilters
from backtest_simulator.venue.types import FillModelConfig, PendingOrder

_BTCUSDT = 'BTCUSDT'
_HIGH_PRICE = Decimal('75000.00')
_LOW_PRICE = Decimal('65000.00')
_INITIAL_CAPITAL = Decimal('100000')
_N_SIGNALS = 20  # 10 round-trip pairs
# A causal fill model on this tape produces ~ -10_000 USD per pair *
# 0.01428 BTC ≈ -142.86 USD per pair, ~10 pairs = -1428.6 USD on
# 100_000 USD capital ≈ -1.43 % gross return. The threshold is set
# stricter than -1 % so a non-causal regression that gives the
# inverse strategy an honest fill (= zero) shows up loudly.
_CATASTROPHIC_THRESHOLD = Decimal('-0.01')


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
        capital_available=_INITIAL_CAPITAL,
        operational_mode=OperationalMode.ACTIVE,
    )


def _extract_side(cmd: object) -> str:
    side = getattr(cmd, 'side', None)
    if side is None:
        msg = f'TradeCommand has no `side` attribute: {cmd!r}'
        raise AssertionError(msg)
    return getattr(side, 'name', str(side))


def _extract_qty(cmd: object) -> Decimal:
    """Return the order quantity from a Praxis TradeCommand-shaped object."""
    for attr in ('quantity', 'qty', 'size'):
        value = getattr(cmd, attr, None)
        if value is not None:
            return Decimal(str(value))
    msg = f'TradeCommand has no qty/quantity/size attribute: {cmd!r}'
    raise AssertionError(msg)


def _dense_burst(
    at: datetime, price: Decimal, n: int = 30, step_secs: int = 1,
) -> list[dict[str, object]]:
    """Synthetic tick burst at a fixed price; provides liquidity for one walk."""
    return [
        {
            'time': at + timedelta(seconds=i * step_secs),
            'price': float(price),
            'qty': 1.0,
            'trade_id': hash((at, price, i)) & 0x7FFFFFFF,
        }
        for i in range(n)
    ]


def _alternating_tape() -> tuple[pl.DataFrame, list[datetime], list[Decimal]]:
    """Tape with strict alternation HIGH/LOW/HIGH/LOW... per bar.

    Returns (trades_df, signal_timestamps, per_signal_prices). Per-bar
    price equals `_HIGH_PRICE` if bar index is even else `_LOW_PRICE`,
    so the next-bar direction at bar i is unambiguous: down if i is
    even, up if i is odd.
    """
    base_ts = datetime(2026, 4, 20, 0, 0, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    signal_times: list[datetime] = []
    prices: list[Decimal] = []
    for i in range(_N_SIGNALS):
        price = _HIGH_PRICE if i % 2 == 0 else _LOW_PRICE
        ts = base_ts + timedelta(minutes=i * 5)
        rows.extend(_dense_burst(ts, price, n=30, step_secs=1))
        signal_times.append(ts)
        prices.append(price)
    return (
        pl.DataFrame(rows).with_columns(pl.col('time').dt.replace_time_zone('UTC')),
        signal_times,
        prices,
    )


def _walk_one(
    side: OrderSide, qty: Decimal, submit_time: datetime,
    trades: pl.DataFrame,
) -> tuple[Decimal, Decimal]:
    """Walk one MARKET order; return (filled_qty, avg_fill_price). No fees."""
    filters = BinanceSpotFilters.binance_spot(_BTCUSDT)
    order = PendingOrder(
        order_id=f'order-{side.name}-{submit_time.isoformat()}',
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
        f'no fills for {side.name} qty={qty} at {submit_time}; tape too thin.'
    )
    total_qty = Decimal('0')
    total_notional = Decimal('0')
    for fill in fills:
        total_qty += fill.fill_qty
        total_notional += fill.fill_qty * fill.fill_price
    return total_qty, total_notional / total_qty


def _next_bar_label(prices: list[Decimal], i: int) -> int:
    """Perfect causal label: 1 if bar[i+1] > bar[i] else 0. Last bar = 0."""
    if i + 1 >= len(prices):
        return 0  # no next bar → "down" by convention; the strategy's
                  # last action with this label is a SELL (close), which
                  # is what we want at the horizon.
    return 1 if prices[i + 1] > prices[i] else 0


def test_inverse_prescient_loses_catastrophically() -> None:
    """Sign-flipped prescient on alternating tape: per-pair PnL < 0, total <-1%."""
    trades, signal_times, prices = _alternating_tape()
    pipeline, _controller, capital_state = build_validation_pipeline(
        capital_pool=_INITIAL_CAPITAL,
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
        strategy_budget=_INITIAL_CAPITAL,
    )
    submit = build_action_submitter(bindings)

    strategy = InversePrescientStrategy(
        'inverse-prescient-fixture',
        capital=_INITIAL_CAPITAL,
        kelly_pct=Decimal('1'),
        estimated_price=_HIGH_PRICE,  # so quantize lands at 0.00133 BTC
        stop_bps=Decimal('50'),
    )
    params = StrategyParams(raw={})
    context = _empty_context()

    pairs: list[tuple[Decimal, Decimal, Decimal, Decimal]] = []
    open_buy: tuple[Decimal, Decimal] | None = None
    sides_emitted: list[str] = []
    qtys_emitted: list[Decimal] = []
    for i, ts in enumerate(signal_times):
        actions = strategy.on_signal(
            Signal(
                predictor_fn_id='inverse-prescient',
                timestamp=ts,
                values={'_future_pred': _next_bar_label(prices, i)},
            ),
            params,
            context,
        )
        if not actions:
            continue
        assert len(actions) == 1, (
            f'InversePrescientStrategy emitted {len(actions)} actions on '
            f'signal #{i}; the test harness only handles single-action '
            f'signals.'
        )
        action = actions[0]
        sides_emitted.append(action.direction.name)
        qtys_emitted.append(action.size)
        submit(actions, 'inverse-prescient')
        qty_filled, fill_px = _walk_one(
            action.direction, action.size, ts, trades,
        )
        if action.direction == OrderSide.BUY:
            open_buy = (qty_filled, fill_px)
        else:
            assert open_buy is not None, (
                f'SELL at signal #{i} without an open BUY — strategy state desync.'
            )
            buy_qty, buy_px = open_buy
            sell_qty, sell_px = qty_filled, fill_px
            assert buy_qty == sell_qty, (
                f'pair qty mismatch buy={buy_qty} sell={sell_qty}'
            )
            gross_pnl = (sell_px - buy_px) * sell_qty
            pairs.append((gross_pnl, buy_px, sell_px, sell_qty))
            open_buy = None

    # ---- Horizon close. The last-bar's "next bar" doesn't exist, so
    #      `_next_bar_label` returns 0 (down-by-convention), which the
    #      inverse strategy maps to inverse=1 → BUY/HOLD: a long would
    #      otherwise stay open at the horizon and its loss would not
    #      be counted. Force-walk a SELL through the last bar's burst
    #      so the open pair closes against the actual final tape price.
    if open_buy is not None:
        buy_qty, buy_px = open_buy
        sell_qty, sell_px = _walk_one(
            OrderSide.SELL, buy_qty, signal_times[-1], trades,
        )
        assert sell_qty == buy_qty, (
            f'horizon liquidation qty mismatch buy={buy_qty} sell={sell_qty}'
        )
        gross_pnl = (sell_px - buy_px) * sell_qty
        pairs.append((gross_pnl, buy_px, sell_px, sell_qty))
        open_buy = None

    assert open_buy is None, (
        'open BUY at horizon was not closed; horizon liquidation hook '
        'failed to lift the residual long.'
    )

    # ---- Production submitter received every emitted action with the
    #      same side AND qty: rules out a regression where the submitter
    #      drops, reorders, OR resizes the inverse strategy's flips.
    #      Without the qty assertion a submitter that zeroes a command
    #      would slip past — PnL is computed from the strategy's
    #      `action.size` so a downstream qty mutation would be invisible
    #      to every other check.
    assert len(outbound.commands) == len(sides_emitted), (
        f'outbound forwarded {len(outbound.commands)} != emitted '
        f'{len(sides_emitted)} commands — submitter dropped or '
        f'duplicated an inverse-prescient flip.'
    )
    cmd_sides = [_extract_side(c) for c in outbound.commands]
    assert cmd_sides == sides_emitted, (
        f'outbound side sequence {cmd_sides} != emitted {sides_emitted}; '
        f'submitter is sending the wrong side or reordering legs.'
    )
    cmd_qtys = [_extract_qty(c) for c in outbound.commands]
    assert cmd_qtys == qtys_emitted, (
        f'outbound qty sequence {cmd_qtys} != emitted {qtys_emitted}; '
        f'submitter is mutating order size between strategy and Praxis.'
    )

    # ---- Exact pair count. With 20 alternating-tape signals the
    #      inverse strategy emits a BUY at every even bar (label=0
    #      → inverse=1) and a SELL at every odd bar (label=1 →
    #      inverse=0), closing 9 pairs in the loop. The horizon hook
    #      above closes the 10th pair (BUY at i=18, SELL force-walked
    #      at i=19). A degenerate strategy that never trades would
    #      silently pass the per-pair loss assertion below.
    assert len(pairs) == 10, (
        f'InversePrescient closed {len(pairs)} pairs on a 20-bar '
        f'alternating tape — expected exactly 10. Sides emitted: '
        f'{sides_emitted}; qtys: {qtys_emitted}'
    )

    # ---- Exact emitted-side pattern. With the (label=0,1,0,1,...,1,0)
    #      sequence the inverse strategy must emit exactly the BUY/SELL
    #      flip-flop. Catches a regression where the strategy emits a
    #      different action shape that happens to sum to ten pairs.
    assert sides_emitted == ['BUY', 'SELL'] * 9 + ['BUY'], (
        f'sides_emitted {sides_emitted} != expected `[BUY,SELL]*9 + [BUY]`; '
        f'the strategy is not the canonical inverse-prescient.'
    )

    # ---- Per-pair gross PnL strictly negative: every BUY entered at
    #      75_000 (HIGH) and every matching SELL exited at 65_000 (LOW)
    #      because the inverse strategy is wrong on every signal. A
    #      causal fill model produces -10_000 * qty per pair, no
    #      exceptions. A non-causal fill model would let some pairs
    #      come out flat or positive.
    for net, buy_px, sell_px, qty in pairs:
        assert net < Decimal('0'), (
            f'inverse-prescient pair has non-negative gross PnL — fill '
            f'model is non-causal. buy={buy_px} sell={sell_px} qty={qty} '
            f'gross={net}'
        )

    # ---- Total gross return is catastrophic: the inverse strategy on
    #      an alternating tape must hemorrhage ~ -1.4 % of capital with
    #      a causal fill model. A non-causal regression that gives the
    #      strategy "fair" fills (zero return) would lift this above
    #      the threshold and trip the assertion.
    total_pnl = sum((net for (net, _, _, _) in pairs), Decimal('0'))
    total_return = total_pnl / _INITIAL_CAPITAL
    assert total_return < _CATASTROPHIC_THRESHOLD, (
        f'inverse-prescient total return {total_return} is above the '
        f'catastrophic-loss threshold {_CATASTROPHIC_THRESHOLD}. The '
        f'fill model is non-causal: a sign-flipped peeking strategy '
        f'should never break even. pairs={pairs}'
    )
