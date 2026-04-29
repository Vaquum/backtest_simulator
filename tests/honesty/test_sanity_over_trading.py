"""Sanity baseline #3: over-trading is unprofitable by construction — fees, not drift.

Pins slice #17 Task 4 MVC. An OverTradingStrategy that flip-flops on
every signal — buy / sell / buy / sell ... — must over a zero-mean-drift
tape with realistic fees produce both:

    mean R/trade < 0
    profit factor (sum_winners / sum_losers) < 1

If the test passes with PF >= 1 or mean R >= 0, the simulator is
silently more generous than live: either the fee model is undercharging
or walk_trades is giving free price improvement on every fill. Either
way an honesty violation that matters before the strategy can be paper
traded.

Tape design — paired zero-gross construction:
  * 10 round-trip pairs over 20 signals.
  * Pair k=0,2,4,...: BUY at +1% / SELL at -1% (gross loss before fees).
  * Pair k=1,3,5,...: BUY at -1% / SELL at +1% (gross win before fees).
  * Across N pairs the gross sums cancel — *fees alone* push mean R
    negative and PF below 1. With zero fees mean R sits near +0.04
    (the small +/- 1% denominator asymmetry between high-side and
    low-side pairs makes the win-pair's R slightly larger in magnitude
    than the loss-pair's R), and PF lands at 1.0 — both invariants
    fail, catching the regression. The third invariant pins the fee
    rate itself — round-trip fees must land in [19.5, 20.5] bps of one
    leg's notional for the 10 bps taker schedule. A half-rate
    undercharge (5 bps taker -> 10 bps round-trip) would still pass
    mean R<0 / PF<1 but fails this band.
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
from nexus.strategy.action import Action
from nexus.strategy.context import StrategyContext
from nexus.strategy.params import StrategyParams
from nexus.strategy.signal import Signal

from backtest_simulator.honesty import build_validation_pipeline
from backtest_simulator.launcher.action_submitter import (
    SubmitterBindings,
    build_action_submitter,
)
from backtest_simulator.strategies import OverTradingStrategy
from backtest_simulator.venue.fees import FeeSchedule
from backtest_simulator.venue.fills import walk_trades
from backtest_simulator.venue.filters import BinanceSpotFilters
from backtest_simulator.venue.types import FillModelConfig, PendingOrder

_BTCUSDT = 'BTCUSDT'
_BASE_PRICE = Decimal('70000.00')
_DELTA = Decimal('700')  # +/- 1% swings around base


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
    """Return the order side name from a Praxis TradeCommand-shaped object."""
    side = getattr(cmd, 'side', None)
    if side is None:
        msg = f'TradeCommand has no `side` attribute: {cmd!r}'
        raise AssertionError(msg)
    return getattr(side, 'name', str(side))


def _extract_qty(cmd: object) -> Decimal:
    """Return the order quantity as Decimal from a TradeCommand-shaped object."""
    for attr in ('quantity', 'qty', 'size'):
        value = getattr(cmd, attr, None)
        if value is not None:
            return Decimal(str(value))
    msg = f'TradeCommand has no qty/quantity/size attribute: {cmd!r}'
    raise AssertionError(msg)


def _dense_burst(at: datetime, price: Decimal, n: int = 30, step_secs: int = 1) -> list[dict[str, object]]:
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


def _paired_zero_gross_tape(n_signals: int) -> tuple[pl.DataFrame, list[datetime]]:
    """Build a tape whose BUY/SELL pair-grosses sum to zero across N pairs.

    For pair k (signals 2k=BUY, 2k+1=SELL):
      * k even: BUY at base+delta, SELL at base-delta (gross loss).
      * k odd:  BUY at base-delta, SELL at base+delta (gross win).
    Mean gross PnL across pairs == 0 by construction; only fees + tick
    rounding remain to pull mean R below zero and PF below one.
    """
    assert n_signals % 2 == 0, 'pair construction requires even signal count'
    base_ts = datetime(2026, 4, 20, 0, 0, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    signal_times: list[datetime] = []
    for i in range(n_signals):
        pair_idx = i // 2
        leg = i % 2  # 0=BUY, 1=SELL
        direction = Decimal('1') if pair_idx % 2 == 0 else Decimal('-1')
        leg_sign = Decimal('1') if leg == 0 else Decimal('-1')
        price = _BASE_PRICE + direction * leg_sign * _DELTA
        ts = base_ts + timedelta(minutes=i * 5)
        rows.extend(_dense_burst(ts, price, n=30, step_secs=1))
        signal_times.append(ts)
    return (
        pl.DataFrame(rows).with_columns(pl.col('time').dt.replace_time_zone('UTC')),
        signal_times,
    )


def _walk_one(
    side: OrderSide,
    qty: Decimal,
    submit_time: datetime,
    trades: pl.DataFrame,
) -> tuple[Decimal, Decimal, Decimal]:
    """Walk one MARKET order; return (filled_qty, avg_fill_price, total_fee).

    No `stop_price` is attached to the PendingOrder — for MARKET orders
    `walk_trades` interprets a non-None stop as a mid-walk stop-out
    (refusing to fill if the tape has already breached it), which would
    block the over-trader's BUY legs that the strategy *declares* with
    a fixed-estimated-price stop above the actual fill on low-side bars.
    R below is computed from the strategy's declared stop_bps directly.
    """
    filters = BinanceSpotFilters.binance_spot(_BTCUSDT)
    fees = FeeSchedule()
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
    total_fee = Decimal('0')
    for fill in fills:
        total_qty += fill.fill_qty
        notional = fill.fill_qty * fill.fill_price
        total_notional += notional
        total_fee += fees.fee(_BTCUSDT, notional, is_maker=fill.is_maker)
    return total_qty, total_notional / total_qty, total_fee


def _declared_stop_bps(action: Action) -> Decimal:
    """Read the BUY action's declared stop_bps and validate the stop_price math."""
    raw_bps = action.execution_params.get('stop_bps')
    raw_px = action.execution_params.get('stop_price')
    assert raw_bps is not None and raw_px is not None, (
        f'BUY action missing stop_bps/stop_price; '
        f'execution_params={action.execution_params}'
    )
    bps = Decimal(str(raw_bps))
    declared_px = Decimal(str(raw_px))
    # The strategy must emit a stop_price consistent with stop_bps off the
    # configured estimated_price. A drift here (e.g. wrong bps multiplier,
    # wrong sign for a long, % vs bps confusion) would silently change R
    # denominators across the run.
    expected_px = _BASE_PRICE * (Decimal('1') - bps / Decimal('10000'))
    assert abs(declared_px - expected_px) <= Decimal('0.01'), (
        f'declared stop_price={declared_px} inconsistent with stop_bps={bps} '
        f'off estimated_price={_BASE_PRICE}; expected~{expected_px}'
    )
    return bps


def test_sanity_over_trading() -> None:
    """OverTradingStrategy: mean R<0 and PF<1 on a zero-gross tape; fees alone."""
    n_signals = 20  # 10 BUY + 10 SELL = 10 round-trip pairs
    trades, signal_times = _paired_zero_gross_tape(n_signals)
    initial_pool = Decimal('100000')
    pipeline, controller, capital_state = build_validation_pipeline(
        nexus_config=NexusInstanceConfig(account_id='bts-test', venue='binance_spot_simulated'),
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
            capital_controller=controller,
        strategy_budget=initial_pool,
    )
    submit = build_action_submitter(bindings)

    strategy = OverTradingStrategy(
        'sanity-over-trading',
        capital=initial_pool,
        kelly_pct=Decimal('1'),
        estimated_price=_BASE_PRICE,
        stop_bps=Decimal('50'),
    )
    params = StrategyParams(raw={})
    context = _empty_context()

    pairs: list[tuple[Decimal, Decimal, Decimal, Decimal, Decimal]] = []
    open_buy: tuple[Decimal, Decimal, Decimal, Decimal] | None = None
    sides_emitted: list[str] = []
    qtys_emitted: list[Decimal] = []
    for i, ts in enumerate(signal_times):
        actions = strategy.on_signal(
            Signal(
                predictor_fn_id='sanity-over-trading',
                timestamp=ts,
                values={'_preds': i % 2, '_probs': 0.5},
            ),
            params,
            context,
        )
        assert len(actions) == 1, (
            f'OverTradingStrategy must emit exactly 1 action per signal; '
            f'got {len(actions)} on signal #{i}'
        )
        action = actions[0]
        sides_emitted.append(action.direction.name)
        qtys_emitted.append(action.size)
        # Validate / extract the strategy's declared stop_bps at BUY time.
        # SELL is a position close, no stop attached. R is computed off
        # the declared bps later, not off a test-local hardcoded value.
        declared_bps = (
            _declared_stop_bps(action) if action.direction == OrderSide.BUY else None
        )
        submit(actions, 'sanity-over-trading')
        qty_filled, fill_px, fee = _walk_one(
            action.direction, action.size, ts, trades,
        )
        if action.direction == OrderSide.BUY:
            assert declared_bps is not None
            open_buy = (qty_filled, fill_px, fee, declared_bps)
        else:
            assert open_buy is not None, (
                f'SELL at signal #{i} without an open BUY — strategy state desync.'
            )
            buy_qty, buy_px, buy_fee, buy_bps = open_buy
            sell_qty, sell_px, sell_fee = qty_filled, fill_px, fee
            assert buy_qty == sell_qty, (
                f'pair qty mismatch buy={buy_qty} sell={sell_qty}'
            )
            # R denominator from the *declared* bps the strategy emitted,
            # applied to the actual entry fill — not a test-local hardcoded
            # 50. Catches a strategy regression where the action carries
            # the wrong stop_bps (e.g. unit confusion: 0.005 vs 50).
            gross_pnl = (sell_px - buy_px) * sell_qty
            total_fee = buy_fee + sell_fee
            net_pnl = gross_pnl - total_fee
            risk_per_unit = buy_px * buy_bps / Decimal('10000')
            risk = risk_per_unit * sell_qty
            r_mult = net_pnl / risk
            pairs.append((net_pnl, gross_pnl, total_fee, r_mult, buy_px * sell_qty))
            open_buy = None

    # Strategy alternates BUY / SELL deterministically.
    assert sides_emitted == ['BUY', 'SELL'] * (n_signals // 2), (
        f'OverTradingStrategy did not flip-flop perfectly; sides={sides_emitted}'
    )
    assert len(pairs) == n_signals // 2, (
        f'expected {n_signals // 2} round-trip pairs, got {len(pairs)}'
    )

    # ---- Production submitter actually saw every flip with the right side AND qty.
    # If `build_action_submitter` dropped, mangled, or duplicated a leg, this
    # catches it before any PnL math runs.
    assert len(outbound.commands) == n_signals, (
        f'submitter forwarded {len(outbound.commands)} != {n_signals} commands; '
        f'a flip leg was dropped or duplicated.'
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

    # ---- Sanity-baseline #3 invariants. The tape's gross PnL across pairs sums
    # to zero by construction (5 +1400*qty pairs cancel 5 -1400*qty pairs), so
    # any non-zero PF and below-zero mean R must come from fees + tick rounding.
    # With zero fees mean R ~= +0.040004 (the small high-side / low-side R-
    # denominator asymmetry pushes it slightly positive) and PF == 1.0 — both
    # invariants fail, which is the regression detector. Fee-undercharging or
    # free-price-improvement in walk_trades flip mean R back above zero and
    # PF back to >= 1.
    r_mults = [r for (_, _, _, r, _) in pairs]
    mean_r = sum(r_mults, Decimal('0')) / len(r_mults)
    winners = sum(
        (net for (net, _, _, _, _) in pairs if net > 0), Decimal('0'),
    )
    losers = sum(
        (-net for (net, _, _, _, _) in pairs if net < 0), Decimal('0'),
    )
    assert losers > 0, f'no losing pairs — fees did not bite. pairs={pairs}'
    pf = winners / losers

    assert mean_r < Decimal('0'), (
        f'mean R/trade={mean_r}; over-trading must be R-negative on a '
        f'zero-gross tape because fees alone drag every pair. Pairs: {pairs}'
    )
    assert pf < Decimal('1'), (
        f'profit factor={pf}; over-trading must have PF<1 on a zero-gross tape '
        f'because fees on round trips strictly exceed any positive gross. '
        f'winners_sum={winners} losers_sum={losers} pairs={pairs}'
    )

    # ---- Fee model is actually charging at ~10 bps taker (round-trip ~20 bps).
    # Without this assertion, the test would still pass if fees were charged
    # at half-rate (e.g. 5 bps): mean R<0 and PF<1 hold for any fee > 0. That
    # would silently hide a fee-undercharging regression. Pin the rate band.
    total_fees = sum((fee for (_, _, fee, _, _) in pairs), Decimal('0'))
    total_one_leg_notional = sum(
        (notional for (_, _, _, _, notional) in pairs), Decimal('0'),
    )
    fees_round_trip_bps = (
        total_fees / total_one_leg_notional * Decimal('10000')
    )
    assert Decimal('19.5') <= fees_round_trip_bps <= Decimal('20.5'), (
        f'round-trip fee={fees_round_trip_bps} bps of one-leg notional; '
        f'expected band [19.5, 20.5] bps for the 10 bps taker schedule. '
        f'A fee model regression is hiding behind the mean-R<0/PF<1 invariants.'
    )
