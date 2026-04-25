"""Honesty gate: bar-shuffled tape collapses any structural alpha to ~0.

Pins slice #17 Task 8 MVC and SPEC §9.3 — "profit on shuffled bars =
exploiting simulator artefacts, not market structure."

A buy-and-hold strategy on a directional tape captures a real
structural move (a +500/bar uptrend across 20 bars, ~+1.36 % gross
when 10 % of capital is deployed) and posts a positive return.
Permuting the same bars destroys the temporal structure: the BUY and
SELL prices become two exchangeable elements of the same shuffled
multiset (drawn without replacement, not strictly independent, but
their joint distribution is symmetric so `E[sell - buy] = 0`).
Across many seeds the cross-seed mean realised gross return
therefore collapses close to zero. If the shuffled mean stays
meaningfully positive, the simulator is generating phantom alpha
from order-of-operations artefacts — e.g., walk_trades binding fills
to bar indices instead of submit timestamps, or the fill engine
privileging early bars over later ones independent of price.

Returns are GROSS (no fees subtracted). `FeeSchedule.fee()` is
proportional to notional, so the per-seed fee amount varies with
the shuffled BUY/SELL prices; subtracting it would still leave the
structural-alpha question intact (fees are honest and don't depend
on bar *order*) but would muddy the assertion's geometry. Working
in gross removes the fee component as a concern entirely so the
gate measures only whether bar order leaks into PnL.

Test runs:
  1. **In-order baseline** — BuyAndHoldStrategy on the deterministic
     uptrend tape, driven through the production
     `build_action_submitter` chain. Returns >= 1 % gross. The
     baseline also asserts that both legs reach Praxis with the
     correct side and qty (catches a CAPITAL denial of the 10 % Kelly
     BUY, or a submitter that drops / mangles a command — without
     this the bulk loop's `walk_trades` would still produce a number
     and the structural assertion would silently miss it).
  2. **N-seed shuffle bulk** — `_N_SEEDS = 1000` seeds, each
     permutes the same 20 prices (timestamps unchanged) and re-runs
     the strategy. The bulk loop skips the submitter for performance;
     the venue fill engine (`walk_trades`) is the load-bearing piece
     and runs every seed.
  3. **Magnitude collapse** — `|cross-seed mean| <= 10 %` of the
     in-order return. Magnitude-based instead of CI-brackets-zero
     for a different reason than power: at N=1000 the 95 % CI
     half-width (~0.037 %) is well below the ratio threshold (~0.136 %),
     so a CI-zero gate has the *power* to fire on the same regressions.
     The honesty argument is calibration: the ratio threshold is tied
     to the *real* in-order alpha magnitude (the thing the operator
     cares about preserving structure for); CI-on-zero is a null-
     hypothesis test that produces a false positive at the chosen
     5 % rate even when the simulator is honest, which is exactly the
     kind of flakiness this gate must not have. The ratio gate fires
     iff the shuffled alpha is meaningfully comparable to the real
     alpha.
  4. **Sign symmetry** — `min(positive_count, negative_count) >=
     30 %` of seeds. Catches a one-sided drag/lift bias the magnitude
     check might miss: a symmetric shuffle should produce roughly
     balanced signs, so a one-sided distribution indicates a
     directional simulator artefact.
"""
from __future__ import annotations

import math
import random
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, Decimal
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
from backtest_simulator.strategies import BuyAndHoldStrategy
from backtest_simulator.venue.fees import FeeSchedule
from backtest_simulator.venue.fills import walk_trades
from backtest_simulator.venue.filters import BinanceSpotFilters
from backtest_simulator.venue.types import FillModelConfig, PendingOrder

_BTCUSDT = 'BTCUSDT'
_BASE_PRICE = Decimal('70000.00')
# +500 USD per bar -> +9_500 USD across 19 hops. With `_KELLY_PCT = 10`
# (10 % per-trade allocation under the CAPITAL stage's 15 % cap) the
# in-order *gross* return on this tape lands ~+1.36 %, comfortably
# above the 1 % structural-alpha floor. Smaller steps leave the
# baseline indistinguishable from shuffle-dispersion noise.
_PER_BAR_STEP = Decimal('500.00')
_N_BARS = 20
_KELLY_PCT = Decimal('10')  # 10 % of capital -> 10 % per-trade allocation
_INITIAL_CAPITAL = Decimal('100000')
# Per-seed gross-return std lands around 0.6 % of capital (exact
# first/last permutation math on the 20-element price set). At 1000
# seeds the SE on the cross-seed mean is ~0.019 % of capital — about
# 7x below `_COLLAPSE_RATIO * in_order_return` (~0.136 %), and the
# 95 % CI half-width (~0.037 %) is about 3.7x below the same
# threshold. The ratio gate fires before SE noise can dominate.
_N_SEEDS = 1000
# Shuffled mean's magnitude must be < 10 % of the in-order return.
# A simulator that preserves bar-order alpha would lift this above
# the threshold; a finite-sample noise pull-down would have to be
# huge to trip it.
_COLLAPSE_RATIO = Decimal('0.10')


def _build_tape_from_prices(
    prices: list[Decimal],
) -> tuple[pl.DataFrame, list[datetime]]:
    """Convert per-bar prices into a polars trade-tape + signal timestamps.

    Each bar gets a 30-tick burst at the same price so walk_trades has
    enough liquidity for the buy-and-hold qty. Timestamps are fixed
    every 5 minutes from the same base — only the prices vary across
    in-order vs shuffled runs.
    """
    base_ts = datetime(2026, 4, 20, 0, 0, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    signal_times: list[datetime] = []
    for i, price in enumerate(prices):
        ts = base_ts + timedelta(minutes=i * 5)
        for j in range(30):
            rows.append({
                'time': ts + timedelta(seconds=j),
                'price': float(price),
                'qty': 1.0,
                'trade_id': hash((ts, price, j)) & 0x7FFFFFFF,
            })
        signal_times.append(ts)
    return (
        pl.DataFrame(rows).with_columns(pl.col('time').dt.replace_time_zone('UTC')),
        signal_times,
    )


def _ordered_uptrend_prices() -> list[Decimal]:
    """Per-bar prices: linear uptrend `_BASE_PRICE + i * _PER_BAR_STEP`."""
    return [_BASE_PRICE + Decimal(i) * _PER_BAR_STEP for i in range(_N_BARS)]


def _shuffled_prices(seed: int) -> list[Decimal]:
    """Per-bar prices: deterministic permutation of `_ordered_uptrend_prices`."""
    rng = random.Random(seed * 99991 + 7)
    prices = _ordered_uptrend_prices()
    rng.shuffle(prices)
    return prices


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
    """Return the order quantity from a TradeCommand-shaped object."""
    for attr in ('quantity', 'qty', 'size'):
        value = getattr(cmd, attr, None)
        if value is not None:
            return Decimal(str(value))
    msg = f'TradeCommand has no qty/quantity/size attribute: {cmd!r}'
    raise AssertionError(msg)


def _walk_one(
    side: OrderSide, qty: Decimal, submit_time: datetime, trades: pl.DataFrame,
) -> tuple[Decimal, Decimal, Decimal]:
    """Walk one MARKET order; return (filled_qty, fill_price, total_fee)."""
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


def _run_buy_hold_on_tape(
    prices: list[Decimal], *, outbound: _OutboundCapture | None,
) -> Decimal:
    """Run BuyAndHoldStrategy across `prices`; return *gross* return (no fees).

    Returns the gross PnL fraction of `_INITIAL_CAPITAL`.
    `FeeSchedule.fee()` is proportional to per-leg notional, so the
    per-seed fee total varies with the shuffled BUY/SELL prices.
    Working in gross PnL removes that variation as a confound — the
    structural-alpha question is purely whether bar order leaks into
    realised PnL, and gross is the cleanest way to ask it.

    If `outbound` is provided, the production `build_action_submitter`
    chain runs (validation + capital pipeline + Praxis forward); the
    caller asserts on `outbound.commands` after the run to verify the
    plumbing is load-bearing. The bulk shuffle loop passes `None` to
    skip the submitter for performance — `walk_trades` is still
    invoked for every fill so the venue-side fill-engine path is
    unchanged.

    `estimated_price=_BASE_PRICE` (not `prices[0]`) is intentional:
    the strategy's qty must be independent of the per-seed first-bar
    price, otherwise the qty would vary across seeds and confound the
    cross-seed mean (a Jensen's-inequality bias on `E[sell/buy]`).
    Holding qty constant across all runs makes
    `E[(sell - buy) * qty] = qty * E[sell - buy] = 0` exact under the
    null.
    """
    trades, signal_times = _build_tape_from_prices(prices)
    strategy = BuyAndHoldStrategy(
        'shuffle-bars',
        symbol=_BTCUSDT,
        capital=_INITIAL_CAPITAL,
        kelly_pct=_KELLY_PCT,
        estimated_price=_BASE_PRICE,
        stop_bps=Decimal('500'),
    )
    submit = None
    if outbound is not None:
        pipeline, _controller, capital_state = build_validation_pipeline(
            capital_pool=_INITIAL_CAPITAL,
        )
        bindings = SubmitterBindings(
            nexus_config=NexusInstanceConfig(
                account_id='bts-acct', venue='binance_spot_simulated',
            ),
            state=InstanceState(capital=capital_state),
            praxis_outbound=cast(PraxisOutbound, outbound),
            validation_pipeline=pipeline,
            strategy_budget=_INITIAL_CAPITAL,
        )
        submit = build_action_submitter(bindings)

    params = StrategyParams(raw={})
    context = _empty_context()
    open_ts = signal_times[0]
    close_ts = signal_times[-1]

    open_actions = strategy.on_signal(
        Signal(
            predictor_fn_id='shuffle-bars', timestamp=open_ts, values={},
        ),
        params, context,
    )
    assert len(open_actions) == 1
    if submit is not None:
        submit(open_actions, 'shuffle-bars')
    buy_qty, buy_px, buy_fee = _walk_one(
        OrderSide.BUY, open_actions[0].size, open_ts, trades,
    )

    close_actions = strategy.on_signal(
        Signal(
            predictor_fn_id='shuffle-bars', timestamp=close_ts,
            values={'_close_position': True},
        ),
        params, context,
    )
    assert len(close_actions) == 1
    if submit is not None:
        submit(close_actions, 'shuffle-bars')
    sell_qty, sell_px, sell_fee = _walk_one(
        OrderSide.SELL, close_actions[0].size, close_ts, trades,
    )
    del buy_fee, sell_fee  # gross-only return; fees vary with shuffled
                            # prices and are dropped to keep the gate
                            # measuring bar-order leak, not fee noise.
    assert buy_qty == sell_qty
    gross_pnl = (sell_px - buy_px) * sell_qty
    return gross_pnl / _INITIAL_CAPITAL


def test_shuffle_bars_collapses_structural_alpha() -> None:
    """Permuted bars: cross-seed gross mean magnitude << in-order; signs balance."""
    # ---- 1. In-order baseline: real structural alpha. Drive through
    #         the production submitter once so this branch covers the
    #         full chain — and assert outbound carries both legs with
    #         the right side and qty so a CAPITAL denial of the 10 %
    #         Kelly BUY (or a submitter that drops/mangles a command)
    #         is caught loudly. Without these assertions the bulk
    #         loop's `walk_trades` would still produce a number even
    #         if the submitter side is broken, and the structural
    #         test would silently degrade.
    in_order_prices = _ordered_uptrend_prices()
    in_order_outbound = _OutboundCapture()
    in_order_return = _run_buy_hold_on_tape(
        in_order_prices, outbound=in_order_outbound,
    )
    assert in_order_return >= Decimal('0.01'), (
        f'in-order buy-and-hold return = {in_order_return}; the +500/bar '
        f'uptrend with `_KELLY_PCT=10` should yield ~+1.36 % gross. The '
        f'test cannot pin "structural alpha collapses" if the baseline '
        f'does not show structural alpha.'
    )
    # Both legs must reach Praxis. The submitter dispatch is the
    # production callback BacktestLauncher installs; if CAPITAL
    # denies the BUY here, the gate is misconfigured and the rest
    # of the test cannot mean what it claims.
    assert len(in_order_outbound.commands) == 2, (
        f'in-order baseline only forwarded {len(in_order_outbound.commands)} '
        f'TradeCommand(s) to Praxis; expected exactly 2 (BUY open, SELL '
        f'close). CAPITAL may be denying the 10 % Kelly BUY (cap 15 %), '
        f'or the submitter is dropping a leg. commands={in_order_outbound.commands}'
    )
    cmd_sides = [_extract_side(c) for c in in_order_outbound.commands]
    assert cmd_sides == ['BUY', 'SELL'], (
        f'in-order baseline outbound side sequence = {cmd_sides}; '
        f'expected [BUY, SELL]. The submitter is reordering or '
        f'mangling legs — the structural test below cannot rely on '
        f'this baseline.'
    )
    cmd_qtys = [_extract_qty(c) for c in in_order_outbound.commands]
    # `BuyAndHoldStrategy` quantizes qty with `ROUND_DOWN` to match
    # the venue step_size; mirror that here so the assertion compares
    # apples to apples.
    expected_qty = (
        _INITIAL_CAPITAL * _KELLY_PCT / Decimal('100') / _BASE_PRICE
    ).quantize(Decimal('0.00001'), rounding=ROUND_DOWN)
    assert all(q == expected_qty for q in cmd_qtys), (
        f'in-order baseline outbound qtys = {cmd_qtys}; expected each '
        f'== {expected_qty} (10 % Kelly off `_BASE_PRICE`). Submitter '
        f'is mutating order size between strategy and Praxis.'
    )

    # ---- 2. Bulk shuffle loop: per-seed permutation of the same prices.
    seed_returns: list[float] = []
    for seed in range(_N_SEEDS):
        prices = _shuffled_prices(seed)
        r = _run_buy_hold_on_tape(prices, outbound=None)
        seed_returns.append(float(r))

    n = float(_N_SEEDS)
    mean = sum(seed_returns) / n
    var = sum((r - mean) ** 2 for r in seed_returns) / (n - 1)
    se = math.sqrt(var / n)

    # ---- 3. Magnitude collapse. The shuffled mean must shrink to <
    #         `_COLLAPSE_RATIO` of the in-order return: structural
    #         alpha cannot survive bar permutation if the simulator
    #         is honest about bar order. Magnitude-based instead of
    #         CI-brackets-zero because shuffle-mean SE at finite N is
    #         dominated by per-seed gross dispersion (~0.6 % of
    #         capital, SE at N=1000 ~0.019 %); a CI-on-zero assertion
    #         would still occasionally miss zero by random chance
    #         and become brittle to seed selection. Ratio-on-in-order
    #         is what actually matters — preserve only insignificant
    #         noise.
    in_order_f = float(in_order_return)
    collapse_threshold = float(_COLLAPSE_RATIO) * abs(in_order_f)
    assert abs(mean) <= collapse_threshold, (
        f'shuffle-bars cross-seed mean return |{mean:.6f}| exceeds '
        f'{float(_COLLAPSE_RATIO):.0%} of in-order return '
        f'|{in_order_f:.6f}| (threshold {collapse_threshold:.6f}). '
        f'Permuting the bars should collapse the structural alpha; '
        f'the simulator is preserving alpha that depends on bar order '
        f'— a non-causal artefact, not market structure. '
        f'n_seeds={int(n)} se={se:.6e}.'
    )

    # ---- 4. Sign-symmetry sanity. Across 1000 seeds the shuffled
    #         returns should be roughly half positive and half
    #         negative — a simulator that consistently produces
    #         negative-only or positive-only shuffled returns is
    #         imposing a non-shuffle-related drag/lift that the
    #         magnitude check might miss if the bias is small but
    #         systematic. Pin at >=30 % of seeds in the minority sign.
    pos = sum(1 for r in seed_returns if r > 0)
    neg = sum(1 for r in seed_returns if r < 0)
    minority = min(pos, neg)
    assert minority >= int(0.3 * _N_SEEDS), (
        f'shuffle-bars sign distribution is skewed: {pos} positive, '
        f'{neg} negative across {_N_SEEDS} seeds (minority={minority}, '
        f'threshold={int(0.3 * _N_SEEDS)}). A symmetric shuffle should '
        f'produce roughly balanced signs; one-sided returns indicate '
        f'a directional simulator artefact.'
    )
