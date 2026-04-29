"""Sanity baseline #4: random-timing strategy has zero gross alpha across seeds.

Pins slice #17 Task 5 MVC. A strategy that flips long/flat with prob
0.5 on each signal — independently of `_preds`, the tape, or any
future bar — must, over many seeds on a per-seed random-shuffled
tape, produce a mean gross return whose 95% confidence interval
brackets zero.

Why this baseline matters: random-timing strategies are the canonical
alpha-leakage probe — for the *venue + strategy* part of the chain.
Any positive systematic mean across thousands of seeds means
`walk_trades` is filling at a future-tick price (venue-side lookahead)
or the strategy itself is observing future tape bars. A negative
systematic mean of magnitude > expected fee drag means realised
slippage is being silently injected on every fill. Either way an
honesty violation that matters before the strategy can be paper
traded.

The 1000-seed bulk loop *deliberately* bypasses `build_action_submitter`
and the capital validation pipeline for performance — so this baseline
does NOT cover capital-state lookahead or signal/timestamp misalignment
inside the submitter. Those paths are pinned by the production-path
seed below (plumbing-only) and by Tasks 2/3/4 (statistical-edge cases
through the full chain).

Tape design — per-seed random shuffle: each seed independently samples
which of the 20 bars are at +1 % vs -1 % from `_BASE_PRICE` (10 +1
and 10 -1, balanced and shuffled). A *shared* fixed tape would couple
the random-timing strategy's first-passage distribution (BUYs cluster
early, SELLs late) to the tape's bar-index price pattern and bias the
per-seed mean even when no alpha leakage exists. Per-seed shuffling
breaks that coupling: across seeds the BUY/SELL bar choice and the
H/L bar location are independent, so E[buy_price] = E[sell_price] =
`_BASE_PRICE` and the gross-PnL null is honest.

Each bar's "dense burst" interleaves three distinct prices —
`center - tick_jitter`, `center`, `center + tick_jitter` — across
30 ticks in a fixed period-3 pattern (low, mid, high, low, mid,
high, ...). A *symmetric* one-tick peek would shift BUY and SELL
fills by the same delta and cancel in gross PnL — so this test
deliberately is not the place to catch symmetric peek (Task 3
pins exact-fill-price for that). What the jittered burst *does*
catch is *asymmetric* lookahead — e.g., a "best-of-K" scan that
gives BUYs the minimum and SELLs the maximum of the next K ticks.
That bias is one-sided; with same-price bursts there is no signal,
so any test that wants to catch it needs a tape where consecutive
ticks differ.

Test runs:
  * 1 production-path seed: drives one seed through the real
    `build_action_submitter` callback and asserts every emitted action
    reached Praxis with side+qty matched. Plumbing proof for the path
    the bulk loop skips.
  * 1000 statistical seeds: per seed, walks every emitted action
    through `walk_trades` (no fees — gross test). Open positions at
    the horizon are mark-to-market closed via a final walked SELL
    on the last burst, so seeds that end long contribute their
    unrealised PnL — leakage on entries that never closed inside
    the window can't hide. Across seeds, asserts the 95% CI of the
    mean brackets zero (i.e. zero is within
    `mean +/- 1.96 * sample_std / sqrt(n_seeds)`).

Performance: the bulk loop skips the action_submitter so 1000 seeds
land in single-digit seconds. Production-path coverage is in the
out-of-loop seed.
"""
from __future__ import annotations

import math
import random
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
from backtest_simulator.strategies import RandomTimingStrategy
from backtest_simulator.venue.fills import walk_trades
from backtest_simulator.venue.filters import BinanceSpotFilters
from backtest_simulator.venue.types import FillModelConfig, PendingOrder

_BTCUSDT = 'BTCUSDT'
_BASE_PRICE = Decimal('70000.00')
_DELTA = Decimal('700')  # +/- 1% swings around base
_TICK_JITTER = Decimal('0.10')  # within-burst tick variation; one BTCUSDT tick.
_INITIAL_CAPITAL = Decimal('100000')
_N_SIGNALS_PER_SEED = 20
_N_SEEDS_BULK = 1000
_PROD_PATH_SEED = 42


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
    for attr in ('quantity', 'qty', 'size'):
        value = getattr(cmd, attr, None)
        if value is not None:
            return Decimal(str(value))
    msg = f'TradeCommand has no qty/quantity/size attribute: {cmd!r}'
    raise AssertionError(msg)


def _dense_burst(at: datetime, price: Decimal, n: int = 30, step_secs: int = 1) -> list[dict[str, object]]:
    """Per-tick price interleaved center-minus / center / center-plus.

    The ticks have THREE distinct prices in a fixed period-3 pattern.
    A *symmetric* one-tick peek (consume tick i+1 rather than tick i)
    cancels in BUY/SELL gross PnL by construction so this test does not
    catch it — Task 3 pins exact-fill-price for that path. The jittered
    burst makes the test sensitive to *asymmetric* lookahead — e.g. a
    "best-of-K" scan that gives BUYs the minimum and SELLs the maximum
    of the next K ticks. With same-price bursts there is no signal and
    asymmetric peek is invisible too.
    """
    pattern = (Decimal('-1'), Decimal('0'), Decimal('1'))
    return [
        {
            'time': at + timedelta(seconds=i * step_secs),
            'price': float(price + pattern[i % 3] * _TICK_JITTER),
            'qty': 1.0,
            'trade_id': hash((at, price, i)) & 0x7FFFFFFF,
        }
        for i in range(n)
    ]


def _per_seed_shuffled_tape(seed: int, n_signals: int) -> tuple[pl.DataFrame, list[datetime]]:
    """Tape with per-seed random shuffle of 10 H + 10 L bars; net drift == 0.

    Each seed shuffles which bar indices are at +1 % vs -1 % from
    `_BASE_PRICE`. Half are +1 %, half are -1 %, so net drift across
    the window is exactly zero. Across seeds, the H/L position
    distribution is uniform, so the random-timing strategy's BUY/SELL
    first-passage distribution (BUYs early, SELLs late) decouples from
    the price pattern.

    The tape seed is offset from the strategy seed by a large prime so
    the two random streams are independent; otherwise a strategy seed
    that happens to flip on the same bars where the tape is high
    would correlate by accident.
    """
    assert n_signals == 20, 'shuffle expects 20 bars (10 H + 10 L)'
    rng = random.Random(seed * 99991 + 7)
    signs: list[int] = [1] * 10 + [-1] * 10
    rng.shuffle(signs)
    base_ts = datetime(2026, 4, 20, 0, 0, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    signal_times: list[datetime] = []
    for i in range(n_signals):
        price = _BASE_PRICE + Decimal(signs[i]) * _DELTA
        ts = base_ts + timedelta(minutes=i * 5)
        rows.extend(_dense_burst(ts, price, n=30, step_secs=1))
        signal_times.append(ts)
    return (
        pl.DataFrame(rows).with_columns(pl.col('time').dt.replace_time_zone('UTC')),
        signal_times,
    )


def _walk_one(
    side: OrderSide, qty: Decimal, submit_time: datetime, trades: pl.DataFrame,
) -> tuple[Decimal, Decimal]:
    """Walk one MARKET order; return (filled_qty, avg_fill_price). No fees here."""
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


def _seed_gross_return(seed: int) -> tuple[Decimal, int]:
    """Run one seed on its own shuffled tape; return (total_gross_pnl, n_pairs).

    Skips the action_submitter for performance — the production-path
    coverage is in `_run_production_path` once. `walk_trades` is still
    called for every emitted action so any venue-side lookahead would
    surface in the per-seed return. Open positions at the horizon are
    force-liquidated by walking a final SELL through the last burst —
    leakage on entries that never closed inside the window cannot hide
    behind the open-long being silently dropped.
    """
    trades, signal_times = _per_seed_shuffled_tape(seed, _N_SIGNALS_PER_SEED)
    strategy = RandomTimingStrategy(
        f'sanity-random-{seed}',
        seed=seed,
        capital=_INITIAL_CAPITAL,
        kelly_pct=Decimal('1'),
        estimated_price=_BASE_PRICE,
        stop_bps=Decimal('50'),
    )
    params = StrategyParams(raw={})
    context = _empty_context()
    open_buy: tuple[Decimal, Decimal] | None = None
    total_pnl = Decimal('0')
    n_pairs = 0
    for i, ts in enumerate(signal_times):
        actions = strategy.on_signal(
            Signal(
                predictor_fn_id='sanity-random',
                timestamp=ts,
                values={'_preds': i % 2, '_probs': 0.5},
            ),
            params,
            context,
        )
        if not actions:
            continue
        # The strategy is single-shot per signal by construction. If
        # this assertion ever fires the test no longer covers what it
        # claims — a future regression that emits multiple actions per
        # signal would otherwise have its non-zeroth actions skipped
        # by the bulk harness.
        assert len(actions) == 1, (
            f'RandomTimingStrategy emitted {len(actions)} actions on signal #{i} '
            f'(seed={seed}); the bulk loop only walks single-action signals.'
        )
        action = actions[0]
        qty_filled, fill_px = _walk_one(action.direction, action.size, ts, trades)
        if action.direction == OrderSide.BUY:
            open_buy = (qty_filled, fill_px)
        else:
            assert open_buy is not None
            buy_qty, buy_px = open_buy
            sell_qty, sell_px = qty_filled, fill_px
            assert buy_qty == sell_qty
            total_pnl += (sell_px - buy_px) * sell_qty
            n_pairs += 1
            open_buy = None
    if open_buy is not None:
        # Horizon force-liquidation: SELL the held qty through the LAST
        # bar's tape so the seed's PnL captures the unclosed leg. A
        # leakage that only manifests on the BUY entry would otherwise
        # not surface in the seed's gross return when the position
        # never closes inside the window.
        buy_qty, buy_px = open_buy
        sell_qty, sell_px = _walk_one(
            OrderSide.SELL, buy_qty, signal_times[-1], trades,
        )
        assert sell_qty == buy_qty, (
            f'horizon liquidation qty mismatch buy={buy_qty} sell={sell_qty}'
        )
        total_pnl += (sell_px - buy_px) * sell_qty
        n_pairs += 1
    return total_pnl, n_pairs


def _run_production_path(seed: int, trades: pl.DataFrame, signal_times: list[datetime]) -> int:
    """Run one seed through the production submitter; return # outbound commands.

    Verifies that the random-timing strategy's actions reach Praxis
    untouched in side and qty, mirroring the production-path proofs
    in Tasks 2/3/4. Returns the outbound command count for assertion
    by the caller against the strategy's emitted action count.
    """
    pipeline, controller, capital_state = build_validation_pipeline(
        nexus_config=NexusInstanceConfig(account_id='bts-test', venue='binance_spot_simulated'),
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
            capital_controller=controller,
        strategy_budget=_INITIAL_CAPITAL,
    )
    submit = build_action_submitter(bindings)
    strategy = RandomTimingStrategy(
        f'sanity-random-prod-{seed}',
        seed=seed,
        capital=_INITIAL_CAPITAL,
        kelly_pct=Decimal('1'),
        estimated_price=_BASE_PRICE,
        stop_bps=Decimal('50'),
    )
    params = StrategyParams(raw={})
    context = _empty_context()
    sides_emitted: list[str] = []
    qtys_emitted: list[Decimal] = []
    for i, ts in enumerate(signal_times):
        actions = strategy.on_signal(
            Signal(
                predictor_fn_id='sanity-random-prod',
                timestamp=ts,
                values={'_preds': i % 2, '_probs': 0.5},
            ),
            params,
            context,
        )
        if not actions:
            continue
        assert len(actions) == 1, (
            f'RandomTimingStrategy emitted {len(actions)} actions on signal #{i}; '
            f'production-path runner only handles single-action signals.'
        )
        action = actions[0]
        sides_emitted.append(action.direction.name)
        qtys_emitted.append(action.size)
        # Walk for completeness (verifies the tape supports the order
        # at the action's timestamp), but PnL comes from the bulk loop.
        _walk_one(action.direction, action.size, ts, trades)
        submit(actions, 'sanity-random-prod')
    cmd_sides = [_extract_side(c) for c in outbound.commands]
    cmd_qtys = [_extract_qty(c) for c in outbound.commands]
    assert cmd_sides == sides_emitted, (
        f'production-path sides {cmd_sides} != emitted {sides_emitted}; '
        f'submitter mangled the random-timing flips.'
    )
    assert cmd_qtys == qtys_emitted, (
        f'production-path qtys {cmd_qtys} != emitted {qtys_emitted}; '
        f'submitter mutated order size.'
    )
    return len(outbound.commands)


def test_sanity_random_brackets_zero() -> None:
    """1000 seeds, mean gross-return 95% CI brackets 0 on per-seed shuffled tapes."""
    # Production-path coverage on one seed — proves the action_submitter
    # forwards every random-timing action to Praxis with side+qty intact.
    prod_trades, prod_signal_times = _per_seed_shuffled_tape(
        _PROD_PATH_SEED, _N_SIGNALS_PER_SEED,
    )
    n_outbound = _run_production_path(
        _PROD_PATH_SEED, prod_trades, prod_signal_times,
    )
    assert n_outbound > 0, (
        f'production-path seed {_PROD_PATH_SEED} produced zero submissions; '
        f'the Bernoulli flip is degenerate.'
    )

    # Bulk statistical loop — 1000 seeds, gross PnL only (no fees), per seed
    # divided by initial capital to express as a return.
    seed_returns: list[float] = []
    n_seeds_with_pairs = 0
    for seed in range(_N_SEEDS_BULK):
        pnl, n_pairs = _seed_gross_return(seed)
        if n_pairs == 0:
            # The Bernoulli mask can produce a seed with no closed pairs.
            # Such a seed contributes zero return, which is consistent
            # with the zero-mean null. Don't drop it — that would bias.
            seed_returns.append(0.0)
            continue
        n_seeds_with_pairs += 1
        seed_returns.append(float(pnl / _INITIAL_CAPITAL))

    # Sanity floor: most seeds should produce at least one pair (with
    # p_flip=0.5 over 20 signals, ~99.99 % of seeds emit something).
    assert n_seeds_with_pairs >= int(0.95 * _N_SEEDS_BULK), (
        f'only {n_seeds_with_pairs}/{_N_SEEDS_BULK} seeds produced any '
        f'closed pair; the random-timing distribution is degenerate.'
    )

    # 95 % confidence interval of the sample mean. With true mean 0 the
    # interval brackets zero in expectation, and any drift / lookahead
    # leak shows as a one-sided shift exceeding 1.96 SE.
    n = float(_N_SEEDS_BULK)
    mean = sum(seed_returns) / n
    var = sum((r - mean) ** 2 for r in seed_returns) / (n - 1)
    se = math.sqrt(var / n)
    ci_half = 1.96 * se
    ci_lo, ci_hi = mean - ci_half, mean + ci_half
    assert ci_lo <= 0.0 <= ci_hi, (
        f'mean gross return {mean:.6f} is outside the 95% CI brackets-zero '
        f'band [{ci_lo:.6f}, {ci_hi:.6f}]. Either the simulator has '
        f'asymmetric venue-side lookahead (e.g. best-of-K next-tick scan '
        f'in walk_trades) or a strategy-side cheat that observes a future '
        f'bar, or silent slippage on every fill. n_seeds={int(n)} '
        f'se={se:.6e}. Capital-state lookahead is not covered here — see '
        f'Tasks 2/3/4 for the production-path coverage.'
    )
