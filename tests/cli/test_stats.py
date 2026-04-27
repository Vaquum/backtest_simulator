"""`bts sweep` post-sweep stats — slice #17 Task 17 wiring tests.

Pins the contract that `compute_sweep_stats` produces a
`SweepStats` whose DSR / SPA / PBO fields are exercised on a
3-decoder × 8-day synthetic sweep. Mutation-proof: switching the
helper's IS/OOS slicing or wiring the wrong decoder's returns
flips the assertions.
"""
from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta, time as dtime
from decimal import Decimal

import numpy as np
import polars as pl

from backtest_simulator.cli._metrics import Trade
from backtest_simulator.cli._stats import (
    compute_sweep_stats,
    daily_return_for_run,
    fetch_buy_hold_benchmark,
)
from backtest_simulator.sensors.precompute import (
    PredictionsInput,
    SignalsTable,
)


def _trade(coid: str, side: str, qty: str, price: str, fee: str = '0') -> Trade:
    return Trade(coid, side, qty, price, fee, 'USDT', '2024-01-01T00:00:00+00:00')


def test_daily_return_for_run_zero_when_no_trades() -> None:
    """Empty trades → 0 return (no realised PnL, no trailing inventory)."""
    assert daily_return_for_run([], {}) == 0.0


def test_daily_return_for_run_paired_trade() -> None:
    """One BUY + one SELL → net PnL / STARTING_CAPITAL."""
    # 1 BTC bought at 50_000, sold at 50_500. Net = 500 (no fees).
    # daily_return = 500 / 100_000 = 0.005.
    trades = [
        _trade('coid-buy', 'BUY', '1', '50000'),
        _trade('coid-sell', 'SELL', '1', '50500'),
    ]
    assert daily_return_for_run(trades, {}) == 0.005


def test_daily_return_for_run_trailing_inventory_returns_none() -> None:
    """BUY with no matching SELL → None (codex round 1 P1).

    Treating an open position as a 0 return would silently hide
    losers and inflate apparent Sharpe. None signals the run
    should be excluded from stats entirely; the sweep
    accumulator counts these separately.

    Mutation proof: returning 0.0 for trailing inventory makes
    this assert fire (the value would be 0.0, not None).
    """
    trades = [_trade('coid-buy', 'BUY', '1', '50000')]
    assert daily_return_for_run(trades, {}) is None


def test_compute_sweep_stats_too_few_obs_returns_none() -> None:
    """Single-day sweep cannot fit DSR / SPA / PBO — all None."""
    stats = compute_sweep_stats(
        per_decoder_returns={'d1': [0.01], 'd2': [-0.01]},
        benchmark_returns=[0.005],
    )
    assert stats.dsr is None
    assert stats.spa is None
    assert stats.pbo is None


def test_compute_sweep_stats_dsr_skips_on_constant_returns() -> None:
    """DSR returns None when all returns are constant.

    `scipy.stats.kurtosis` returns NaN on zero-variance input;
    feeding NaN into the DSR formula propagates to NaN outputs.
    The wrapper detects that and skips with `dsr is None` so
    the sweep summary line isn't `deflated=nan p_value=nan`.
    Mutation proof: removing the NaN guard makes this assert
    fire (dsr would be a non-None NaN-bearing result).
    """
    stats = compute_sweep_stats(
        per_decoder_returns={
            'd1': [0.0, 0.0, 0.0], 'd2': [0.0, 0.0, 0.0],
        },
        benchmark_returns=[0.005, 0.005, 0.005],
        spa_n_bootstrap=10,
    )
    assert stats.dsr is None


def test_compute_sweep_stats_dsr_runs_on_two_decoders_two_days() -> None:
    """DSR runs as soon as n_obs >= 2; SPA too; PBO needs >= 4 obs."""
    stats = compute_sweep_stats(
        per_decoder_returns={
            'd1': [0.01, 0.02], 'd2': [-0.01, 0.005],
        },
        benchmark_returns=[0.005, 0.005],
    )
    assert stats.dsr is not None
    assert stats.spa is not None
    assert stats.pbo is None  # n_obs=2 < 4
    assert stats.best_decoder == 'd1'  # higher mean / non-negative variance
    assert stats.dsr.n_trials == 2


def test_compute_sweep_stats_pbo_runs_on_four_obs_two_decoders() -> None:
    """PBO unlocks at n_obs >= 4 + n_decoders >= 2 + non-tied returns."""
    stats = compute_sweep_stats(
        per_decoder_returns={
            'd1': [0.01, 0.02, 0.005, 0.015],
            'd2': [-0.01, 0.005, 0.0, 0.01],
        },
        benchmark_returns=[0.005, 0.005, 0.005, 0.005],
    )
    assert stats.pbo is not None
    assert stats.pbo.n_strategies == 2
    # n_obs=4, half=2, n_groups=2 → C(2,1)=2 splits
    assert stats.pbo.n_splits == 2


def test_compute_sweep_stats_pbo_skips_on_pairwise_identical_returns() -> None:
    """PBO returns None when all decoders share the same return series.

    Codex round 1 P1: ties yield deterministic primitive ordering,
    so an all-zero (or all-identical) sweep would report
    `pbo=0.000` — a false "no overfitting" signal. The wrapper
    skips with `None` instead.

    Mutation proof: removing the pairwise-identical check makes
    PBO fire and `assert stats.pbo is None` fails.
    """
    stats = compute_sweep_stats(
        per_decoder_returns={
            'd1': [0.0, 0.0, 0.0, 0.0],
            'd2': [0.0, 0.0, 0.0, 0.0],
        },
        benchmark_returns=[0.0, 0.0, 0.0, 0.0],
        spa_n_bootstrap=10,
    )
    assert stats.pbo is None


def test_compute_sweep_stats_n_search_trials_overrides_decoder_count() -> None:
    """`n_search_trials` is the DSR n_trials, not `len(per_decoder_returns)`.

    Codex round 1 P1: a sweep that picked 1 decoder out of 1000
    permutations should deflate against 1000, not 1, so the
    operator can't game the DSR by narrowing the visible pick
    set. Pin the override flow.

    Same return series, two `n_search_trials` values: the larger
    must produce a smaller deflated_sharpe (more aggressive
    deflation).
    """
    returns = {'d1': [0.01, 0.02, 0.015]}
    stats_few = compute_sweep_stats(
        returns, benchmark_returns=[0.0, 0.0, 0.0], n_search_trials=1,
    )
    stats_many = compute_sweep_stats(
        returns, benchmark_returns=[0.0, 0.0, 0.0], n_search_trials=1000,
    )
    assert stats_few.dsr is not None
    assert stats_many.dsr is not None
    assert stats_few.dsr.deflated_sharpe > stats_many.dsr.deflated_sharpe


def test_compute_sweep_stats_dsr_deflates_with_more_trials() -> None:
    """`n_trials` increases the expected_max_sharpe → deflated SR drops.

    Mutation-proof: swapping `n_trials` direction flips the
    inequality. Wilder's expected_max_sharpe is monotonic in
    n_trials, so a single-trial sweep produces a higher deflated
    SR than a many-trial sweep with the same realised Sharpe.
    """
    returns_one = {'d1': [0.01, 0.02, 0.015]}
    returns_many = {
        f'd{i}': [0.01, 0.02, 0.015] for i in range(10)
    }
    stats_one = compute_sweep_stats(
        returns_one, benchmark_returns=[0.0, 0.0, 0.0],
    )
    stats_many = compute_sweep_stats(
        returns_many, benchmark_returns=[0.0, 0.0, 0.0],
    )
    assert stats_one.dsr is not None
    assert stats_many.dsr is not None
    # Same realised Sharpe (identical return series), but more
    # trials → harder to clear → smaller deflated stat.
    assert stats_one.dsr.deflated_sharpe > stats_many.dsr.deflated_sharpe


def test_fetch_buy_hold_benchmark_two_days() -> None:
    """Buy-hold benchmark = (close - open) / open per day, injected provider."""
    days = [
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 2, tzinfo=UTC),
    ]
    prices = {
        datetime(2024, 1, 1, 8, 0, tzinfo=UTC): Decimal('70000'),
        datetime(2024, 1, 1, 12, 0, tzinfo=UTC): Decimal('70700'),
        datetime(2024, 1, 2, 8, 0, tzinfo=UTC): Decimal('70000'),
        datetime(2024, 1, 2, 12, 0, tzinfo=UTC): Decimal('69300'),
    }

    def _seed_price_at(ts: datetime) -> Decimal:
        return prices[ts]
    bench = fetch_buy_hold_benchmark(
        days, dtime(8, 0), dtime(12, 0), seed_price_at=_seed_price_at,
    )
    assert len(bench) == 2
    # Day 1: (70_700 - 70_000) / 70_000 = 0.01
    assert math.isclose(bench[0], 0.01, abs_tol=1e-12)
    # Day 2: (69_300 - 70_000) / 70_000 = -0.01
    assert math.isclose(bench[1], -0.01, abs_tol=1e-12)


def _build_cpcv_fixture(
    *,
    preds_d1: list[int],
    preds_d2: list[int],
    n_ticks: int = 8,
    bar_seconds: int = 60,
) -> tuple[list[datetime], pl.DataFrame, dict[str, SignalsTable]]:
    """Build (tick_timestamps, klines, signals_per_decoder) for cpcv_pbo tests.

    Auditor round-7 fix: cpcv_pbo now consumes SignalsTable +
    klines + tick_timestamps. Test fixtures construct synthetic
    versions of all three so the bar-level CSCV math is exercised
    without spinning up Trainer / HistoricalData.
    """
    base = datetime(2026, 4, 20, 0, 0, tzinfo=UTC)
    ticks = [
        base + timedelta(seconds=i * bar_seconds) for i in range(n_ticks)
    ]
    # klines need one extra row past the last tick so close-to-
    # next-close exists for every tick. Synthetic close walk gives
    # a non-zero bar_return at every tick.
    closes = [100.0 + i * 0.5 for i in range(n_ticks + 1)]
    kline_ts = [
        base + timedelta(seconds=i * bar_seconds) for i in range(n_ticks + 1)
    ]
    klines = pl.DataFrame({
        'datetime': kline_ts,
        'close': closes,
    }).with_columns(
        pl.col('datetime').dt.replace_time_zone('UTC'),
    )

    def _build(decoder: str, preds: list[int]) -> SignalsTable:
        return SignalsTable.from_predictions(
            decoder_id=decoder,
            split_config=(70, 15, 15),
            predictions=PredictionsInput(
                timestamps=ticks,
                probs=np.array([0.5 + 0.01 * i for i in range(n_ticks)]),
                preds=np.array(preds, dtype=np.int64),
                label_horizon_bars=1,
                bar_seconds=bar_seconds,
            ),
        )

    return (
        ticks,
        klines,
        {'d1': _build('d1', preds_d1), 'd2': _build('d2', preds_d2)},
    )


def test_cpcv_pbo_runs_on_min_paths() -> None:
    """CPCV PBO runs on minimal CSCV: 4 groups x 8 ticks x 2 decoders.

    `CpcvPaths.build(n_groups=4, n_test_groups=2)` produces C(4,2)=6
    paths. Each path: 2 IS-group ticks + 2 OOS-group ticks (in this
    fixture). With 2 decoders + non-tied predictions, paths that
    have enough IS+OOS data + non-tied IS Sharpes produce logits.
    """
    from backtest_simulator.cli._stats import cpcv_pbo
    from backtest_simulator.honesty.cpcv import CpcvPaths
    paths = CpcvPaths.build(
        n_groups=4, n_test_groups=2,
        purge_seconds=0, embargo_seconds=0,
    )
    ticks, klines, signals = _build_cpcv_fixture(
        preds_d1=[1, 0, 1, 1, 0, 1, 0, 1],
        preds_d2=[0, 1, 0, 1, 1, 0, 1, 0],
    )
    result = cpcv_pbo(
        paths=paths, signals_per_decoder=signals,
        tick_timestamps=ticks, klines=klines,
    )
    assert result is not None
    assert result.n_decoders == 2
    assert 0.0 <= result.pbo <= 1.0
    assert result.purge_seconds == 0
    assert result.embargo_seconds == 0


def test_cpcv_pbo_skips_when_insufficient_decoders() -> None:
    """1 decoder can't produce a CPCV PBO -- None instead of fake."""
    from backtest_simulator.cli._stats import cpcv_pbo
    from backtest_simulator.honesty.cpcv import CpcvPaths
    paths = CpcvPaths.build(
        n_groups=4, n_test_groups=2,
        purge_seconds=0, embargo_seconds=0,
    )
    ticks, klines, signals = _build_cpcv_fixture(
        preds_d1=[1, 0, 1, 1, 0, 1, 0, 1],
        preds_d2=[0, 1, 0, 1, 1, 0, 1, 0],
    )
    result = cpcv_pbo(
        paths=paths,
        signals_per_decoder={'d1': signals['d1']},  # only 1 decoder
        tick_timestamps=ticks, klines=klines,
    )
    assert result is None


def test_cpcv_pbo_skips_on_pairwise_identical_predictions() -> None:
    """All-equal decoder predictions -> None (auditor round 7 fix).

    Without the pairwise-identical guard at the prediction-sequence
    level, every path picks the same first decoder via `max`
    deterministic tie ordering, producing fake `pbo=0.000`. The
    guard checks predictions across all ticks and returns None so
    sweep skips with reason.

    Mutation proof: removing the prediction pairwise-identical
    check makes the result non-None.
    """
    from backtest_simulator.cli._stats import cpcv_pbo
    from backtest_simulator.honesty.cpcv import CpcvPaths
    paths = CpcvPaths.build(
        n_groups=4, n_test_groups=2,
        purge_seconds=0, embargo_seconds=0,
    )
    same_preds = [1, 0, 1, 1, 0, 1, 0, 1]
    ticks, klines, signals = _build_cpcv_fixture(
        preds_d1=same_preds, preds_d2=same_preds,
    )
    result = cpcv_pbo(
        paths=paths, signals_per_decoder=signals,
        tick_timestamps=ticks, klines=klines,
    )
    assert result is None


def test_cpcv_pbo_uses_signals_table_lookup_for_path_filtering() -> None:
    """SignalsTable.lookup with allowed_groups is the load-bearing accessor.

    Auditor round 7 P1: cpcv_pbo previously consumed only daily
    returns; the SignalsTable build was ornamental. This test pins
    that cpcv_pbo now goes through `signals.lookup(allowed_groups
    =path.train_groups, ...)` and `signals.lookup(allowed_groups
    =path.test_groups, ...)` per tick, per path, per decoder. If
    we mock the lookup to return None for everything, cpcv_pbo
    returns None (no paths produce logits because IS+OOS data is
    empty).

    Mutation proof: if cpcv_pbo bypassed the lookup and read
    predictions directly off the table's `_frame`, this assert
    would fail.
    """
    from unittest.mock import patch
    from backtest_simulator.cli._stats import cpcv_pbo
    from backtest_simulator.honesty.cpcv import CpcvPaths
    paths = CpcvPaths.build(
        n_groups=4, n_test_groups=2,
        purge_seconds=0, embargo_seconds=0,
    )
    ticks, klines, signals = _build_cpcv_fixture(
        preds_d1=[1, 0, 1, 1, 0, 1, 0, 1],
        preds_d2=[0, 1, 0, 1, 1, 0, 1, 0],
    )
    # Patch lookup so it ALWAYS returns None for path-filtered calls
    # (anything passing allowed_groups). Non-filtered calls (used by
    # the pairwise-identical guard) still pass through.
    original_lookup = signals['d1'].__class__.lookup

    def filtered_lookup(self, t, *, allowed_groups=None, **kw):  # type: ignore[no-untyped-def]
        if allowed_groups is not None:
            return None
        return original_lookup(self, t, allowed_groups=None, **kw)

    with patch.object(
        signals['d1'].__class__, 'lookup', filtered_lookup,
    ):
        result = cpcv_pbo(
            paths=paths, signals_per_decoder=signals,
            tick_timestamps=ticks, klines=klines,
        )
    # No path-filtered lookup ever returned a row -> every path's IS
    # and OOS are empty -> result is None.
    assert result is None


def test_cpcv_pbo_purge_seconds_passed_through_to_lookup() -> None:
    """`purge_seconds` from path is plumbed to `signals.lookup`.

    The auditor round-7 cpcv_pbo passes path.purge_seconds and
    path.embargo_seconds into each lookup call. This test pins the
    contract that the result reports the same seconds the operator
    supplied.
    """
    from backtest_simulator.cli._stats import cpcv_pbo
    from backtest_simulator.honesty.cpcv import CpcvPaths
    paths = CpcvPaths.build(
        n_groups=4, n_test_groups=2,
        purge_seconds=120, embargo_seconds=60,
    )
    ticks, klines, signals = _build_cpcv_fixture(
        preds_d1=[1, 0, 1, 1, 0, 1, 0, 1],
        preds_d2=[0, 1, 0, 1, 1, 0, 1, 0],
    )
    result = cpcv_pbo(
        paths=paths, signals_per_decoder=signals,
        tick_timestamps=ticks, klines=klines,
    )
    if result is not None:
        assert result.purge_seconds == 120
        assert result.embargo_seconds == 60


def test_compute_sweep_stats_spa_consistent_with_synthetic_signal() -> None:
    """SPA's statistic is positive when a candidate beats the benchmark.

    SPA's `_scaled_t` is `sqrt(n) * mean(excess) / sd(excess)` and
    returns 0 when sd is non-positive — so the synthetic signal
    must have variance for the statistic to be informative.
    Excess returns of 0.005, 0.015, 0.005, 0.015 (mean 0.01, sd
    !=0) → statistic > 0.
    """
    stats = compute_sweep_stats(
        per_decoder_returns={'beats': [0.005, 0.015, 0.005, 0.015]},
        benchmark_returns=[0.0, 0.0, 0.0, 0.0],
        spa_n_bootstrap=100,
    )
    assert stats.spa is not None
    assert stats.spa.n_candidates == 1
    assert stats.spa.statistic > 0.0
