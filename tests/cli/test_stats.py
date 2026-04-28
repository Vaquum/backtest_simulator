"""`bts sweep` post-sweep stats — slice #17 Task 17 wiring tests.

Pins the contract that `compute_sweep_stats` produces a
`SweepStats` whose DSR / SPA fields are exercised on a synthetic
sweep, and that `cpcv_pbo` consumes deployed-strategy daily
returns directly (auditor post-v2.0.1 fix). Mutation-proof:
switching the helper's IS/OOS slicing or wiring the wrong
decoder's returns flips the assertions.
"""
from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta, time as dtime
from decimal import Decimal

from backtest_simulator.cli._metrics import Trade
from backtest_simulator.cli._stats import (
    compute_sweep_stats,
    daily_return_for_run,
    fetch_buy_hold_benchmark,
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
    """Single-day sweep cannot fit DSR / SPA — all None."""
    stats = compute_sweep_stats(
        per_decoder_returns={'d1': [0.01], 'd2': [-0.01]},
        benchmark_returns=[0.005],
    )
    assert stats.dsr is None
    assert stats.spa is None


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
    """DSR runs as soon as n_obs >= 2; SPA too.

    Auditor: legacy half-split PBO (`_safe_pbo`) was removed —
    only `sweep cpcv_pbo` is the PBO surface now (driven by
    `cpcv_pbo` in the same module, exercised in `bts sweep`).
    `SweepStats` no longer carries a `.pbo` field.
    """
    stats = compute_sweep_stats(
        per_decoder_returns={
            'd1': [0.01, 0.02], 'd2': [-0.01, 0.005],
        },
        benchmark_returns=[0.005, 0.005],
    )
    assert stats.dsr is not None
    assert stats.spa is not None
    assert stats.best_decoder == 'd1'  # higher mean / non-negative variance
    assert stats.dsr.n_trials == 2


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


def _cpcv_fixture(
    *,
    returns_d1: list[float],
    returns_d2: list[float],
    n_days: int | None = None,
) -> tuple[list[datetime], dict[str, list[float]]]:
    """Build (days, per_decoder_returns) for cpcv_pbo tests.

    Auditor (post-v2.0.1): cpcv_pbo now consumes deployed-strategy
    daily returns (`per_decoder_returns`) directly, NOT bar-level
    SignalsTable + klines. The fixture builds a synthetic
    contiguous-day timeline + per-decoder daily return series.
    """
    if n_days is None:
        n_days = max(len(returns_d1), len(returns_d2))
    base = datetime(2026, 4, 20, 0, 0, tzinfo=UTC)
    days = [base + timedelta(days=i) for i in range(n_days)]
    return days, {'d1': returns_d1, 'd2': returns_d2}


def test_cpcv_pbo_runs_on_min_paths() -> None:
    """CPCV PBO runs on minimal CSCV: 4 groups x 8 days x 2 decoders.

    `CpcvPaths.build(n_groups=4, n_test_groups=2)` produces C(4,2)=6
    paths. With 8 days, each group has 2 days; each path partitions
    into 4 IS days + 4 OOS days. 2 non-tied per-decoder daily return
    series produce per-path Sharpes; surviving paths emit logits.
    """
    from backtest_simulator.cli._stats import cpcv_pbo
    from backtest_simulator.honesty.cpcv import CpcvPaths
    paths = CpcvPaths.build(
        n_groups=4, n_test_groups=2,
        purge_seconds=0, embargo_seconds=0,
    )
    days, per_decoder = _cpcv_fixture(
        returns_d1=[0.01, -0.005, 0.02, -0.01, 0.015, -0.008, 0.012, -0.003],
        returns_d2=[-0.01, 0.008, -0.015, 0.005, -0.012, 0.01, -0.008, 0.005],
    )
    result = cpcv_pbo(
        paths=paths,
        per_decoder_returns=per_decoder,
        days=days,
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
    days, per_decoder = _cpcv_fixture(
        returns_d1=[0.01, -0.005, 0.02, -0.01, 0.015, -0.008, 0.012, -0.003],
        returns_d2=[-0.01, 0.008, -0.015, 0.005, -0.012, 0.01, -0.008, 0.005],
    )
    result = cpcv_pbo(
        paths=paths,
        per_decoder_returns={'d1': per_decoder['d1']},  # only 1 decoder
        days=days,
    )
    assert result is None


def test_cpcv_pbo_skips_on_pairwise_identical_returns() -> None:
    """All-equal decoder per-day return series -> None.

    Without the pairwise-identical guard, every path picks the same
    first decoder via `max` deterministic tie ordering, producing
    fake `pbo=0.000`. The guard returns None so sweep skips with
    reason.

    Mutation proof: removing the pairwise-identical check at the
    series level makes the result non-None.
    """
    from backtest_simulator.cli._stats import cpcv_pbo
    from backtest_simulator.honesty.cpcv import CpcvPaths
    paths = CpcvPaths.build(
        n_groups=4, n_test_groups=2,
        purge_seconds=0, embargo_seconds=0,
    )
    same_returns = [0.01, -0.005, 0.02, -0.01, 0.015, -0.008, 0.012, -0.003]
    days, per_decoder = _cpcv_fixture(
        returns_d1=same_returns, returns_d2=list(same_returns),
    )
    result = cpcv_pbo(
        paths=paths,
        per_decoder_returns=per_decoder,
        days=days,
    )
    assert result is None


def test_cpcv_pbo_uses_deployed_strategy_returns_not_proxy() -> None:
    """cpcv_pbo's input IS the deployed-strategy returns dict.

    Auditor (post-v2.0.1): the prior bar-level implementation
    ranked decoders on `pred * close_to_next_close_return` — a
    signal-return proxy that ignored stops, slippage, impact,
    maker-fill, and trailing-inventory exclusions. The new
    function takes `per_decoder_returns` directly: the same
    `daily_return_for_run` output that already drives DSR/SPA.

    This test pins the input shape — `cpcv_pbo` accepts a
    `dict[str, list[float]]` keyed by decoder_id, with each
    value being net PnL fractions per clean day. Mutation proof:
    if `cpcv_pbo` reverts to taking `signals_per_decoder /
    klines / tick_timestamps`, calling it with the new arg names
    raises TypeError and this test fires.
    """
    from backtest_simulator.cli._stats import cpcv_pbo
    from backtest_simulator.honesty.cpcv import CpcvPaths
    paths = CpcvPaths.build(
        n_groups=4, n_test_groups=2,
        purge_seconds=0, embargo_seconds=0,
    )
    days, per_decoder = _cpcv_fixture(
        returns_d1=[0.01, -0.005, 0.02, -0.01, 0.015, -0.008, 0.012, -0.003],
        returns_d2=[-0.01, 0.008, -0.015, 0.005, -0.012, 0.01, -0.008, 0.005],
    )
    # Calling with the new kwargs MUST type-check; calling with
    # old kwargs (signals_per_decoder, tick_timestamps, klines)
    # MUST raise TypeError. Verify the latter as a contract pin.
    import pytest
    with pytest.raises(TypeError):
        cpcv_pbo(  # type: ignore[call-arg]
            paths=paths,
            signals_per_decoder={},
            tick_timestamps=days,
            klines=None,
        )
    # And the new signature works.
    result = cpcv_pbo(
        paths=paths,
        per_decoder_returns=per_decoder,
        days=days,
    )
    assert result is not None


def test_cpcv_pbo_purge_seconds_recorded_in_result() -> None:
    """`purge_seconds` and `embargo_seconds` flow from path to result.

    The result echoes the seconds the operator configured. With
    purge=86400 + embargo=86400 on a 12-day fixture, train days
    adjacent to test days are dropped; surviving paths still
    record the supplied seconds.
    """
    from backtest_simulator.cli._stats import cpcv_pbo
    from backtest_simulator.honesty.cpcv import CpcvPaths
    paths = CpcvPaths.build(
        n_groups=4, n_test_groups=2,
        purge_seconds=86400, embargo_seconds=86400,
    )
    # 12 days so post-purge IS still has >=2 days per path.
    days, per_decoder = _cpcv_fixture(
        returns_d1=[
            0.01, -0.005, 0.02, -0.01, 0.015, -0.008,
            0.012, -0.003, 0.018, -0.011, 0.014, -0.006,
        ],
        returns_d2=[
            -0.01, 0.008, -0.015, 0.005, -0.012, 0.01,
            -0.008, 0.005, -0.013, 0.009, -0.011, 0.004,
        ],
    )
    result = cpcv_pbo(
        paths=paths,
        per_decoder_returns=per_decoder,
        days=days,
    )
    if result is not None:
        assert result.purge_seconds == 86400
        assert result.embargo_seconds == 86400


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
