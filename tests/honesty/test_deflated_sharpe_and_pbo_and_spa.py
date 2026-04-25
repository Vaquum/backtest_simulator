"""Honesty gate: Deflated Sharpe + PBO + SPA agree with their inputs.

Pins slice #17 Task 17b / 17c / 17d.
"""
from __future__ import annotations

import math

import polars as pl
import pytest

from backtest_simulator.honesty.deflated_sharpe import (
    DeflatedSharpeResult,
    deflated_sharpe,
)
from backtest_simulator.honesty.pbo import (
    PboResult,
    probability_of_backtest_overfitting,
)
from backtest_simulator.honesty.spa import SpaResult, spa_test


def test_deflated_sharpe_and_pbo_and_spa() -> None:
    """All three statistics produce sane results on a known fixture."""

    # Deflated Sharpe: single-trial DSR uses E[max]=0 so z scales
    # the raw SR by sqrt(n-1)/sqrt(sr_var_factor). With sharpe=2,
    # skew=0, raw_kurt=3, n=252: sr_var_factor = 1 + (3-1)/4 * 4 = 3
    # so expected z is `2.0 * sqrt(251) / sqrt(3.0)`.
    dsr = deflated_sharpe(
        sharpe=2.0,
        n_trials=1,
        skew=0.0,
        kurtosis=0.0,
        n_observations=252,
    )
    assert isinstance(dsr, DeflatedSharpeResult)
    assert dsr.n_trials == 1
    expected_z = 2.0 * math.sqrt(251) / math.sqrt(3.0)
    assert abs(dsr.deflated_sharpe - expected_z) < 1e-6, (
        f'expected z ~= {expected_z}, got {dsr.deflated_sharpe}'
    )
    # Multi-trial inflation: more trials → larger E[max], smaller
    # deflated z, larger p_value.
    dsr_many = deflated_sharpe(
        sharpe=2.0, n_trials=1000,
        skew=0.0, kurtosis=0.0, n_observations=252,
    )
    assert dsr_many.deflated_sharpe < dsr.deflated_sharpe, (
        f'1000 trials must deflate the SR (lower z); got '
        f'1-trial={dsr.deflated_sharpe}, 1000-trial='
        f'{dsr_many.deflated_sharpe}'
    )
    assert dsr_many.p_value < dsr.p_value, (
        f'1000 trials must produce a smaller p-value (less skill '
        f'evidence); got 1-trial={dsr.p_value}, 1000-trial='
        f'{dsr_many.p_value}'
    )

    # ---- PBO ----
    # Build a 2-strategy fixture where strategy A consistently beats
    # B in-sample but they perform identically out-of-sample. PBO
    # should approach 0.5 — A is overfit-noise.
    n = 64
    in_sample_a = [0.001 * (i % 4 - 1) for i in range(n)]
    in_sample_b = [-x for x in in_sample_a]  # A wins IS by construction
    oos_random_a = [0.001 if i % 2 == 0 else -0.001 for i in range(n)]
    oos_random_b = [-x for x in oos_random_a]  # tied OOS in absolute terms
    is_pnl = pl.DataFrame({
        'A': in_sample_a, 'B': in_sample_b,
    })
    oos_pnl = pl.DataFrame({
        'A': oos_random_a, 'B': oos_random_b,
    })
    pbo = probability_of_backtest_overfitting(
        in_sample_pnl=is_pnl,
        out_of_sample_pnl=oos_pnl,
        n_groups=4,
    )
    assert isinstance(pbo, PboResult)
    assert pbo.n_strategies == 2
    assert pbo.n_splits == math.comb(4, 2)
    assert 0.0 <= pbo.pbo <= 1.0, (
        f'PBO must be a probability in [0,1]; got {pbo.pbo}'
    )

    # ---- SPA ----
    # Two candidates vs. a benchmark of zeros:
    #   candidate_a: tiny positive drift (should be insignificant
    #     after multiple-testing).
    #   candidate_b: zero drift.
    n_obs = 100
    candidates_df = pl.DataFrame({
        'A': [0.001 if i % 2 == 0 else -0.001 for i in range(n_obs)],
        'B': [0.0] * n_obs,
    })
    benchmark = pl.Series('benchmark', [0.0] * n_obs)
    spa = spa_test(
        candidate_returns=candidates_df,
        benchmark_returns=benchmark,
        block_size=10,
        n_bootstrap=200,
        seed=42,
    )
    assert isinstance(spa, SpaResult)
    assert spa.n_candidates == 2
    assert 0.0 <= spa.p_value <= 1.0, (
        f'SPA p_value must be in [0,1]; got {spa.p_value}'
    )
    # Determinism check: same seed → same statistic.
    spa_again = spa_test(
        candidate_returns=candidates_df,
        benchmark_returns=benchmark,
        block_size=10, n_bootstrap=200, seed=42,
    )
    assert spa.statistic == spa_again.statistic, (
        f'SPA must be deterministic at same seed; '
        f'got {spa.statistic} vs {spa_again.statistic}'
    )
    assert spa.p_value == spa_again.p_value, (
        f'SPA p_value must be deterministic at same seed; '
        f'got {spa.p_value} vs {spa_again.p_value}'
    )


def test_deflated_sharpe_rejects_invalid_inputs() -> None:
    """n_observations<=1 and n_trials<1 raise loud."""
    with pytest.raises(ValueError, match='n_observations must be > 1'):
        deflated_sharpe(
            sharpe=1.0, n_trials=1,
            skew=0.0, kurtosis=0.0, n_observations=1,
        )
    with pytest.raises(ValueError, match='n_trials must be >= 1'):
        deflated_sharpe(
            sharpe=1.0, n_trials=0,
            skew=0.0, kurtosis=0.0, n_observations=100,
        )


def test_pbo_rejects_invalid_inputs() -> None:
    """Odd n_groups, mismatched columns, empty df all raise."""
    pnl = pl.DataFrame({'A': [0.0, 0.1, 0.0, -0.1]})
    with pytest.raises(ValueError, match='n_groups must be even'):
        probability_of_backtest_overfitting(
            in_sample_pnl=pnl, out_of_sample_pnl=pnl, n_groups=3,
        )


def test_spa_rejects_length_mismatch() -> None:
    """Length-mismatch between candidates and benchmark raises."""
    candidates = pl.DataFrame({'A': [0.0, 0.1, 0.0]})
    bench = pl.Series('b', [0.0, 0.0])
    with pytest.raises(ValueError, match='length mismatch'):
        spa_test(
            candidate_returns=candidates,
            benchmark_returns=bench,
            block_size=2, n_bootstrap=10, seed=0,
        )
