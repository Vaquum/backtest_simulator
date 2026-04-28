"""Hansen's Superior Predictive Ability (SPA) test."""
from __future__ import annotations

# Slice #17 Task 17 (SPA portion). Hansen (2005), "A test for
# superior predictive ability". Tests the null that NO candidate
# strategy outperforms the benchmark, controlling for
# multiple-testing across the candidate set. The test statistic is
# the maximum standardized excess return; its p-value is computed
# via a stationary bootstrap with `block_size` mean block length.
import math
import random
from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class SpaResult:
    """Hansen SPA test outcome."""

    statistic: float
    p_value: float
    n_candidates: int


def _bootstrap_indices(
    n: int, block_size: int, rng: random.Random,
) -> list[int]:
    # Politis-Romano stationary bootstrap: at each step, with
    # probability 1/block_size start a new block at a random index;
    # otherwise step forward by 1.
    if block_size <= 0:
        msg = f'block_size must be positive; got {block_size}'
        raise ValueError(msg)
    p_new = 1.0 / float(block_size)
    indices: list[int] = []
    current = rng.randrange(n)
    for _ in range(n):
        indices.append(current)
        if rng.random() < p_new:
            current = rng.randrange(n)
        else:
            current = (current + 1) % n
    return indices


def _scaled_t(values: list[float], n: int) -> float:
    """Studentised mean: `sqrt(n) * mean / sd`. Returns 0 when var is non-positive."""
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / max(n - 1, 1)
    if var <= 0.0:
        return 0.0
    return math.sqrt(n) * mean / math.sqrt(var)


def _bootstrap_max_t(
    d_matrix: list[list[float]],
    n: int,
    block_size: int,
    rng: random.Random,
) -> float:
    bs_indices = _bootstrap_indices(n, block_size, rng)
    bs_stats: list[float] = []
    for d in d_matrix:
        mean_d_full = sum(d) / n
        recentered = [d[idx] - max(mean_d_full, 0.0) for idx in bs_indices]
        bs_stats.append(_scaled_t(recentered, n))
    return max(bs_stats)


def spa_test(
    *,
    candidate_returns: pl.DataFrame,
    benchmark_returns: pl.Series,
    block_size: int,
    n_bootstrap: int,
    seed: int,
) -> SpaResult:
    """Run Hansen's SPA test.

    Args:
      candidate_returns: rows = observations, columns = candidate
        strategies' return series.
      benchmark_returns: same length, the reference series.
      block_size: stationary bootstrap mean block length.
      n_bootstrap: number of bootstrap replications.
      seed: deterministic RNG seed.
    """
    if candidate_returns.is_empty() or benchmark_returns.is_empty():
        msg = (
            'spa_test: candidate_returns and benchmark_returns must '
            'be non-empty.'
        )
        raise ValueError(msg)
    n = len(candidate_returns)
    if len(benchmark_returns) != n:
        msg = (
            f'spa_test: length mismatch — candidate has {n} rows, '
            f'benchmark has {len(benchmark_returns)}.'
        )
        raise ValueError(msg)
    if n_bootstrap < 1:
        msg = f'spa_test: n_bootstrap must be >= 1; got {n_bootstrap}'
        raise ValueError(msg)
    candidates = list(candidate_returns.columns)
    n_candidates = len(candidates)
    bench = benchmark_returns.to_list()
    # Excess returns matrix: d_{i,t} = candidate_i(t) - benchmark(t).
    d_matrix: list[list[float]] = [
        [candidate_returns[c].to_list()[t] - bench[t] for t in range(n)]
        for c in candidates
    ]
    realised_t = max(_scaled_t(d, n) for d in d_matrix)
    rng = random.Random(seed)
    exceed = sum(
        1
        for _ in range(n_bootstrap)
        if _bootstrap_max_t(d_matrix, n, block_size, rng) >= realised_t
    )
    p_value = exceed / n_bootstrap
    return SpaResult(
        statistic=realised_t,
        p_value=p_value,
        n_candidates=n_candidates,
    )
