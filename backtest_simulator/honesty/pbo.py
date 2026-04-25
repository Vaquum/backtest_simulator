"""Probability of Backtest Overfitting (Bailey, Borwein, López de Prado)."""
from __future__ import annotations

# Slice #17 Task 17 (PBO portion). PBO measures how often the
# best in-sample strategy under-performs out-of-sample. A high
# PBO (> 0.5) means the in-sample winner is essentially random
# noise — the backtest is overfit. Reference: Bailey, Borwein,
# López de Prado, Zhu (2017), "The Probability of Backtest
# Overfitting".
#
# Implementation: combinatorial-symmetric cross-validation. Given
# `n_groups` time-blocks of returns, partition into all
# C(n_groups, n_groups/2) (in-sample, out-of-sample) splits.
# For each split:
#   - rank each strategy by in-sample Sharpe.
#   - find the in-sample winner.
#   - rank that winner out-of-sample.
#   - logit transform the relative rank.
# PBO = fraction of splits where the in-sample winner ranks below
# median (logit < 0) out-of-sample.
import math
from dataclasses import dataclass
from itertools import combinations

import polars as pl


@dataclass(frozen=True)
class PboResult:
    """Probability of backtest overfitting + the logit-rank distribution."""

    pbo: float
    n_splits: int
    n_strategies: int


def _sharpe(returns: list[float]) -> float:
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    if var <= 0.0:
        return 0.0
    return mean / math.sqrt(var)


def probability_of_backtest_overfitting(
    *,
    in_sample_pnl: pl.DataFrame,
    out_of_sample_pnl: pl.DataFrame,
    n_groups: int,
) -> PboResult:
    """Compute PBO from a combinatorial split of strategy returns.

    Args:
      in_sample_pnl: DataFrame where each column is one strategy's
        return series over the in-sample window. Rows are
        observations; columns are strategy ids.
      out_of_sample_pnl: same shape, holdout window.
      n_groups: number of contiguous time-blocks the rows split
        into. Must be even (CSCV requires balanced halves).
    """
    if n_groups % 2 != 0 or n_groups < 2:
        msg = (
            f'probability_of_backtest_overfitting: n_groups must be even '
            f'and >= 2 for CSCV; got {n_groups}.'
        )
        raise ValueError(msg)
    if in_sample_pnl.is_empty() or out_of_sample_pnl.is_empty():
        msg = (
            'probability_of_backtest_overfitting: PnL DataFrames must '
            'be non-empty. Calibrate over a window with realised returns.'
        )
        raise ValueError(msg)
    strategies = list(in_sample_pnl.columns)
    if list(out_of_sample_pnl.columns) != strategies:
        msg = (
            'probability_of_backtest_overfitting: in-sample and '
            'out-of-sample DataFrames must share strategy columns. '
            f'in={strategies} oos={list(out_of_sample_pnl.columns)}'
        )
        raise ValueError(msg)
    in_n = len(in_sample_pnl)
    oos_n = len(out_of_sample_pnl)
    if in_n < n_groups or oos_n < n_groups:
        msg = (
            f'probability_of_backtest_overfitting: each PnL needs at '
            f'least {n_groups} rows; got in={in_n}, oos={oos_n}.'
        )
        raise ValueError(msg)
    # Concatenate IS+OOS, partition into n_groups, then iterate over
    # combinations of n_groups/2 group-indices as IS and remainder
    # as OOS.
    full = pl.concat([in_sample_pnl, out_of_sample_pnl], how='vertical')
    rows_per_group = len(full) // n_groups
    group_slices: list[pl.DataFrame] = []
    for g in range(n_groups):
        start = g * rows_per_group
        end = start + rows_per_group if g < n_groups - 1 else len(full)
        group_slices.append(full.slice(start, end - start))
    half = n_groups // 2
    n_strategies = len(strategies)
    overfit_count = 0
    n_splits = 0
    for is_group_ids in combinations(range(n_groups), half):
        oos_group_ids = tuple(
            g for g in range(n_groups) if g not in is_group_ids
        )
        is_data = pl.concat(
            [group_slices[g] for g in is_group_ids],
            how='vertical',
        )
        oos_data = pl.concat(
            [group_slices[g] for g in oos_group_ids],
            how='vertical',
        )
        is_sharpes = [
            _sharpe(is_data[s].to_list()) for s in strategies
        ]
        oos_sharpes = [
            _sharpe(oos_data[s].to_list()) for s in strategies
        ]
        # In-sample winner.
        best_is = is_sharpes.index(max(is_sharpes))
        # Out-of-sample rank (1-indexed): sort descending, find the
        # winner's rank.
        oos_sorted_desc = sorted(
            range(n_strategies), key=lambda i: -oos_sharpes[i],
        )
        winner_rank = oos_sorted_desc.index(best_is) + 1
        relative_rank = winner_rank / (n_strategies + 1)
        # Logit < 0 ⇔ rank below median ⇔ overfit.
        if relative_rank > 0.5:
            overfit_count += 1
        n_splits += 1
    pbo = overfit_count / max(n_splits, 1)
    return PboResult(pbo=pbo, n_splits=n_splits, n_strategies=n_strategies)
