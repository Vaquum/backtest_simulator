"""Deflated Sharpe Ratio — Bailey & López de Prado (2014)."""
from __future__ import annotations

# Slice #17 Task 17 (DSR portion). The unadjusted Sharpe ratio
# overstates skill when the strategy has been selected from many
# trials. The Deflated Sharpe Ratio (DSR) is the probability that
# the realised Sharpe exceeds the maximum expected Sharpe under
# the null hypothesis, given:
#   - n_trials: number of independent strategies considered.
#   - skew, kurtosis: higher moments of the realised return series.
#   - n_observations: number of return observations.
# Reference: Bailey, Borwein, López de Prado, Zhu (2014),
# "The Probability of Backtest Overfitting".
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """Probability-of-skill given multiple-testing inflation."""

    deflated_sharpe: float
    p_value: float
    n_trials: int


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _expected_max_sharpe(n_trials: int) -> float:
    # Bailey-LdP approximation: E[max_n SR] ≈ sqrt(2 ln n) for large
    # n, with the Euler-Mascheroni adjustment for finite n.
    if n_trials <= 1:
        return 0.0
    n = float(n_trials)
    euler_mascheroni = 0.5772156649015329
    inv_phi_term = math.sqrt(
        max(2.0 * math.log(n), 1e-12),
    )
    return inv_phi_term - euler_mascheroni / inv_phi_term


def deflated_sharpe(
    *,
    sharpe: float,
    n_trials: int,
    skew: float,
    kurtosis: float,
    n_observations: int,
) -> DeflatedSharpeResult:
    """Compute the deflated Sharpe ratio + its p-value.

    Args:
      sharpe: realised Sharpe ratio of the candidate strategy.
      n_trials: number of strategies considered (multiple-testing
        inflation factor).
      skew: realised return-series skewness.
      kurtosis: realised return-series kurtosis (excess kurtosis;
        normal = 0). Bailey-LdP uses raw kurtosis = excess + 3.
      n_observations: number of return observations used to
        compute `sharpe`. Must be > 1.
    """
    if n_observations <= 1:
        msg = (
            f'deflated_sharpe: n_observations must be > 1; got '
            f'{n_observations}. The variance of SR estimates is '
            f'undefined for a single observation.'
        )
        raise ValueError(msg)
    if n_trials < 1:
        msg = f'deflated_sharpe: n_trials must be >= 1; got {n_trials}.'
        raise ValueError(msg)
    raw_kurtosis = kurtosis + 3.0
    expected_max_sr = _expected_max_sharpe(n_trials)
    sr_variance_factor = (
        1.0 - skew * sharpe + (raw_kurtosis - 1.0) / 4.0 * sharpe * sharpe
    )
    if sr_variance_factor <= 0.0:
        # Pathological: variance estimate non-positive. Use the
        # original (un-deflated) Sharpe; downstream operators see
        # the raw value alongside p_value=NaN-equivalent.
        return DeflatedSharpeResult(
            deflated_sharpe=sharpe,
            p_value=0.5,
            n_trials=n_trials,
        )
    z = (
        (sharpe - expected_max_sr)
        * math.sqrt(max(n_observations - 1, 1))
        / math.sqrt(sr_variance_factor)
    )
    p = _norm_cdf(z)
    return DeflatedSharpeResult(
        deflated_sharpe=z,
        p_value=p,
        n_trials=n_trials,
    )
