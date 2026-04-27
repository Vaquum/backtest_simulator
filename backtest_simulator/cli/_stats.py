"""Sweep-level statistical-honesty stats for `bts sweep` summary."""

# Slice #17 Task 17 (DSR + SPA + PBO portion). Wires the standalone
# primitives (`backtest_simulator.honesty.deflated_sharpe`,
# `.spa`, `.pbo`) into the bts sweep summary so the operator sees
# multiple-testing / overfitting / superiority-vs-benchmark signals
# alongside the existing per-run mechanics. CPCV is deferred until
# Task 16 (`SignalsTable.lookup` path-aware wiring) lands; without
# Task 16 there is no path-aware data source to feed CPCV's
# combinatorial splits — printing a CPCV line off the same per-run
# returns would be decorative, not meaningful.
#
# Signal flow:
#   1. `bts sweep` calls `run_window_in_subprocess` per (decoder, day)
#      → returns trades.
#   2. `daily_return_for_run(trades, declared_stops)` summarises each
#      run as a single-day return scalar.
#   3. After all runs, returns are accumulated per-decoder.
#   4. `compute_sweep_stats(per_decoder_returns, benchmark_returns)`
#      runs DSR (best decoder, deflated for n_trials) + SPA (all
#      decoders vs buy-hold) + PBO (combinatorial-symmetric
#      cross-validation across decoders).
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from datetime import time as dtime
from decimal import Decimal

import polars as pl
from scipy.stats import kurtosis, skew

from backtest_simulator.cli._metrics import (
    STARTING_CAPITAL,
    Trade,
    pair_metrics,
    pair_trades,
)
from backtest_simulator.honesty.deflated_sharpe import (
    DeflatedSharpeResult,
    deflated_sharpe,
)
from backtest_simulator.honesty.pbo import (
    PboResult,
    probability_of_backtest_overfitting,
)
from backtest_simulator.honesty.spa import SpaResult, spa_test


@dataclass(frozen=True)
class SweepStats:
    """Aggregated DSR + PBO + SPA results for a finished sweep.

    Any of the three may be `None` when the input is too small
    for the underlying primitive to run honestly (e.g. fewer than
    2 decoders for PBO/SPA, fewer than 2 days for DSR). Each
    result carries the primitive's own dataclass; the `bts sweep`
    summary printer reads them directly.
    """

    dsr: DeflatedSharpeResult | None
    spa: SpaResult | None
    pbo: PboResult | None
    best_decoder: str | None
    best_sharpe: float | None
    n_decoders: int
    n_observations: int


def daily_return_for_run(
    trades: list[Trade], declared_stops: dict[str, Decimal],
) -> float | None:
    """One run's net PnL expressed as a fraction of starting capital.

    Sums net PnL across all closed BUY→SELL pairs and divides by
    `STARTING_CAPITAL` so cross-decoder comparisons stay
    dimensionless. **Returns `None` when the run has a trailing
    un-closed BUY** — the open position's PnL is unrealised at
    window close; treating it as zero would silently hide losers
    and inflate apparent Sharpe (codex round 1 P1). Caller filters
    `None` out of the per-decoder return series and surfaces a
    `n_runs_with_trailing_inventory` counter so the operator
    sees how many runs were excluded from the stats.
    """
    pairs, trailing = pair_trades(trades)
    if trailing:
        return None
    net = Decimal('0')
    for pair in pairs:
        declared = declared_stops.get(pair[0].client_order_id)
        pair_net, _, _ = pair_metrics(pair, declared)
        net += pair_net
    return float(net / STARTING_CAPITAL)


def fetch_buy_hold_benchmark(
    days: list[datetime], hours_start: dtime, hours_end: dtime,
    *, seed_price_at: Callable[[datetime], Decimal],
) -> list[float]:
    """Per-day buy-hold returns over the same trading-hours window.

    For each day, fetch the first trade at-or-after `hours_start`
    (open) and the first trade at-or-after `hours_end` (close); the
    return is `(close - open) / open`. Used as the SPA benchmark.
    `seed_price_at` is injected to keep this module IO-free for
    unit testing.
    """
    out: list[float] = []
    for day in days:
        ws = datetime.combine(day.date(), hours_start, tzinfo=UTC)
        we = datetime.combine(day.date(), hours_end, tzinfo=UTC)
        open_p = seed_price_at(ws)
        close_p = seed_price_at(we)
        out.append(float((close_p - open_p) / open_p))
    return out


def compute_sweep_stats(
    per_decoder_returns: dict[str, list[float]],
    benchmark_returns: list[float],
    *, n_search_trials: int | None = None,
    seed: int = 42, spa_n_bootstrap: int = 1000,
) -> SweepStats:
    """Run DSR + SPA + PBO across the sweep's per-decoder daily returns.

    `n_search_trials` is the multiple-testing inflation factor for
    DSR — the size of the candidate search space the selected
    decoders were picked from (typically `--n-permutations` from
    the sweep args). Defaults to `len(per_decoder_returns)` (the
    visible-pick count, a lower bound that under-deflates the
    selected winner — codex round 1 P1 caught this); operators
    must pass `n_permutations` explicitly to capture the true
    search space.
    """
    n_decoders = len(per_decoder_returns)
    if n_decoders == 0:
        return SweepStats(None, None, None, None, None, 0, 0)
    n_obs = min((len(rs) for rs in per_decoder_returns.values()), default=0)
    if n_obs < 2:
        return SweepStats(None, None, None, None, None, n_decoders, n_obs)
    n_trials = n_search_trials if n_search_trials is not None else n_decoders
    sharpes = {d: _sharpe(rs) for d, rs in per_decoder_returns.items()}
    best_decoder, best_sharpe = max(sharpes.items(), key=lambda kv: kv[1])
    dsr = _safe_dsr(
        per_decoder_returns[best_decoder], best_sharpe, n_trials, n_obs,
    )
    spa = _safe_spa(
        per_decoder_returns, benchmark_returns, seed, spa_n_bootstrap,
    )
    pbo = _safe_pbo(per_decoder_returns, n_obs)
    return SweepStats(
        dsr=dsr, spa=spa, pbo=pbo,
        best_decoder=best_decoder, best_sharpe=best_sharpe,
        n_decoders=n_decoders, n_observations=n_obs,
    )


def _sharpe(returns: list[float]) -> float:
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    if var <= 0.0:
        return 0.0
    return mean / math.sqrt(var)


def _safe_dsr(
    returns: list[float], sharpe: float, n_trials: int, n_obs: int,
) -> DeflatedSharpeResult | None:
    """Wrap `deflated_sharpe` with constant-returns guard.

    `scipy.stats.kurtosis` returns NaN on constant input
    (zero variance, kurtosis is 0/0 indeterminate). NaN
    propagates through the DSR formula and surfaces as
    `deflated=nan p_value=nan` in the sweep summary, which
    misleads more than it informs. Skip cleanly when skew /
    kurtosis cannot be computed honestly.
    """
    if n_obs < 2 or n_trials < 1:
        return None
    skew_v = float(skew(returns))
    kurt_v = float(kurtosis(returns))
    if (
        math.isnan(skew_v) or math.isnan(kurt_v)
        or math.isinf(skew_v) or math.isinf(kurt_v)
    ):
        return None
    return deflated_sharpe(
        sharpe=sharpe, n_trials=n_trials,
        skew=skew_v, kurtosis=kurt_v, n_observations=n_obs,
    )


def _safe_spa(
    per_decoder_returns: dict[str, list[float]],
    benchmark_returns: list[float],
    seed: int, n_bootstrap: int,
) -> SpaResult | None:
    if not per_decoder_returns or not benchmark_returns:
        return None
    n = len(benchmark_returns)
    if any(len(rs) != n for rs in per_decoder_returns.values()):
        return None
    candidate_df = pl.DataFrame(per_decoder_returns)
    benchmark_series = pl.Series('benchmark', benchmark_returns)
    return spa_test(
        candidate_returns=candidate_df,
        benchmark_returns=benchmark_series,
        block_size=max(1, min(5, n)),
        n_bootstrap=n_bootstrap, seed=seed,
    )


def _safe_pbo(
    per_decoder_returns: dict[str, list[float]], n_obs: int,
) -> PboResult | None:
    """PBO across decoders with first-half/second-half IS/OOS split.

    Skipped when fewer than 2 decoders, fewer than 4 observations
    (n_groups=2 needs 2 IS + 2 OOS), or when all decoder return
    series are pairwise identical (codex round 1 P1: ties yield
    deterministic primitive ordering, so all-zero sweeps report
    `pbo=0.0` — falsely "no overfitting signal" — instead of
    skipping). The minimum non-trivial sweep is 2 decoders by 4
    days with at least one decoder differing.
    """
    n_decoders = len(per_decoder_returns)
    if n_decoders < 2:
        return None
    if n_obs < 4:
        return None
    series = list(per_decoder_returns.values())
    if all(s == series[0] for s in series[1:]):
        # Degenerate: all candidates are the same return series →
        # combinatorial-symmetric ranking is deterministic and
        # uninformative.
        return None
    half = n_obs // 2
    n_groups = min(half, 4)
    if n_groups % 2 == 1:
        n_groups -= 1
    if n_groups < 2:
        return None
    is_data = pl.DataFrame({
        d: rs[:half] for d, rs in per_decoder_returns.items()
    })
    oos_data = pl.DataFrame({
        d: rs[half:half * 2] for d, rs in per_decoder_returns.items()
    })
    return probability_of_backtest_overfitting(
        in_sample_pnl=is_data, out_of_sample_pnl=oos_data, n_groups=n_groups,
    )
