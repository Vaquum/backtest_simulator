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
from backtest_simulator.honesty.cpcv import CpcvPaths
from backtest_simulator.honesty.deflated_sharpe import (
    DeflatedSharpeResult,
    deflated_sharpe,
)
from backtest_simulator.honesty.spa import SpaResult, spa_test


@dataclass(frozen=True)
class SweepStats:
    """Aggregated DSR + SPA results for a finished sweep.

    Either may be `None` when the input is too small for the
    underlying primitive to run honestly (e.g. fewer than 2
    decoders for SPA, fewer than 2 days for DSR). Each result
    carries the primitive's own dataclass; the `bts sweep`
    summary printer reads them directly.

    Auditor: legacy half-split PBO (`_safe_pbo`) was removed —
    `sweep cpcv_pbo` is the honest replacement (driven directly
    by `cpcv_pbo` in this file). Carrying a `pbo` field that
    nothing prints would maintain a parallel BTS-only statistic
    against the bts-only-no-ornamentation rule.
    """

    dsr: DeflatedSharpeResult | None
    spa: SpaResult | None
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
        return SweepStats(None, None, None, None, 0, 0)
    n_obs = min((len(rs) for rs in per_decoder_returns.values()), default=0)
    if n_obs < 2:
        return SweepStats(None, None, None, None, n_decoders, n_obs)
    n_trials = n_search_trials if n_search_trials is not None else n_decoders
    sharpes = {d: _sharpe(rs) for d, rs in per_decoder_returns.items()}
    best_decoder, best_sharpe = max(sharpes.items(), key=lambda kv: kv[1])
    dsr = _safe_dsr(
        per_decoder_returns[best_decoder], best_sharpe, n_trials, n_obs,
    )
    spa = _safe_spa(
        per_decoder_returns, benchmark_returns, seed, spa_n_bootstrap,
    )
    # Auditor: legacy half-split PBO removed — `sweep cpcv_pbo`
    # is the honest replacement. Computing `_safe_pbo` here when
    # nothing reads its result was BTS-only ornamentation.
    return SweepStats(
        dsr=dsr, spa=spa,
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


@dataclass(frozen=True)
class CpcvPboResult:
    """Day-level CSCV PBO via `CpcvPaths` + deployed-strategy returns.

    Auditor (post-v2.0.1): the previous bar-level implementation
    ranked decoders on `pred * close_to_next_close_return` —
    a SIGNAL-RETURN PROXY that ignored stops, maker-fill, pending-
    order state, realized slippage / impact, and trailing-
    inventory exclusions. Rewired to consume `per_decoder_returns`
    directly: the same `daily_return_for_run` output `bts sweep`
    feeds into DSR/SPA, which IS net PnL / starting capital from
    the actual closed BUY→SELL pairs of the deployed
    long_on_signal strategy. Now the PBO surface ranks the
    DEPLOYED path, not a proxy.

    `pbo` is the overfitting proportion. `n_paths` counts paths
    that produced a logit (post per-path tie skip).
    `n_paths_skipped` counts paths where IS or OOS data was too
    thin or top-2 IS Sharpes tied / OOS rank was ambiguous.
    """

    pbo: float
    n_paths: int
    n_decoders: int
    purge_seconds: int
    embargo_seconds: int
    n_paths_skipped: int


def cpcv_pbo(
    *,
    paths: CpcvPaths,
    per_decoder_returns: dict[str, list[float]],
    days: list[datetime],
) -> CpcvPboResult | None:
    """Day-level CSCV PBO using deployed-strategy daily returns.

    Args:
        paths: CpcvPaths.build output. Group indices map to
            contiguous DAY blocks (not bars) — e.g. with
            n_groups=4 and 12 days, group 0 covers days 0-2,
            group 1 covers 3-5, etc. Each path's train_groups +
            test_groups define which days are IS vs OOS.
        per_decoder_returns: deployed-strategy daily returns from
            `daily_return_for_run` — net PnL / starting_capital,
            already excluding runs with trailing inventory.
            Length is the count of "clean" days (those where
            EVERY decoder produced a closed-pair return).
        days: the clean-day timestamps in time order, parallel
            to each `per_decoder_returns[d]` list.

    PBO = fraction of paths where the IS-best decoder underperforms
    the median OOS Sharpe (López de Prado §11 logit aggregation).

    Returns None when:
      - <2 decoders or no paths,
      - fewer days than n_groups (can't partition contiguously),
      - all per-decoder series byte-equal (degenerate),
      - every path's IS or OOS subset is < 2 days,
      - per-path top-2 IS Sharpes tie (no unique best-IS) for
        every path.
    """
    n_decoders = len(per_decoder_returns)
    if n_decoders < 2 or len(paths) == 0 or not days:
        return None
    decoder_ids = list(per_decoder_returns.keys())
    n_days = len(days)
    n_groups = max(
        max(p.train_groups + p.test_groups) for p in paths
    ) + 1
    if n_groups < 2 or n_days < n_groups:
        return None
    # Pairwise-identical guard: if every decoder produced the
    # SAME deployed-strategy returns across all clean days, max()
    # below would deterministically pick the first key and
    # fabricate a zero-PBO. Skip cleanly.
    sequences = [tuple(per_decoder_returns[d]) for d in decoder_ids]
    if all(s == sequences[0] for s in sequences[1:]):
        return None
    # Map day_index -> group_index. Contiguous blocks: day i goes
    # to floor(i * n_groups / n_days). Equivalent to dividing the
    # n_days into n_groups roughly-equal chunks.
    group_of_day = [
        min(i * n_groups // n_days, n_groups - 1)
        for i in range(n_days)
    ]
    first_path = paths.paths()[0]
    purge_seconds = first_path.purge_seconds
    embargo_seconds = first_path.embargo_seconds

    logits: list[float] = []
    n_paths_skipped = 0
    for path in paths:
        is_idx = [
            i for i in range(n_days)
            if group_of_day[i] in path.train_groups
        ]
        oos_idx = [
            i for i in range(n_days)
            if group_of_day[i] in path.test_groups
        ]
        # Purge: drop train days within `purge_seconds` of any
        # test-group boundary (both sides). Embargo: drop train
        # days that come AFTER each test block within
        # `embargo_seconds` (López de Prado direction).
        if purge_seconds > 0 or embargo_seconds > 0:
            is_idx = _apply_purge_embargo(
                is_idx, oos_idx, days,
                purge_seconds=purge_seconds,
                embargo_seconds=embargo_seconds,
            )
        if len(is_idx) < 2 or len(oos_idx) < 2:
            n_paths_skipped += 1
            continue
        is_sharpes: dict[str, float] = {}
        oos_sharpes: dict[str, float] = {}
        for did in decoder_ids:
            series = per_decoder_returns[did]
            is_sharpes[did] = _sharpe([series[i] for i in is_idx])
            oos_sharpes[did] = _sharpe([series[i] for i in oos_idx])
        # Per-path tie skip: top-2 IS Sharpes equal -> no unique
        # best-IS, max() deterministic ordering would fabricate.
        is_values_sorted = sorted(is_sharpes.values(), reverse=True)
        if is_values_sorted[0] == is_values_sorted[1]:
            n_paths_skipped += 1
            continue
        best_is = max(is_sharpes, key=lambda d: is_sharpes[d])
        best_oos_value = oos_sharpes[best_is]
        n_oos_ties = sum(
            1 for v in oos_sharpes.values() if v == best_oos_value
        )
        if n_oos_ties > 1:
            n_paths_skipped += 1
            continue
        sorted_oos = sorted(
            decoder_ids, key=lambda d: oos_sharpes[d], reverse=True,
        )
        rank = sorted_oos.index(best_is) + 1
        omega = (rank - 1) / (n_decoders - 1)
        omega = max(min(omega, 1 - 1e-9), 1e-9)
        logits.append(math.log(omega / (1 - omega)))

    if not logits:
        return None
    pbo = sum(1 for x in logits if x > 0) / len(logits)
    return CpcvPboResult(
        pbo=pbo, n_paths=len(logits), n_decoders=n_decoders,
        purge_seconds=purge_seconds, embargo_seconds=embargo_seconds,
        n_paths_skipped=n_paths_skipped,
    )


def _apply_purge_embargo(
    is_idx: list[int], oos_idx: list[int],
    days: list[datetime],
    *, purge_seconds: int, embargo_seconds: int,
) -> list[int]:
    """Drop is_idx days too close to oos_idx days (purge + embargo).

    Purge: |t_train - t_test| <= purge_seconds for ANY test day
        -> drop the train day (both directions).
    Embargo: 0 < (t_train - t_test) <= embargo_seconds for ANY
        test day -> drop the train day (test FIRST, train AFTER).

    Operator-side defaults are typically 0 / 0; the implementation
    falls through unchanged when both are zero.
    """
    if purge_seconds == 0 and embargo_seconds == 0:
        return is_idx
    out: list[int] = []
    for i in is_idx:
        t_i = days[i]
        keep = True
        for j in oos_idx:
            t_j = days[j]
            delta = (t_i - t_j).total_seconds()
            if abs(delta) <= purge_seconds:
                keep = False
                break
            if 0 < delta <= embargo_seconds:
                keep = False
                break
        if keep:
            out.append(i)
    return out


