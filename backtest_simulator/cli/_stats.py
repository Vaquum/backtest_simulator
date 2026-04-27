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
from typing import TYPE_CHECKING

import polars as pl
from scipy.stats import kurtosis, skew

if TYPE_CHECKING:
    from backtest_simulator.sensors.precompute import SignalsTable

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


@dataclass(frozen=True)
class CpcvPboResult:
    """Bar-level CSCV PBO via `CpcvPaths` + `SignalsTable.lookup`.

    Slice #17 Task 17 CPCV (auditor round-7 fix). Per path, per
    decoder, walk the runtime tick_timestamps and call
    `signals.lookup(t, allowed_groups=path.test_groups, ...)` to
    get OOS preds + same with `allowed_groups=path.train_groups`
    for IS preds. Multiply pred * bar_return (close-to-next-close)
    to get strategy returns; compute Sharpe per partition. PBO =
    fraction of paths where the IS-best decoder underperformed
    median OOS (López de Prado §11 logit aggregation).

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
    signals_per_decoder: dict[str, SignalsTable],
    tick_timestamps: list[datetime],
    klines: pl.DataFrame,
) -> CpcvPboResult | None:
    """Bar-level CSCV PBO using SignalsTable.lookup for path filtering.

    Args:
        paths: CpcvPaths.build output. Each path's test_groups +
            train_groups are passed as `allowed_groups` to
            `SignalsTable.lookup` to filter the per-bar predictions
            into IS / OOS subsets at lookup time.
        signals_per_decoder: per-decoder SignalsTables built from
            per-tick runtime replay (Nexus's exact recipe). The
            lookup IS the path filter — auditor round-7 fix
            making both SignalsTable and CpcvPaths load-bearing.
        tick_timestamps: the runtime tick instants the strategy
            sees a signal at (epoch-aligned interval boundaries
            within the operator's replay window). Same list the
            sweep used for SignalsTable build.
        klines: pre-fetched klines (BacktestMarketDataPoller's
            source) used to compute per-bar close-to-next-close
            returns. The strategy return at tick `t` is
            `pred(t) * bar_return(t)` — long-flat semantics
            mirroring `long_on_signal.py`.

    Returns None when:
      - <2 decoders or no paths,
      - all decoder predictions are byte-equal (degenerate),
      - every path's IS or OOS subset is < 2 bars,
      - per-path top-2 IS Sharpes tie (no unique best-IS) for
        every path.
    """
    n_decoders = len(signals_per_decoder)
    if n_decoders < 2 or len(paths) == 0 or not tick_timestamps:
        return None
    decoder_ids = list(signals_per_decoder.keys())
    # Derive n_groups from path partitions (CpcvPaths.build already
    # validates 2 <= n_test_groups < n_groups).
    n_groups = max(
        max(p.train_groups + p.test_groups) for p in paths
    ) + 1
    if n_groups < 2:
        return None
    # Per-bar returns from klines: (close[next] - close[this]) /
    # close[this]. Build a tick → bar_return dict for fast lookup.
    bar_returns = _bar_returns_at_ticks(klines, tick_timestamps)
    if not bar_returns:
        return None
    # Pairwise-identical guard. Look at the predictions across all
    # ticks for each decoder; if all decoders produce identical
    # pred sequences, the rank logic below would fabricate a
    # zero-PBO via deterministic max().
    pred_sequences: dict[str, tuple[int, ...]] = {}
    for did, signals in signals_per_decoder.items():
        seq: list[int] = []
        for t in tick_timestamps:
            row = signals.lookup(t)
            seq.append(int(row.pred) if row is not None else 0)
        pred_sequences[did] = tuple(seq)
    sequences = list(pred_sequences.values())
    if all(s == sequences[0] for s in sequences[1:]):
        return None
    first_path = paths.paths()[0]
    purge_seconds = first_path.purge_seconds
    embargo_seconds = first_path.embargo_seconds

    logits: list[float] = []
    n_paths_skipped = 0
    for path in paths:
        is_sharpes: dict[str, float] = {}
        oos_sharpes: dict[str, float] = {}
        too_thin = False
        for did, signals in signals_per_decoder.items():
            is_returns: list[float] = []
            oos_returns: list[float] = []
            for t in tick_timestamps:
                bret = bar_returns.get(t)
                if bret is None:
                    continue
                is_row = signals.lookup(
                    t,
                    allowed_groups=path.train_groups,
                    n_groups=n_groups,
                    purge_seconds=purge_seconds,
                    embargo_seconds=embargo_seconds,
                )
                if is_row is not None:
                    is_returns.append(int(is_row.pred) * bret)
                oos_row = signals.lookup(
                    t,
                    allowed_groups=path.test_groups,
                    n_groups=n_groups,
                    purge_seconds=purge_seconds,
                    embargo_seconds=embargo_seconds,
                )
                if oos_row is not None:
                    oos_returns.append(int(oos_row.pred) * bret)
            if len(is_returns) < 2 or len(oos_returns) < 2:
                too_thin = True
                break
            is_sharpes[did] = _sharpe(is_returns)
            oos_sharpes[did] = _sharpe(oos_returns)
        if too_thin:
            n_paths_skipped += 1
            continue
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


def _bar_returns_at_ticks(
    klines: pl.DataFrame, tick_timestamps: list[datetime],
) -> dict[datetime, float]:
    """Per-tick close-to-next-close return, keyed by tick timestamp.

    For each tick t, finds the latest kline at or before t and the
    next kline. `bar_return(t) = (next.close - tick.close) /
    tick.close`. Ticks for which either side is missing (last
    kline, or tick falls outside klines coverage) are omitted —
    the caller treats them as "no bar data here".
    """
    if klines.is_empty() or 'datetime' not in klines.columns:
        return {}
    sorted_klines = klines.sort('datetime')
    timestamps = sorted_klines['datetime'].to_list()
    closes = sorted_klines['close'].to_list()
    out: dict[datetime, float] = {}
    for t in tick_timestamps:
        # Find the LATEST kline with datetime <= t
        idx_at_or_before = -1
        for i, kt in enumerate(timestamps):
            if kt <= t:
                idx_at_or_before = i
            else:
                break
        if idx_at_or_before < 0 or idx_at_or_before >= len(timestamps) - 1:
            continue
        this_close = float(closes[idx_at_or_before])
        next_close = float(closes[idx_at_or_before + 1])
        if this_close <= 0:
            continue
        out[t] = (next_close - this_close) / this_close
    return out


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
