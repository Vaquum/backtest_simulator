"""Sweep-level statistical-honesty stats for `bts sweep` summary."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from datetime import time as dtime
from decimal import Decimal
from pathlib import Path

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

    dsr: DeflatedSharpeResult | None
    spa: SpaResult | None
    best_decoder: str | None
    best_sharpe: float | None
    n_decoders: int
    n_observations: int

def daily_return_for_run(
    trades: list[Trade], declared_stops: dict[str, Decimal],
) -> float | None:
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
    out: list[float] = []
    for day in days:
        ws = datetime.combine(day.date(), hours_start, tzinfo=UTC)
        we = datetime.combine(day.date(), hours_end, tzinfo=UTC)
        open_p = seed_price_at(ws)
        close_p = seed_price_at(we)
        out.append(float((close_p - open_p) / open_p))
    return out

def make_seed_price_from_parquet(
    tape_path: Path,
) -> Callable[[datetime], Decimal]:
    import polars as _pl
    frame = _pl.read_parquet(str(tape_path)).set_sorted('time')
    def _seed(ts: datetime) -> Decimal:
        rows = frame.filter(_pl.col('time') >= ts).head(1)
        if rows.is_empty():
            msg = (
                f'--trades-tape {tape_path} has no tick at or after '
                f'{ts.isoformat()}; tape must cover every replay day '
                f'open + close.'
            )
            raise RuntimeError(msg)
        return Decimal(str(rows['price'][0]))
    return _seed

def announce_operator_trades_tape(tape_path: Path, t_start: float) -> str:
    import time as _time
    if not tape_path.is_file():
        msg = f'bts sweep: --trades-tape file not found: {tape_path}'
        raise FileNotFoundError(msg)
    elapsed = _time.perf_counter() - t_start
    print(
        f'[{elapsed:7.2f}s] trades tape ready: {tape_path.name} '
        f'(operator-supplied)',
        flush=True,
    )
    return str(tape_path)

def compute_sweep_stats(
    per_decoder_returns: dict[str, list[float]],
    benchmark_returns: list[float],
    *, n_search_trials: int | None = None,
    seed: int = 42, spa_n_bootstrap: int = 1000,
) -> SweepStats:
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
    sequences = [tuple(per_decoder_returns[d]) for d in decoder_ids]
    if all(s == sequences[0] for s in sequences[1:]):
        return None
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

