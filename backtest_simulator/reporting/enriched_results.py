"""Enrich Limen results.csv with backtest columns keyed by round_id."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import polars as pl

ENRICHED_COLUMNS: Final[frozenset[str]] = frozenset({
    'round_id',
    'round_params',
    'n_trades',
    'total_fees',
    'r_mean',
    'r_median',
    'profit_factor',
    'sum_pnl_net',
    'probs_var',
    'preds_long_share',
    'tradable',
    'train_s',
    'backtest_wall_s',
    'window_start',
    'window_end',
    'n_paths',
    'per_path_r_mean',
    'per_path_profit_factor',
})


def build_enriched_table(
    experiment_dir: Path,
    backtest_results_parquet: Path,
    *,
    out_csv: Path | None = None,
) -> pl.DataFrame:
    """Join Limen's `results.csv` with the sweep's `backtest_results.parquet`.

    Keyed on `round_id`. Every decoder in Limen's `round_data.jsonl` MUST
    have exactly one row in the enriched output; missing rows or extra
    rows raise `ValueError`. Column order: Limen-original columns first,
    then the appended backtest columns in the order declared in
    `ENRICHED_COLUMNS`.
    """
    limen_results = _read_limen_results(experiment_dir)
    bt = pl.read_parquet(backtest_results_parquet) if backtest_results_parquet.is_file() else pl.DataFrame()
    limen_ids = set(limen_results['round_id'].to_list()) if not limen_results.is_empty() else set()
    bt_ids = set(bt['round_id'].to_list()) if 'round_id' in bt.columns else set()
    _assert_bijection(limen_ids, bt_ids)
    joined = limen_results.join(bt, on='round_id', how='left') if not bt.is_empty() else limen_results
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        joined.write_csv(out_csv)
    return joined


def _read_limen_results(experiment_dir: Path) -> pl.DataFrame:
    results_csv = experiment_dir / 'results.csv'
    round_data_jsonl = experiment_dir / 'round_data.jsonl'
    if results_csv.is_file():
        return pl.read_csv(results_csv)
    if round_data_jsonl.is_file():
        rows = [json.loads(line) for line in round_data_jsonl.read_text(encoding='utf-8').splitlines() if line.strip()]
        return pl.DataFrame({
            'round_id': [int(r['round_id']) for r in rows],
            'round_params': [json.dumps(r['round_params']) for r in rows],
        })
    return pl.DataFrame(schema={'round_id': pl.Int64, 'round_params': pl.Utf8})


def _assert_bijection(limen_ids: set[int], backtest_ids: set[int]) -> None:
    missing = limen_ids - backtest_ids
    extra = backtest_ids - limen_ids
    problems: list[str] = []
    if missing and backtest_ids:
        problems.append(f'decoder(s) present in Limen results.csv but missing from backtest parquet: {sorted(missing)}')
    if extra:
        problems.append(f'decoder(s) present in backtest parquet but not declared in Limen results.csv: {sorted(extra)}')
    if problems:
        raise ValueError('; '.join(problems))
