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

# Columns the backtest run contributes to the enriched table. The
# missing-parquet path (see `build_enriched_table`) fills these with
# nulls so downstream tooling sees a stable schema whether a backtest
# ran or not. Types are pinned so the null columns cast to the same
# dtypes the backtest would produce.
_BACKTEST_COLUMN_DTYPES: Final[tuple[tuple[str, pl.DataType], ...]] = (
    ('n_trades', pl.Int64()),
    ('total_fees', pl.Float64()),
    ('r_mean', pl.Float64()),
    ('r_median', pl.Float64()),
    ('profit_factor', pl.Float64()),
    ('sum_pnl_net', pl.Float64()),
    ('probs_var', pl.Float64()),
    ('preds_long_share', pl.Float64()),
    ('tradable', pl.Boolean()),
    ('train_s', pl.Float64()),
    ('backtest_wall_s', pl.Float64()),
    ('window_start', pl.Utf8()),
    ('window_end', pl.Utf8()),
    ('n_paths', pl.Int64()),
    ('per_path_r_mean', pl.Float64()),
    ('per_path_profit_factor', pl.Float64()),
)


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
    limen_ids: set[object] = (
        set(limen_results['round_id'].to_list())
        if not limen_results.is_empty()
        else set[object]()
    )
    bt_ids: set[object] = (
        set(bt['round_id'].to_list())
        if 'round_id' in bt.columns
        else set[object]()
    )
    _assert_bijection(limen_ids, bt_ids)
    if not bt.is_empty():
        joined = limen_results.join(bt, on='round_id', how='left')
    else:
        # No backtest parquet on disk: produce the stable enriched
        # schema anyway with nulls in every backtest-contributed column.
        # Downstream tooling must be able to read this CSV blind and
        # get the same columns whether a backtest has run yet or not.
        joined = limen_results.with_columns([
            pl.lit(None, dtype=dtype).alias(name)
            for name, dtype in _BACKTEST_COLUMN_DTYPES
        ])
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        joined.write_csv(out_csv)
    return joined


def _read_limen_results(experiment_dir: Path) -> pl.DataFrame:
    results_csv = experiment_dir / 'results.csv'
    round_data_jsonl = experiment_dir / 'round_data.jsonl'
    if results_csv.is_file():
        df = pl.read_csv(results_csv)
        # Limen's own `results.csv` uses `id` as the decoder identifier;
        # older round_data.jsonl used `round_id`. We normalise to
        # `round_id` so the rest of the enrichment path is column-stable.
        if 'round_id' not in df.columns and 'id' in df.columns:
            df = df.rename({'id': 'round_id'})
        return df
    if round_data_jsonl.is_file():
        rows = [json.loads(line) for line in round_data_jsonl.read_text(encoding='utf-8').splitlines() if line.strip()]
        return pl.DataFrame({
            'round_id': [int(r['round_id']) for r in rows],
            'round_params': [json.dumps(r['round_params']) for r in rows],
        })
    return pl.DataFrame(schema={'round_id': pl.Int64, 'round_params': pl.Utf8})


def _assert_bijection(limen_ids: set[object], backtest_ids: set[object]) -> None:
    missing = limen_ids - backtest_ids
    extra = backtest_ids - limen_ids
    problems: list[str] = []
    # `set[object]` is not directly sort()-comparable; sort by repr so
    # the diagnostic order is stable regardless of the underlying cell
    # type (polars values can be int, str, or extension types).
    if missing and backtest_ids:
        problems.append(
            f'decoder(s) present in Limen results.csv but missing from backtest parquet: '
            f'{sorted(missing, key=repr)}',
        )
    if extra:
        problems.append(
            f'decoder(s) present in backtest parquet but not declared in Limen results.csv: '
            f'{sorted(extra, key=repr)}',
        )
    if problems:
        raise ValueError('; '.join(problems))
