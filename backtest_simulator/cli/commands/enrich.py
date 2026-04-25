"""`bts enrich` — join Limen `results.csv` with optional backtest parquet."""

# This was the entire scope of `backtest_simulator.cli` pre-M2.
# Behaviour is unchanged; only the shape moved into the master CLI's
# subcommand package.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backtest_simulator.cli._verbosity import add_verbosity_arg, configure
from backtest_simulator.reporting.enriched_results import build_enriched_table


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        'enrich',
        help='Join <experiment_dir>/results.csv with optional '
             'backtest_results.parquet into results_with_backtest.csv.',
    )
    p.add_argument('--experiment', required=True, type=Path)
    p.add_argument('--backtest-parquet', type=Path, default=None)
    p.add_argument('--out', type=Path, default=None)
    add_verbosity_arg(p)
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    configure(args.verbose)
    exp_dir = Path(args.experiment).resolve()
    if not exp_dir.is_dir():
        sys.stderr.write(f'bts enrich: experiment dir not found: {exp_dir}\n')
        return 2
    backtest_parquet = (
        Path(args.backtest_parquet).resolve() if args.backtest_parquet is not None
        else exp_dir / 'backtest_results.parquet'
    )
    out_csv = (
        Path(args.out).resolve() if args.out is not None
        else exp_dir / 'results_with_backtest.csv'
    )
    build_enriched_table(exp_dir, backtest_parquet, out_csv=out_csv)
    sys.stdout.write(f'bts enrich: wrote {out_csv}\n')
    return 0
