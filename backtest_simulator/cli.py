"""bts CLI — sweep + analyze subcommands."""
from __future__ import annotations

import argparse
from pathlib import Path

from backtest_simulator.reporting.enriched_results import build_enriched_table


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='bts', description='backtest_simulator CLI')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p_sweep = sub.add_parser('sweep', help='Run a backtest sweep against a Limen experiment.')
    p_sweep.add_argument('--experiment', required=True, type=Path)
    p_sweep.add_argument('--strategy', default='long_on_signal')
    p_sweep.add_argument('--window', default='fixture')
    p_sweep.add_argument('--workers', type=int, default=1)
    p_sweep.add_argument('--seeds', default='42')
    p_sweep.add_argument('--n-paths', type=int, default=1)
    p_sweep.add_argument('--risk-rule', default='stop_on_entry', choices=['stop_on_entry'])
    p_sweep.set_defaults(func=_cmd_sweep)

    p_analyze = sub.add_parser('analyze', help='Emit results_with_backtest.csv from sweep output.')
    p_analyze.add_argument('--experiment', required=True, type=Path)
    p_analyze.add_argument('--backtest-parquet', type=Path, default=None)
    p_analyze.add_argument('--out', type=Path, default=None)
    p_analyze.set_defaults(func=_cmd_analyze)

    args = ap.parse_args(argv)
    return int(args.func(args))


def _cmd_sweep(args: argparse.Namespace) -> int:
    exp_dir = Path(args.experiment)
    if not exp_dir.is_dir():
        return 2
    # M1 scope-box: sweep execution lives in the runnable notebook so the
    # user can see every step. CLI-driven headless sweep is a follow-up.
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    exp_dir = Path(args.experiment)
    backtest_parquet = Path(args.backtest_parquet) if args.backtest_parquet else exp_dir / 'backtest_results.parquet'
    out_csv = Path(args.out) if args.out else exp_dir / 'results_with_backtest.csv'
    build_enriched_table(exp_dir, backtest_parquet, out_csv=out_csv)
    return 0
