"""bts CLI — enrich + analyze subcommands."""
from __future__ import annotations

# `bts enrich` joins Limen UEL `results.csv` with an optional
# `backtest_results.parquet` into a single enriched CSV at
# `<experiment_dir>/results_with_backtest.csv`. If no backtest
# parquet exists yet, the output carries the Limen columns plus
# NaN placeholders for the backtest columns.
#
# This command does NOT run a backtest sweep. It only enriches
# existing results. `bts sweep` and `bts analyze` are retained
# as aliases for backwards compatibility.
import argparse
import sys
from pathlib import Path

from backtest_simulator.reporting.enriched_results import build_enriched_table


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='bts', description='backtest_simulator CLI')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p_enrich = sub.add_parser(
        'enrich',
        help='Join <experiment_dir>/results.csv with optional '
             'backtest_results.parquet into results_with_backtest.csv.',
    )
    p_enrich.add_argument('--experiment', required=True, type=Path)
    p_enrich.add_argument('--backtest-parquet', type=Path, default=None)
    p_enrich.add_argument('--out', type=Path, default=None)
    p_enrich.set_defaults(func=_cmd_enrich)

    p_sweep = sub.add_parser(
        'sweep',
        help='Alias of `enrich` (does NOT run a backtest sweep).',
    )
    p_sweep.add_argument('--experiment', required=True, type=Path)
    p_sweep.add_argument('--backtest-parquet', type=Path, default=None)
    p_sweep.add_argument('--out', type=Path, default=None)
    p_sweep.set_defaults(func=_cmd_enrich)

    p_analyze = sub.add_parser(
        'analyze',
        help='Alias of `enrich`.',
    )
    p_analyze.add_argument('--experiment', required=True, type=Path)
    p_analyze.add_argument('--backtest-parquet', type=Path, default=None)
    p_analyze.add_argument('--out', type=Path, default=None)
    p_analyze.set_defaults(func=_cmd_enrich)

    args = ap.parse_args(argv)
    return int(args.func(args))


def _cmd_enrich(args: argparse.Namespace) -> int:
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


if __name__ == '__main__':  # pragma: no cover - manual invocation path
    sys.exit(main())
