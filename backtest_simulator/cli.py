"""bts CLI — sweep + analyze subcommands."""
from __future__ import annotations

# `bts sweep` is the Part 2 entry point that produces
# `<experiment_dir>/results_with_backtest.csv` enriched with backtest
# columns. It reads the Limen UEL `results.csv` (and optionally an
# already-produced `backtest_results.parquet`) and joins them into a
# single enriched CSV. If no backtest parquet exists yet, the output
# carries the Limen columns plus NaN placeholders for the backtest
# columns — an operator-visible HONEST null rather than a silently
# missing file.
#
# `bts analyze` is the same operation retained under its historical
# name, so existing automation that calls `bts analyze` keeps working.
import argparse
import sys
from pathlib import Path

from backtest_simulator.reporting.enriched_results import build_enriched_table


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='bts', description='backtest_simulator CLI')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p_sweep = sub.add_parser(
        'sweep',
        help='Produce <experiment_dir>/results_with_backtest.csv from UEL results + '
             'optional backtest_results.parquet.',
    )
    p_sweep.add_argument('--experiment', required=True, type=Path)
    p_sweep.add_argument('--backtest-parquet', type=Path, default=None)
    p_sweep.add_argument('--out', type=Path, default=None)
    p_sweep.set_defaults(func=_cmd_sweep)

    p_analyze = sub.add_parser(
        'analyze',
        help='Alias of `sweep`: emit results_with_backtest.csv from sweep output.',
    )
    p_analyze.add_argument('--experiment', required=True, type=Path)
    p_analyze.add_argument('--backtest-parquet', type=Path, default=None)
    p_analyze.add_argument('--out', type=Path, default=None)
    p_analyze.set_defaults(func=_cmd_sweep)

    args = ap.parse_args(argv)
    return int(args.func(args))


def _cmd_sweep(args: argparse.Namespace) -> int:
    exp_dir = Path(args.experiment).resolve()
    if not exp_dir.is_dir():
        sys.stderr.write(f'bts sweep: experiment dir not found: {exp_dir}\n')
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
    sys.stdout.write(f'bts sweep: wrote {out_csv}\n')
    return 0


if __name__ == '__main__':  # pragma: no cover - manual invocation path
    sys.exit(main())
