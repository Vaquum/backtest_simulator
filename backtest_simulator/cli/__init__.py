"""bts master CLI — `bts sweep` is the only subcommand."""
from __future__ import annotations

import argparse
import os
from typing import Final

_THREAD_ENV_VARS: Final[tuple[str, ...]] = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'POLARS_MAX_THREADS', 'TOKENIZERS_PARALLELISM')
for _k in _THREAD_ENV_VARS:
    os.environ.setdefault(_k, '1')

SUBCOMMANDS: Final[tuple[str, ...]] = ('sweep',)

def _build_parser() -> argparse.ArgumentParser:
    from backtest_simulator.cli.commands import sweep as _sweep
    ap = argparse.ArgumentParser(prog='bts', description='backtest_simulator master CLI')
    sub = ap.add_subparsers(dest='cmd', required=True)

    def add_parser(name: str, help_: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help_)
    _sweep.register(add_parser)
    return ap

def main(argv: list[str] | None=None) -> int:
    ap = _build_parser()
    args = ap.parse_args(argv)
    return int(args.func(args))
