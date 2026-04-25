"""`bts version` — print the package version."""
from __future__ import annotations

import argparse
import importlib.metadata
import sys


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser('version', help='Print the bts version.')
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    del args
    version = importlib.metadata.version('backtest_simulator')
    sys.stdout.write(f'bts {version}\n')
    return 0
