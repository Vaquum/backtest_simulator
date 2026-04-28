"""`bts version` — print the package version."""
from __future__ import annotations

import argparse
import importlib.metadata
import sys
from collections.abc import Callable


def register(add_parser: Callable[[str, str], argparse.ArgumentParser]) -> None:
    p = add_parser('version', 'Print the bts version.')
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    del args
    version = importlib.metadata.version('backtest_simulator')
    sys.stdout.write(f'bts {version}\n')
    return 0
