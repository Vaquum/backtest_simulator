"""bts master CLI — `bts sweep` is the only subcommand."""
from __future__ import annotations

import argparse
import sys
from typing import Final

from backtest_simulator.cli.commands import (
    sweep as _sweep,
)

SUBCOMMANDS: Final[tuple[str, ...]] = ('sweep',)

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog='bts', description='backtest_simulator master CLI')
    sub = ap.add_subparsers(dest='cmd', required=True)

    def add_parser(name: str, help_: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help_)

    _sweep.register(add_parser)
    return ap

def main(argv: list[str] | None = None) -> int:
    ap = _build_parser()
    args = ap.parse_args(argv)
    return int(args.func(args))

if __name__ == '__main__':
    sys.exit(main())
