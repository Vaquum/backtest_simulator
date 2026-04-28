"""`bts typecheck` — pyright strict wrapper."""

# Runs pyright in strict mode against the configured paths (defaults to
# the package root, matching `pyproject.toml`'s `[tool.pyright].include`).
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Final

from backtest_simulator.cli._verbosity import add_verbosity_arg, configure

_DEFAULT_PATHS: Final[tuple[str, ...]] = ('backtest_simulator',)


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser('typecheck', help='Run pyright strict.')
    p.add_argument('--paths', nargs='*', type=Path, default=None,
                   help='Override paths (default: backtest_simulator).')
    add_verbosity_arg(p)
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    configure(args.verbose)
    repo_root = Path(__file__).resolve().parents[3]
    paths = args.paths or [repo_root / p for p in _DEFAULT_PATHS]
    cmd: list[str] = [sys.executable, '-m', 'pyright']
    cmd.extend(str(p) for p in paths)
    if args.verbose:
        cmd.append('--verbose')
    return subprocess.run(cmd, check=False).returncode
