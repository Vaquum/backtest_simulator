"""`bts lint` — ruff wrapper."""

# Runs `ruff check` on the configured paths. Verbosity at level 1+ forwards
# ruff's `--verbose` so the operator sees per-file linting decisions.
from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final

from backtest_simulator.cli._verbosity import add_verbosity_arg, configure

# Same path set CI runs (.github/workflows/pr_checks_lint.yml).
_DEFAULT_PATHS: Final[tuple[str, ...]] = (
    'backtest_simulator', 'tools', 'tests',
)


def register(add_parser: Callable[[str, str], argparse.ArgumentParser]) -> None:
    p = add_parser('lint', 'Run ruff check.')
    p.add_argument('--paths', nargs='*', type=Path, default=None,
                   help='Override paths (default: backtest_simulator tools tests).')
    p.add_argument('--fix', action='store_true', help='Apply ruff --fix.')
    add_verbosity_arg(p)
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    configure(args.verbose)
    repo_root = Path(__file__).resolve().parents[3]
    paths = args.paths or [repo_root / p for p in _DEFAULT_PATHS]
    cmd: list[str] = [sys.executable, '-m', 'ruff', 'check']
    if args.fix:
        cmd.append('--fix')
    if args.verbose:
        cmd.append('--verbose')
    cmd.extend(str(p) for p in paths)
    return subprocess.run(cmd, check=False).returncode
