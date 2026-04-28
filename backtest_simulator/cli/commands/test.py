"""`bts test` — pytest wrapper, the operator-mandated test entry point."""

# The slice's operational protocol forbids invoking `pytest` directly;
# all test runs go through this subcommand so verbosity, marker filters,
# and the integration / honesty / cli scope are unified under one tool.
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from backtest_simulator.cli._verbosity import add_verbosity_arg, configure


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser('test', help='Run pytest under bts.')
    p.add_argument('-k', dest='pattern', type=str, default=None,
                   help='pytest -k pattern.')
    p.add_argument('--honesty', action='store_true',
                   help='Run only tests/honesty/.')
    p.add_argument('--integration', action='store_true',
                   help='Run only tests/integration/.')
    p.add_argument('--cli', action='store_true',
                   help='Run only tests/cli/.')
    p.add_argument('extra', nargs=argparse.REMAINDER,
                   help='Anything after `--` is forwarded to pytest verbatim.')
    add_verbosity_arg(p)
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    configure(args.verbose)
    cmd: list[str] = [sys.executable, '-m', 'pytest']
    paths = _resolve_paths(args)
    cmd.extend(str(p) for p in paths)
    if args.pattern:
        cmd.extend(['-k', args.pattern])
    if args.verbose:
        cmd.append('-' + 'v' * min(args.verbose, 2))
    extra = list(args.extra or [])
    if extra and extra[0] == '--':
        extra = extra[1:]
    cmd.extend(extra)
    return subprocess.run(cmd, check=False).returncode


def _resolve_paths(args: argparse.Namespace) -> list[Path]:
    repo_root = Path(__file__).resolve().parents[3]
    selected: list[Path] = []
    if args.honesty:
        selected.append(repo_root / 'tests/honesty')
    if args.integration:
        selected.append(repo_root / 'tests/integration')
    if args.cli:
        selected.append(repo_root / 'tests/cli')
    if not selected:
        selected.append(repo_root / 'tests')
    return selected
