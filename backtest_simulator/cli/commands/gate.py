"""`bts gate <name>` — invoke a specific CI gate locally."""

# Wraps the existing `tools/check_*.py` and `tools/*_gate.py` scripts
# so the operator runs the same gate locally that CI runs at PR time.
from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final

from backtest_simulator.cli._verbosity import add_verbosity_arg, configure

# Honesty test paths must exist at command construction time. Test
# directories landing in later M2 tasks (e.g. tests/runtime/ in Task 22,
# tests/integration/ already present, tests/cli/ added in Task 1) are
# included only if the directory is on disk; the helper `_honesty_paths`
# below filters at invocation time so `bts gate honesty` always points
# at a real path set.
_HONESTY_TEST_DIR_CANDIDATES: Final[tuple[str, ...]] = (
    'tests/honesty',
    'tests/cli',
    'tests/runtime',
    'tests/integration',
    'tests/tools',
    'tests/launcher',
    'tests/pipeline',
)


def _honesty_paths(repo_root: Path) -> list[str]:
    return [str(repo_root / p) for p in _HONESTY_TEST_DIR_CANDIDATES if (repo_root / p).is_dir()]


def _build_command(name: str, repo_root: Path) -> list[str]:
    if name == 'lint':
        return [sys.executable, '-m', 'ruff', 'check',
                'backtest_simulator', 'tools', 'tests']
    if name == 'typing':
        # Mirror `pr_checks_typing.yml` byte-for-byte: plant PEP 561
        # `py.typed` markers in sibling libs (CI's "Plant py.typed
        # markers in sibling libraries" step), invoke pyright with
        # `--pythonpath sys.executable` so the venv's site-packages
        # is discovered (CI uses system Python where auto-detection
        # works), and resolve the base-budget oracle with the same
        # bootstrap-or-fail-loud conditional CI uses (origin/main +
        # HEAD-introduces-file check, never silent fallback).
        #
        # Implementation lives in `backtest_simulator.cli._typing_runner`
        # so the conditional is readable Python rather than a dense
        # one-liner string. Each step there has a docstring naming the
        # workflow step it mirrors.
        return [sys.executable, '-m', 'backtest_simulator.cli._typing_runner']
    if name == 'honesty':
        return [sys.executable, '-m', 'pytest', *_honesty_paths(repo_root),
                '-v', '--tb=short']
    return _STATIC_GATE_COMMANDS[name]


_STATIC_GATE_COMMANDS: Final[dict[str, list[str]]] = {
    'fail_loud': [sys.executable, 'tools/fail_loud_gate.py'],
    'cc': [sys.executable, 'tools/cc_gate.py'],
    'slice': [sys.executable, 'tools/slice_gate.py'],
    'version': [sys.executable, 'tools/version_gate.py'],
    'ruleset': [sys.executable, 'tools/ruleset_gate.py'],
    'module_budgets': [sys.executable, 'tools/check_module_budgets.py'],
    'docstrings': [sys.executable, 'tools/check_module_docstrings.py'],
    'file_size_balance': [sys.executable, 'tools/check_file_size_balance.py'],
    'test_code_ratio': [sys.executable, 'tools/check_test_code_ratio.py'],
    'no_swallowed_violations': [
        sys.executable, 'tools/check_no_swallowed_violations.py',
    ],
}

_ALL_GATES: Final[tuple[str, ...]] = (
    'lint', 'typing', 'honesty',
    *sorted(_STATIC_GATE_COMMANDS.keys()),
)


def register(add_parser: Callable[[str, str], argparse.ArgumentParser]) -> None:
    p = add_parser('gate', 'Run a specific CI gate locally.')
    p.add_argument('name', choices=[*_ALL_GATES, 'all'],
                   help='Gate name. `all` runs every gate sequentially '
                        'and stops at the first non-zero exit.')
    add_verbosity_arg(p)
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    configure(args.verbose)
    repo_root = Path(__file__).resolve().parents[3]
    if args.name == 'all':
        for gate_name in _ALL_GATES:
            rc = _run_one(gate_name, repo_root)
            if rc != 0:
                return rc
        return 0
    return _run_one(args.name, repo_root)


def _run_one(name: str, cwd: Path) -> int:
    cmd = _build_command(name, cwd)
    return subprocess.run(cmd, check=False, cwd=cwd).returncode
