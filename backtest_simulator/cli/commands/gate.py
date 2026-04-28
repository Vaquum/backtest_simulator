"""`bts gate <name>` — invoke a specific CI gate locally."""

# Wraps the existing `scripts/check_*.py` and `tools/*_gate.py` scripts
# so the operator runs the same gate locally that CI runs at PR time.
from __future__ import annotations

import argparse
import subprocess
import sys
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
                'backtest_simulator', 'tools', 'tests', 'scripts']
    if name == 'typing':
        # Mirror the pr_checks_typing workflow: pyright JSON output
        # piped into tools/typing_gate.py against the protected base-
        # ref budget. The local invocation uses
        # `git show origin/main:.github/typing_budget.json` (or
        # `HEAD:` as a fallback) as the base, NOT `--bootstrap` —
        # bootstrap disables the ratchet and would let a local gate
        # pass after raising the budget while CI still rejected.
        return [
            sys.executable, '-c',
            'import subprocess, sys; '
            'pyr = subprocess.run([sys.executable, "-m", "pyright", "--outputjson", "backtest_simulator"], '
            'capture_output=True, text=True, check=False); '
            'open("pyright_output.json", "w").write(pyr.stdout); '
            'base = subprocess.run(["git", "show", "origin/main:.github/typing_budget.json"], '
            'capture_output=True, text=True, check=False); '
            'base = base if base.returncode == 0 else '
            'subprocess.run(["git", "show", "HEAD:.github/typing_budget.json"], '
            'capture_output=True, text=True, check=True); '
            'open("base_budget.json", "w").write(base.stdout); '
            'sys.exit(subprocess.run([sys.executable, "tools/typing_gate.py", '
            '"--pyright-json", "pyright_output.json", '
            '"--base-budget", "base_budget.json"]).returncode)',
        ]
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
    'module_budgets': [sys.executable, 'scripts/check_module_budgets.py'],
    'docstrings': [sys.executable, 'scripts/check_module_docstrings.py'],
    'file_size_balance': [sys.executable, 'scripts/check_file_size_balance.py'],
    'test_code_ratio': [sys.executable, 'scripts/check_test_code_ratio.py'],
    'no_swallowed_violations': [
        sys.executable, 'scripts/check_no_swallowed_violations.py',
    ],
}

_ALL_GATES: Final[tuple[str, ...]] = (
    'lint', 'typing', 'honesty',
    *sorted(_STATIC_GATE_COMMANDS.keys()),
)


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser('gate', help='Run a specific CI gate locally.')
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
