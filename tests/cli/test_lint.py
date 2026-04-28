"""`bts lint` runs ruff and prints the success banner on a clean tree."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_lint_subcommand_prints_all_checks_passed() -> None:
    proc = subprocess.run(
        [sys.executable, '-m', 'backtest_simulator.cli', 'lint',
         '--paths', str(REPO_ROOT / 'backtest_simulator')],
        check=False, capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'All checks passed!' in (proc.stdout + proc.stderr), (
        f'expected ruff success banner; got stdout={proc.stdout!r} '
        f'stderr={proc.stderr!r}'
    )
