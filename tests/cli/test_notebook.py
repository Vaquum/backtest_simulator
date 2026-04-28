"""`bts notebook` rejects missing files; integration covers full execution."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_notebook_subcommand_executes_notebook(tmp_path: Path) -> None:
    # The full-execution path requires `notebooks/sweep_and_analyze.ipynb`,
    # which lands in Task 25. Until then this test pins the missing-file
    # error behaviour: the wrapper exits non-zero with a clear message
    # rather than silently passing.
    missing = tmp_path / 'does-not-exist.ipynb'
    proc = subprocess.run(
        [sys.executable, '-m', 'backtest_simulator.cli', 'notebook',
         '--path', str(missing)],
        check=False, capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode != 0
    assert 'not found' in proc.stderr or 'not found' in proc.stdout
