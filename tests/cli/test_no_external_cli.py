"""The `bts_sweep.py` legacy CLI no longer lives anywhere git tracks."""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_no_bts_sweep_tracked() -> None:
    result = subprocess.run(
        ['git', 'ls-files'],
        check=True, capture_output=True, text=True, cwd=REPO_ROOT,
    )
    bad = [
        line for line in result.stdout.splitlines()
        if line.endswith('bts_sweep.py') or line.endswith('/bts_sweep.py')
    ]
    assert bad == [], f'bts_sweep.py is tracked: {bad}'
