"""`bts gate <name>` runs the named CI gate locally."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, '-m', 'backtest_simulator.cli', *args],
        check=False, capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_gate_lint() -> None:
    proc = _run(['gate', 'lint'])
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_gate_honesty() -> None:
    # The dispatcher's command construction is the actual contract; we
    # don't run pytest from inside this test (the full honesty suite is
    # the slice's regression gate, not a CLI-test concern).
    from backtest_simulator.cli.commands.gate import _build_command
    cmd = _build_command('honesty', REPO_ROOT)
    assert 'pytest' in ' '.join(cmd)
    # honesty test paths are dynamically filtered to existing dirs.
    joined = ' '.join(cmd)
    assert 'tests/honesty' in joined


def test_gate_typing() -> None:
    from backtest_simulator.cli.commands.gate import _build_command
    cmd = _build_command('typing', REPO_ROOT)
    assert cmd[1:] == ['tools/local_typing_gate.py'], (
        f'`bts gate typing` must invoke tools/local_typing_gate.py. cmd[1:]={cmd[1:]!r}'
    )
