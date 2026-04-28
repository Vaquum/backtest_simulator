"""`bts version` prints `bts <pyproject version>` and exits 0."""
from __future__ import annotations

import importlib.metadata
import io
import tomllib
from contextlib import redirect_stdout
from pathlib import Path

from backtest_simulator.cli import main

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_version_subcommand_exit_code() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(['version'])
    assert rc == 0


def test_version_subcommand_prints_version_string() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(['version'])
    expected = importlib.metadata.version('backtest_simulator')
    assert buf.getvalue().strip() == f'bts {expected}'


def test_pyproject_version_matches() -> None:
    pyproject = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    declared = pyproject['project']['version']
    installed = importlib.metadata.version('backtest_simulator')
    assert declared == installed, (
        f'pyproject.toml declares {declared!r} but the installed package '
        f'reports {installed!r}; reinstall the editable build.'
    )
