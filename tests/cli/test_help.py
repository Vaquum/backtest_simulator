"""`bts --help` lists exactly the 9 documented subcommands."""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from backtest_simulator.cli import SUBCOMMANDS, _build_parser

EXPECTED: tuple[str, ...] = (
    'run', 'sweep', 'enrich', 'test', 'lint',
    'typecheck', 'gate', 'notebook', 'version',
)


def test_help_lists_all_subcommands() -> None:
    ap = _build_parser()
    buf = io.StringIO()
    with redirect_stdout(buf), pytest.raises(SystemExit) as exc:
        ap.parse_args(['--help'])
    assert exc.value.code == 0
    out = buf.getvalue()
    for name in EXPECTED:
        assert name in out, f'subcommand {name!r} missing from --help output:\n{out}'


def test_subcommands_constant_matches_expected() -> None:
    assert tuple(SUBCOMMANDS) == EXPECTED
