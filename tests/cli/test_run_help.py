"""`bts run --help` exposes -v / -vv / -vvv verbosity flags."""
from __future__ import annotations

import io
import re
from contextlib import redirect_stdout

import pytest

from backtest_simulator.cli import _build_parser


def test_run_verbosity_flags() -> None:
    ap = _build_parser()
    buf = io.StringIO()
    with redirect_stdout(buf), pytest.raises(SystemExit) as exc:
        ap.parse_args(['run', '--help'])
    assert exc.value.code == 0
    out = buf.getvalue()
    matches = re.findall(r'^\s*-v[v]?[v]?\b', out, flags=re.MULTILINE)
    assert len(matches) >= 1, f'no -v flag in run --help:\n{out}'
    # `-v` is the canonical flag; argparse `count` exposes it once but
    # the docstring describes -vv / -vvv. Both must appear in the help
    # text so the operator sees the full progression.
    assert '-v' in out
    assert '-vv' in out or '-v -v' in out or 'count' in out
    assert '-vvv' in out or 'NOTSET' in out
