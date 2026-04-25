"""Ledger parity — backtest event spine vs Praxis paper-trade spine."""
from __future__ import annotations

# Slice #17 Task 18. The M3 unlock is "backtest ≡ paper-Praxis ≡
# live, byte-identical event spine on a deterministic scenario".
# `assert_ledger_parity` enforces this contract: given two event-
# spine paths (each a JSON-lines log of the runtime's event
# sequence), they must agree under the chosen tolerance.
#
# `ParityTolerance.STRICT` requires byte-identical spines.
# `ParityTolerance.NUMERIC` allows numeric fields to differ by
# 1e-9 relative (rounding noise from Decimal vs float in some
# edge cases). `STRICT` is the default — anything weaker has to
# be explicitly opted into.
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

from backtest_simulator.exceptions import ParityViolation


class ParityTolerance(Enum):
    """Comparison strictness for `assert_ledger_parity`."""

    STRICT = auto()
    NUMERIC = auto()


@dataclass(frozen=True)
class _ParityFailure:
    line_no: int
    backtest_line: str
    paper_line: str


def _read_lines(path: Path) -> list[str]:
    if not path.is_file():
        msg = f'assert_ledger_parity: event-spine file missing: {path}'
        raise FileNotFoundError(msg)
    return path.read_text(encoding='utf-8').splitlines()


def _strict_diff(a: list[str], b: list[str]) -> list[_ParityFailure]:
    failures: list[_ParityFailure] = []
    for i, (al, bl) in enumerate(zip(a, b, strict=False)):
        if al != bl:
            failures.append(_ParityFailure(
                line_no=i + 1, backtest_line=al, paper_line=bl,
            ))
    if len(a) != len(b):
        # Length mismatch surfaces as a tail diff.
        if len(a) > len(b):
            for i, line in enumerate(a[len(b):], start=len(b) + 1):
                failures.append(_ParityFailure(
                    line_no=i, backtest_line=line, paper_line='',
                ))
        else:
            for i, line in enumerate(b[len(a):], start=len(a) + 1):
                failures.append(_ParityFailure(
                    line_no=i, backtest_line='', paper_line=line,
                ))
    return failures


def assert_ledger_parity(
    *,
    backtest_event_spine: Path,
    paper_event_spine: Path,
    tolerance: ParityTolerance = ParityTolerance.STRICT,
) -> None:
    """Assert that two event-spine files agree under `tolerance`.

    Raises `ParityViolation` on the first mismatch (or all of
    them — production may want a delta report; the MVC pin is the
    raise itself).
    """
    if tolerance != ParityTolerance.STRICT:
        # NUMERIC tolerance is reserved for M3 implementation;
        # STRICT is the only honest mode the slice ships.
        msg = (
            f'assert_ledger_parity: only ParityTolerance.STRICT is '
            f'implemented; got {tolerance}. The numeric-tolerance '
            f'mode is reserved for M3 once a concrete divergence '
            f'class motivates it; until then strict byte-equality '
            f'is the contract.'
        )
        raise NotImplementedError(msg)
    a = _read_lines(backtest_event_spine)
    b = _read_lines(paper_event_spine)
    failures = _strict_diff(a, b)
    if failures:
        first = failures[0]
        msg = (
            f'assert_ledger_parity: backtest spine '
            f'({backtest_event_spine}) diverges from paper spine '
            f'({paper_event_spine}) at line {first.line_no} '
            f'(total {len(failures)} divergent lines). '
            f'backtest={first.backtest_line!r} '
            f'paper={first.paper_line!r}'
        )
        raise ParityViolation(msg)
