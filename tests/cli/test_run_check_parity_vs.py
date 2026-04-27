"""Honesty gate: `bts run --check-parity-vs` failure modes.

Slice #17 Task 18 / auditor round 4 P1: parity gate must not
silently fail-open when the event_spine artifact is missing or
when `assert_ledger_parity` raises.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from backtest_simulator.cli.commands.run import _maybe_assert_parity


def _args(check: Path | None, tolerance: str = 'strict') -> argparse.Namespace:
    return argparse.Namespace(
        check_parity_vs=check, parity_tolerance=tolerance,
    )


def test_maybe_assert_parity_no_check_returns_zero(tmp_path: Path) -> None:
    """No --check-parity-vs given -> exit 0, log spine path."""
    spine = tmp_path / 'spine.jsonl'
    spine.write_text('', encoding='utf-8')
    result: dict[str, Any] = {
        'event_spine_jsonl': str(spine),
        'event_spine_n_events': 0,
    }
    rc = _maybe_assert_parity(_args(check=None), result)
    assert rc == 0


def test_maybe_assert_parity_pass_returns_zero(tmp_path: Path) -> None:
    """Identical files + STRICT -> parity passes -> exit 0."""
    spine = tmp_path / 'spine.jsonl'
    ref = tmp_path / 'ref.jsonl'
    content = '{"epoch_id":1,"event_seq":1,"timestamp":"x","event_type":"X","payload_raw_b64":"YWJj"}\n'
    spine.write_text(content, encoding='utf-8')
    ref.write_text(content, encoding='utf-8')
    result: dict[str, Any] = {
        'event_spine_jsonl': str(spine),
        'event_spine_n_events': 1,
    }
    rc = _maybe_assert_parity(_args(check=ref), result)
    assert rc == 0


def test_maybe_assert_parity_violation_returns_one(tmp_path: Path) -> None:
    """Mismatched files + STRICT -> ParityViolation -> exit 1.

    The CLI must propagate ParityViolation as a non-zero exit so
    CI / shell scripts can gate on the parity check.
    """
    spine = tmp_path / 'spine.jsonl'
    ref = tmp_path / 'ref.jsonl'
    spine.write_text(
        '{"epoch_id":1,"event_seq":1,"payload_raw_b64":"YWJj"}\n',
        encoding='utf-8',
    )
    ref.write_text(
        '{"epoch_id":1,"event_seq":1,"payload_raw_b64":"ZGVm"}\n',
        encoding='utf-8',
    )
    result: dict[str, Any] = {
        'event_spine_jsonl': str(spine),
        'event_spine_n_events': 1,
    }
    rc = _maybe_assert_parity(_args(check=ref), result)
    assert rc == 1


def test_maybe_assert_parity_missing_spine_with_check_fails_loud(
    tmp_path: Path,
) -> None:
    """`--check-parity-vs` requested but spine path missing -> exit 1.

    Auditor round 4 P1: the parity gate must NEVER fail-open. If
    the subprocess didn't return event_spine_jsonl (or returned a
    non-str), a parity check is impossible — fail loudly with
    exit 1 so the operator sees the gate didn't run, not pass.

    Mutation proof: if the gate returned 0 here, a broken pipeline
    would silently pass the parity check.
    """
    ref = tmp_path / 'ref.jsonl'
    ref.write_text('', encoding='utf-8')
    result: dict[str, Any] = {
        # Subprocess result drift: event_spine_jsonl key absent.
        'event_spine_n_events': 0,
    }
    rc = _maybe_assert_parity(_args(check=ref), result)
    assert rc == 1


def test_maybe_assert_parity_missing_spine_without_check_returns_zero(
    tmp_path: Path,
) -> None:
    """No parity check requested + missing spine -> exit 0 (no gate)."""
    result: dict[str, Any] = {
        'event_spine_n_events': 0,
    }
    rc = _maybe_assert_parity(_args(check=None), result)
    assert rc == 0


def test_maybe_assert_parity_emit_human_false_keeps_stdout_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """`emit_human=False` writes nothing to stdout (JSON mode contract).

    Codex round-5 P1: `bts run --output-format json` must emit a
    single parseable JSON object on stdout. Parity status writes
    polluted that contract previously. The fix routes them via
    return code + stderr only when emit_human=False.

    Mutation proof: removing the emit_human guard prints to stdout
    and this test fires.
    """
    spine = tmp_path / 'spine.jsonl'
    ref = tmp_path / 'ref.jsonl'
    content = '{"epoch_id":1,"event_seq":1,"payload_raw_b64":"YWJj"}\n'
    spine.write_text(content, encoding='utf-8')
    ref.write_text(content, encoding='utf-8')
    result: dict[str, Any] = {
        'event_spine_jsonl': str(spine),
        'event_spine_n_events': 1,
    }
    capsys.readouterr()  # clear any prior captured
    rc = _maybe_assert_parity(_args(check=ref), result, emit_human=False)
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == '', (
        f'JSON-mode stdout must be empty; got {captured.out!r}'
    )
