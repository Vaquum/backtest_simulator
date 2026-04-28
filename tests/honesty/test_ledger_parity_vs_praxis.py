"""Honesty gate: backtest event spine == paper-Praxis spine on a fixture.

Pins slice #17 Task 18 / SPEC §9.6 layer 4.
"""
from __future__ import annotations

import pytest

from backtest_simulator.exceptions import ParityViolation
from backtest_simulator.honesty.ledger_parity import (
    ParityTolerance,
    assert_ledger_parity,
)


def test_ledger_parity_vs_praxis(tmp_path: object) -> None:
    """Identical event-spine files pass; one-byte divergence raises.

    The MVC: write two identical JSON-lines event-spine files and
    assert parity. Then mutate one byte in the second file and
    assert ParityViolation.
    """
    from pathlib import Path
    base_dir = Path(str(tmp_path))
    spine_a = base_dir / 'backtest.jsonl'
    spine_b = base_dir / 'paper.jsonl'
    content = (
        '{"event":"start","t":"2024-01-01T00:00:00Z"}\n'
        '{"event":"submit","cmd_id":"abc","qty":"0.001"}\n'
        '{"event":"fill","cmd_id":"abc","price":"42700.0"}\n'
        '{"event":"end","t":"2024-01-01T00:01:00Z"}\n'
    )
    spine_a.write_text(content, encoding='utf-8')
    spine_b.write_text(content, encoding='utf-8')
    # Identical files → no raise.
    assert_ledger_parity(
        backtest_event_spine=spine_a,
        paper_event_spine=spine_b,
        tolerance=ParityTolerance.STRICT,
    )
    # Mutate one line.
    spine_b.write_text(
        content.replace('"qty":"0.001"', '"qty":"0.0011"'),
        encoding='utf-8',
    )
    with pytest.raises(ParityViolation, match='diverges from paper'):
        assert_ledger_parity(
            backtest_event_spine=spine_a,
            paper_event_spine=spine_b,
            tolerance=ParityTolerance.STRICT,
        )


def test_ledger_parity_default_tolerance_is_strict() -> None:
    """No-tolerance arg defaults to STRICT — never silently relaxes."""
    import inspect
    sig = inspect.signature(assert_ledger_parity)
    default = sig.parameters['tolerance'].default
    assert default == ParityTolerance.STRICT, (
        f'default tolerance must be STRICT to prevent silent '
        f'numeric drift; got {default}'
    )


def test_ledger_parity_clock_normalized_strips_envelope(
    tmp_path: object,
) -> None:
    """CLOCK_NORMALIZED strips event_seq + envelope timestamp.

    Slice #17 Task 18 codex round-3: cross-runtime mode. Two
    spines with identical payload bytes but different event_seq /
    timestamp values pass under CLOCK_NORMALIZED, fail under
    STRICT. Mutation proof: if the normalize stripper drops more
    than `event_seq` + `timestamp`, the payload-diff test below
    breaks.
    """
    from pathlib import Path
    base = Path(str(tmp_path))
    spine_a = base / 'a.jsonl'
    spine_b = base / 'b.jsonl'
    # Same payload_raw_b64; different event_seq + timestamp.
    spine_a.write_text(
        '{"event_seq":1,"timestamp":"2024-01-01T00:00:00","payload_raw_b64":"YWJj"}\n'
        '{"event_seq":2,"timestamp":"2024-01-01T00:00:01","payload_raw_b64":"ZGVm"}\n',
        encoding='utf-8',
    )
    spine_b.write_text(
        '{"event_seq":42,"timestamp":"2099-12-31T23:59:59","payload_raw_b64":"YWJj"}\n'
        '{"event_seq":43,"timestamp":"2099-12-31T23:59:58","payload_raw_b64":"ZGVm"}\n',
        encoding='utf-8',
    )
    # CLOCK_NORMALIZED: passes (only envelope differs).
    assert_ledger_parity(
        backtest_event_spine=spine_a, paper_event_spine=spine_b,
        tolerance=ParityTolerance.CLOCK_NORMALIZED,
    )
    # STRICT: same input fails because event_seq + timestamp bytes differ.
    with pytest.raises(ParityViolation, match='diverges'):
        assert_ledger_parity(
            backtest_event_spine=spine_a, paper_event_spine=spine_b,
            tolerance=ParityTolerance.STRICT,
        )


def test_ledger_parity_clock_normalized_catches_payload_diff(
    tmp_path: object,
) -> None:
    """CLOCK_NORMALIZED still raises on payload byte drift.

    The whole point of the mode: timestamps + seqs may differ
    cross-runtime, but payload bytes must not. Different
    `payload_raw_b64` -> raise even under CLOCK_NORMALIZED.
    """
    from pathlib import Path
    base = Path(str(tmp_path))
    spine_a = base / 'a.jsonl'
    spine_b = base / 'b.jsonl'
    spine_a.write_text(
        '{"event_seq":1,"timestamp":"x","payload_raw_b64":"YWJj"}\n',
        encoding='utf-8',
    )
    # Same envelope, different payload bytes.
    spine_b.write_text(
        '{"event_seq":1,"timestamp":"x","payload_raw_b64":"ZGVm"}\n',
        encoding='utf-8',
    )
    with pytest.raises(ParityViolation, match='CLOCK_NORMALIZED'):
        assert_ledger_parity(
            backtest_event_spine=spine_a, paper_event_spine=spine_b,
            tolerance=ParityTolerance.CLOCK_NORMALIZED,
        )


def test_ledger_parity_missing_file_raises(tmp_path: object) -> None:
    """Missing event-spine path raises FileNotFoundError loudly."""
    from pathlib import Path
    spine_a = Path(str(tmp_path)) / 'present.jsonl'
    spine_b = Path(str(tmp_path)) / 'missing.jsonl'
    spine_a.write_text('{}\n', encoding='utf-8')
    with pytest.raises(FileNotFoundError, match='event-spine file missing'):
        assert_ledger_parity(
            backtest_event_spine=spine_a, paper_event_spine=spine_b,
        )


def test_ledger_parity_length_mismatch_raises(tmp_path: object) -> None:
    """Files of different line counts raise ParityViolation."""
    from pathlib import Path
    spine_a = Path(str(tmp_path)) / 'a.jsonl'
    spine_b = Path(str(tmp_path)) / 'b.jsonl'
    spine_a.write_text('{"x":1}\n{"x":2}\n', encoding='utf-8')
    spine_b.write_text('{"x":1}\n', encoding='utf-8')
    with pytest.raises(ParityViolation):
        assert_ledger_parity(
            backtest_event_spine=spine_a, paper_event_spine=spine_b,
        )
