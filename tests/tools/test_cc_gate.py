from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_issue_title_fetch_is_non_nullable_and_gate_has_no_none_guard() -> None:
    source = (REPO_ROOT / 'tools/cc_gate.py').read_text(encoding='utf-8')
    start = source.index('def fetch_issue_title(repo: str, number: int) -> str:')
    end = source.index('def find_closing_references(body: str) -> list[int]:')
    block = source[start:end]

    assert '-> str | None' not in block
    assert 'return None' not in block
    assert 'or None if the issue cannot be' not in block
    assert 'raise SystemExit(2)' in block
    assert 'if issue_title is not None' not in source
