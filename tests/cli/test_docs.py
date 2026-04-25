"""`docs/cli.md` exists and documents every subcommand."""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / 'docs/cli.md'

EXPECTED_SECTIONS: tuple[str, ...] = (
    'run', 'sweep', 'enrich', 'test', 'lint',
    'typecheck', 'gate', 'notebook', 'version',
)


def test_docs_cli_md_exists() -> None:
    assert DOC.is_file(), f'{DOC} missing'


def test_docs_cli_md_has_all_subcommand_sections() -> None:
    text = DOC.read_text(encoding='utf-8')
    headings = set(re.findall(r'^##\s+([a-z_]+)\b', text, flags=re.MULTILINE))
    missing = set(EXPECTED_SECTIONS) - headings
    assert not missing, f'docs/cli.md missing sections for: {sorted(missing)}'
