from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_post_merge_changelog_workflow_removed() -> None:
    assert not (REPO_ROOT / '.github/workflows/pr_post_changelog.yml').exists()


def test_update_changelog_script_removed() -> None:
    assert not (REPO_ROOT / 'scripts/update_changelog.py').exists()


def test_typing_gate_setup_failures_exit_2() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / 'repo'
        shutil.copytree(
            REPO_ROOT,
            tmp,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns('.git'),
        )
        (tmp / 'pyproject.toml').write_text('not = [valid\n', encoding='utf-8')

        result = subprocess.run(
            [
                'python3',
                'tools/typing_gate.py',
                '--pyright-json',
                '/tmp/missing-pyright.json',
                '--bootstrap',
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=tmp,
        )

    assert result.returncode == 2
    assert result.stdout == ''
    assert 'typing_gate: cannot parse pyproject.toml:' in result.stderr
