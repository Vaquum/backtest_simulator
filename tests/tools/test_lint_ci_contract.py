from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Final

REPO_ROOT = Path(__file__).resolve().parents[2]
LINT_WORKFLOW: Final[Path] = REPO_ROOT / '.github/workflows/pr_checks_lint.yml'
RULESET_WORKFLOW: Final[Path] = REPO_ROOT / '.github/workflows/pr_checks_ruleset.yml'
RULESET_SNAPSHOT: Final[Path] = REPO_ROOT / '.github/rulesets/main.json'
BAD_FIXTURE: Final[Path] = REPO_ROOT / 'tests/fixtures/lint/bad_imports.py'
RUFF_VERSION: Final[str] = '0.15.11'
EXPECTED_RUFF_POLICY: Final[dict[str, object]] = {
    'exclude': [
        '.git',
        '__pycache__',
        'build',
        'dist',
        'quickstart_etl_tests',
    ],
    'select': [
        'E',
        'F',
        'I',
        'UP',
        'RUF',
        'BLE',
        'ANN',
    ],
    'ignore': ['E501'],
    'per-file-ignores': {
        'quickstart_etl_tests/**/*.py': ['S101', 'ANN', 'BLE001'],
    },
}


def _required_status_contexts() -> list[str]:
    payload = json.loads(RULESET_SNAPSHOT.read_text(encoding='utf-8'))
    for rule in payload['rules']:
        if rule['type'] == 'required_status_checks':
            checks = rule['parameters']['required_status_checks']
            return [entry['context'] for entry in checks]
    raise AssertionError('required_status_checks rule missing from ruleset snapshot')


def _run_ruff(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, '-m', 'ruff', *args],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_pr_checks_lint_workflow_exists() -> None:
    assert LINT_WORKFLOW.exists()


def test_ruleset_snapshot_requires_pr_checks_lint() -> None:
    assert 'pr_checks_lint' in _required_status_contexts()


def test_pr_checks_lint_runs_pinned_ruff_on_tools_and_tests_tools() -> None:
    workflow = LINT_WORKFLOW.read_text(encoding='utf-8')

    assert f"'ruff=={RUFF_VERSION}'" in workflow
    assert '.venv-lint/bin/python -m ruff check tools tests/tools' in workflow
    assert 'continue-on-error' not in workflow


def test_pr_checks_ruleset_runs_test_lint_ci_contract() -> None:
    workflow = RULESET_WORKFLOW.read_text(encoding='utf-8')

    assert f"'ruff=={RUFF_VERSION}'" in workflow
    assert 'tests/tools/test_lint_ci_contract.py' in workflow


def test_pinned_ruff_fails_on_known_bad_fixture() -> None:
    version = _run_ruff('--version')
    assert version.returncode == 0, version.stderr
    assert version.stdout.strip() == f'ruff {RUFF_VERSION}'

    result = _run_ruff('check', str(BAD_FIXTURE))

    assert result.returncode == 1
    assert 'bad_imports.py' in f'{result.stdout}\n{result.stderr}'


def test_ruff_pin_is_consistent_across_workflows() -> None:
    files = [LINT_WORKFLOW, RULESET_WORKFLOW]
    pins = sorted({
        pin
        for workflow in files
        for pin in re.findall(r'ruff==([0-9.]+)', workflow.read_text(encoding='utf-8'))
    })

    assert pins == [RUFF_VERSION]


def test_pyproject_ruff_policy_contract() -> None:
    data = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    ruff = data['tool']['ruff']
    actual_policy = {
        'exclude': ruff.get('exclude'),
        'select': ruff['lint'].get('select'),
        'ignore': ruff['lint'].get('ignore'),
        'per-file-ignores': ruff['lint'].get('per-file-ignores'),
    }

    assert actual_policy == EXPECTED_RUFF_POLICY

    result = _run_ruff('check', '--isolated', '--select', 'BLE001', 'tools', 'tests/tools')
    assert result.returncode == 0, result.stdout + result.stderr
