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
        'demo',
        'tests/fixtures',
    ],
    'select': [
        'E',
        'F',
        'I',
        'UP',
        'RUF',
        'BLE',
        'ANN',
        'C901',
        'PLR0912',
        'PLR0913',
        'PLR0915',
        'T201',
        'FIX001',
        'FIX002',
        'FIX003',
        'FIX004',
        'ERA001',
        'D200',
        'D205',
        'D415',
        'PIE790',
    ],
    'ignore': ['E501'],
    'per-file-ignores': {
        'tests/**/*.py': [
            'S101', 'ANN', 'BLE001',
            'PLR0912', 'PLR0913', 'PLR0915',
            'D200', 'D205', 'D415',
        ],
        'tools/**/*.py': [
            'C901', 'PLR0912', 'PLR0913', 'PLR0915',
            'T201',
            'FIX001', 'FIX002', 'FIX003', 'FIX004',
            'ERA001', 'D200', 'D205', 'D415',
        ],
        'scripts/**/*.py': [
            'C901', 'PLR0912', 'PLR0913', 'PLR0915',
            'T201',
            'FIX001', 'FIX002', 'FIX003', 'FIX004',
            'ERA001', 'D200', 'D205', 'D415',
        ],
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
    # Name kept for slice #11 Tests-table backward compatibility. The
    # assertions inside now cover the broader post-#11 surface
    # (ruff + every gate script + no-soft-fail). A future slice may
    # split or rename this; doing so requires updating the slice body's
    # Tests table in lockstep.
    workflow = LINT_WORKFLOW.read_text(encoding='utf-8')

    assert f"'ruff=={RUFF_VERSION}'" in workflow
    assert '.venv-lint/bin/python -m ruff check backtest_simulator tools tests scripts' in workflow
    assert 'continue-on-error' not in workflow
    # Hard-mechanical gate surfaces from slice #11 — each invocation
    # must appear verbatim somewhere in the workflow.
    assert 'scripts/check_module_budgets.py' in workflow
    assert 'scripts/check_module_docstrings.py' in workflow
    assert 'scripts/check_file_size_balance.py' in workflow
    assert 'scripts/check_test_code_ratio.py' in workflow
    assert 'scripts/check_coverage_floor.py' in workflow
    assert 'scripts/check_budget_ratchet.py' in workflow
    assert 'vulture' in workflow
    # No soft-fail pathway: no `|| true`, no continue-on-error on any step.
    assert '|| true' not in workflow


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
