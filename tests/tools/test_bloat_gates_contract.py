"""Contract tests for the hard-mechanical bloat gates (slice #11)."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
BUDGET_JSON: Final[Path] = REPO_ROOT / '.github/module_budgets.json'
LINT_WORKFLOW: Final[Path] = REPO_ROOT / '.github/workflows/pr_checks_lint.yml'
SCRIPTS_DIR: Final[Path] = REPO_ROOT / 'scripts'
PYPROJECT: Final[Path] = REPO_ROOT / 'pyproject.toml'

GATE_SCRIPTS: Final[list[str]] = [
    'check_module_budgets.py',
    'check_test_code_ratio.py',
    'check_module_docstrings.py',
    'check_file_size_balance.py',
    'check_coverage_floor.py',
    'check_budget_ratchet.py',
]

GATE_BANNERS: Final[dict[str, str]] = {
    'check_module_budgets.py': 'MODULE BUDGET GATE',
    'check_test_code_ratio.py': 'TEST/CODE RATIO GATE',
    'check_module_docstrings.py': 'MODULE DOCSTRING GATE',
    'check_file_size_balance.py': 'FILE SIZE BALANCE GATE',
    'check_coverage_floor.py': 'COVERAGE FLOOR GATE',
    'check_budget_ratchet.py': 'BUDGET RATCHET GATE',
}


def _run(script: str, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(SCRIPTS_DIR / script), *args]
    return subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=cwd or REPO_ROOT)


def test_module_budgets_is_valid_json() -> None:
    data = json.loads(BUDGET_JSON.read_text(encoding='utf-8'))
    assert isinstance(data, dict)
    assert all(isinstance(k, str) for k in data)
    assert all(isinstance(v, int) and v > 0 for v in data.values())


def test_module_budgets_covers_every_m1_path_from_issue_10() -> None:
    data = json.loads(BUDGET_JSON.read_text(encoding='utf-8'))
    bts_paths = {p for p in data if p.startswith('backtest_simulator/')}
    script_paths = {p for p in data if p.startswith('scripts/')}
    # Lower bound relaxed after the fake-purge commit deleted driver.py,
    # environment.py, and the runtime/* shim modules. Every remaining
    # module must still have a budget entry; new modules added in future
    # PRs bump this by their own count.
    assert len(bts_paths) >= 22, f'expected >= 22 backtest_simulator paths, got {len(bts_paths)}'
    assert len(script_paths) == 7, f'expected 7 scripts paths, got {len(script_paths)}'


def test_all_gate_scripts_exist_and_are_executable() -> None:
    for name in GATE_SCRIPTS:
        path = SCRIPTS_DIR / name
        assert path.is_file(), f'{path} missing'
        assert path.stat().st_mode & 0o111, f'{path} not executable'


def test_all_scripts_pass_on_current_repo() -> None:
    _run('check_module_budgets.py').check_returncode()
    _run('check_test_code_ratio.py').check_returncode()
    _run('check_module_docstrings.py').check_returncode()
    _run('check_file_size_balance.py').check_returncode()
    _run(
        'check_budget_ratchet.py',
        '--base-file', '/dev/null',
        '--pr-body-file', '/dev/null',
    ).check_returncode()


def test_pass_banners_printed_on_success() -> None:
    for name in ('check_module_budgets.py', 'check_test_code_ratio.py',
                 'check_module_docstrings.py', 'check_file_size_balance.py'):
        result = _run(name)
        assert result.returncode == 0, result.stderr
        banner = GATE_BANNERS[name]
        assert f'{banner} -- PASS' in result.stdout, f'{name} stdout: {result.stdout!r}'


def test_fail_banners_are_declared_in_each_script_source() -> None:
    for name in GATE_SCRIPTS:
        source = (SCRIPTS_DIR / name).read_text(encoding='utf-8')
        banner = GATE_BANNERS[name]
        assert f'{banner} -- FAIL' in source, f'{name} missing FAIL banner literal'
        assert f'{banner} -- PASS' in source, f'{name} missing PASS banner literal'


def test_workflow_invokes_every_gate() -> None:
    workflow = LINT_WORKFLOW.read_text(encoding='utf-8')
    for script in GATE_SCRIPTS:
        assert f'scripts/{script}' in workflow, f'{script} not invoked by workflow'
    assert 'vulture backtest_simulator/' in workflow
    assert 'ruff check backtest_simulator tools tests scripts' in workflow


def test_no_soft_fail_pathway_in_workflow() -> None:
    workflow = LINT_WORKFLOW.read_text(encoding='utf-8')
    assert '|| true' not in workflow
    assert 'continue-on-error' not in workflow
    forbidden_flags = re.compile(r'--warn-only|--no-fail|--soft(-fail)?')
    assert forbidden_flags.search(workflow) is None


def test_scripts_are_self_budgeted() -> None:
    data = json.loads(BUDGET_JSON.read_text(encoding='utf-8'))
    for name in GATE_SCRIPTS:
        key = f'scripts/{name}'
        assert key in data, f'{key} missing from module_budgets.json'
        assert data[key] <= 120, f'{key} budget {data[key]} exceeds the 120-line self-limit'


def test_ruff_select_includes_new_rules() -> None:
    cfg = tomllib.loads(PYPROJECT.read_text(encoding='utf-8'))
    select = cfg['tool']['ruff']['lint']['select']
    for rule in ('C901', 'PLR0912', 'PLR0913', 'PLR0915', 'T201',
                 'FIX001', 'FIX002', 'FIX003', 'FIX004',
                 'ERA001', 'D200', 'D205', 'D415', 'PIE790'):
        assert rule in select, f'ruff select missing {rule}'


def test_budget_ratchet_vacuous_when_base_missing() -> None:
    result = _run('check_budget_ratchet.py', '--base-file', '/dev/null', '--pr-body-file', '/dev/null')
    assert result.returncode == 0
    assert 'BUDGET RATCHET GATE -- PASS' in result.stdout
    assert 'vacuous' in result.stdout.lower()


def test_budget_ratchet_accepts_marker(tmp_path: Path) -> None:
    # Build a self-contained repo layout in tmp_path that actually has a
    # budget raise between base and head. Previous version ran against
    # the real head budget, so the base's `foo.py` key never matched
    # anything in head and the marker logic was never exercised.
    (tmp_path / '.github').mkdir()
    head = {'backtest_simulator/foo.py': 200}
    base = {'backtest_simulator/foo.py': 100}
    (tmp_path / '.github' / 'module_budgets.json').write_text(json.dumps(head), encoding='utf-8')
    base_file = tmp_path / 'base.json'
    base_file.write_text(json.dumps(base), encoding='utf-8')
    body_file = tmp_path / 'body.txt'
    body_file.write_text(
        '[budget-raise: backtest_simulator/foo.py: legitimate growth]\n',
        encoding='utf-8',
    )
    scripts_dir = tmp_path / 'scripts'
    scripts_dir.mkdir()
    (scripts_dir / '__init__.py').write_text('', encoding='utf-8')
    import shutil
    shutil.copy2(SCRIPTS_DIR / 'check_budget_ratchet.py', scripts_dir / 'check_budget_ratchet.py')
    result = subprocess.run(
        [sys.executable, str(scripts_dir / 'check_budget_ratchet.py'),
         '--base-file', str(base_file), '--pr-body-file', str(body_file)],
        check=False, capture_output=True, text=True, cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert 'BUDGET RATCHET GATE -- PASS' in result.stdout
