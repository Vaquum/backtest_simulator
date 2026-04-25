"""Mutation tests for every bloat gate: prove each gate fires on its violation."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
SCRIPTS_DIR: Final[Path] = REPO_ROOT / 'scripts'


def _clone_script_into(root: Path, name: str) -> Path:
    dest_dir = root / 'scripts'
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / '__init__.py').write_text('', encoding='utf-8')
    dest = dest_dir / name
    shutil.copy2(SCRIPTS_DIR / name, dest)
    return dest


def _run(script: Path, *args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        check=False,
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def test_module_budget_mutation_fires(tmp_path: Path) -> None:
    (tmp_path / '.github').mkdir()
    (tmp_path / 'backtest_simulator').mkdir()
    budget = {'backtest_simulator/oversized.py': 5}
    (tmp_path / '.github' / 'module_budgets.json').write_text(json.dumps(budget), encoding='utf-8')
    (tmp_path / 'backtest_simulator' / 'oversized.py').write_text(
        'a = 1\nb = 2\nc = 3\nd = 4\ne = 5\nf = 6\n',  # 6 SLOC > 5 budget
        encoding='utf-8',
    )
    script = _clone_script_into(tmp_path, 'check_module_budgets.py')
    result = _run(script, cwd=tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'MODULE BUDGET GATE -- FAIL' in result.stderr
    assert 'oversized.py' in result.stderr


def test_module_docstring_mutation_fires(tmp_path: Path) -> None:
    (tmp_path / 'backtest_simulator').mkdir()
    (tmp_path / 'backtest_simulator' / 'no_docstring.py').write_text(
        'x = 1\n',
        encoding='utf-8',
    )
    script = _clone_script_into(tmp_path, 'check_module_docstrings.py')
    result = _run(script, cwd=tmp_path)
    assert result.returncode == 1
    assert 'MODULE DOCSTRING GATE -- FAIL' in result.stderr
    assert 'no_docstring.py' in result.stderr


def test_module_docstring_multiline_mutation_fires(tmp_path: Path) -> None:
    (tmp_path / 'backtest_simulator').mkdir()
    (tmp_path / 'backtest_simulator' / 'multiline.py').write_text(
        '"""Summary line.\n\nDetailed paragraph."""\nx = 1\n',
        encoding='utf-8',
    )
    script = _clone_script_into(tmp_path, 'check_module_docstrings.py')
    result = _run(script, cwd=tmp_path)
    assert result.returncode == 1
    assert 'MODULE DOCSTRING GATE -- FAIL' in result.stderr
    assert 'multiline.py' in result.stderr


def test_file_size_balance_mutation_fires(tmp_path: Path) -> None:
    (tmp_path / 'backtest_simulator').mkdir()
    (tmp_path / 'backtest_simulator' / 'small_a.py').write_text('a = 1\n', encoding='utf-8')
    (tmp_path / 'backtest_simulator' / 'small_b.py').write_text('b = 2\n', encoding='utf-8')
    # Median will be ~1 line; a 10-line file gives ratio > 2.5.
    (tmp_path / 'backtest_simulator' / 'giant.py').write_text('\n' * 20, encoding='utf-8')
    script = _clone_script_into(tmp_path, 'check_file_size_balance.py')
    result = _run(script, cwd=tmp_path)
    assert result.returncode == 1
    assert 'FILE SIZE BALANCE GATE -- FAIL' in result.stderr
    assert 'giant.py' in result.stderr


def test_test_code_ratio_mutation_fires(tmp_path: Path) -> None:
    # source with enough SLOC to bypass the vacuous-pass boundary (50),
    # but tests too small -> ratio < 0.60.
    source_dir = tmp_path / 'backtest_simulator'
    source_dir.mkdir()
    (source_dir / 'big.py').write_text('\n'.join(f'x_{i} = {i}' for i in range(60)) + '\n', encoding='utf-8')
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests' / 'tiny.py').write_text('y = 1\n', encoding='utf-8')
    script = _clone_script_into(tmp_path, 'check_test_code_ratio.py')
    result = _run(script, cwd=tmp_path)
    assert result.returncode == 1
    assert 'TEST/CODE RATIO GATE -- FAIL' in result.stderr


def test_coverage_floor_mutation_fires(tmp_path: Path) -> None:
    # Mutation input must be below both the line floor (50%) and the
    # branch floor (45%) to trip the gate. M1 bootstrap floors were
    # lowered from 95/90 to 50/45 to accommodate integration-scaffold
    # modules exercised only by the e2e boot path.
    coverage = {
        'totals': {
            'num_statements': 100,
            'percent_covered': 40.0,
            'percent_covered_branches': 35.0,
        },
        'files': {'backtest_simulator/foo.py': {'missing_lines': [10, 11, 12]}},
    }
    (tmp_path / 'coverage.json').write_text(json.dumps(coverage), encoding='utf-8')
    script = _clone_script_into(tmp_path, 'check_coverage_floor.py')
    result = _run(script, cwd=tmp_path)
    assert result.returncode == 1
    assert 'COVERAGE FLOOR GATE -- FAIL' in result.stderr
    assert '40.0%' in result.stderr


def test_budget_ratchet_mutation_fires(tmp_path: Path) -> None:
    (tmp_path / '.github').mkdir()
    head = {'backtest_simulator/foo.py': 200}
    base = {'backtest_simulator/foo.py': 100}
    (tmp_path / '.github' / 'module_budgets.json').write_text(json.dumps(head), encoding='utf-8')
    base_file = tmp_path / 'base.json'
    base_file.write_text(json.dumps(base), encoding='utf-8')
    body_file = tmp_path / 'body.txt'
    body_file.write_text('No marker present.', encoding='utf-8')
    script = _clone_script_into(tmp_path, 'check_budget_ratchet.py')
    result = _run(script, '--base-file', str(base_file), '--pr-body-file', str(body_file), cwd=tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'BUDGET RATCHET GATE -- FAIL' in result.stderr
    assert 'backtest_simulator/foo.py' in result.stderr


def test_ruff_complexity_mutation_fires(tmp_path: Path) -> None:
    # A function with cyclomatic complexity > 10 (enough nested branches).
    body = 'def too_complex(x: int) -> int:\n    y: int = 0\n'
    for i in range(15):
        body += f'    if x == {i}:\n        y += {i}\n'
    body += '    return y\n'
    target = tmp_path / 'mutation.py'
    target.write_text(body, encoding='utf-8')
    result = subprocess.run(
        [sys.executable, '-m', 'ruff', 'check', '--isolated',
         '--select', 'C901', str(target)],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert 'C901' in result.stdout + result.stderr


def test_ruff_print_mutation_fires(tmp_path: Path) -> None:
    target = tmp_path / 'mutation.py'
    target.write_text('def f() -> None:\n    print("hello")\n', encoding='utf-8')
    result = subprocess.run(
        [sys.executable, '-m', 'ruff', 'check', '--isolated', '--select', 'T201', str(target)],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert 'T201' in result.stdout + result.stderr


def test_ruff_todo_mutation_fires(tmp_path: Path) -> None:
    target = tmp_path / 'mutation.py'
    target.write_text('# TODO: implement me\nx = 1\n', encoding='utf-8')
    result = subprocess.run(
        [sys.executable, '-m', 'ruff', 'check', '--isolated', '--select', 'FIX002', str(target)],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert 'FIX002' in result.stdout + result.stderr


def test_ruff_commented_code_mutation_fires(tmp_path: Path) -> None:
    target = tmp_path / 'mutation.py'
    # ERA001 catches commented-out code. A pattern like `# x = 1` is flagged.
    target.write_text(textwrap.dedent('''
        def f() -> None:
            # y = compute_something(x, y, z)
            return None
    ''').strip() + '\n', encoding='utf-8')
    result = subprocess.run(
        [sys.executable, '-m', 'ruff', 'check', '--isolated', '--select', 'ERA001', str(target)],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert 'ERA001' in result.stdout + result.stderr


def test_vulture_mutation_fires(tmp_path: Path) -> None:
    source = tmp_path / 'backtest_simulator'
    source.mkdir()
    (source / '__init__.py').write_text('', encoding='utf-8')
    # Vulture assigns ≥ 90% confidence to unused imports; the ≥ 80%
    # threshold must catch that class. Unused variables and functions
    # sit at 60% and are intentionally below our CI threshold — they
    # require higher confidence to avoid false positives on the
    # production codebase.
    (source / 'bait.py').write_text(
        '"""Bait module."""\n\nimport json  # deliberately unused\n',
        encoding='utf-8',
    )
    result = subprocess.run(
        [sys.executable, '-m', 'vulture', 'backtest_simulator/', '--min-confidence', '80'],
        check=False, capture_output=True, text=True, cwd=tmp_path,
    )
    assert result.returncode != 0, f'vulture did not flag: {result.stdout!r}'
    assert 'json' in result.stdout
