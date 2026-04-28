#!/usr/bin/env python3
"""Module line-count budget gate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUDGET_PATH = REPO_ROOT / '.github' / 'module_budgets.json'
SOURCE_DIR = REPO_ROOT / 'backtest_simulator'


def count_significant_lines(path: Path) -> int:
    # Budgets are measured against non-blank, non-comment-only lines.
    count = 0
    for line in path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            count += 1
    return count


def _fail_setup(message: str) -> None:
    print('MODULE BUDGET GATE -- FAIL', file=sys.stderr)
    print(f'  {message}', file=sys.stderr)
    sys.exit(2)


def load_budgets() -> dict[str, int]:
    if not BUDGET_PATH.is_file():
        _fail_setup(f'missing budget file: {BUDGET_PATH}')
    try:
        raw = json.loads(BUDGET_PATH.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError) as exc:
        _fail_setup(f'cannot read {BUDGET_PATH}: {exc}')
        raise  # unreachable; _fail_setup sys.exits
    if not isinstance(raw, dict):
        _fail_setup(f'{BUDGET_PATH} is not a JSON object')
    parsed: dict[str, int] = {}
    for key, value in raw.items():
        if not isinstance(value, int) or value <= 0:
            _fail_setup(f'{BUDGET_PATH}["{key}"] must be positive int, got {value!r}')
        parsed[str(key)] = value
    return parsed


def check(budgets: dict[str, int]) -> tuple[list[tuple[str, int, int]], list[str]]:
    # Returns (over_budget, unbudgeted_source_paths). Any path under
    # backtest_simulator/ that is NOT declared in module_budgets.json is
    # a violation — otherwise a new module could silently escape the gate.
    over: list[tuple[str, int, int]] = []
    for rel_path, budget in budgets.items():
        path = REPO_ROOT / rel_path
        if not path.is_file():
            continue
        actual = count_significant_lines(path)
        if actual > budget:
            over.append((rel_path, actual, budget))
    unbudgeted: list[str] = []
    if SOURCE_DIR.is_dir():
        declared = {key for key in budgets if key.startswith('backtest_simulator/')}
        for path in sorted(SOURCE_DIR.rglob('*.py')):
            rel = str(path.relative_to(REPO_ROOT))
            if rel not in declared:
                unbudgeted.append(rel)
    return over, unbudgeted


def main() -> int:
    budgets = load_budgets()
    over, unbudgeted = check(budgets)
    if over or unbudgeted:
        print('MODULE BUDGET GATE -- FAIL', file=sys.stderr)
        print('', file=sys.stderr)
        for rel_path, actual, budget in over:
            overage = actual - budget
            print(
                f'  - {rel_path}: {actual} lines '
                f'(budget {budget}, overage +{overage})',
                file=sys.stderr,
            )
        for rel_path in unbudgeted:
            print(
                f'  - {rel_path}: undeclared module; '
                f'add an entry to .github/module_budgets.json',
                file=sys.stderr,
            )
        print('', file=sys.stderr)
        total = len(over) + len(unbudgeted)
        print(f'{total} violation(s). Merge blocked.', file=sys.stderr)
        return 1
    print('MODULE BUDGET GATE -- PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
