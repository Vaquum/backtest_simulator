#!/usr/bin/env python3
"""Module line-count budget gate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUDGET_PATH = REPO_ROOT / '.github' / 'module_budgets.json'


def count_significant_lines(path: Path) -> int:
    # Budgets are measured against non-blank, non-comment-only lines.
    count = 0
    for line in path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            count += 1
    return count


def load_budgets() -> dict[str, int]:
    if not BUDGET_PATH.is_file():
        print('MODULE BUDGET GATE -- FAIL', file=sys.stderr)
        print(f'  missing budget file: {BUDGET_PATH}', file=sys.stderr)
        sys.exit(2)
    data = json.loads(BUDGET_PATH.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        print('MODULE BUDGET GATE -- FAIL', file=sys.stderr)
        print('  budget file is not a JSON object', file=sys.stderr)
        sys.exit(2)
    return {str(k): int(v) for k, v in data.items()}


def check(budgets: dict[str, int]) -> list[tuple[str, int, int]]:
    violations: list[tuple[str, int, int]] = []
    for rel_path, budget in budgets.items():
        path = REPO_ROOT / rel_path
        if not path.is_file():
            continue
        actual = count_significant_lines(path)
        if actual > budget:
            violations.append((rel_path, actual, budget))
    return violations


def main() -> int:
    budgets = load_budgets()
    violations = check(budgets)
    if violations:
        print('MODULE BUDGET GATE -- FAIL', file=sys.stderr)
        print('', file=sys.stderr)
        for rel_path, actual, budget in violations:
            overage = actual - budget
            print(
                f'  - {rel_path}: {actual} lines '
                f'(budget {budget}, overage +{overage})',
                file=sys.stderr,
            )
        print('', file=sys.stderr)
        print(
            f'{len(violations)} module(s) over budget. Merge blocked.',
            file=sys.stderr,
        )
        return 1
    print('MODULE BUDGET GATE -- PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
