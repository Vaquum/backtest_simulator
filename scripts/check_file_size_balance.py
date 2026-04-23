#!/usr/bin/env python3
"""File size balance gate: largest <= MAX_RATIO x median."""
from __future__ import annotations

import statistics
import sys
from pathlib import Path
from typing import Final

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / 'backtest_simulator'

MAX_RATIO: Final[float] = 2.50
MIN_FILES_FOR_GATE: Final[int] = 3


def count_lines(path: Path) -> int:
    # Total physical line count, blank lines included (file-size balance
    # cares about actual file size on disk, not logical SLOC).
    return len(path.read_text(encoding='utf-8').splitlines())


def main() -> int:
    if not SOURCE_DIR.is_dir():
        print('FILE SIZE BALANCE GATE -- PASS (vacuous: backtest_simulator/ missing)')
        return 0
    sized: list[tuple[Path, int]] = [
        (p, count_lines(p)) for p in sorted(SOURCE_DIR.rglob('*.py'))
    ]
    if len(sized) < MIN_FILES_FOR_GATE:
        print(
            f'FILE SIZE BALANCE GATE -- PASS '
            f'(vacuous: only {len(sized)} source file(s), need >= {MIN_FILES_FOR_GATE})'
        )
        return 0
    sizes = [s for _, s in sized]
    largest_path, largest_size = max(sized, key=lambda item: item[1])
    median = statistics.median(sizes)
    ratio = largest_size / median if median > 0 else float('inf')
    if ratio > MAX_RATIO:
        print('FILE SIZE BALANCE GATE -- FAIL', file=sys.stderr)
        print('', file=sys.stderr)
        rel = largest_path.relative_to(REPO_ROOT)
        print(f'  largest file:    {rel} ({largest_size} lines)', file=sys.stderr)
        print(f'  median file size: {int(median)} lines', file=sys.stderr)
        print(
            f'  ratio:            {ratio:.2f} (max allowed: {MAX_RATIO:.2f})',
            file=sys.stderr,
        )
        print('', file=sys.stderr)
        print('Merge blocked.', file=sys.stderr)
        return 1
    print(
        f'FILE SIZE BALANCE GATE -- PASS '
        f'(largest={largest_size}, median={int(median)}, ratio={ratio:.2f})'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
