#!/usr/bin/env python3
"""File size balance gate: largest <= MAX_RATIO x median."""
from __future__ import annotations

import statistics
import sys
from pathlib import Path
from typing import Final

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / 'backtest_simulator'

# 16.00 accommodates:
#   (a) Protocol-conformance files (SimulatedVenueAdapter implements the
#       15-method praxis.infrastructure.venue_adapter.VenueAdapter Protocol;
#       runtime_checkable isinstance() requires those methods to live on
#       the class itself, not helpers).
#   (b) BacktestLauncher in launcher/launcher.py: a real praxis.Launcher
#       subclass that overrides the full boot/shutdown path plus the
#       freezegun+Timer+asyncio integration. Every override reads from
#       the same launcher instance state, so splitting across sibling
#       modules breaks the framework's state plumbing.
#   (c) action_submitter's `_to_praxis_enums` enum-bridge: every
#       Nexus-to-Praxis enum pair is expressed inline so pyright can
#       type-narrow the conversion on each branch.
# Private helpers do live in sibling `_*` modules where the class/file
# boundary is natural (e.g. `venue/_adapter_internals.py`,
# `honesty/capital.py`'s `_PendingLifecycle`); the ratio cap is set
# against the residual Protocol-sized modules.
MAX_RATIO: Final[float] = 16.00
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
    # Exclude zero-line files from the median: a package with many
    # empty __init__.py files would otherwise produce median=0 and
    # ratio=inf for the smallest real file. Empty files don't have a
    # meaningful size to balance against.
    nonzero_sizes = [s for _, s in sized if s > 0]
    if len(nonzero_sizes) < MIN_FILES_FOR_GATE:
        print(
            f'FILE SIZE BALANCE GATE -- PASS '
            f'(vacuous: only {len(nonzero_sizes)} non-empty source file(s))'
        )
        return 0
    largest_path, largest_size = max(sized, key=lambda item: item[1])
    median = statistics.median(nonzero_sizes)
    ratio = largest_size / median
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
