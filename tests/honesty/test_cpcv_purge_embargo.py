"""Honesty gate: CpcvPaths enumerate purge+embargo-aware splits.

Pins slice #17 Task 17a / SPEC §9.5 statistical-honesty sub-rule.
"""
from __future__ import annotations

import math

import pytest

from backtest_simulator.honesty.cpcv import CpcvPath, CpcvPaths


def test_cpcv_paths_purge_and_embargo() -> None:
    """`build` enumerates C(n_groups, n_test_groups) paths.

    Each path has the right train/test partition + carries the
    purge/embargo as supplied. With n_groups=6, n_test_groups=2,
    there are C(6, 2) = 15 paths.
    """
    paths = CpcvPaths.build(
        n_groups=6,
        n_test_groups=2,
        purge_seconds=120,
        embargo_seconds=60,
    )
    expected_n = math.comb(6, 2)  # 15
    assert len(paths) == expected_n, (
        f'expected {expected_n} paths for C(6, 2); got {len(paths)}'
    )

    seen_pairs: set[tuple[int, ...]] = set()
    for i, p in enumerate(paths):
        assert isinstance(p, CpcvPath)
        assert p.path_id == i, (
            f'path #{i} has path_id={p.path_id}; ids must be sequential'
        )
        assert len(p.test_groups) == 2
        assert len(p.train_groups) == 4
        assert set(p.test_groups).isdisjoint(set(p.train_groups)), (
            f'path {p.path_id}: train and test groups overlap '
            f'(train={p.train_groups}, test={p.test_groups})'
        )
        assert p.purge_seconds == 120
        assert p.embargo_seconds == 60
        seen_pairs.add(p.test_groups)

    assert len(seen_pairs) == expected_n, (
        f'CpcvPaths.build produced duplicate test-group partitions; '
        f'unique={len(seen_pairs)}, expected={expected_n}'
    )


def test_cpcv_paths_rejects_invalid_n_groups() -> None:
    """`n_groups < 2` is undefined for CPCV — raise loud."""
    with pytest.raises(ValueError, match='n_groups must be >= 2'):
        CpcvPaths.build(
            n_groups=1, n_test_groups=1, purge_seconds=0, embargo_seconds=0,
        )


def test_cpcv_paths_rejects_invalid_n_test_groups() -> None:
    """`n_test_groups` must be in [1, n_groups - 1]."""
    with pytest.raises(ValueError, match='n_test_groups must be in'):
        CpcvPaths.build(
            n_groups=4, n_test_groups=0, purge_seconds=0, embargo_seconds=0,
        )
    with pytest.raises(ValueError, match='n_test_groups must be in'):
        CpcvPaths.build(
            n_groups=4, n_test_groups=4, purge_seconds=0, embargo_seconds=0,
        )


def test_cpcv_paths_rejects_negative_purge_or_embargo() -> None:
    """purge_seconds and embargo_seconds must be non-negative."""
    with pytest.raises(ValueError, match='must be non-negative'):
        CpcvPaths.build(
            n_groups=4, n_test_groups=1,
            purge_seconds=-1, embargo_seconds=0,
        )
    with pytest.raises(ValueError, match='must be non-negative'):
        CpcvPaths.build(
            n_groups=4, n_test_groups=1,
            purge_seconds=0, embargo_seconds=-1,
        )
