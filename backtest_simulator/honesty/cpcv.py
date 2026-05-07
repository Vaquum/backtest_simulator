"""CPCV — Combinatorially Purged Cross-Validation paths."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import combinations


@dataclass(frozen=True)
class CpcvPath:

    path_id: int
    train_groups: tuple[int, ...]
    test_groups: tuple[int, ...]
    purge_seconds: int
    embargo_seconds: int

@dataclass
class CpcvPaths:

    _paths: tuple[CpcvPath, ...] = field(default_factory=tuple)

    @classmethod
    def build(
        cls,
        *,
        n_groups: int,
        n_test_groups: int,
        purge_seconds: int,
        embargo_seconds: int,
    ) -> CpcvPaths:
        if n_groups < 2:
            msg = (
                f'CpcvPaths.build: n_groups must be >= 2 (got {n_groups}). '
                f'CPCV is undefined on a single block — fall back to a '
                f'plain holdout split.'
            )
            raise ValueError(msg)
        if n_test_groups < 1 or n_test_groups >= n_groups:
            msg = (
                f'CpcvPaths.build: n_test_groups must be in '
                f'[1, n_groups - 1]; got {n_test_groups} of {n_groups}.'
            )
            raise ValueError(msg)
        if purge_seconds < 0 or embargo_seconds < 0:
            msg = (
                f'CpcvPaths.build: purge_seconds and embargo_seconds must '
                f'be non-negative; got purge={purge_seconds}, '
                f'embargo={embargo_seconds}.'
            )
            raise ValueError(msg)
        all_groups = list(range(n_groups))
        paths: list[CpcvPath] = []
        for path_id, test_groups in enumerate(
            combinations(all_groups, n_test_groups),
        ):
            train_groups = tuple(g for g in all_groups if g not in test_groups)
            paths.append(CpcvPath(
                path_id=path_id,
                train_groups=train_groups,
                test_groups=test_groups,
                purge_seconds=purge_seconds,
                embargo_seconds=embargo_seconds,
            ))
        return cls(_paths=tuple(paths))

    def __iter__(self) -> Iterator[CpcvPath]:
        return iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)

    def paths(self) -> tuple[CpcvPath, ...]:
        return self._paths
