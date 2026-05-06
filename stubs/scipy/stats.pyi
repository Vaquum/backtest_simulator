"""Minimal scipy.stats stub: only the kurtosis + skew + ttest_ind we use."""
from __future__ import annotations

from collections.abc import Sequence

def kurtosis(
    a: Sequence[float],
    axis: int = 0,
    *,
    bias: bool = True,
    fisher: bool = True,
) -> float: ...


def skew(
    a: Sequence[float],
    axis: int = 0,
    *,
    bias: bool = True,
) -> float: ...
