"""Minimal stub for limen exports we use.

The real `UniversalExperimentLoop.run` accepts `Callable | None` /
`dict | None` parameters whose Any propagation flags the call site
as partially-unknown. This stub pins the concrete signature for the
keyword-args we actually pass.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class HistoricalData:
    data: Any  # noqa: ANN401 - Polars DataFrame; we cast at consumer boundary
    data_columns: list[str]

    def __init__(self) -> None: ...

    def get_spot_klines(
        self,
        n_rows: int | None = ...,
        kline_size: int = ...,
        start_date_limit: str | None = ...,
    ) -> Any: ...  # noqa: ANN401


class Sensor:
    def __init__(self, *args: object, **kwargs: object) -> None: ...
    def predict(self, *args: object, **kwargs: object) -> Any: ...  # noqa: ANN401


class Trainer:
    def __init__(self, *, experiment_dir: str | Path) -> None: ...
    def train(self, permutation_ids: object) -> list[Sensor]: ...


class UniversalExperimentLoop:
    def __init__(
        self,
        *,
        sfd: object,
        search_strategy: object,
        experiment_dir: str | Path,
    ) -> None: ...

    def run(
        self,
        *,
        experiment_name: str,
        n_permutations: int = ...,
        resume: bool = ...,
    ) -> None: ...
