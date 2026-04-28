"""Minimal stub for limen exports we use.

The real `UniversalExperimentLoop.run` accepts `Callable | None` /
`dict | None` parameters whose Any propagation flags the call site
as partially-unknown. This stub pins the concrete signature for the
keyword-args we actually pass.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from polars import DataFrame as _DataFrame


class HistoricalData:
    data: _DataFrame
    data_columns: list[str]

    def __init__(self) -> None: ...

    def get_spot_klines(
        self,
        n_rows: int | None = ...,
        kline_size: int = ...,
        start_date_limit: str | None = ...,
    ) -> _DataFrame: ...


class Sensor:
    def __init__(self, *args: object, **kwargs: object) -> None: ...
    def predict(self, *args: object, **kwargs: object) -> Any: ...  # noqa: ANN401


class Trainer:
    # The real Limen Trainer sets `_manifest` and `_round_data`
    # in `__init__`. We expose them here so the bts sweep can read
    # the manifest config + per-permutation params without
    # reaching past Limen's underscore at the call site (which
    # pyright would flag as `reportPrivateUsage`).
    _manifest: Any  # noqa: ANN401 - Limen's Manifest; consumed via .data_source_config / .split_config
    _round_data: dict[int, dict[str, Any]]  # noqa: ANN401 - per-pid run params dict

    def __init__(
        self,
        experiment_dir: str | Path,
        data: Any = ...,  # noqa: ANN401 - Polars DataFrame | None
    ) -> None: ...
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
