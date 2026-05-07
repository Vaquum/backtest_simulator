"""ExperimentPipeline — load user experiment file, run Limen UEL, filter results."""
from __future__ import annotations

import importlib.util
import json
import logging
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import polars as pl
from limen import Sensor, Trainer, UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import RandomStrategy
from limen.sfd import logreg_binary

_log = logging.getLogger(__name__)
FilterValue = float | int | str | bool | tuple[float, float] | set[object] | frozenset[object]
FilterCriteria = dict[str, FilterValue]

@dataclass(frozen=True)
class ExperimentFile:
    source_path: Path
    module: ModuleType
    params: Callable[[], dict[str, list[object]]]
    manifest: Callable[[], object]

class ExperimentPipeline:

    def __init__(self, experiment_dir: Path, sfd: object = logreg_binary) -> None:
        self._experiment_dir = Path(experiment_dir).resolve()
        self._sfd = sfd

    @staticmethod
    def load_from_file(path: Path) -> ExperimentFile:
        source_path = Path(path).resolve()
        spec = importlib.util.spec_from_file_location(source_path.stem, source_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module.__name__] = module
        spec.loader.exec_module(module)
        params_fn_raw = cast('Callable[[], object]', getattr(module, 'params'))
        manifest_fn_raw = cast('Callable[[], object]', getattr(module, 'manifest'))

        def params_fn() -> dict[str, list[object]]:
            result: object = params_fn_raw()
            typed_result = cast('Mapping[object, object]', result)
            typed: dict[str, list[object]] = {}
            for raw_key, raw_value in typed_result.items():
                value_list: list[object] = list(cast('list[object]', raw_value))
                typed[cast('str', raw_key)] = value_list
            return typed

        def manifest_fn() -> object:
            return manifest_fn_raw()
        return ExperimentFile(source_path=source_path, module=module, params=params_fn, manifest=manifest_fn)

    def run(self, experiment_file: ExperimentFile, experiment_name: str, n_permutations: int=100, *, seed: int=42, resume: bool=False) -> Path:
        self._experiment_dir.mkdir(parents=True, exist_ok=True)
        domain = ParamDomain(experiment_file.params())
        strategy = RandomStrategy(domain, seed=seed)
        uel = UniversalExperimentLoop(sfd=experiment_file.module, search_strategy=strategy, experiment_dir=self._experiment_dir)
        _run_uel(uel, experiment_name, n_permutations, resume=resume)
        _log.info('experiment finished', extra={'dir': str(self._experiment_dir), 'n_permutations': n_permutations})
        return self._experiment_dir

    def read_results(self) -> pl.DataFrame:
        results_path = self._experiment_dir / 'results.csv'
        df = pl.read_csv(results_path)
        parsed: list[dict[str, object]] = []
        for s in df['round_params']:
            if isinstance(s, str):
                obj: object = json.loads(s)
                if isinstance(obj, dict):
                    typed_obj = cast('Mapping[object, object]', obj)
                    parsed.append({str(k): v for k, v in typed_obj.items()})
                else:
                    parsed.append({})
            else:
                parsed.append({})
        keys: list[str] = sorted({k for row in parsed for k in row})
        new_cols: dict[str, list[object]] = {k: [row.get(k) for row in parsed] for k in keys if k not in df.columns}
        return df.with_columns([pl.Series(k, v) for k, v in new_cols.items()])

    def train(self, permutation_ids: Iterable[int]) -> list[Sensor]:
        ids = sorted({int(pid) for pid in permutation_ids})
        trainer = Trainer(experiment_dir=self._experiment_dir)
        sensors = trainer.train(ids)
        _log.info('trained sensors', extra={'count': len(sensors), 'permutation_ids': ids})
        return sensors

    @property
    def experiment_dir(self) -> Path:
        return self._experiment_dir

class _UELRunProtocol(Protocol):

    def __call__(self, *, experiment_name: str, n_permutations: int, resume: bool) -> None:
        ...

def _run_uel(uel: UniversalExperimentLoop, experiment_name: str, n_permutations: int, *, resume: bool) -> None:
    run_fn = cast('_UELRunProtocol', uel.run)
    run_fn(experiment_name=experiment_name, n_permutations=n_permutations, resume=resume)
