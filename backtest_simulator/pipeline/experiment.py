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

    @staticmethod
    def load_from_file(path: Path) -> ExperimentFile:
        source_path = Path(path).resolve()
        if not source_path.is_file():
            msg = f'experiment file not found: {source_path}'
            raise FileNotFoundError(msg)
        spec = importlib.util.spec_from_file_location(source_path.stem, source_path)
        if spec is None or spec.loader is None:
            msg = f'could not build import spec for {source_path}'
            raise ImportError(msg)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module.__name__] = module
        spec.loader.exec_module(module)
        params_fn_raw = getattr(module, 'params', None)
        manifest_fn_raw = getattr(module, 'manifest', None)
        if not callable(params_fn_raw) or not callable(manifest_fn_raw):
            msg = f'experiment file {source_path} must define callable `params` and `manifest` at module level'
            raise ValueError(msg)

        def params_fn() -> dict[str, list[object]]:
            result: object = params_fn_raw()
            if not isinstance(result, dict):
                msg = f'experiment file {source_path}: params() must return dict[str, list[object]], got {type(result).__name__}'
                raise TypeError(msg)
            typed_result = cast('Mapping[object, object]', result)
            typed: dict[str, list[object]] = {}
            for raw_key, raw_value in typed_result.items():
                if not isinstance(raw_key, str):
                    msg = f'experiment file {source_path}: params() keys must be str, got {type(raw_key).__name__}={raw_key!r}'
                    raise TypeError(msg)
                if not isinstance(raw_value, list):
                    msg = f'experiment file {source_path}: params()[{raw_key!r}] must be a list, got {type(raw_value).__name__}'
                    raise TypeError(msg)
                value_list: list[object] = list(cast('list[object]', raw_value))
                typed[raw_key] = value_list
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
        if not results_path.is_file():
            msg = f'results.csv not found at {results_path}; run the experiment first'
            raise FileNotFoundError(msg)
        df = pl.read_csv(results_path)
        if 'round_params' not in df.columns:
            return df
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
        if not parsed:
            return df
        keys: list[str] = sorted({k for row in parsed for k in row})
        new_cols: dict[str, list[object]] = {k: [row.get(k) for row in parsed] for k in keys if k not in df.columns}
        return df.with_columns([pl.Series(k, v) for k, v in new_cols.items()])

    def train(self, permutation_ids: Iterable[int]) -> list[Sensor]:
        ids = sorted({int(pid) for pid in permutation_ids})
        if not ids:
            msg = 'train requires at least one permutation_id'
            raise ValueError(msg)
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
