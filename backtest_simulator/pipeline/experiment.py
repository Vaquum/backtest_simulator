"""ExperimentPipeline — load user experiment file, run Limen UEL, filter results."""
from __future__ import annotations

import importlib.util
import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import polars as pl
from limen import Sensor, Trainer, UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import RandomStrategy
from limen.sfd import logreg_binary

_log = logging.getLogger(__name__)

FilterValue = (
    float | int | str | bool
    | tuple[float, float]
    | set[object] | frozenset[object]
)
FilterCriteria = dict[str, FilterValue]


@dataclass(frozen=True)
class ExperimentFile:
    """Loaded user experiment file with `params()` and `manifest()` callables.

    `params()` returns the grid dict passed to `ParamDomain`.
    `manifest()` returns the experiment manifest Limen consumes; its exact
    shape belongs to Limen, so we carry it as `object` and hand it through.
    """

    source_path: Path
    module: ModuleType
    params: Callable[[], dict[str, object]]
    manifest: Callable[[], object]


class ExperimentPipeline:
    """Three-step pipeline: load → run → filter → train the selected sensors."""

    def __init__(self, experiment_dir: Path, sfd: object = logreg_binary) -> None:
        # `sfd` is Limen's plug-in strategy-family-descriptor — a module or
        # instance exposing the required hooks. Limen itself leaves it
        # untyped; `object` is the honest narrowest Python type for a
        # duck-typed arg.
        self._experiment_dir = Path(experiment_dir).resolve()
        self._sfd = sfd

    @staticmethod
    def load_from_file(path: Path) -> ExperimentFile:
        """Import a user-supplied Python module by path; return its `params`/`manifest`.

        The file must define two module-level callables (module functions OR
        class instances whose methods match):

          def params() -> dict[str, list[object]]
          def manifest() -> limen.experiment.Manifest
        """
        source_path = Path(path).resolve()
        if not source_path.is_file():
            msg = f'experiment file not found: {source_path}'
            raise FileNotFoundError(msg)

        spec = importlib.util.spec_from_file_location(source_path.stem, source_path)
        if spec is None or spec.loader is None:
            msg = f'could not build import spec for {source_path}'
            raise ImportError(msg)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        params_fn = getattr(module, 'params', None)
        manifest_fn = getattr(module, 'manifest', None)
        if not callable(params_fn) or not callable(manifest_fn):
            msg = (
                f'experiment file {source_path} must define callable '
                '`params` and `manifest` at module level'
            )
            raise ValueError(msg)

        return ExperimentFile(
            source_path=source_path, module=module,
            params=params_fn, manifest=manifest_fn,
        )

    def run(
        self,
        experiment_file: ExperimentFile,
        experiment_name: str,
        n_permutations: int = 100,
        *,
        seed: int = 42,
        resume: bool = False,
    ) -> Path:
        """Run UEL via the MSQ path; persists every artifact Trainer needs.

        The legacy UEL path keeps results in-memory and does NOT write
        `metadata.json`, `round_data.jsonl`, or the per-permutation model
        artifacts. Only the MSQ (search-strategy) path writes them, and
        `Trainer` hard-requires all three. So we always go through MSQ
        — a `RandomStrategy` constructed from the user's `params()` dict
        is the thinnest wrapper that unlocks the full persistence.

        Every param-grid coordinate still comes from the user's experiment
        file: `ParamDomain(experiment_file.params())` is authoritative.
        """
        self._experiment_dir.mkdir(parents=True, exist_ok=True)
        domain = ParamDomain(experiment_file.params())
        strategy = RandomStrategy(domain, seed=seed)
        uel = UniversalExperimentLoop(
            sfd=self._sfd, search_strategy=strategy,
            experiment_dir=self._experiment_dir,
        )
        uel.run(
            experiment_name=experiment_name,
            n_permutations=n_permutations,
            resume=resume,
        )
        _log.info(
            'experiment finished',
            extra={'dir': str(self._experiment_dir), 'n_permutations': n_permutations},
        )
        return self._experiment_dir

    def read_results(self) -> pl.DataFrame:
        """Read `results.csv` and flatten `round_params` JSON into top-level columns."""
        results_path = self._experiment_dir / 'results.csv'
        if not results_path.is_file():
            msg = f'results.csv not found at {results_path}; run the experiment first'
            raise FileNotFoundError(msg)

        df = pl.read_csv(results_path)
        if 'round_params' not in df.columns:
            return df

        parsed = [json.loads(s) if isinstance(s, str) else {} for s in df['round_params']]
        if not parsed:
            return df
        keys = sorted({k for row in parsed for k in row})
        new_cols = {
            k: [row.get(k) for row in parsed]
            for k in keys if k not in df.columns
        }
        return df.with_columns([pl.Series(k, v) for k, v in new_cols.items()])

    @staticmethod
    def filter_results(df: pl.DataFrame, criteria: FilterCriteria) -> pl.DataFrame:
        """Apply user filter: {col: value|(lo,hi)|set} — equality | range | membership."""
        result = df
        for column, spec in criteria.items():
            if column not in result.columns:
                msg = f'filter references unknown column {column!r} (available: {result.columns})'
                raise ValueError(msg)
            result = result.filter(_column_predicate(column, spec))
        return result

    def train(self, permutation_ids: Iterable[int]) -> list[Sensor]:
        """Train the selected permutations via Limen Trainer; return Sensor objects."""
        ids = sorted({int(pid) for pid in permutation_ids})
        if not ids:
            msg = 'train requires at least one permutation_id'
            raise ValueError(msg)
        trainer = Trainer(experiment_dir=self._experiment_dir)
        sensors = trainer.train(ids)
        _log.info(
            'trained sensors', extra={'count': len(sensors), 'permutation_ids': ids},
        )
        return sensors

    @property
    def experiment_dir(self) -> Path:
        return self._experiment_dir


def _column_predicate(column: str, spec: FilterValue) -> pl.Expr:
    col = pl.col(column)
    if isinstance(spec, tuple) and len(spec) == 2:
        lo, hi = spec
        return (col >= lo) & (col <= hi)
    if isinstance(spec, (set, frozenset)):
        return col.is_in(list(spec))
    return col == spec
