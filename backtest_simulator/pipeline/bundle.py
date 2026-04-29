"""Limen bundle loader: `<name>__rNNNN.zip` -> ExperimentFile shim + CSV pool."""
from __future__ import annotations

import argparse
import ast
import contextlib
import dataclasses
import fcntl
import importlib.util
import json
import sys
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import cast

import limen
from limen.experiment import Manifest as LimenManifest
from limen.experiment.manifest_core import DataSourceConfig

from backtest_simulator.pipeline.experiment import ExperimentFile

_RUNTIME_BODY_MARKERS: frozenset[str] = frozenset({
    'HistoricalData',
    'get_spot_klines',
    'UniversalExperimentLoop',
    'run',
})


@dataclass(frozen=True)
class BundleSpec:
    bundle_path: Path
    work_dir: Path
    py_path: Path
    json_path: Path
    csv_path: Path
    sfd_identifier: str
    data_source: dict[str, object]
    uel_run: dict[str, object]


def extract_bundle(bundle_zip: Path, work_dir: Path) -> BundleSpec:
    """Unzip + parse JSON. Validates the three-file shape and required keys."""
    bundle_zip = bundle_zip.expanduser().resolve()
    if not bundle_zip.is_file():
        msg = f'extract_bundle: bundle zip not found: {bundle_zip}'
        raise FileNotFoundError(msg)
    work_dir = work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_zip) as zf:
        zf.extractall(work_dir)
        names = zf.namelist()
    py_names = [n for n in names if n.endswith('.py')]
    json_names = [n for n in names if n.endswith('.json')]
    csv_names = [n for n in names if n.endswith('.csv')]
    if len(py_names) != 1 or len(json_names) != 1 or len(csv_names) != 1:
        msg = (
            f'extract_bundle: {bundle_zip} must contain exactly one .py, '
            f'one .json, and one .csv; got py={py_names!r} '
            f'json={json_names!r} csv={csv_names!r}'
        )
        raise ValueError(msg)
    py_path = work_dir / py_names[0]
    json_path = work_dir / json_names[0]
    csv_path = work_dir / csv_names[0]
    meta = json.loads(json_path.read_text(encoding='utf-8'))
    sfd_identifier = meta.get('sfd_identifier')
    if not isinstance(sfd_identifier, str) or not sfd_identifier:
        msg = (
            f'extract_bundle: {json_path}::sfd_identifier must be a '
            f'non-empty string, got {sfd_identifier!r}'
        )
        raise ValueError(msg)
    # Identifier flows into generated Python source for the shim;
    # validate it's a real Python identifier so a malformed bundle
    # cannot inject arbitrary code (`"Foo; rm -rf /"`).
    if not sfd_identifier.isidentifier():
        msg = (
            f'extract_bundle: {json_path}::sfd_identifier must be a valid '
            f'Python identifier, got {sfd_identifier!r}'
        )
        raise ValueError(msg)
    data_source = meta.get('data_source')
    if not isinstance(data_source, dict):
        msg = (
            f'extract_bundle: {json_path}::data_source must be a JSON '
            f'object, got {type(data_source).__name__}'
        )
        raise ValueError(msg)
    uel_run = meta.get('uel_run')
    if not isinstance(uel_run, dict):
        msg = (
            f'extract_bundle: {json_path}::uel_run must be a JSON object, '
            f'got {type(uel_run).__name__}'
        )
        raise ValueError(msg)
    return BundleSpec(
        bundle_path=bundle_zip,
        work_dir=work_dir,
        py_path=py_path,
        json_path=json_path,
        csv_path=csv_path,
        sfd_identifier=sfd_identifier,
        data_source=cast('dict[str, object]', data_source),
        uel_run=cast('dict[str, object]', uel_run),
    )


def _strip_runtime_body(source: str) -> str:
    """Cut at first top-level Call to a runtime-body marker; drop the rest."""
    tree = ast.parse(source)
    kept: list[ast.stmt] = []
    for stmt in tree.body:
        if _is_runtime_body_stmt(stmt):
            break
        kept.append(stmt)
    new_tree = ast.Module(body=kept, type_ignores=[])
    return ast.unparse(new_tree)


def _is_runtime_body_stmt(stmt: ast.stmt) -> bool:
    # FunctionDef / AsyncFunctionDef / ClassDef / Lambda are
    # definitions, not runtime side effects — their bodies fire on
    # CALL, not on module import. The cut must look only at statements
    # whose execution at import time runs arbitrary code (Expr,
    # Assign with Call value, AugAssign, AnnAssign with Call value, etc.).
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        # Decorator expressions DO run at import; check those alone.
        for dec in stmt.decorator_list:
            if _expr_calls_runtime_marker(dec):
                return True
        return False
    # Walk the statement, but stop descending when we hit a nested
    # def / class / lambda — its body is scoped, not runtime.
    return _expr_calls_runtime_marker(stmt)


def _expr_calls_runtime_marker(node: ast.AST) -> bool:
    """Walk `node` for a runtime-marker Call without entering nested scopes."""
    stack: list[ast.AST] = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, ast.Call):
            func = current.func
            if isinstance(func, ast.Attribute) and func.attr in _RUNTIME_BODY_MARKERS:
                return True
            if isinstance(func, ast.Name) and func.id in _RUNTIME_BODY_MARKERS:
                return True
        for child in ast.iter_child_nodes(current):
            if isinstance(child, (
                ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
            )):
                continue
            stack.append(child)
    return False


def load_bundled_experiment(spec: BundleSpec) -> ExperimentFile:
    """Exec stripped .py in a fresh module; expose SFD's params + JSON-overridden manifest."""
    module = _exec_stripped_module(spec)
    sfd_class = _get_sfd_class(module, spec)
    params_fn_raw, base_manifest_fn = _get_sfd_callables(sfd_class, spec)

    def validated_params() -> dict[str, list[object]]:
        return _validate_params_shape(params_fn_raw(), spec.sfd_identifier)

    def manifest_with_override() -> object:
        base = base_manifest_fn()
        if not isinstance(base, LimenManifest):
            msg = (
                f'{spec.sfd_identifier}.manifest() must return Manifest, '
                f'got {type(base).__name__}'
            )
            raise TypeError(msg)
        return _override_data_source(base, spec.data_source)

    return ExperimentFile(
        source_path=spec.py_path,
        module=module,
        params=validated_params,
        manifest=manifest_with_override,
    )


def _exec_stripped_module(spec: BundleSpec) -> ModuleType:
    stripped = _strip_runtime_body(spec.py_path.read_text(encoding='utf-8'))
    module_name = f'_bts_bundle_{spec.py_path.stem.replace("-", "_")}'
    spec_obj = importlib.util.spec_from_loader(module_name, loader=None)
    if spec_obj is None:
        msg = f'load_bundled_experiment: failed to build module spec for {module_name}'
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec_obj)
    sys.modules[module_name] = module
    exec(compile(stripped, str(spec.py_path), 'exec'), module.__dict__)
    return module


def _get_sfd_class(module: ModuleType, spec: BundleSpec) -> object:
    sfd_class = getattr(module, spec.sfd_identifier, None)
    if sfd_class is None:
        msg = (
            f'SFD class {spec.sfd_identifier!r} not found in {spec.py_path}'
        )
        raise AttributeError(msg)
    return sfd_class


def _get_sfd_callables(
    sfd_class: object, spec: BundleSpec,
) -> tuple[Callable[[], object], Callable[[], object]]:
    params_fn = getattr(sfd_class, 'params', None)
    manifest_fn = getattr(sfd_class, 'manifest', None)
    if not callable(params_fn) or not callable(manifest_fn):
        msg = (
            f'{spec.sfd_identifier} must expose callable `params` and `manifest`'
        )
        raise TypeError(msg)
    return (
        cast('Callable[[], object]', params_fn),
        cast('Callable[[], object]', manifest_fn),
    )


def _validate_params_shape(
    result: object, sfd_identifier: str,
) -> dict[str, list[object]]:
    if not isinstance(result, dict):
        msg = (
            f'{sfd_identifier}.params() must return dict, '
            f'got {type(result).__name__}'
        )
        raise TypeError(msg)
    typed: dict[str, list[object]] = {}
    for k, v in cast('dict[object, object]', result).items():
        if not isinstance(k, str):
            msg = f'{sfd_identifier}.params() keys must be str'
            raise TypeError(msg)
        if not isinstance(v, list):
            msg = (
                f'{sfd_identifier}.params()[{k!r}] must be list, '
                f'got {type(v).__name__}'
            )
            raise TypeError(msg)
        typed[k] = cast('list[object]', v)
    return typed


def _override_data_source(
    base: LimenManifest, override: dict[str, object],
) -> LimenManifest:
    method_name = override.get('method')
    if not isinstance(method_name, str) or not method_name:
        msg = (
            f'data_source.method must be a non-empty string, '
            f'got {method_name!r}'
        )
        raise ValueError(msg)
    raw_params = override.get('params', {})
    if not isinstance(raw_params, dict):
        msg = (
            f'data_source.params must be a JSON object, '
            f'got {type(raw_params).__name__}'
        )
        raise ValueError(msg)
    historical = limen.HistoricalData()
    method = getattr(historical, method_name, None)
    if not callable(method):
        msg = f'limen.HistoricalData has no method {method_name!r}'
        raise AttributeError(msg)
    return dataclasses.replace(
        base,
        data_source_config=DataSourceConfig(
            method=method,
            params=cast('dict[str, object]', raw_params),
        ),
    )


# Single source of truth: the shim source the CLI writes to disk imports
# `_override_data_source` and calls it, so the in-memory and CLI paths
# share the same code (no f-string-templated lookalike to drift).
_SHIM_TEMPLATE: str = '''
from backtest_simulator.pipeline.bundle import _override_data_source as _bts_override

_BTS_DATA_SOURCE = {data_source!r}
_BTS_SFD_CLASS = {sfd_identifier}

params = _BTS_SFD_CLASS.params

def manifest():
    return _bts_override(_BTS_SFD_CLASS.manifest(), _BTS_DATA_SOURCE)
'''


def materialize_bundle_for_cli(
    bundle_zip: Path, work_dir: Path,
) -> tuple[Path, Path, BundleSpec]:
    """Extract a bundle, write a synthetic shim, return (shim_path, csv_path, spec).

    Concurrent `bts run --bundle X` invocations race on extract+shim
    writes; serialize via `fcntl.LOCK_EX` on a sibling lock file. The
    spec is returned alongside the paths so the caller does not need
    to re-call `extract_bundle` outside the lock to recover JSON
    metadata (which would race with the very serialization this
    helper guarantees).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    lock_path = work_dir / '.bundle.lock'
    with _exclusive_dir_lock(lock_path):
        spec = extract_bundle(bundle_zip, work_dir)
        stripped = _strip_runtime_body(spec.py_path.read_text(encoding='utf-8'))
        shim_source = stripped + _SHIM_TEMPLATE.format(
            data_source=spec.data_source,
            sfd_identifier=spec.sfd_identifier,
        )
        shim_path = work_dir / f'{spec.py_path.stem}__bts_shim.py'
        shim_path.write_text(shim_source, encoding='utf-8')
        return shim_path, spec.csv_path, spec


@contextlib.contextmanager
def _exclusive_dir_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, 'w') as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


# Sentinel default for --n-permutations on both run.py and sweep.py.
# argparse stores `None` when the flag is unsupplied; the resolver
# below picks the bundle's value (if any) or the canonical fallback.
# The operator's explicit value (any int) is identity-comparable
# against `None`, so `--n-permutations 30` (operator chose 30) is
# distinguishable from operator silence.
N_PERMUTATIONS_DEFAULT: int = 30


def materialize_bundle_on_args(
    args: argparse.Namespace, work_root: Path,
) -> None:
    """If --bundle was supplied, populate args from the bundle's JSON.

    Resolution order for `n_permutations`:
      1. Operator explicit `--n-permutations <int>` wins.
      2. Otherwise, bundle's JSON `uel_run.n_permutations` wins.
      3. Otherwise, fall back to `N_PERMUTATIONS_DEFAULT`.

    The parser's default is `None`; the resolver below applies the
    fallback. This sentinel-default pattern is the only way to
    distinguish operator silence from operator explicit-equal-to-default.
    """
    bundle = getattr(args, 'bundle', None)
    if bundle is None:
        if getattr(args, 'n_permutations', None) is None:
            args.n_permutations = N_PERMUTATIONS_DEFAULT
        return
    bundle_path = Path(bundle).expanduser().resolve()
    cache_dir = work_root / 'bundles' / bundle_path.stem
    shim_path, csv_path, spec = materialize_bundle_for_cli(bundle_path, cache_dir)
    args.exp_code = shim_path
    if getattr(args, 'input_from_file', None) is None:
        args.input_from_file = str(csv_path)
    cli_n_perm = getattr(args, 'n_permutations', None)
    if cli_n_perm is None:
        json_n_perm = spec.uel_run.get('n_permutations')
        if isinstance(json_n_perm, int):
            args.n_permutations = json_n_perm
        else:
            args.n_permutations = N_PERMUTATIONS_DEFAULT
