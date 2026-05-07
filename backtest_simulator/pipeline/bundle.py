"""Limen bundle loader: `<name>__rNNNN.zip` -> ExperimentFile shim + CSV pool."""
from __future__ import annotations

import argparse
import ast
import contextlib
import dataclasses
import fcntl
import json
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import limen
from limen.experiment import Manifest as LimenManifest
from limen.experiment.manifest_core import DataSourceConfig

__all__ = ['extract_bundle', 'BundleSpec', 'materialize_bundle_for_cli', 'materialize_bundle_on_args', '_override_data_source', 'N_PERMUTATIONS_DEFAULT']

_RUNTIME_BODY_MARKERS: frozenset[str] = frozenset({'HistoricalData', 'get_spot_klines', 'UniversalExperimentLoop', 'run'})

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
        msg = f'extract_bundle: {bundle_zip} must contain exactly one .py, one .json, and one .csv; got py={py_names!r} json={json_names!r} csv={csv_names!r}'
        raise ValueError(msg)
    py_path = work_dir / py_names[0]
    json_path = work_dir / json_names[0]
    csv_path = work_dir / csv_names[0]
    meta = json.loads(json_path.read_text(encoding='utf-8'))
    sfd_identifier = meta.get('sfd_identifier')
    if not isinstance(sfd_identifier, str) or not sfd_identifier:
        msg = f'extract_bundle: {json_path}::sfd_identifier must be a non-empty string, got {sfd_identifier!r}'
        raise ValueError(msg)
    if not sfd_identifier.isidentifier():
        msg = f'extract_bundle: {json_path}::sfd_identifier must be a valid Python identifier, got {sfd_identifier!r}'
        raise ValueError(msg)
    data_source = meta.get('data_source')
    if not isinstance(data_source, dict):
        msg = f'extract_bundle: {json_path}::data_source must be a JSON object, got {type(data_source).__name__}'
        raise ValueError(msg)
    uel_run = meta.get('uel_run')
    if not isinstance(uel_run, dict):
        msg = f'extract_bundle: {json_path}::uel_run must be a JSON object, got {type(uel_run).__name__}'
        raise ValueError(msg)
    return BundleSpec(bundle_path=bundle_zip, work_dir=work_dir, py_path=py_path, json_path=json_path, csv_path=csv_path, sfd_identifier=sfd_identifier, data_source=cast('dict[str, object]', data_source), uel_run=cast('dict[str, object]', uel_run))

def _strip_runtime_body(source: str) -> str:
    tree = ast.parse(source)
    kept: list[ast.stmt] = []
    for stmt in tree.body:
        if _is_runtime_body_stmt(stmt):
            break
        kept.append(stmt)
    new_tree = ast.Module(body=kept, type_ignores=[])
    return ast.unparse(new_tree)

def _is_runtime_body_stmt(stmt: ast.stmt) -> bool:
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        for dec in stmt.decorator_list:
            if _expr_calls_runtime_marker(dec):
                return True
        return False
    return _expr_calls_runtime_marker(stmt)

def _expr_calls_runtime_marker(node: ast.AST) -> bool:
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
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            stack.append(child)
    return False

def _override_data_source(base: LimenManifest, override: dict[str, object]) -> LimenManifest:
    method_name = override.get('method')
    if not isinstance(method_name, str) or not method_name:
        msg = f'data_source.method must be a non-empty string, got {method_name!r}'
        raise ValueError(msg)
    raw_params = override.get('params', {})
    if not isinstance(raw_params, dict):
        msg = f'data_source.params must be a JSON object, got {type(raw_params).__name__}'
        raise ValueError(msg)
    historical = limen.HistoricalData()
    method = getattr(historical, method_name, None)
    if not callable(method):
        msg = f'limen.HistoricalData has no method {method_name!r}'
        raise AttributeError(msg)
    return dataclasses.replace(base, data_source_config=DataSourceConfig(method=method, params=cast('dict[str, object]', raw_params)))
_SHIM_TEMPLATE: str = '\nfrom backtest_simulator.pipeline.bundle import _override_data_source as _bts_override\n\n_BTS_DATA_SOURCE = {data_source!r}\n_BTS_SFD_CLASS = {sfd_identifier}\n\nparams = _BTS_SFD_CLASS.params\n\ndef manifest():\n    return _bts_override(_BTS_SFD_CLASS.manifest(), _BTS_DATA_SOURCE)\n'

def materialize_bundle_for_cli(bundle_zip: Path, work_dir: Path) -> tuple[Path, Path, BundleSpec]:
    work_dir.mkdir(parents=True, exist_ok=True)
    lock_path = work_dir / '.bundle.lock'
    with _exclusive_dir_lock(lock_path):
        spec = extract_bundle(bundle_zip, work_dir)
        stripped = _strip_runtime_body(spec.py_path.read_text(encoding='utf-8'))
        shim_source = stripped + _SHIM_TEMPLATE.format(data_source=spec.data_source, sfd_identifier=spec.sfd_identifier)
        shim_path = work_dir / f'{spec.py_path.stem}__bts_shim.py'
        shim_path.write_text(shim_source, encoding='utf-8')
        return (shim_path, spec.csv_path, spec)

@contextlib.contextmanager
def _exclusive_dir_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, 'w') as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
N_PERMUTATIONS_DEFAULT: int = 30

def materialize_bundle_on_args(args: argparse.Namespace, work_root: Path) -> None:
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
