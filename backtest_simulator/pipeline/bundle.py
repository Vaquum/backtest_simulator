"""Limen bundle loader — `<name>__rNNNN.zip` (.py + .json + .csv)."""
# The upstream contract:
#
#   .py — canonical SFD definition. Imports + helpers + the SFD class
#         (with `params()` and `manifest()` static methods). At the
#         bottom, a script body that does the data pull and `uel.run(...)`
#         for direct re-execution; bts skips those statements.
#   .json — overrides for `data_source` (method + params) and `uel_run`
#         (experiment_name, n_permutations, prep_each_round, etc.).
#         JSON wins where it overlaps with the .py's script body.
#   .csv — the trained-and-filtered results pool. One row per decoder
#         permutation; `id` column matches `--decoder-id`.
#
# bts uses the .py's prefix (everything above the first runtime side
# effect — data pull or uel.run) to get the SFD class object, then drives
# UEL itself with JSON-provided params.
from __future__ import annotations

import ast
import importlib.util
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from limen.experiment import Manifest as LimenManifest
from limen.experiment.manifest_core import DataSourceConfig

from backtest_simulator.pipeline.experiment import ExperimentFile

# AST node names whose appearance in a top-level statement marks the
# start of the runtime body bts must skip. The convention is:
# `limen.HistoricalData(...)` (direct), `historical.get_spot_klines(...)`
# (instance method off HistoricalData), `limen.UniversalExperimentLoop(...)`
# (UEL constructor), `uel.run(...)` (UEL execution). We match by attribute
# name; aliased imports are out of slice scope (Risks).
_RUNTIME_BODY_MARKERS: frozenset[str] = frozenset({
    'HistoricalData',
    'get_spot_klines',
    'UniversalExperimentLoop',
    'run',
})


@dataclass(frozen=True)
class BundleSpec:
    """Filesystem layout + JSON metadata extracted from a bundle zip."""

    bundle_path: Path
    work_dir: Path
    py_path: Path
    json_path: Path
    csv_path: Path
    sfd_identifier: str
    data_source: dict[str, object]
    uel_run: dict[str, object]


def extract_bundle(bundle_zip: Path, work_dir: Path) -> BundleSpec:
    """Unzip bundle into work_dir and parse the JSON metadata.

    Expects three sibling files inside the zip with a shared stem:
    `<stem>.py`, `<stem>.json`, `<stem>.csv`. Raises if any are
    missing or if the JSON omits required keys.
    """
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
    """Return source with the data-pull / uel-run statements removed.

    Walks top-level statements; drops every statement whose AST contains
    a `Call` against any name in `_RUNTIME_BODY_MARKERS` (HistoricalData,
    get_spot_klines, UniversalExperimentLoop, run). Imports, function
    definitions, class definitions, and pure assignments are kept.
    Once a runtime statement is dropped, every statement that follows
    is also dropped (the script body is contiguous by convention).
    """
    tree = ast.parse(source)
    kept: list[ast.stmt] = []
    for stmt in tree.body:
        if _is_runtime_body_stmt(stmt):
            break
        kept.append(stmt)
    new_tree = ast.Module(body=kept, type_ignores=[])
    return ast.unparse(new_tree)


def _is_runtime_body_stmt(stmt: ast.stmt) -> bool:
    """True if the statement contains a Call to a runtime-body marker."""
    for node in ast.walk(stmt):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _RUNTIME_BODY_MARKERS:
            return True
        if isinstance(func, ast.Name) and func.id in _RUNTIME_BODY_MARKERS:
            return True
    return False


def load_bundled_experiment(spec: BundleSpec) -> ExperimentFile:
    """Load the bundle's .py with the runtime body stripped; return ExperimentFile.

    The returned ExperimentFile carries:
      - `params`: the SFD class's `params()` static method.
      - `manifest`: a wrapper that calls the SFD class's `manifest()`,
        then replaces `data_source_config` with the JSON-driven values
        (method + params from `BundleSpec.data_source`).
    """
    stripped = _strip_runtime_body(spec.py_path.read_text(encoding='utf-8'))
    module_name = f'_bts_bundle_{spec.py_path.stem.replace("-", "_")}'
    spec_obj = importlib.util.spec_from_loader(module_name, loader=None)
    if spec_obj is None:
        msg = f'load_bundled_experiment: failed to build module spec for {module_name}'
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec_obj)
    sys.modules[module_name] = module
    exec(
        compile(stripped, str(spec.py_path), 'exec'),
        module.__dict__,
    )
    sfd_class = getattr(module, spec.sfd_identifier, None)
    if sfd_class is None:
        msg = (
            f'load_bundled_experiment: SFD class {spec.sfd_identifier!r} '
            f'not found in {spec.py_path}; the JSON sfd_identifier and '
            f'the .py must agree.'
        )
        raise AttributeError(msg)
    params_fn_raw = getattr(sfd_class, 'params', None)
    base_manifest_fn = getattr(sfd_class, 'manifest', None)
    if not callable(params_fn_raw) or not callable(base_manifest_fn):
        msg = (
            f'load_bundled_experiment: {spec.sfd_identifier} must expose '
            f'callable `params` and `manifest`'
        )
        raise TypeError(msg)

    def manifest_with_override() -> object:
        base = base_manifest_fn()
        if not isinstance(base, LimenManifest):
            msg = (
                f'load_bundled_experiment: {spec.sfd_identifier}.manifest() '
                f'must return a limen.experiment.Manifest, got {type(base).__name__}'
            )
            raise TypeError(msg)
        return _override_data_source(base, spec.data_source)

    return ExperimentFile(
        source_path=spec.py_path,
        module=module,
        params=cast(
            'object', params_fn_raw,  # narrows to dict[str, list[object]] at call
        ),  # type: ignore[arg-type]
        manifest=manifest_with_override,
    )


def materialize_bundle_for_cli(
    bundle_zip: Path, work_dir: Path,
) -> tuple[Path, Path]:
    """Extract a bundle and write a synthetic exp-code shim the CLI can load.

    Returns `(shim_py_path, csv_path)`. The shim is the bundle's `.py`
    with the runtime body stripped, plus module-level `params` and
    `manifest` aliases (manifest applies the JSON `data_source` override
    in-line). Existing CLI machinery in `_pipeline.py` consumes the shim
    via the same path-based loader as a hand-written `--exp-code`.

    The CLI supplies `--bundle <zip>`; this helper populates
    `args.exp_code = shim_py_path` and `args.input_from_file = csv_path`
    so the rest of the run / sweep flow is unchanged.
    """
    spec = extract_bundle(bundle_zip, work_dir)
    stripped = _strip_runtime_body(spec.py_path.read_text(encoding='utf-8'))
    method_name = spec.data_source.get('method')
    raw_params = spec.data_source.get('params', {})
    if not isinstance(method_name, str) or not method_name:
        msg = (
            f'materialize_bundle_for_cli: data_source.method must be a '
            f'non-empty string, got {method_name!r}'
        )
        raise ValueError(msg)
    if not isinstance(raw_params, dict):
        msg = (
            f'materialize_bundle_for_cli: data_source.params must be a '
            f'JSON object, got {type(raw_params).__name__}'
        )
        raise ValueError(msg)
    # The shim appends module-level `params` / `manifest` aliases that
    # delegate to the SFD class and apply the JSON data_source override.
    # The aliases are at module top level so
    # `ExperimentPipeline.load_from_file` accepts them without changes.
    shim_extension = (
        '\n\n'
        '# bts-bundle: module-level params/manifest aliases for the\n'
        '# SFD class above; manifest applies the JSON-driven data_source\n'
        '# override required by the upstream contract.\n'
        f'import dataclasses as _bts_dc\n'
        f'from limen.experiment.manifest_core import DataSourceConfig as _bts_DSC\n'
        f'from limen import HistoricalData as _bts_HD\n'
        f'params = {spec.sfd_identifier}.params\n'
        f'def manifest():\n'
        f'    _base = {spec.sfd_identifier}.manifest()\n'
        f'    _hist = _bts_HD()\n'
        f'    _method = getattr(_hist, {method_name!r})\n'
        f'    return _bts_dc.replace(\n'
        f'        _base,\n'
        f'        data_source_config=_bts_DSC(\n'
        f'            method=_method,\n'
        f'            params={raw_params!r},\n'
        f'        ),\n'
        f'    )\n'
    )
    shim_path = spec.work_dir / f'{spec.py_path.stem}__bts_shim.py'
    shim_path.write_text(stripped + shim_extension, encoding='utf-8')
    return shim_path, spec.csv_path


def _override_data_source(
    base: LimenManifest, override: dict[str, object],
) -> LimenManifest:
    """Replace `base.data_source_config` with the bundle JSON's values.

    Resolves `data_source.method` (string name) to a bound method on a
    fresh `limen.HistoricalData()` instance, with `data_source.params`
    forwarded as the call's kwargs.
    """
    method_name = override.get('method')
    if not isinstance(method_name, str) or not method_name:
        msg = (
            f'_override_data_source: data_source.method must be a non-empty '
            f'string, got {method_name!r}'
        )
        raise ValueError(msg)
    raw_params = override.get('params', {})
    if not isinstance(raw_params, dict):
        msg = (
            f'_override_data_source: data_source.params must be a JSON object, '
            f'got {type(raw_params).__name__}'
        )
        raise ValueError(msg)
    params: dict[str, object] = cast('dict[str, object]', raw_params)
    import limen
    historical = limen.HistoricalData()
    method = getattr(historical, method_name, None)
    if not callable(method):
        msg = (
            f'_override_data_source: limen.HistoricalData has no method '
            f'{method_name!r}'
        )
        raise AttributeError(msg)
    new_config = DataSourceConfig(method=method, params=params)
    # `Manifest` is a frozen dataclass; replace via dataclasses.replace.
    import dataclasses
    return dataclasses.replace(base, data_source_config=new_config)
