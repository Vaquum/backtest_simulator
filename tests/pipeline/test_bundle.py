"""Limen bundle loader contract tests."""
from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from backtest_simulator.pipeline import ExperimentPipeline
from backtest_simulator.pipeline.bundle import (
    _is_runtime_body_stmt,
    _strip_runtime_body,
    extract_bundle,
    load_bundled_experiment,
    materialize_bundle_for_cli,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# Synthetic bundle: a real Limen-shaped SFD that delegates to
# `logreg_binary`, so the loader exercises the same code paths as a
# real upstream bundle without depending on an external `_examples/`
# zip. CI consumes this directly; the tests below run unconditionally.
_SYNTHETIC_PY = '''
import warnings
warnings.filterwarnings("ignore")

import limen
from limen.sfd.foundational_sfd import logreg_binary as base_sfd


class StubSfd:
    @staticmethod
    def params():
        return {
            **base_sfd.params(),
            "q": [0.32, 0.4, 0.5],
        }

    @staticmethod
    def manifest():
        return base_sfd.manifest()


# Reproducer body bts must skip:
historical = limen.HistoricalData()
data = historical.get_spot_klines(kline_size=3600, start_date_limit="2025-01-01")
uel = limen.UniversalExperimentLoop(data=data, sfd=StubSfd)
uel.run(experiment_name="stub", n_permutations=10)
'''

_SYNTHETIC_JSON = {
    'sfd_identifier': 'StubSfd',
    'data_source': {
        'method': 'get_spot_klines',
        'params': {'kline_size': 7200, 'start_date_limit': '2024-01-01'},
    },
    'uel_run': {
        'experiment_name': 'stub-experiment',
        'n_permutations': 50,
        'prep_each_round': True,
        'random_search': True,
    },
}


def _write_bundle(tmp_path: Path, *, py_source: str = _SYNTHETIC_PY,
                  meta: dict[str, object] | None = None,
                  csv_body: str = 'id,backtest_mean_kelly_pct\n0,1.0\n1,0.5\n') -> Path:
    """Build a 3-file bundle zip in tmp_path; return the zip path."""
    stem = 'StubBundle__r0001'
    work = tmp_path / 'src'
    work.mkdir()
    (work / f'{stem}.py').write_text(py_source, encoding='utf-8')
    (work / f'{stem}.json').write_text(
        json.dumps(meta if meta is not None else _SYNTHETIC_JSON),
        encoding='utf-8',
    )
    (work / f'{stem}.csv').write_text(csv_body, encoding='utf-8')
    bundle_path = tmp_path / f'{stem}.zip'
    with zipfile.ZipFile(bundle_path, 'w') as zf:
        for child in work.iterdir():
            zf.write(child, arcname=child.name)
    return bundle_path


# ---- extract_bundle --------------------------------------------------------


def test_extract_bundle_reads_json_meta(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path)
    spec = extract_bundle(bundle, tmp_path / 'extract')
    assert spec.sfd_identifier == 'StubSfd'
    assert spec.data_source['method'] == 'get_spot_klines'
    assert spec.data_source['params']['kline_size'] == 7200  # type: ignore[index]
    assert spec.uel_run['experiment_name'] == 'stub-experiment'
    assert spec.uel_run['n_permutations'] == 50


def test_extract_bundle_lays_out_three_files(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path)
    spec = extract_bundle(bundle, tmp_path / 'extract')
    assert spec.py_path.is_file() and spec.py_path.suffix == '.py'
    assert spec.json_path.is_file() and spec.json_path.suffix == '.json'
    assert spec.csv_path.is_file() and spec.csv_path.suffix == '.csv'


def test_extract_bundle_rejects_zip_with_extra_files(tmp_path: Path) -> None:
    work = tmp_path / 'src'
    work.mkdir()
    (work / 'a.py').write_text('', encoding='utf-8')
    (work / 'a.json').write_text('{}', encoding='utf-8')
    (work / 'a.csv').write_text('', encoding='utf-8')
    (work / 'extra.py').write_text('', encoding='utf-8')
    bundle = tmp_path / 'bad.zip'
    with zipfile.ZipFile(bundle, 'w') as zf:
        for child in work.iterdir():
            zf.write(child, arcname=child.name)
    with pytest.raises(ValueError, match=r'exactly one \.py'):
        extract_bundle(bundle, tmp_path / 'extract')


# ---- _strip_runtime_body ---------------------------------------------------


def test_strip_runtime_body_drops_uel_run() -> None:
    src = (
        'class Sfd:\n'
        '    @staticmethod\n'
        '    def params(): return {}\n'
        '\n'
        'uel = object()\n'
        'uel.run(experiment_name="x", n_permutations=10)\n'
    )
    stripped = _strip_runtime_body(src)
    assert 'class Sfd' in stripped
    assert 'uel.run' not in stripped


def test_strip_runtime_body_drops_data_pull() -> None:
    src = (
        'import limen\n'
        'class Sfd:\n'
        '    @staticmethod\n'
        '    def params(): return {}\n'
        '\n'
        'historical = limen.HistoricalData()\n'
        'data = historical.get_spot_klines(kline_size=3600)\n'
    )
    stripped = _strip_runtime_body(src)
    assert 'class Sfd' in stripped
    assert 'HistoricalData' not in stripped
    assert 'get_spot_klines' not in stripped


def test_strip_runtime_body_keeps_class_def() -> None:
    src = (
        'def helper(x): return x + 1\n'
        '\n'
        'class Sfd:\n'
        '    CONSTANT = 42\n'
        '    @staticmethod\n'
        '    def params(): return {}\n'
        '\n'
        'uel.run(n_permutations=5)\n'
    )
    stripped = _strip_runtime_body(src)
    assert 'def helper' in stripped
    assert 'class Sfd' in stripped
    assert 'CONSTANT = 42' in stripped
    assert 'uel.run' not in stripped


def test_strip_runtime_body_drops_everything_after_first_runtime_stmt() -> None:
    src = (
        'class Sfd:\n'
        '    @staticmethod\n'
        '    def params(): return {}\n'
        '\n'
        'uel.run(n_permutations=5)\n'
        'class TooLate:\n'
        '    pass\n'
    )
    stripped = _strip_runtime_body(src)
    assert 'class Sfd' in stripped
    assert 'TooLate' not in stripped


def test_strip_runtime_body_does_not_cut_on_run_inside_function() -> None:
    """A helper function calling .run() must NOT trigger the cut."""
    src = (
        'def runner(executor):\n'
        '    return executor.run(payload="x")\n'
        '\n'
        'class Sfd:\n'
        '    @staticmethod\n'
        '    def params(): return {"q": [0.5]}\n'
    )
    stripped = _strip_runtime_body(src)
    assert 'def runner' in stripped
    assert 'class Sfd' in stripped
    assert 'q' in stripped


def test_strip_runtime_body_does_not_cut_on_run_inside_class_body() -> None:
    """A class with a method named `run` (or one that calls `.run()`)
    must NOT lose its definition to the AST cut."""
    src = (
        'class Sfd:\n'
        '    @staticmethod\n'
        '    def params(): return {"q": [0.5]}\n'
        '\n'
        '    def run(self):\n'
        '        return "fine — this is a method, not module-level"\n'
        '\n'
        '    def helper(self, executor):\n'
        '        return executor.run()\n'
    )
    stripped = _strip_runtime_body(src)
    assert 'class Sfd' in stripped
    assert 'def params' in stripped


def test_is_runtime_body_stmt_detects_dotted_run() -> None:
    stmt = ast.parse('uel.run(n_permutations=5)').body[0]
    assert _is_runtime_body_stmt(stmt) is True


def test_is_runtime_body_stmt_ignores_pure_assigns() -> None:
    stmt = ast.parse('CONSTANT = [1, 2, 3]').body[0]
    assert _is_runtime_body_stmt(stmt) is False


def test_is_runtime_body_stmt_ignores_run_inside_function_body() -> None:
    """Codex post-auditor caught `ast.walk` descending into nested
    function bodies; the cut must look at top-level expressions only."""
    stmt = ast.parse('def helper():\n    obj.run()\n').body[0]
    assert _is_runtime_body_stmt(stmt) is False


# ---- load_bundled_experiment -----------------------------------------------


def test_load_bundled_experiment_exposes_params(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path)
    spec = extract_bundle(bundle, tmp_path / 'extract')
    exp = load_bundled_experiment(spec)
    keys = list(exp.params().keys())
    assert keys
    assert 'q' in keys


def test_load_bundled_experiment_overrides_data_source(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path)
    spec = extract_bundle(bundle, tmp_path / 'extract')
    exp = load_bundled_experiment(spec)
    manifest = exp.manifest()
    cfg_params = manifest.data_source_config.params  # type: ignore[attr-defined]
    assert cfg_params['kline_size'] == 7200
    assert cfg_params['start_date_limit'] == '2024-01-01'


def test_load_bundled_experiment_validates_params_shape(tmp_path: Path) -> None:
    """A wrong-shape `params()` must raise here, not deep inside Limen."""
    bad_py = (
        'class StubSfd:\n'
        '    @staticmethod\n'
        '    def params(): return [1, 2, 3]\n'  # not a dict
        '    @staticmethod\n'
        '    def manifest():\n'
        '        from limen.sfd.foundational_sfd import logreg_binary\n'
        '        return logreg_binary.manifest()\n'
    )
    bundle = _write_bundle(tmp_path, py_source=bad_py)
    spec = extract_bundle(bundle, tmp_path / 'extract')
    exp = load_bundled_experiment(spec)
    with pytest.raises(TypeError, match='must return dict'):
        exp.params()


# ---- materialize_bundle_for_cli (shim path) -------------------------------


def test_materialize_bundle_for_cli_writes_loadable_shim(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path)
    shim_path, csv_path = materialize_bundle_for_cli(bundle, tmp_path / 'cache')
    assert shim_path.is_file()
    assert csv_path.is_file()
    exp = ExperimentPipeline.load_from_file(shim_path)
    keys = list(exp.params().keys())
    assert 'q' in keys
    cfg_params = exp.manifest().data_source_config.params  # type: ignore[attr-defined]
    assert cfg_params['kline_size'] == 7200


def test_materialize_bundle_for_cli_shim_calls_shared_override(
    tmp_path: Path,
) -> None:
    """The shim and the in-memory `_override_data_source` must be the
    same code path. The shim text must `import _override_data_source`
    rather than re-implement the override inline.
    """
    bundle = _write_bundle(tmp_path)
    shim_path, _ = materialize_bundle_for_cli(bundle, tmp_path / 'cache')
    src = shim_path.read_text(encoding='utf-8')
    assert '_override_data_source' in src
    # No second f-string-templated DataSourceConfig construction.
    assert src.count('DataSourceConfig(') == 0


# ---- CLI surface -----------------------------------------------------------


def test_cli_run_help_advertises_bundle() -> None:
    out = subprocess.run(
        [sys.executable, '-m', 'backtest_simulator.cli', 'run', '--help'],
        check=False, capture_output=True, text=True,
    )
    assert out.returncode == 0
    assert '--bundle' in out.stdout
    assert '--exp-code' in out.stdout


def test_cli_sweep_help_advertises_bundle() -> None:
    out = subprocess.run(
        [sys.executable, '-m', 'backtest_simulator.cli', 'sweep', '--help'],
        check=False, capture_output=True, text=True,
    )
    assert out.returncode == 0
    assert '--bundle' in out.stdout


def test_cli_bundle_exp_code_mutually_exclusive(tmp_path: Path) -> None:
    fake_zip = tmp_path / 'fake.zip'
    fake_py = tmp_path / 'fake.py'
    fake_zip.write_bytes(b'PK')
    fake_py.write_text('', encoding='utf-8')
    out = subprocess.run(
        [
            sys.executable, '-m', 'backtest_simulator.cli', 'run',
            '--bundle', str(fake_zip),
            '--exp-code', str(fake_py),
            '--window-start', '2026-01-01T00:00:00+00:00',
            '--window-end', '2026-01-02T00:00:00+00:00',
        ],
        check=False, capture_output=True, text=True,
    )
    assert out.returncode != 0
    assert 'not allowed with' in (out.stderr + out.stdout).lower()


def test_bts_run_bundle_end_to_end_routes_through_shim(tmp_path: Path) -> None:
    """`bts run --bundle <zip>` rewrites args to point at the synthesized
    shim + CSV before the rest of the run flow executes.

    The full backtest needs ClickHouse + a Limen training run; that's a
    minutes-long integration that doesn't belong in unit-test scope.
    What this test pins is the bundle->args plumbing: after
    `materialize_bundle_on_args(args, work_root)`, `args.exp_code`
    points at a shim file that loads cleanly through
    `ExperimentPipeline.load_from_file`, `args.input_from_file` is the
    bundle's CSV, and `args.n_permutations` reflects the JSON's value
    when the operator did not pass an explicit `--n-permutations`.
    """
    import argparse as _ap

    from backtest_simulator.pipeline.bundle import materialize_bundle_on_args
    bundle = _write_bundle(tmp_path)
    args = _ap.Namespace(
        bundle=bundle,
        exp_code=None,
        input_from_file=None,
        n_permutations=30,  # bts default; bundle JSON's 50 should win
    )
    work_root = tmp_path / 'work'
    materialize_bundle_on_args(args, work_root)
    assert args.exp_code is not None and args.exp_code.is_file()
    assert args.input_from_file is not None and Path(args.input_from_file).is_file()
    # Bundle's JSON `uel_run.n_permutations=50` overrides the bts default 30.
    assert args.n_permutations == 50
    # Shim is loadable via the existing path-based machinery.
    exp = ExperimentPipeline.load_from_file(args.exp_code)
    assert callable(exp.params)
    assert callable(exp.manifest)


def test_bts_run_bundle_respects_explicit_n_permutations(tmp_path: Path) -> None:
    """Operator-supplied --n-permutations beats the bundle's JSON value."""
    import argparse as _ap

    from backtest_simulator.pipeline.bundle import materialize_bundle_on_args
    bundle = _write_bundle(tmp_path)
    args = _ap.Namespace(
        bundle=bundle, exp_code=None, input_from_file=None,
        n_permutations=7,  # explicit; not the default
    )
    materialize_bundle_on_args(args, tmp_path / 'work')
    assert args.n_permutations == 7  # operator wins


@pytest.fixture(autouse=True)
def _cleanup() -> None:
    for name in list(sys.modules):
        if name.startswith('_bts_bundle_'):
            del sys.modules[name]
    yield
    for name in list(sys.modules):
        if name.startswith('_bts_bundle_'):
            del sys.modules[name]
    cache = Path('/tmp/bts_sweep/run/bundles')
    if cache.is_dir():
        shutil.rmtree(cache, ignore_errors=True)
