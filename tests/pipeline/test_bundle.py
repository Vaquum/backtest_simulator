"""Slice #36 — Limen bundle loader contract tests."""
from __future__ import annotations

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
EXAMPLE_BUNDLE = REPO_ROOT.parent / '_examples' / 'LogReg-Placeholder__r0007.zip'


def _bundle_or_skip() -> Path:
    """Skip the test if the example bundle isn't available locally."""
    if not EXAMPLE_BUNDLE.is_file():
        pytest.skip(f'example bundle not present at {EXAMPLE_BUNDLE}')
    return EXAMPLE_BUNDLE


def _make_synthetic_bundle(tmp_path: Path, py_source: str) -> Path:
    """Build a 3-file bundle zip in tmp_path for AST tests that don't need Limen."""
    stem = 'TestBundle__r0001'
    work = tmp_path / 'src'
    work.mkdir()
    (work / f'{stem}.py').write_text(py_source, encoding='utf-8')
    (work / f'{stem}.json').write_text(
        json.dumps({
            'sfd_identifier': 'StubSfd',
            'data_source': {
                'method': 'get_spot_klines',
                'params': {'kline_size': 1234, 'start_date_limit': '2024-06-01'},
            },
            'uel_run': {
                'experiment_name': 'test',
                'n_permutations': 5,
                'prep_each_round': False,
            },
        }), encoding='utf-8',
    )
    # CSV body irrelevant for these tests; just a valid header.
    (work / f'{stem}.csv').write_text('id,kelly_pct\n0,1.0\n', encoding='utf-8')
    bundle_path = tmp_path / f'{stem}.zip'
    with zipfile.ZipFile(bundle_path, 'w') as zf:
        for child in work.iterdir():
            zf.write(child, arcname=child.name)
    return bundle_path


# ---- extract_bundle --------------------------------------------------------


def test_extract_bundle_reads_json_meta(tmp_path: Path) -> None:
    bundle_zip = _bundle_or_skip()
    spec = extract_bundle(bundle_zip, tmp_path)
    assert spec.sfd_identifier == 'Round3SFD'
    assert spec.data_source['method'] == 'get_spot_klines'
    assert spec.data_source['params']['kline_size'] == 7200  # type: ignore[index]
    assert spec.uel_run['experiment_name'] == 'LogReg-Placeholder'
    assert spec.uel_run['n_permutations'] == 1000


def test_extract_bundle_lays_out_three_files(tmp_path: Path) -> None:
    bundle_zip = _bundle_or_skip()
    spec = extract_bundle(bundle_zip, tmp_path)
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
    bundle_path = tmp_path / 'bad.zip'
    with zipfile.ZipFile(bundle_path, 'w') as zf:
        for child in work.iterdir():
            zf.write(child, arcname=child.name)
    with pytest.raises(ValueError, match=r'exactly one \.py'):
        extract_bundle(bundle_path, tmp_path / 'extract')


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


def test_strip_runtime_body_keeps_class_def_and_helpers() -> None:
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
    # Operator's contract: script body is contiguous; once we see uel.run,
    # everything after is also script (even if it textually looks like a
    # class def).
    assert 'TooLate' not in stripped


def test_is_runtime_body_stmt_detects_dotted_run() -> None:
    import ast
    stmt = ast.parse('uel.run(n_permutations=5)').body[0]
    assert _is_runtime_body_stmt(stmt) is True


def test_is_runtime_body_stmt_ignores_pure_assigns() -> None:
    import ast
    stmt = ast.parse('CONSTANT = [1, 2, 3]').body[0]
    assert _is_runtime_body_stmt(stmt) is False


# ---- load_bundled_experiment ----------------------------------------------


def test_load_bundled_experiment_exposes_params(tmp_path: Path) -> None:
    bundle_zip = _bundle_or_skip()
    spec = extract_bundle(bundle_zip, tmp_path)
    exp = load_bundled_experiment(spec)
    keys = list(exp.params().keys())
    assert keys, 'SFD params() returned no keys'
    # The bundle ships an SFD with hyperparam keys including these:
    assert 'q' in keys
    assert 'class_weight' in keys


def test_load_bundled_experiment_overrides_data_source(tmp_path: Path) -> None:
    bundle_zip = _bundle_or_skip()
    spec = extract_bundle(bundle_zip, tmp_path)
    exp = load_bundled_experiment(spec)
    manifest = exp.manifest()
    # JSON's data_source.params: kline_size=7200, start_date_limit='2024-01-01'.
    # Base SFD would have kline_size=3600, start_date_limit='2025-01-01'.
    cfg_params = manifest.data_source_config.params  # type: ignore[attr-defined]
    assert cfg_params == {
        'kline_size': 7200,
        'start_date_limit': '2024-01-01',
    }


# ---- materialize_bundle_for_cli (synthesizes shim) ------------------------


def test_materialize_bundle_for_cli_writes_loadable_shim(tmp_path: Path) -> None:
    bundle_zip = _bundle_or_skip()
    shim_path, csv_path = materialize_bundle_for_cli(bundle_zip, tmp_path)
    assert shim_path.is_file()
    assert csv_path.is_file() and csv_path.suffix == '.csv'
    # The shim must be loadable by the existing path-based loader.
    exp = ExperimentPipeline.load_from_file(shim_path)
    assert callable(exp.params)
    assert callable(exp.manifest)
    keys = list(exp.params().keys())
    assert 'q' in keys
    cfg_params = exp.manifest().data_source_config.params  # type: ignore[attr-defined]
    assert cfg_params['kline_size'] == 7200


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
    """Passing both --bundle and --exp-code must fail at argparse."""
    fake_zip = tmp_path / 'fake.zip'
    fake_py = tmp_path / 'fake.py'
    fake_zip.write_bytes(b'PK')  # invalid zip; argparse rejects before we open it
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
    # argparse's mutually-exclusive group emits "not allowed with argument"
    # in stderr.
    assert 'not allowed with' in (out.stderr + out.stdout).lower()


# Cleanup: the loader writes shims under tmp_path; pytest handles teardown.
# Module-loaded modules are added to sys.modules with `_bts_bundle_*` names;
# they're harmless to leave (each test uses a unique tmp_path so the
# module-name collision is avoided in practice).
def _cleanup_module_cache() -> None:
    for name in list(sys.modules):
        if name.startswith('_bts_bundle_'):
            del sys.modules[name]


@pytest.fixture(autouse=True)
def _cleanup() -> None:
    _cleanup_module_cache()
    yield
    _cleanup_module_cache()
    # Also wipe any /tmp/bts_sweep/run/bundles dirs the CLI helper made.
    cache = Path('/tmp/bts_sweep/run/bundles')
    if cache.is_dir():
        shutil.rmtree(cache, ignore_errors=True)
