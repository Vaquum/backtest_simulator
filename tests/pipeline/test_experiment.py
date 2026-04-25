"""ExperimentPipeline: file loading + filter semantics + results flattening."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from backtest_simulator.pipeline import ExperimentPipeline

_FIXTURES = Path(__file__).parent / 'fixtures'


def test_load_from_file_exposes_params_and_manifest() -> None:
    loaded = ExperimentPipeline.load_from_file(_FIXTURES / 'toy_experiment.py')
    assert callable(loaded.params)
    assert callable(loaded.manifest)
    p = loaded.params()
    assert p['lookback'] == [10, 20, 30]
    assert p['threshold'] == [0.55, 0.60]


def test_load_from_file_rejects_missing_callables(tmp_path: Path) -> None:
    bad = tmp_path / 'bad.py'
    bad.write_text('x = 1\n', encoding='utf-8')
    with pytest.raises(ValueError, match='must define callable'):
        ExperimentPipeline.load_from_file(bad)


def test_load_from_file_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ExperimentPipeline.load_from_file(tmp_path / 'nope.py')


def _write_results(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    df = pl.DataFrame(rows)
    results_path = tmp_path / 'results.csv'
    df.write_csv(results_path)
    return results_path


def test_read_results_flattens_round_params(tmp_path: Path) -> None:
    _write_results(tmp_path, [
        {'permutation_id': 1, 'reliability': 0.7,
         'round_params': json.dumps({'q': 0.35, 'roc_period': 4})},
        {'permutation_id': 2, 'reliability': 0.9,
         'round_params': json.dumps({'q': 0.41, 'roc_period': 12})},
    ])
    pipeline = ExperimentPipeline(experiment_dir=tmp_path)
    df = pipeline.read_results()
    assert 'q' in df.columns
    assert 'roc_period' in df.columns
    assert sorted(df['q'].to_list()) == [0.35, 0.41]


def test_filter_equality() -> None:
    df = pl.DataFrame({'reliability': [0.5, 0.7, 0.9], 'kind': ['a', 'b', 'a']})
    out = ExperimentPipeline.filter_results(df, {'kind': 'a'})
    assert out.height == 2


def test_filter_range() -> None:
    df = pl.DataFrame({'reliability': [0.5, 0.7, 0.9], 'sharpe': [0.1, 1.2, 2.3]})
    out = ExperimentPipeline.filter_results(df, {'sharpe': (1.0, 2.0)})
    assert out['sharpe'].to_list() == [1.2]


def test_filter_set_membership() -> None:
    df = pl.DataFrame({'permutation_id': [1, 2, 3, 4, 5]})
    out = ExperimentPipeline.filter_results(df, {'permutation_id': {2, 4}})
    assert sorted(out['permutation_id'].to_list()) == [2, 4]


def test_filter_composes_all_three_kinds() -> None:
    df = pl.DataFrame({
        'permutation_id': [1, 2, 3, 4, 5],
        'reliability': [0.3, 0.7, 0.9, 0.8, 0.6],
        'kind': ['a', 'a', 'b', 'a', 'a'],
    })
    out = ExperimentPipeline.filter_results(df, {
        'permutation_id': {1, 2, 3, 4},  # drops 5
        'reliability': (0.5, 1.0),       # drops 1
        'kind': 'a',                       # drops 3
    })
    assert sorted(out['permutation_id'].to_list()) == [2, 4]


def test_filter_rejects_unknown_column() -> None:
    df = pl.DataFrame({'reliability': [0.5]})
    with pytest.raises(ValueError, match='unknown column'):
        ExperimentPipeline.filter_results(df, {'sharpe': 1.0})


def test_read_results_missing_file_raises(tmp_path: Path) -> None:
    pipeline = ExperimentPipeline(experiment_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        pipeline.read_results()
