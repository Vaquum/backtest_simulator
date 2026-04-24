"""ManifestBuilder: emit real Nexus YAML + strategy file; round-trip via load_manifest."""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from nexus.infrastructure.manifest import load_manifest

from backtest_simulator.pipeline import ManifestBuilder
from backtest_simulator.pipeline.manifest_builder import (
    SensorBinding,
    StrategyParamsSpec,
)


def _make_experiment_dir(tmp_path: Path) -> Path:
    d = tmp_path / 'exp'
    d.mkdir()
    (d / 'results.csv').write_text('permutation_id\n1\n', encoding='utf-8')
    return d


def _spec() -> StrategyParamsSpec:
    return StrategyParamsSpec(
        symbol='BTCUSDT',
        capital=Decimal('100000'),
        kelly_pct=Decimal('10'),
        estimated_price=Decimal('65000'),
        stop_bps=Decimal('50'),
    )


def test_build_emits_valid_nexus_manifest(tmp_path: Path) -> None:
    exp = _make_experiment_dir(tmp_path)
    builder = ManifestBuilder(output_dir=tmp_path / 'out')
    built = builder.build(
        account_id='bts-acct-0',
        allocated_capital=Decimal('100000'),
        capital_pool=Decimal('10000'),
        strategy_id='long_on_signal_q35',
        sensor=SensorBinding(
            experiment_dir=exp, permutation_ids=(1, 5), interval_seconds=3600,
        ),
        strategy_params=_spec(),
    )
    assert built.manifest_path.is_file()
    assert built.strategies_base_path.is_dir()
    assert (built.strategies_base_path / 'long_on_signal.py').is_file()
    # The manifest and strategy file are co-located so Nexus's file-path
    # resolution (relative to manifest.parent) finds the strategy without
    # any implicit base-path joining.
    assert built.strategies_base_path == built.manifest_path.parent


def test_manifest_round_trips_via_nexus_loader(tmp_path: Path) -> None:
    exp = _make_experiment_dir(tmp_path)
    builder = ManifestBuilder(output_dir=tmp_path / 'out')
    built = builder.build(
        account_id='bts-acct-0',
        allocated_capital=Decimal('100000'),
        capital_pool=Decimal('10000'),
        strategy_id='long_on_signal_q35',
        sensor=SensorBinding(
            experiment_dir=exp, permutation_ids=(1,), interval_seconds=3600,
        ),
        strategy_params=_spec(),
    )
    reloaded = load_manifest(built.manifest_path)
    assert reloaded.account_id == 'bts-acct-0'
    assert reloaded.allocated_capital == Decimal('100000')
    assert len(reloaded.strategies) == 1
    assert reloaded.strategies[0].strategy_id == 'long_on_signal_q35'
    assert reloaded.strategies[0].sensors[0].permutation_ids == (1,)
    assert reloaded.strategies[0].sensors[0].interval_seconds == 3600


def test_build_rejects_missing_experiment_dir(tmp_path: Path) -> None:
    builder = ManifestBuilder(output_dir=tmp_path / 'out')
    with pytest.raises(ValueError, match='experiment_dir not found'):
        builder.build(
            account_id='bts-acct-0',
            allocated_capital=Decimal('100000'),
            capital_pool=Decimal('10000'),
            strategy_id='s',
            sensor=SensorBinding(
                experiment_dir=tmp_path / 'missing',
                permutation_ids=(1,), interval_seconds=3600,
            ),
            strategy_params=_spec(),
        )


def test_build_rejects_missing_template(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match='strategy template'):
        ManifestBuilder(output_dir=tmp_path, template_name='no_such.py')


def test_strategy_file_has_baked_params_substituted(tmp_path: Path) -> None:
    # The strategy template contains `__BTS_PARAMS__`; ManifestBuilder
    # must substitute it with the actual JSON config so Nexus's dynamic
    # loader sees valid Python after `exec_module`.
    exp = _make_experiment_dir(tmp_path)
    builder = ManifestBuilder(output_dir=tmp_path / 'out')
    built = builder.build(
        account_id='bts-acct-0',
        allocated_capital=Decimal('100000'),
        capital_pool=Decimal('10000'),
        strategy_id='s',
        sensor=SensorBinding(
            experiment_dir=exp, permutation_ids=(1,), interval_seconds=3600,
        ),
        strategy_params=StrategyParamsSpec(
            symbol='BTCUSDT',
            capital=Decimal('50000'),
            kelly_pct=Decimal('7.5'),
            estimated_price=Decimal('63250.5'),
            stop_bps=Decimal('75'),
        ),
    )
    body = (built.strategies_base_path / 'long_on_signal.py').read_text(encoding='utf-8')
    assert '__BTS_PARAMS__' not in body, 'template placeholder was not substituted'
    assert '"symbol": "BTCUSDT"' in body
    assert '"capital": "50000"' in body
    assert '"kelly_pct": "7.5"' in body
    assert '"estimated_price": "63250.5"' in body


def test_strategy_file_is_syntactically_valid_python(tmp_path: Path) -> None:
    import ast
    exp = _make_experiment_dir(tmp_path)
    builder = ManifestBuilder(output_dir=tmp_path / 'out')
    built = builder.build(
        account_id='bts-acct-0',
        allocated_capital=Decimal('100000'),
        capital_pool=Decimal('10000'),
        strategy_id='s',
        sensor=SensorBinding(
            experiment_dir=exp, permutation_ids=(1,), interval_seconds=3600,
        ),
        strategy_params=_spec(),
    )
    body = (built.strategies_base_path / 'long_on_signal.py').read_text(encoding='utf-8')
    # Parse must not raise — any substitution error would leave invalid Python.
    tree = ast.parse(body)
    class_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    assert 'Strategy' in class_names, f'generated strategy missing Strategy class: {class_names}'


def test_strategy_params_round_trip_to_yaml(tmp_path: Path) -> None:
    import yaml
    exp = _make_experiment_dir(tmp_path)
    builder = ManifestBuilder(output_dir=tmp_path / 'out')
    built = builder.build(
        account_id='bts-acct-0',
        allocated_capital=Decimal('100000'),
        capital_pool=Decimal('10000'),
        strategy_id='s1',
        sensor=SensorBinding(
            experiment_dir=exp, permutation_ids=(1,), interval_seconds=3600,
        ),
        strategy_params=StrategyParamsSpec(
            symbol='BTCUSDT',
            capital=Decimal('50000'),
            kelly_pct=Decimal('7.5'),
            estimated_price=Decimal('63250.5'),
            stop_bps=Decimal('75'),
        ),
    )
    data = yaml.safe_load(built.manifest_path.read_text(encoding='utf-8'))
    params = data['strategies'][0]['params']
    assert params['symbol'] == 'BTCUSDT'
    assert params['capital'] == '50000'
    assert params['kelly_pct'] == '7.5'
    assert params['estimated_price'] == '63250.5'
    assert params['stop_bps'] == '75'
