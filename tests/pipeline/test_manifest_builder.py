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
        strategy_params=StrategyParamsSpec(
            symbol='BTCUSDT', enter_threshold=0.6,
            stop_bps=Decimal('50'), qty=Decimal('0.001'),
        ),
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
        strategy_params=StrategyParamsSpec(
            symbol='BTCUSDT', enter_threshold=0.6,
            stop_bps=Decimal('50'), qty=Decimal('0.001'),
        ),
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
            strategy_params=StrategyParamsSpec(
                symbol='BTCUSDT', enter_threshold=0.6,
                stop_bps=Decimal('50'), qty=Decimal('0.001'),
            ),
        )


def test_build_rejects_missing_template(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match='strategy template'):
        ManifestBuilder(output_dir=tmp_path, template_name='no_such.py')


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
            symbol='BTCUSDT', enter_threshold=0.58, stop_bps=Decimal('75'),
            qty=Decimal('0.002'), side='BUY', prob_key='p_up',
        ),
    )
    data = yaml.safe_load(built.manifest_path.read_text(encoding='utf-8'))
    params = data['strategies'][0]['params']
    assert params['symbol'] == 'BTCUSDT'
    assert params['enter_threshold'] == 0.58
    assert params['stop_bps'] == '75'
    assert params['prob_key'] == 'p_up'
