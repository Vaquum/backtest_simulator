"""Tests for `read_kline_size_from_experiment_dir` and `_resolve_grid_interval`.

The bundle path's correctness rests on these two helpers. Without
them, a bundle's `kline_size=7200` would silently fall back to the
hardcoded 3600s, the runtime PredictLoop and the parity grid would
disagree, and every fire would raise `ParityViolation`.
"""
from __future__ import annotations

import json
import sys
import textwrap
from decimal import Decimal
from pathlib import Path

import pytest

from backtest_simulator.cli._run_window import read_kline_size_from_experiment_dir
from backtest_simulator.cli.commands.sweep import _resolve_grid_interval


def _write_fake_experiment_dir(
    tmp_path: Path, *, kline_size: int, sub: str = 'exp',
) -> Path:
    """Build a stub experiment_dir + a stub `_kline_module_<kline_size>.py`
    SFD module such that `read_kline_size_from_experiment_dir` returns
    `kline_size`. The helper imports `metadata['sfd_module']`'s
    `manifest()`. We give it a tiny module on `sys.path` and write
    metadata.json pointing at it.
    """
    exp_dir = tmp_path / sub
    exp_dir.mkdir()
    mod_name = f'_kline_module_{sub}_{kline_size}'
    mod_path = tmp_path / f'{mod_name}.py'
    mod_path.write_text(textwrap.dedent(f'''
        from dataclasses import dataclass

        @dataclass
        class _Cfg:
            params: dict

        @dataclass
        class _M:
            data_source_config: _Cfg

        def manifest():
            return _M(data_source_config=_Cfg(params={{"kline_size": {kline_size}}}))
    ''').strip(), encoding='utf-8')
    metadata = {'sfd_module': mod_name}
    (exp_dir / 'metadata.json').write_text(
        json.dumps(metadata), encoding='utf-8',
    )
    sys.path.insert(0, str(tmp_path))
    return exp_dir


@pytest.fixture(autouse=True)
def _cleanup_sys_path() -> object:
    yield
    # Drop any per-test paths added by the helper
    sys.path[:] = [p for p in sys.path if not p.startswith('/private/var/folders')]
    # And evict our stub modules
    for name in list(sys.modules):
        if name.startswith('_kline_module_'):
            del sys.modules[name]


def test_read_kline_size_returns_manifest_value(tmp_path: Path) -> None:
    exp = _write_fake_experiment_dir(tmp_path, kline_size=7200)
    assert read_kline_size_from_experiment_dir(exp) == 7200


def test_read_kline_size_handles_3600_baseline(tmp_path: Path) -> None:
    exp = _write_fake_experiment_dir(tmp_path, kline_size=3600)
    assert read_kline_size_from_experiment_dir(exp) == 3600


def test_read_kline_size_handles_5min_kline(tmp_path: Path) -> None:
    """5-minute klines (kline_size=300) — sub-hour cadence the parity
    grid must also honor.
    """
    exp = _write_fake_experiment_dir(tmp_path, kline_size=300)
    assert read_kline_size_from_experiment_dir(exp) == 300


def test_read_kline_size_missing_metadata_raises(tmp_path: Path) -> None:
    """Helper must raise when metadata.json isn't there — silent
    fallback to a default would mask the bundle path's data-source
    declaration.
    """
    exp = tmp_path / 'no_meta'
    exp.mkdir()
    with pytest.raises(FileNotFoundError, match=r'metadata\.json not found'):
        read_kline_size_from_experiment_dir(exp)


def test_read_kline_size_missing_kline_size_key_raises(tmp_path: Path) -> None:
    """If the SFD's manifest declines to declare `kline_size`, the
    helper must raise — silent default would diverge the runtime
    Timer cadence from the data feed.
    """
    exp = tmp_path / 'no_kline'
    exp.mkdir()
    mod_name = '_kline_module_no_kline'
    mod_path = tmp_path / f'{mod_name}.py'
    mod_path.write_text(
        'from dataclasses import dataclass\n'
        '@dataclass\n'
        'class _Cfg:\n'
        '    params: dict\n'
        '@dataclass\n'
        'class _M:\n'
        '    data_source_config: _Cfg\n'
        '\n'
        'def manifest():\n'
        # No kline_size key in params
        '    return _M(data_source_config=_Cfg(params={"start_date_limit": "2024-01-01"}))\n',
        encoding='utf-8',
    )
    (exp / 'metadata.json').write_text(
        json.dumps({'sfd_module': mod_name}), encoding='utf-8',
    )
    sys.path.insert(0, str(tmp_path))
    with pytest.raises(KeyError):
        read_kline_size_from_experiment_dir(exp)


def test_resolve_grid_interval_homogeneous_picks(tmp_path: Path) -> None:
    """All picks point at experiment_dirs with the same kline_size →
    return that kline_size as the parity grid cadence.
    """
    exp_a = _write_fake_experiment_dir(tmp_path, kline_size=7200, sub='a')
    exp_b = _write_fake_experiment_dir(tmp_path, kline_size=7200, sub='b')
    picks = [
        (1, Decimal('1.5'), exp_a, 1),
        (2, Decimal('1.2'), exp_b, 2),
    ]
    assert _resolve_grid_interval(picks) == 7200


def test_resolve_grid_interval_mixed_kline_sizes_raises(
    tmp_path: Path,
) -> None:
    """Picks declaring different kline_sizes → raise. The parity grid
    can't be one cadence if the picks disagree, so this is a hard
    error rather than a silent pick-the-min.
    """
    exp_2h = _write_fake_experiment_dir(tmp_path, kline_size=7200, sub='two')
    exp_1h = _write_fake_experiment_dir(tmp_path, kline_size=3600, sub='one')
    picks = [
        (1, Decimal('1.5'), exp_2h, 1),
        (2, Decimal('1.2'), exp_1h, 2),
    ]
    with pytest.raises(ValueError, match='more than one kline_size'):
        _resolve_grid_interval(picks)
