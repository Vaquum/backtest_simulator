"""Honesty gate: --input-from-file cache key separates by file + params.

Regression pin for the bug where running `bts sweep --input-from-file
max.csv` reused the cached training from a prior `bts sweep
--input-from-file min.csv` run when both files happened to have a
row with the same `id` column. The cache key was `id_{file_id}`
alone — keyed on file_id only. The fix widens the key to
`{file_stem}_id_{file_id}_{params_sha256_full}` so different files
AND different params produce different sub_dirs.

Tests exercise `pick_decoders` directly with a monkeypatched
`train_single_decoder` to capture the sub_dir produced by the
production path. Reconstructing the path string in-test would let
production drift back to the buggy `id_{file_id}` shape without
the tests catching it (codex round-1 P1).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from backtest_simulator.cli import _pipeline

# Minimum columns `pick_decoders` reads: id + 12 _PARAM_COLS + the
# 3 ranking/filter columns it sorts/filters by. Anything else can
# be omitted.
_PARAM_COLS = (
    'frac_diff_d', 'shift', 'q', 'roc_period', 'penalty', 'scaler_type',
    'feature_groups', 'class_weight', 'C', 'max_iter', 'solver', 'tol',
)


def _write_csv(path: Path, *, q_value: float = 0.41) -> None:
    df = pl.DataFrame({
        'id': [0],
        'backtest_mean_kelly_pct': [11.434],
        'backtest_total_return_net_pct': [81.5],
        'backtest_trades_count': [488],
        # _PARAM_COLS:
        'frac_diff_d': [0.0],
        'shift': [-1],
        'q': [q_value],
        'roc_period': [12],
        'penalty': ['l2'],
        'scaler_type': ['robust'],
        'feature_groups': ['momentum'],
        'class_weight': [0.55],
        'C': [1.0],
        'max_iter': [60],
        'solver': ['lbfgs'],
        'tol': [0.01],
    })
    df.write_csv(path)


def _capture_sub_dirs(
    monkeypatch: pytest.MonkeyPatch, csv_path: Path,
) -> list[Path]:
    """Run `pick_decoders` with `train_single_decoder` patched.

    Returns the list of `sub_dir` paths the production path would
    have trained into. We do NOT actually train (saves seconds
    per test) — but the cache-key derivation runs unchanged, so
    the captured path is the production cache key.
    """
    captured: list[Path] = []

    def _capture(sub_dir: Path, params: dict[str, object]) -> None:
        del params
        captured.append(sub_dir)
        # Don't actually train; the cache key is the test's interest.

    monkeypatch.setattr(_pipeline, 'train_single_decoder', _capture)
    _pipeline.pick_decoders(
        n=1,
        input_from_file=str(csv_path),
    )
    return captured


def test_pick_decoders_cache_dir_changes_with_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same row content, different filename -> different sub_dir.

    The operator's reported bug. Mutation proof: dropping
    `file_path.stem` from the cache key makes both produce
    identical paths and this assert fires.
    """
    min_csv = tmp_path / 'min.csv'
    max_csv = tmp_path / 'max.csv'
    _write_csv(min_csv)
    _write_csv(max_csv)
    sub_min = _capture_sub_dirs(monkeypatch, min_csv)[0]
    sub_max = _capture_sub_dirs(monkeypatch, max_csv)[0]
    assert sub_min != sub_max, (
        f'cache sub_dir for min.csv and max.csv with same id + same '
        f'params must differ; got both = {sub_min}'
    )
    assert 'min' in sub_min.name and 'max' in sub_max.name, (
        f'sub_dir names should contain the file stem; got '
        f'min={sub_min.name}, max={sub_max.name}'
    )


def test_pick_decoders_cache_dir_changes_with_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same filename + same id + different params -> different sub_dir.

    Operator overwrites a CSV with the same name but edits the
    row's hyperparameters. The params hash MUST invalidate the
    cache. Mutation proof: dropping the params hash makes both
    versions produce identical paths and this assert fires.
    """
    csv = tmp_path / 'data.csv'
    _write_csv(csv, q_value=0.41)
    sub_v1 = _capture_sub_dirs(monkeypatch, csv)[0]
    _write_csv(csv, q_value=0.50)  # one hyperparameter changed
    sub_v2 = _capture_sub_dirs(monkeypatch, csv)[0]
    assert sub_v1 != sub_v2, (
        f'cache sub_dir must differ when params change; got both = '
        f'{sub_v1}'
    )


def test_pick_decoders_cache_dir_stable_for_identical_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same file + same id + same params -> SAME sub_dir.

    The fix must not break legitimate caching. If the operator runs
    the same filter twice, the second run SHOULD reuse the cached
    training. Pin that the cache key is stable across invocations
    with identical input.
    """
    csv = tmp_path / 'data.csv'
    _write_csv(csv)
    sub_a = _capture_sub_dirs(monkeypatch, csv)[0]
    sub_b = _capture_sub_dirs(monkeypatch, csv)[0]
    assert sub_a == sub_b, (
        f'cache sub_dir must be stable for identical input; got '
        f'{sub_a} vs {sub_b}'
    )


def test_pick_decoders_strips_whitespace_in_numeric_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operator-supplied CSV with leading whitespace on numeric values
    must still cast cleanly.

    Some operator-side exports format floats with a leading space
    (e.g. ` -0.343` from `to_csv(float_format=' %.3f')`). Without
    `str.strip_chars()` before the cast, polars returns null for
    every padded value -> the entire pool drops on the null filter
    -> the operator sees a confusing TypeError downstream.

    Mutation proof: dropping the strip step makes the post-cast
    nulls match the row count, the null-drop empties the pool,
    and the test would either crash or assert n_usable=0 (the
    behaviour observed pre-fix).
    """
    csv = tmp_path / 'whitespace.csv'
    df = pl.DataFrame({
        'id': [0, 1, 2],
        'backtest_mean_kelly_pct': [' 11.434', ' 11.500', ' 11.600'],
        'backtest_total_return_net_pct': [' 81.5', ' 82.0', ' 82.5'],
        'backtest_trades_count': [' 488', ' 490', ' 492'],
        'frac_diff_d': [' 0.0', ' 0.0', ' 0.0'],
        'shift': [-1, -1, -1],
        'q': [0.4, 0.4, 0.4],
        'roc_period': [12, 12, 12],
        'penalty': ['l2', 'l2', 'l2'],
        'scaler_type': ['robust', 'robust', 'robust'],
        'feature_groups': ['momentum', 'momentum', 'momentum'],
        'class_weight': [0.55, 0.55, 0.55],
        'C': [1.0, 1.0, 1.0],
        'max_iter': [60, 60, 60],
        'solver': ['lbfgs', 'lbfgs', 'lbfgs'],
        'tol': [0.01, 0.01, 0.01],
    })
    df.write_csv(csv)
    sub_dirs = _capture_sub_dirs(monkeypatch, csv)
    # If whitespace stripping works, all 3 rows survive the cast +
    # null-drop, and pick_decoders returns 1 pick (n=1 in
    # _capture_sub_dirs).
    assert len(sub_dirs) == 1, (
        f'whitespace-stripped numeric columns must produce a pick; '
        f'got {len(sub_dirs)} (likely all rows dropped to null cast)'
    )


def test_pick_decoders_fails_loudly_on_zero_usable_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All-null rank+kelly columns -> RuntimeError with named columns.

    Pre-fix: the code continued into `_q(col, ...)` which called
    `.quantile()` on an empty Series, returning None, and crashed
    with `TypeError: float() argument must be a string or a real
    number, not 'NoneType'`. The operator had to debug the full
    stack trace to learn that all rows were dropped.

    Post-fix: a clear RuntimeError naming the column set.

    Mutation proof: removing the `if results.height == 0` block
    re-introduces the cryptic TypeError downstream.
    """
    csv = tmp_path / 'all_null.csv'
    df = pl.DataFrame({
        'id': [0, 1, 2],
        # The values cast to null because they are non-numeric.
        'backtest_mean_kelly_pct': ['n/a', 'n/a', 'n/a'],
        'backtest_total_return_net_pct': ['n/a', 'n/a', 'n/a'],
        'backtest_trades_count': [488, 490, 492],
        'frac_diff_d': [0.0, 0.0, 0.0],
        'shift': [-1, -1, -1],
        'q': [0.4, 0.4, 0.4],
        'roc_period': [12, 12, 12],
        'penalty': ['l2', 'l2', 'l2'],
        'scaler_type': ['robust', 'robust', 'robust'],
        'feature_groups': ['momentum', 'momentum', 'momentum'],
        'class_weight': [0.55, 0.55, 0.55],
        'C': [1.0, 1.0, 1.0],
        'max_iter': [60, 60, 60],
        'solver': ['lbfgs', 'lbfgs', 'lbfgs'],
        'tol': [0.01, 0.01, 0.01],
    })
    df.write_csv(csv)

    def _capture(sub_dir: Path, params: dict[str, object]) -> None:
        del sub_dir, params

    monkeypatch.setattr(_pipeline, 'train_single_decoder', _capture)
    with pytest.raises(RuntimeError, match='0 usable rows'):
        _pipeline.pick_decoders(n=1, input_from_file=str(csv))


def test_pick_decoders_cache_uses_full_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache directory uses full 64-char SHA-256 (codex round-1 P1).

    Truncating to 8 hex chars gives a 32-bit hash space — birthday
    paradox makes collisions likely around 65k entries. Operators
    running large CSV-driven sweeps could silently alias different
    rows to the same trained decoder. Full digest closes the
    window.

    Mutation proof: re-truncating to [:8] makes the suffix length
    drop to 8 and this assert fires.
    """
    csv = tmp_path / 'data.csv'
    _write_csv(csv)
    sub_dir = _capture_sub_dirs(monkeypatch, csv)[0]
    # Format: {stem}_id_{file_id}_{hex64}
    parts = sub_dir.name.rsplit('_', 1)
    assert len(parts) == 2, f'unexpected sub_dir shape: {sub_dir.name!r}'
    hash_suffix = parts[1]
    assert len(hash_suffix) == 64, (
        f'cache hash suffix must be the full 64-char SHA-256; got '
        f'{len(hash_suffix)} chars: {hash_suffix!r}'
    )
    # Also confirm it's hex.
    assert all(c in '0123456789abcdef' for c in hash_suffix), (
        f'hash suffix must be lowercase hex; got {hash_suffix!r}'
    )
