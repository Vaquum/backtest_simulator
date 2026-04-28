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

import importlib
import json
import sys
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


def _write_exp_code(path: Path) -> None:
    """Write a minimal UEL-compliant exp.py for tests.

    Has module-level `params()` and `manifest()` callables (the
    `ExperimentPipeline.load_from_file` contract); no `uel.run`
    side effect on import. Tests don't actually train (the
    `train_single_decoder` capture below short-circuits) so the
    manifest content is irrelevant — `pick_decoders` only reads
    `exp_code_path` to bake into the cache key.
    """
    path.write_text(
        'from limen.sfd.foundational_sfd import logreg_binary as _base\n'
        'params = _base.params\n'
        'manifest = _base.manifest\n',
        encoding='utf-8',
    )


def _capture_sub_dirs(
    monkeypatch: pytest.MonkeyPatch, csv_path: Path,
    *, exp_code_path: Path | None = None,
) -> list[Path]:
    """Run `pick_decoders` with `train_single_decoder` patched.

    Returns the list of `sub_dir` paths the production path would
    have trained into. We do NOT actually train (saves seconds
    per test) — but the cache-key derivation runs unchanged, so
    the captured path is the production cache key.
    """
    captured: list[Path] = []

    def _capture(
        sub_dir: Path, params: dict[str, object],
        exp_code: Path, op_param_keys: tuple[str, ...],
    ) -> None:
        del params, exp_code, op_param_keys
        captured.append(sub_dir)

    monkeypatch.setattr(_pipeline, 'train_single_decoder', _capture)
    if exp_code_path is None:
        exp_code_path = csv_path.parent / 'default_exp.py'
        _write_exp_code(exp_code_path)
    _pipeline.pick_decoders(
        n=1,
        exp_code_path=exp_code_path,
        n_permutations=1,
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
    exp_code = csv.parent / 'exp.py'
    _write_exp_code(exp_code)

    def _capture(
        sub_dir: Path, params: dict[str, object], exp: Path,
        op_param_keys: tuple[str, ...],
    ) -> None:
        del sub_dir, params, exp, op_param_keys

    monkeypatch.setattr(_pipeline, 'train_single_decoder', _capture)
    with pytest.raises(RuntimeError, match='0 usable rows'):
        _pipeline.pick_decoders(
            n=1, exp_code_path=exp_code, n_permutations=1,
            input_from_file=str(csv),
        )


def test_pick_decoders_requires_exp_code_file(tmp_path: Path) -> None:
    """Missing --exp-code file -> FileNotFoundError, no fallback.

    Operator-mandated contract: bts must NOT run without an
    explicit UEL-compliant code file. Any prior fallback path
    (auto-writing a logreg_binary `exp.py`) is gone. Mutation
    proof: re-introducing a default file at `EXP_DIR / 'exp.py'`
    would let `pick_decoders` succeed with no exp_code argument
    and this assert would fail.
    """
    csv = tmp_path / 'pool.csv'
    _write_csv(csv)
    nonexistent_exp = tmp_path / 'does_not_exist.py'
    with pytest.raises(FileNotFoundError, match='--exp-code file not found'):
        _pipeline.pick_decoders(
            n=1, exp_code_path=nonexistent_exp, n_permutations=1,
            input_from_file=str(csv),
        )


def test_pick_decoders_cache_dir_changes_with_exp_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same CSV + same params + DIFFERENT exp-code -> different sub_dir.

    Two operator's exp-code files (different SFDs / different
    feature recipes) operating on the same input CSV's row must
    produce different cache sub_dirs — otherwise switching the
    SFD would silently reuse a stale training. Mutation proof:
    dropping `exp_code_path` from the cache hash makes both
    produce identical sub_dirs and this assert fires.
    """
    csv = tmp_path / 'pool.csv'
    _write_csv(csv)
    exp_a = tmp_path / 'sfd_a.py'
    exp_b = tmp_path / 'sfd_b.py'
    _write_exp_code(exp_a)
    # Make exp_b text-different from exp_a so the path hash differs.
    exp_b.write_text(
        'from limen.sfd.foundational_sfd import logreg_binary as _b\n'
        '# different sfd\n'
        'params = _b.params\n'
        'manifest = _b.manifest\n',
        encoding='utf-8',
    )
    sub_a = _capture_sub_dirs(monkeypatch, csv, exp_code_path=exp_a)[0]
    sub_b = _capture_sub_dirs(monkeypatch, csv, exp_code_path=exp_b)[0]
    assert sub_a != sub_b, (
        f'cache sub_dir must differ when exp-code path differs; '
        f'got both = {sub_a}'
    )


def test_pick_decoders_cache_dir_changes_with_exp_code_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same exp-code PATH but different CONTENT -> different sub_dir.

    Codex round-1 P1: hashing only the path lets in-place edits
    silently reuse stale trainings. Hashing the file content
    invalidates the cache on every edit. Mutation proof: hashing
    `str(exp_code_path)` instead of `exp_code_path.read_bytes()`
    makes both versions produce the same sub_dir and this assert
    fires.
    """
    csv = tmp_path / 'pool.csv'
    _write_csv(csv)
    exp = tmp_path / 'sfd.py'
    _write_exp_code(exp)
    sub_v1 = _capture_sub_dirs(monkeypatch, csv, exp_code_path=exp)[0]
    # Edit exp-code in place — same path, different content.
    exp.write_text(
        exp.read_text(encoding='utf-8') + '\n# v2 edit\n',
        encoding='utf-8',
    )
    sub_v2 = _capture_sub_dirs(monkeypatch, csv, exp_code_path=exp)[0]
    assert sub_v1 != sub_v2, (
        f'cache sub_dir must differ when exp-code FILE CONTENT '
        f'changes; got both = {sub_v1}'
    )


def test_run_uses_operator_manifest_in_uel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ExperimentPipeline.run must pass operator's MODULE as SFD to UEL.

    Codex round-1 P0: the prior implementation passed
    `self._sfd` (defaulting to `logreg_binary`) into UEL,
    silently ignoring the operator's manifest. Without the fix,
    every run used logreg_binary's manifest regardless of what
    the operator's `--exp-code` file declared.

    Mutation proof: reverting `sfd=experiment_file.module` to
    `sfd=self._sfd` makes the captured UEL receive logreg_binary
    instead of the operator's module, and this assert fires.
    """
    from backtest_simulator.pipeline import experiment as exp_module
    from backtest_simulator.pipeline.experiment import ExperimentPipeline

    captured: dict[str, object] = {}

    class _FakeUEL:
        def __init__(self, **kw):
            captured['sfd'] = kw.get('sfd')

        def run(self, **kw):
            del kw
            # No-op: don't actually execute uel; we're verifying
            # the SFD passed to its constructor.

    monkeypatch.setattr(exp_module, 'UniversalExperimentLoop', _FakeUEL)

    exp_code = tmp_path / 'op_sfd.py'
    _write_exp_code(exp_code)
    pipe = ExperimentPipeline(experiment_dir=tmp_path / 'work')
    loaded = pipe.load_from_file(exp_code)
    pipe.run(loaded, experiment_name='t', n_permutations=1)

    sfd_passed = captured['sfd']
    assert sfd_passed is loaded.module, (
        f'UEL must receive the operator\'s module as the SFD '
        f'(got {sfd_passed!r}). The constructor default '
        f'`logreg_binary` would silently override the operator\'s '
        f'manifest.'
    )


def test_pick_decoders_rejects_non_uel_compliant_exp_code(
    tmp_path: Path,
) -> None:
    """Exp-code without module-level params/manifest raises loudly.

    The contract: file MUST have module-level `params()` and
    `manifest()` callables. Operator's typical structure
    (`class Round3SFD: @staticmethod def params(): ...`) needs
    module-level aliases (`params = Round3SFD.params; manifest =
    Round3SFD.manifest`) to satisfy the contract. A file without
    them must fail at load time, not silently mis-train.

    The non-input_from_file path triggers
    `ensure_trained_from_exp_code` which calls
    `ExperimentPipeline.load_from_file` — that helper enforces
    the contract.
    """
    bad_exp = tmp_path / 'bad.py'
    bad_exp.write_text(
        'class MySfd:\n'
        '    @staticmethod\n'
        '    def params(): return {}\n'
        '    @staticmethod\n'
        '    def manifest(): return None\n'
        '# NOTE: no module-level params/manifest aliases.\n',
        encoding='utf-8',
    )
    with pytest.raises(
        ValueError, match=r'must define callable.*params.*manifest',
    ):
        _pipeline.ensure_trained_from_exp_code(bad_exp, n_permutations=1)


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


def test_ensure_trained_uses_reimportable_snapshot_module_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex round-2 P0: metadata['sfd_module'] must be reimportable.

    Limen persists `module.__name__` to `metadata.json["sfd_module"]`,
    and `Trainer.train()` later does `importlib.import_module(...)`
    on it — possibly in a fresh subprocess. A path-loaded module's
    `__name__` is the bare file stem (`op_sfd`), which is NOT on
    sys.path, so the reimport fails.

    Fix verified: `ensure_trained_from_exp_code` snapshots the
    operator's file into bts's content-addressed `_OP_SFD_CACHE`,
    and the SFD passed to UEL has `__name__ = _bts_op_<sha16>` —
    a reimportable name once `_OP_SFD_CACHE` is on sys.path.

    This test:
      1. Places exp-code OUTSIDE sys.path
      2. Runs `ensure_trained_from_exp_code` (UEL stubbed)
      3. Reads the captured `metadata['sfd_module']` name
      4. Drops it from sys.modules to simulate a fresh subprocess
      5. Asserts `importlib.import_module(name)` succeeds — the
         path Limen's Trainer takes

    Mutation proof: skipping the snapshot in
    `ensure_trained_from_exp_code` makes the SFD's `__name__`
    revert to the bare operator stem; the assert that the name
    starts with `_bts_op_` fires; even if that assert is removed,
    the `import_module` would fail with `ModuleNotFoundError`.
    """
    isolated = tmp_path / 'isolated'
    isolated.mkdir()
    exp_code = isolated / 'op_sfd.py'
    _write_exp_code(exp_code)

    captured: dict[str, object] = {}

    class _FakeUEL:
        def __init__(self, **kw: object) -> None:
            captured['sfd'] = kw.get('sfd')
            captured['experiment_dir'] = kw.get('experiment_dir')

        def run(self, **kw: object) -> None:
            del kw
            sfd = captured['sfd']
            assert sfd is not None
            exp_dir = captured['experiment_dir']
            assert isinstance(exp_dir, Path)
            exp_dir.mkdir(parents=True, exist_ok=True)
            (exp_dir / 'metadata.json').write_text(
                json.dumps({'sfd_module': sfd.__name__}),  # type: ignore[attr-defined]
                encoding='utf-8',
            )
            (exp_dir / 'results.csv').write_text(
                'id\n0\n', encoding='utf-8',
            )

    from backtest_simulator.pipeline import experiment as exp_module
    monkeypatch.setattr(exp_module, 'UniversalExperimentLoop', _FakeUEL)
    monkeypatch.setattr(_pipeline, 'WORK_DIR', tmp_path / 'work')
    monkeypatch.setattr(
        _pipeline, '_OP_SFD_CACHE', tmp_path / 'work' / 'op_sfds',
    )

    cache_dir = _pipeline.ensure_trained_from_exp_code(
        exp_code, n_permutations=1,
    )

    metadata = json.loads(
        (cache_dir / 'metadata.json').read_text(encoding='utf-8'),
    )
    sfd_module_name = metadata['sfd_module']

    assert sfd_module_name.startswith(_pipeline._OP_SFD_MODULE_PREFIX), (
        f"metadata['sfd_module'] must be a bts content-addressed "
        f'snapshot name (got {sfd_module_name!r}); the operator\'s '
        f'bare stem ({exp_code.stem!r}) is not reimportable across '
        f'subprocess boundaries.'
    )

    sys.modules.pop(sfd_module_name, None)

    reimported = importlib.import_module(sfd_module_name)
    assert hasattr(reimported, 'manifest'), (
        f'reimported module {sfd_module_name} missing `manifest`; '
        f'snapshot is incomplete'
    )
    assert hasattr(reimported, 'params'), (
        f'reimported module {sfd_module_name} missing `params`'
    )


def test_pick_decoders_uses_operator_params_keys_not_hardcoded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex round-2 P1: per-decoder params keys come from operator's params().

    Operator's exp-code declares 3 custom keys (`alpha`, `beta`,
    `gamma`) — NONE of which appear in logreg_binary's 12-key
    grid. The prior `_PARAM_COLS` constant baked logreg's keys
    into `train_single_decoder`, so a non-logreg SFD would
    silently train on the wrong grid.

    This test pins that `train_single_decoder` receives the
    operator's keys (not logreg's). Mutation proof: re-introducing
    a `_PARAM_COLS = ('C', 'penalty', ...)` constant and using it
    in `pick_decoders` flips `op_param_keys` to logreg's set, and
    this assert fires.
    """
    custom_exp = tmp_path / 'custom_sfd.py'
    custom_exp.write_text(
        'def params():\n'
        '    return {"alpha": [1, 2], "beta": ["x", "y"], "gamma": [0.1]}\n'
        'def manifest():\n'
        '    return None\n',
        encoding='utf-8',
    )

    csv = tmp_path / 'pool.csv'
    df = pl.DataFrame({
        'id': [0],
        'backtest_mean_kelly_pct': [11.0],
        'backtest_total_return_net_pct': [80.0],
        'backtest_trades_count': [400],
        'alpha': [1],
        'beta': ['x'],
        'gamma': [0.5],
    })
    df.write_csv(csv)

    captured_keys: list[tuple[str, ...]] = []
    captured_params: list[dict[str, object]] = []

    def _capture(
        sub_dir: Path, params: dict[str, object],
        exp_code: Path, op_param_keys: tuple[str, ...],
    ) -> None:
        del sub_dir, exp_code
        captured_keys.append(op_param_keys)
        captured_params.append(dict(params))

    monkeypatch.setattr(_pipeline, 'train_single_decoder', _capture)
    monkeypatch.setattr(
        _pipeline, '_OP_SFD_CACHE', tmp_path / 'op_sfds',
    )

    _pipeline.pick_decoders(
        n=1, exp_code_path=custom_exp, n_permutations=1,
        input_from_file=str(csv),
    )

    assert captured_keys[0] == ('alpha', 'beta', 'gamma'), (
        f"op_param_keys must come from operator's params() "
        f'({{alpha, beta, gamma}}); got {captured_keys[0]!r}. A '
        f'hardcoded _PARAM_COLS constant would yield logreg\'s '
        f'12 keys (C, penalty, ...).'
    )
    assert captured_params[0] == {
        'alpha': 1, 'beta': 'x', 'gamma': 0.5,
    }, (
        f'per-decoder params dict must contain CSV row values '
        f"for the operator's declared keys; got "
        f'{captured_params[0]!r}'
    )


def test_pick_decoders_rejects_csv_missing_operator_param_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CSV missing operator-declared param columns -> ValueError.

    Codex round-2 P1: with op_param_keys driven by the operator's
    `params()`, a CSV that doesn't have one column per declared
    key cannot supply the per-decoder params dict. Fail loudly
    before the per-decoder train kicks off (vs. crashing inside
    `train_single_decoder` with a confusing KeyError).

    Mutation proof: skipping the columns check would let the bug
    surface deep in `params[k]` lookups inside
    `train_single_decoder` — much harder to diagnose.
    """
    custom_exp = tmp_path / 'custom_sfd.py'
    custom_exp.write_text(
        'def params(): return {"alpha": [1], "beta": ["x"]}\n'
        'def manifest(): return None\n',
        encoding='utf-8',
    )
    csv = tmp_path / 'pool.csv'
    # CSV has alpha but not beta
    df = pl.DataFrame({
        'id': [0],
        'backtest_mean_kelly_pct': [11.0],
        'backtest_total_return_net_pct': [80.0],
        'backtest_trades_count': [400],
        'alpha': [1],
    })
    df.write_csv(csv)

    monkeypatch.setattr(
        _pipeline, '_OP_SFD_CACHE', tmp_path / 'op_sfds',
    )

    with pytest.raises(
        ValueError, match=r'missing columns.*params.*beta',
    ):
        _pipeline.pick_decoders(
            n=1, exp_code_path=custom_exp, n_permutations=1,
            input_from_file=str(csv),
        )


def test_train_single_decoder_per_decoder_module_is_reimportable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex round-3 P0: per-decoder metadata['sfd_module'] must be reimportable.

    The original round-3 fix only snapshotted the OPERATOR'S
    exp-code; the per-decoder `exp.py` generated inside
    `train_single_decoder` was still written to `sub_dir/exp.py`
    and loaded via `spec_from_file_location('exp', ...)`. That
    made `module.__name__ == 'exp'` (the bare stem), which Limen
    persisted to the per-decoder `metadata.json["sfd_module"]`.
    A fresh subprocess (`BacktestLauncher` / `Trainer.train()`)
    later did `importlib.import_module('exp')` — broken because
    `sub_dir` is not on PYTHONPATH.

    Codex's reproduction:
        {"sfd_module": "exp"}
        ModuleNotFoundError: No module named 'exp'

    The fix snapshots the generated per-decoder body to
    `_OP_SFD_CACHE / _bts_pd_<sha16>.py` and loads from there;
    `metadata['sfd_module']` becomes the importable hash-name.

    Mutation proof: reverting `train_single_decoder` to write
    `sub_dir/exp.py` and load it directly makes
    `metadata['sfd_module'] == 'exp'` and `import_module('exp')`
    raises `ModuleNotFoundError` — exactly the codex repro.
    """
    from backtest_simulator.pipeline import experiment as exp_module

    op_cache = tmp_path / 'op_sfds'
    monkeypatch.setattr(_pipeline, '_OP_SFD_CACHE', op_cache)

    exp_code = tmp_path / 'op.py'
    exp_code.write_text(
        'def params(): return {"alpha": [1]}\n'
        'def manifest(): return None\n',
        encoding='utf-8',
    )
    sub_dir = tmp_path / 'trained'

    class _FakeUEL:
        def __init__(self, **kw: object) -> None:
            self.sfd = kw['sfd']
            self.exp_dir = Path(kw['experiment_dir'])  # type: ignore[arg-type]

        def run(self, **kw: object) -> None:
            del kw
            self.exp_dir.mkdir(parents=True, exist_ok=True)
            (self.exp_dir / 'metadata.json').write_text(
                json.dumps({'sfd_module': self.sfd.__name__}),  # type: ignore[attr-defined]
                encoding='utf-8',
            )
            (self.exp_dir / 'results.csv').write_text(
                'id\n0\n', encoding='utf-8',
            )

    monkeypatch.setattr(exp_module, 'UniversalExperimentLoop', _FakeUEL)

    _pipeline.train_single_decoder(
        sub_dir, {'alpha': 1}, exp_code, ('alpha',),
    )

    metadata = json.loads(
        (sub_dir / 'metadata.json').read_text(encoding='utf-8'),
    )
    name = metadata['sfd_module']
    assert name.startswith('_bts_pd_'), (
        f'per-decoder metadata sfd_module must be a bts content-'
        f'addressed snapshot name (got {name!r}); the bare '
        f'`exp` stem is not reimportable across subprocess '
        f'boundaries.'
    )

    # Drop from sys.modules to force a fresh import — same path
    # Limen's Trainer takes inside a subprocess.
    sys.modules.pop(name, None)
    # The cache dir is already on sys.path via _snapshot_exp_code;
    # subprocess workers get it via PYTHONPATH propagation. Reset
    # for hermetic test verification: the snapshot must resolve
    # purely through the cache dir entry on sys.path.
    cache_str = str(op_cache)
    if cache_str not in sys.path:
        sys.path.insert(0, cache_str)
    reimported = importlib.import_module(name)
    assert hasattr(reimported, 'manifest'), (
        f'per-decoder reimport {name} missing `manifest`'
    )
    assert callable(reimported.params), (
        f'per-decoder reimport {name} missing callable `params`'
    )
    assert reimported.params() == {'alpha': [1]}, (
        f'per-decoder params() must return the picked CSV row\'s '
        f'1-element grid; got {reimported.params()!r}'
    )


def test_train_single_decoder_repairs_stale_round3_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex round-4 P0: stale cache-hit ('sfd_module' == 'exp') must self-repair.

    A `sub_dir` left over from the round-3 build (or any older
    bts) records `metadata['sfd_module'] = 'exp'` (the bare
    stem). On cache hit (`results.csv` exists) the round-3
    `train_single_decoder` returned early WITHOUT validating
    that the recorded sfd_module is reimportable. A fresh
    subprocess later calling `importlib.import_module('exp')`
    raises `ModuleNotFoundError` because no `exp.py` lives in
    `_OP_SFD_CACHE`.

    Codex's reproduction (round 4):
        seeded stale dir; existing metadata sfd_module: exp
        # Without fix: train_single_decoder returns immediately,
        # metadata stays 'exp', import fails.
        # With fix: train_single_decoder detects stale, wipes,
        # retrains; metadata now '_bts_pd_<sha16>', import OK.

    This test seeds exactly the stale shape codex described
    (results.csv + metadata['sfd_module']='exp'), calls
    `train_single_decoder`, and asserts the result is repaired
    in place — operator's `--input-from-file` rerun must be
    self-healing without `rm -rf`.

    Mutation proof: removing the
    `_cache_dir_matches_expected_module` check (or weakening it to
    only check `results.csv`) keeps the stale `sfd_module='exp'`
    and `import_module` raises.
    """
    from backtest_simulator.pipeline import experiment as exp_module

    op_cache = tmp_path / 'op_sfds'
    monkeypatch.setattr(_pipeline, '_OP_SFD_CACHE', op_cache)

    exp_code = tmp_path / 'op.py'
    exp_code.write_text(
        'def params(): return {"alpha": [1]}\n'
        'def manifest(): return None\n',
        encoding='utf-8',
    )
    sub_dir = tmp_path / 'trained'
    sub_dir.mkdir()
    # Seed the stale round-3-era artifacts.
    (sub_dir / 'metadata.json').write_text(
        json.dumps({'sfd_module': 'exp'}), encoding='utf-8',
    )
    (sub_dir / 'results.csv').write_text('id\n0\n', encoding='utf-8')
    (sub_dir / 'round_data.jsonl').write_text('', encoding='utf-8')
    (sub_dir / 'exp.py').write_text(
        'def params(): return {"alpha": [1]}\n'
        'def manifest(): return None\n',
        encoding='utf-8',
    )

    class _FakeUEL:
        def __init__(self, **kw: object) -> None:
            self.sfd = kw['sfd']
            self.exp_dir = Path(kw['experiment_dir'])  # type: ignore[arg-type]

        def run(self, **kw: object) -> None:
            del kw
            self.exp_dir.mkdir(parents=True, exist_ok=True)
            (self.exp_dir / 'metadata.json').write_text(
                json.dumps({'sfd_module': self.sfd.__name__}),  # type: ignore[attr-defined]
                encoding='utf-8',
            )
            (self.exp_dir / 'results.csv').write_text(
                'id\n0\n', encoding='utf-8',
            )

    monkeypatch.setattr(exp_module, 'UniversalExperimentLoop', _FakeUEL)

    _pipeline.train_single_decoder(
        sub_dir, {'alpha': 1}, exp_code, ('alpha',),
    )

    metadata = json.loads(
        (sub_dir / 'metadata.json').read_text(encoding='utf-8'),
    )
    name = metadata['sfd_module']
    assert name.startswith('_bts_pd_'), (
        f'stale cache must be repaired in place; got '
        f'sfd_module={name!r} (expected the importable hash-name). '
        f'Without the cache-validity check, the stale `exp` stays '
        f'and Limen\'s Trainer reimport fails downstream.'
    )

    sys.modules.pop(name, None)
    cache_str = str(op_cache)
    if cache_str not in sys.path:
        sys.path.insert(0, cache_str)
    reimported = importlib.import_module(name)
    assert hasattr(reimported, 'manifest'), (
        f'reimport {name} missing manifest after stale repair'
    )
    assert reimported.params() == {'alpha': [1]}


def test_snapshot_exp_code_repairs_corrupt_partial_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex round-5 P1: snapshot existence != snapshot validity.

    A partial write (interrupted by Ctrl-C, OOM, panic mid-write)
    leaves a file at `_OP_SFD_CACHE/_bts_op_<sha16>.py` whose
    content is truncated. The round-3 / round-4 code accepted the
    file purely on `is_file()`, so a subsequent `import_module`
    raised `SyntaxError` on the broken source.

    The round-5 fix adds:
      1. content-hash re-validation on every `_snapshot_exp_code`
         call (mismatched hash -> rewrite atomically)
      2. atomic write via tmp-file + `Path.replace` so partial
         writes never have the final filename

    Mutation proof: removing the `hashlib.sha256(...).hexdigest()
    [:16] != digest` check in `_snapshot_exp_code` keeps the
    corrupt file in place; the assertion that re-snapshot
    rewrites the full content fires.
    """
    monkeypatch.setattr(_pipeline, '_OP_SFD_CACHE', tmp_path / 'op_sfds')

    exp_code = tmp_path / 'op.py'
    exp_code.write_text(
        'def params(): return {"alpha": [1]}\n'
        'def manifest(): return None\n',
        encoding='utf-8',
    )
    op_module_name, snap_path = _pipeline._snapshot_exp_code(exp_code)
    expected_size = snap_path.stat().st_size

    # Simulate a partial-write corruption (truncated body, syntax-broken).
    snap_path.write_text('def manifest(', encoding='utf-8')
    assert snap_path.stat().st_size < expected_size, (
        'corruption setup failed; expected smaller file'
    )

    # Re-snapshot: must detect content-hash mismatch and atomically
    # rewrite the full content.
    op_module_name2, snap_path2 = _pipeline._snapshot_exp_code(exp_code)
    assert op_module_name == op_module_name2
    assert snap_path == snap_path2
    assert snap_path.stat().st_size == expected_size, (
        f'corrupt snapshot must be repaired; got size '
        f'{snap_path.stat().st_size} (expected {expected_size})'
    )

    # Reimport must succeed (was SyntaxError before fix).
    sys.modules.pop(op_module_name, None)
    cache_str = str(_pipeline._OP_SFD_CACHE)
    if cache_str not in sys.path:
        sys.path.insert(0, cache_str)
    reimported = importlib.import_module(op_module_name)
    assert callable(reimported.manifest)
    assert reimported.params() == {'alpha': [1]}


def test_train_single_decoder_under_lock_cache_hit_preserves_sub_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex round-5 P1: under-lock cache hit returns without wiping.

    The TOCTOU race codex flagged: a stale-check + `shutil.rmtree`
    pair WITHOUT a lock lets two concurrent sweeps both see "stale"
    and both wipe. The round-5 fix wraps the validate+wipe+retrain
    block in a per-`sub_dir` `fcntl.flock`, with the validity check
    INSIDE the lock so the second-arriving process sees the first's
    completed work and returns early.

    This test exercises the cache-hit branch of the under-lock
    block: pre-seed `sub_dir` with valid artifacts (results.csv +
    matching metadata + snapshot file in `_OP_SFD_CACHE`), then
    call `train_single_decoder`. The under-lock validity check
    returns True; the function returns immediately. The sentinel
    file in `sub_dir` is preserved.

    Mutation proof: removing the validity check inside the lock
    (so wipe always proceeds) deletes the sentinel; the assert
    fires.
    """
    monkeypatch.setattr(
        _pipeline, '_OP_SFD_CACHE', tmp_path / 'op_sfds',
    )

    exp_code = tmp_path / 'op.py'
    exp_code.write_text(
        'def params(): return {"alpha": [1]}\n'
        'def manifest(): return None\n',
        encoding='utf-8',
    )
    sub_dir = tmp_path / 'trained_from_file' / 'pool_op_id_0_abc'
    sub_dir.mkdir(parents=True)

    # Compute the expected per-decoder body + module name, mirroring
    # the production logic. The body must be byte-equal to what
    # train_single_decoder generates (same operator file + same
    # params + same op_param_keys).
    op_module_name, _ = _pipeline._snapshot_exp_code(exp_code)
    body = (
        f'from {op_module_name} import manifest\n'
        '\n'
        'def params():\n'
        '    return {\n'
        "        'alpha': [1],\n"
        '    }\n'
    )
    pd_digest = (
        __import__('hashlib').sha256(body.encode('utf-8'))
        .hexdigest()[:16]
    )
    pd_module_name = f'_bts_pd_{pd_digest}'
    # Pre-seed the snapshot file in the cache (atomic write
    # mirrored).
    snapshot_path = (
        _pipeline._OP_SFD_CACHE / f'{pd_module_name}.py'
    )
    snapshot_path.write_text(body, encoding='utf-8')
    # Pre-seed `sub_dir` with VALID cache artifacts.
    (sub_dir / 'results.csv').write_text('id\n0\n', encoding='utf-8')
    (sub_dir / 'metadata.json').write_text(
        json.dumps({'sfd_module': pd_module_name}),
        encoding='utf-8',
    )
    sentinel = sub_dir / 'preserve_me.txt'
    sentinel.write_text(
        'this file should survive the cache-hit return',
        encoding='utf-8',
    )

    # Spy on UEL: cache hit means no UEL invocation; if the test
    # accidentally falls through to retrain, this assertion fails
    # with a clear message instead of a Limen crash.
    uel_calls: list[str] = []

    class _FakeUEL:
        def __init__(self, **kw: object) -> None:
            uel_calls.append('init')
            del kw

        def run(self, **kw: object) -> None:
            del kw
            uel_calls.append('run')

    from backtest_simulator.pipeline import experiment as exp_module
    monkeypatch.setattr(exp_module, 'UniversalExperimentLoop', _FakeUEL)

    _pipeline.train_single_decoder(
        sub_dir, {'alpha': 1}, exp_code, ('alpha',),
    )

    assert sentinel.is_file(), (
        'sub_dir contents must be preserved on cache hit; the '
        'sentinel file was wiped, indicating the lock or under-'
        'lock validity check is missing or broken.'
    )
    assert uel_calls == [], (
        f'cache hit must NOT invoke UEL; got {uel_calls!r}'
    )


def test_exclusive_dir_lock_acquires_and_releases(
    tmp_path: Path,
) -> None:
    """Codex round-5 P1: lock helper acquires + releases the file lock.

    A smoke test that the `_exclusive_dir_lock` context manager
    (a) creates the lock file, (b) acquires `fcntl.LOCK_EX` without
    raising, (c) releases on context exit (verified by re-acquiring
    inside the same process — same FD would block; new FD acquires
    fine after the first releases).

    Mutation proof: a missing `fcntl.flock(...)` call in
    `_exclusive_dir_lock` makes this test a no-op (still passes
    since flock is advisory). For a real mutation test of mutual
    exclusion, run two real processes concurrently — covered
    end-to-end by `bts sweep` in operator-side QA.
    """
    lock_path = tmp_path / 'x.lock'

    with _pipeline._exclusive_dir_lock(lock_path):
        assert lock_path.is_file(), (
            f'lock file must exist while context is active; '
            f'got missing file at {lock_path}'
        )
    # After the context exits, the lock is released and the file
    # handle closed. Re-acquire to confirm the lock is reusable.
    with _pipeline._exclusive_dir_lock(lock_path):
        assert lock_path.is_file()


def test_ensure_trained_repairs_stale_fresh_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auditor P0: `ensure_trained_from_exp_code` cache-hit must validate
    metadata + snapshot existence — not just `results.csv`.

    The pre-fix cache-hit path returned as soon as `results.csv`
    existed, so a stale `cache_dir` from an earlier buggy build
    (one whose `metadata['sfd_module']` is the bare operator
    stem rather than `_bts_op_<sha16>`, or one whose
    `_OP_SFD_CACHE` snapshot was deleted) would silently re-serve
    broken semantics with no loud signal — drifting bts back onto
    old semantics.

    The fix brings `ensure_trained_from_exp_code` to FULL parity
    with `train_single_decoder`'s round-4/5 fix:
    `_cache_dir_matches_expected_module(cache_dir, op_module_name)`
    inside `_exclusive_dir_lock`. Stale -> wipe + retrain.

    Mutation proof: reverting to `(cache_dir / 'results.csv').
    is_file()` accepts the seeded stale cache; the assertion that
    `metadata['sfd_module']` was repaired to `_bts_op_<sha16>`
    fires.
    """
    from backtest_simulator.pipeline import experiment as exp_module

    monkeypatch.setattr(_pipeline, 'WORK_DIR', tmp_path / 'work')
    monkeypatch.setattr(
        _pipeline, '_OP_SFD_CACHE', tmp_path / 'work' / 'op_sfds',
    )

    exp_code = tmp_path / 'op_sfd.py'
    _write_exp_code(exp_code)

    # Compute the cache_dir the production path would target
    # (mirroring `ensure_trained_from_exp_code`'s key derivation).
    import hashlib as _hl
    file_hash = _hl.sha256(exp_code.read_bytes()).hexdigest()
    cache_dir = (
        _pipeline.WORK_DIR / 'fresh'
        / f'{exp_code.stem}_n1_{file_hash[:16]}'
    )
    cache_dir.mkdir(parents=True)
    # Seed STALE artifacts: results.csv exists + metadata records
    # the bare operator stem (the round-3-era non-reimportable
    # name). The pre-fix code happily returned this cache_dir.
    (cache_dir / 'results.csv').write_text('id\n0\n', encoding='utf-8')
    (cache_dir / 'metadata.json').write_text(
        json.dumps({'sfd_module': exp_code.stem}),  # 'op_sfd' (bare)
        encoding='utf-8',
    )
    (cache_dir / 'round_data.jsonl').write_text('', encoding='utf-8')

    # Stub UEL so the wipe-then-retrain happy path completes
    # quickly: write a metadata.json reflecting the SFD passed in.
    class _FakeUEL:
        def __init__(self, **kw: object) -> None:
            self.sfd = kw['sfd']
            self.exp_dir = Path(kw['experiment_dir'])  # type: ignore[arg-type]

        def run(self, **kw: object) -> None:
            del kw
            self.exp_dir.mkdir(parents=True, exist_ok=True)
            (self.exp_dir / 'metadata.json').write_text(
                json.dumps({'sfd_module': self.sfd.__name__}),  # type: ignore[attr-defined]
                encoding='utf-8',
            )
            (self.exp_dir / 'results.csv').write_text(
                'id\n0\n', encoding='utf-8',
            )

    monkeypatch.setattr(exp_module, 'UniversalExperimentLoop', _FakeUEL)

    returned_dir = _pipeline.ensure_trained_from_exp_code(
        exp_code, n_permutations=1,
    )
    assert returned_dir == cache_dir, (
        f'ensure_trained_from_exp_code must return the same '
        f'cache_dir; got {returned_dir} vs expected {cache_dir}'
    )

    metadata = json.loads(
        (cache_dir / 'metadata.json').read_text(encoding='utf-8'),
    )
    name = metadata['sfd_module']
    assert name.startswith('_bts_op_'), (
        f'stale fresh-cache must be repaired in place; got '
        f'sfd_module={name!r}. Without the validate-under-lock '
        f'fix, the bare operator stem is silently re-served and '
        f'Limen\'s Trainer reimport fails downstream.'
    )

    # Snapshot file MUST exist for the repaired cache to be a
    # valid hit on the NEXT call.
    snapshot_path = _pipeline._OP_SFD_CACHE / f'{name}.py'
    assert snapshot_path.is_file()

    # And reimport works.
    sys.modules.pop(name, None)
    cache_str = str(_pipeline._OP_SFD_CACHE)
    if cache_str not in sys.path:
        sys.path.insert(0, cache_str)
    reimported = importlib.import_module(name)
    assert callable(reimported.manifest)


def test_cache_dir_matches_expected_module_rejects_missing_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validity helper rejects cache_dirs whose `_OP_SFD_CACHE` snapshot is gone.

    Rule #3 of `_cache_dir_matches_expected_module`: even when
    `metadata['sfd_module']` matches the expected name, the
    snapshot file MUST exist in `_OP_SFD_CACHE`. Otherwise the
    cache_dir is "orphaned" (reimport would fail) and must be
    treated as stale.

    This is a UNIT test of the helper's contract — not an
    integration test against `ensure_trained_from_exp_code`,
    because that function calls `_snapshot_exp_code` UPSTREAM
    of the validity check, which auto-heals a missing snapshot
    by atomic re-write. The orphan case is therefore self-
    healing at the call boundary; what matters is that the
    helper itself rejects the orphan state, so any future caller
    that does NOT pre-snapshot is protected.

    Codex post-auditor P1: my prior `test_ensure_trained_repairs_
    orphaned_cache_when_snapshot_missing` was tautological —
    BOTH the round-7 (results.csv-only) and round-8 (helper-
    based) code passed it because `_snapshot_exp_code` runs
    first and re-creates the snapshot. Replace with this
    helper-level test that DOES distinguish (mutation: removing
    rule #3 makes the helper return True for the orphan, and
    this assert fires).
    """
    monkeypatch.setattr(
        _pipeline, '_OP_SFD_CACHE', tmp_path / 'op_sfds',
    )
    _pipeline._OP_SFD_CACHE.mkdir(parents=True)

    cache_dir = tmp_path / 'cache_dir'
    cache_dir.mkdir()
    (cache_dir / 'results.csv').write_text('id\n0\n', encoding='utf-8')
    expected_module = '_bts_op_deadbeefdeadbeef'
    (cache_dir / 'metadata.json').write_text(
        json.dumps({'sfd_module': expected_module}),
        encoding='utf-8',
    )
    # Orphan: snapshot file is intentionally NOT created.
    snapshot_path = _pipeline._OP_SFD_CACHE / f'{expected_module}.py'
    assert not snapshot_path.exists()

    # Helper MUST reject this orphan state.
    assert not _pipeline._cache_dir_matches_expected_module(
        cache_dir, expected_module,
    ), (
        'validity helper must reject cache_dirs whose snapshot '
        'file is missing from _OP_SFD_CACHE; otherwise an '
        'operator-side `rm -rf op_sfds` leaves bts in a state '
        'where the cache_dir is silently treated as valid but '
        'a fresh-process reimport would raise '
        'ModuleNotFoundError.'
    )

    # Now create the snapshot — helper should return True.
    snapshot_path.write_text(
        'def manifest(): return None\ndef params(): return {}\n',
        encoding='utf-8',
    )
    assert _pipeline._cache_dir_matches_expected_module(
        cache_dir, expected_module,
    )
