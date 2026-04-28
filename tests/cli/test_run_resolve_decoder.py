"""Mutation-proof tests for `bts run`'s `_resolve_decoder` n_permutations plumbing.

Auditor (post-v2.0.1) P0: the explicit `--decoder-id N` path
called `ensure_trained_from_exp_code(exp_code_path, args.n_decoders)`,
where `args.n_decoders` defaults to 1. So `bts run --decoder-id 7`
without `--experiment-dir` would train ONE permutation and then
fail with "id 7 not found" — breaking the new contract that the
operator's request must be internally satisfiable on the most
literal path.

The fix added `--n-permutations` (default 30, parallel with
`bts sweep`) and reroutes BOTH the `--decoder-id` branch and the
`pick_decoders` fallback through it. These tests pin that
`n_permutations=args.n_permutations` flows through correctly,
distinct from `args.n_decoders` (the pick-pool size).
"""
from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path

import pytest


def _make_args(*, decoder_id: int | None, **overrides: object) -> argparse.Namespace:
    """Build a minimal argparse namespace for `_resolve_decoder`.

    Only the fields `_resolve_decoder` reads need defaults; the
    rest of the bts run argspace is unused on this code path.
    """
    base: dict[str, object] = {
        'exp_code': Path('/tmp/bts_test_exp.py'),
        'decoder_id': decoder_id,
        'experiment_dir': None,
        'n_decoders': 1,
        'n_permutations': 30,
        'input_from_file': None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _write_exp_code(path: Path) -> None:
    """Minimal UEL-compliant exp.py — `_resolve_decoder`'s up-front
    `is_file()` check passes; the auto-train path is monkeypatched out.
    """
    path.write_text(
        'from limen.sfd.foundational_sfd import logreg_binary as _base\n'
        'params = _base.params\n'
        'manifest = _base.manifest\n',
        encoding='utf-8',
    )


def test_resolve_decoder_passes_n_permutations_to_auto_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auditor P0: explicit `--decoder-id N` without `--experiment-dir`
    auto-trains via `ensure_trained_from_exp_code` and MUST pass
    `args.n_permutations` (the Limen model size), NOT `args.n_decoders`
    (the pick-pool size).

    Mutation proof: reverting `n_permutations=args.n_permutations` to
    `n_permutations=args.n_decoders` makes `captured['n']` equal 1
    instead of 30, and the assert fires.
    """
    from backtest_simulator.cli.commands import run as run_module

    exp_code = tmp_path / 'op.py'
    _write_exp_code(exp_code)

    captured: dict[str, object] = {}
    fake_dir = tmp_path / 'fake_exp_dir'
    fake_dir.mkdir()

    def _fake_ensure(path: Path, n: int) -> Path:
        captured['exp_code_path'] = path
        captured['n_permutations'] = n
        return fake_dir

    def _fake_kelly(exp_dir: Path, did: int) -> Decimal:
        captured['kelly_exp_dir'] = exp_dir
        captured['kelly_decoder_id'] = did
        return Decimal('0.5')

    monkeypatch.setattr(
        run_module, 'ensure_trained_from_exp_code', _fake_ensure,
    )
    monkeypatch.setattr(run_module, '_kelly_for_decoder', _fake_kelly)

    args = _make_args(
        decoder_id=7, n_decoders=1, n_permutations=30,
    )
    args.exp_code = exp_code  # override the path placeholder

    perm_id, kelly, exp_dir, display_id = run_module._resolve_decoder(args)

    assert captured['n_permutations'] == 30, (
        f'auto-train must use args.n_permutations (30), NOT '
        f'args.n_decoders (1); got {captured["n_permutations"]!r}. '
        f'Reverting the fix to `n_permutations=args.n_decoders` '
        f'makes this assert fire — bts run --decoder-id 7 without '
        f'--experiment-dir would train only 1 permutation and '
        f'fail with "id 7 not found".'
    )
    assert exp_dir == fake_dir
    assert perm_id == 7
    assert display_id == 7
    assert kelly == Decimal('0.5')


def test_resolve_decoder_pick_path_passes_n_permutations_to_pick_decoders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auditor P0 sibling: same fix on the `pick_decoders` fallback.

    `bts run` without `--decoder-id` falls through to `pick_decoders`,
    which previously also passed `args.n_decoders` as `n_permutations`.
    The fix now passes `args.n_permutations` here too. Mutation proof
    parallels the auto-train test.
    """
    from backtest_simulator.cli.commands import run as run_module

    exp_code = tmp_path / 'op.py'
    _write_exp_code(exp_code)

    captured: dict[str, object] = {}
    fake_dir = tmp_path / 'pick_exp_dir'
    fake_dir.mkdir()

    def _fake_pick(
        n: int, *, exp_code_path: Path, n_permutations: int,
        input_from_file: str | None = None,
    ) -> tuple[list[tuple[int, Decimal, Path, int]], int]:
        captured['n'] = n
        captured['n_permutations'] = n_permutations
        captured['exp_code_path'] = exp_code_path
        captured['input_from_file'] = input_from_file
        return ([(0, Decimal('0.7'), fake_dir, 0)], 1)

    monkeypatch.setattr(run_module, 'pick_decoders', _fake_pick)

    args = _make_args(
        decoder_id=None, n_decoders=5, n_permutations=42,
    )
    args.exp_code = exp_code

    run_module._resolve_decoder(args)

    assert captured['n'] == 5, (
        f'pick_decoders must receive n=args.n_decoders (5); '
        f'got {captured["n"]!r}'
    )
    assert captured['n_permutations'] == 42, (
        f'pick_decoders must receive n_permutations=args.n_permutations '
        f'(42), NOT args.n_decoders (5); got '
        f'{captured["n_permutations"]!r}. Reverting the fix to '
        f'`n_permutations=args.n_decoders` makes this assert fire.'
    )


def test_resolve_decoder_uses_experiment_dir_override_without_auto_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--experiment-dir` overrides auto-train (no n_permutations needed).

    When the operator points at a pre-existing experiment_dir, the
    auto-train path is skipped — `ensure_trained_from_exp_code`
    must NOT be called. Pin that contract; otherwise the operator's
    explicit override would be silently re-trained.

    Mutation proof: removing the `if args.experiment_dir is not None`
    branch would auto-train and `captured['called']=True` would fire.
    """
    from backtest_simulator.cli.commands import run as run_module

    exp_code = tmp_path / 'op.py'
    _write_exp_code(exp_code)
    operator_dir = tmp_path / 'operator_exp_dir'
    operator_dir.mkdir()

    captured: dict[str, bool] = {'called': False}

    def _fake_ensure(path: Path, n: int) -> Path:
        captured['called'] = True
        return path  # placeholder

    def _fake_kelly(exp_dir: Path, did: int) -> Decimal:
        return Decimal('0.5')

    monkeypatch.setattr(
        run_module, 'ensure_trained_from_exp_code', _fake_ensure,
    )
    monkeypatch.setattr(run_module, '_kelly_for_decoder', _fake_kelly)

    args = _make_args(decoder_id=3, experiment_dir=operator_dir)
    args.exp_code = exp_code

    _, _, exp_dir, _ = run_module._resolve_decoder(args)

    assert not captured['called'], (
        '--experiment-dir override must skip auto-train; '
        '`ensure_trained_from_exp_code` was called when it should '
        'have been bypassed.'
    )
    assert exp_dir == operator_dir
