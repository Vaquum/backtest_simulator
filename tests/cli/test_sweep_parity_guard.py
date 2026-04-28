"""End-to-end sweep tests for the SignalsTable parity guards.

Codex (post-auditor-4) P1: the in-isolation
`assert_signals_parity` tests pinned the helper's contract, but
NOTHING tested the integration at `commands.sweep._run`. Codex
verified that re-introducing the prior `if runtime_preds_raw:`
guard left the entire 319-test suite green — the unconditional
parity call was unprotected by tests.

These tests monkeypatch the sweep's collaborators (subprocess,
build, preflight) and drive `_run(args)` end-to-end. They pin:
  - empty `runtime_predictions` from a successful subprocess →
    ParityViolation (the helper's per-window strictness reaches
    sweep-level)
  - subprocess RuntimeError → sweep aborts (no silent `continue`)
  - missing `runtime_predictions` key entirely → ParityViolation
  - happy-path: every window's parity ran → `_run` returns 0
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from backtest_simulator.cli.commands import sweep as sweep_module
from backtest_simulator.exceptions import ParityViolation
from backtest_simulator.sensors.precompute import (
    PredictionsInput,
    SignalsTable,
)


def _build_args(
    *, replay_period_start: str, replay_period_end: str,
    cpcv_n_groups: int = 4, cpcv_n_test_groups: int = 2,
) -> argparse.Namespace:
    """Minimum argparse Namespace for `sweep._run`."""
    return argparse.Namespace(
        verbose=0,
        n_decoders=1,
        n_permutations=1,
        replay_period_start=replay_period_start,
        replay_period_end=replay_period_end,
        trading_hours_start=None,
        trading_hours_end=None,
        trades_q_range=None,
        tp_min_q=None,
        fpr_max_q=None,
        kelly_min_q=None,
        trade_count_min_q=None,
        net_return_min_q=None,
        input_from_file=None,
        maker=False,
        strict_impact=False,
        atr_k='0.5',
        atr_window_seconds=900,
        cpcv_n_groups=cpcv_n_groups,
        cpcv_n_test_groups=cpcv_n_test_groups,
        cpcv_purge_seconds=0,
        cpcv_embargo_seconds=0,
        exp_code=Path('/tmp/_unused_sweep_test.py'),
    )


def _build_table_for_ticks(
    decoder_id: str, ticks: list[datetime],
) -> SignalsTable:
    """Build a 2-tick SignalsTable matching the runtime predictions."""
    n = len(ticks)
    return SignalsTable.from_predictions(
        decoder_id=decoder_id, split_config=(70, 15, 15),
        predictions=PredictionsInput(
            timestamps=ticks,
            probs=np.array([0.5 + 0.01 * i for i in range(n)]),
            preds=np.array([1] * n, dtype=np.int64),
            label_horizon_bars=1, bar_seconds=3600,
        ),
    )


def _install_sweep_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *, picks: list[tuple[int, Decimal, Path, int]],
    signals_per_decoder: dict[str, SignalsTable],
    subprocess_result: dict[str, object] | type[Exception] | None,
) -> None:
    """Patch sweep collaborators to drive `_run` deterministically."""

    def _fake_pick_decoders(*_a: object, **_kw: object) -> tuple[
        list[tuple[int, Decimal, Path, int]], int,
    ]:
        return picks, len(picks)

    def _fake_build(*_a: object, **_kw: object) -> dict[str, SignalsTable]:
        return signals_per_decoder

    def _fake_preflight() -> None:
        return None

    def _fake_seed_price(_ts: datetime) -> Decimal:
        return Decimal('70000')

    def _fake_run_window(
        *_a: object, **_kw: object,
    ) -> dict[str, object]:
        if isinstance(subprocess_result, type) and issubclass(
            subprocess_result, Exception,
        ):
            raise subprocess_result('child boom')
        if subprocess_result is None:
            msg = 'subprocess_result must be set'
            raise RuntimeError(msg)
        return subprocess_result

    monkeypatch.setattr(sweep_module, 'pick_decoders', _fake_pick_decoders)
    monkeypatch.setattr(
        sweep_module, '_build_and_save_signals_tables', _fake_build,
    )
    monkeypatch.setattr(sweep_module, 'preflight_tunnel', _fake_preflight)
    monkeypatch.setattr(sweep_module, 'seed_price_at', _fake_seed_price)
    monkeypatch.setattr(
        sweep_module, 'run_window_in_subprocess', _fake_run_window,
    )


def _result_template(
    runtime_predictions: list[dict[str, object]],
) -> dict[str, object]:
    """Default subprocess result dict shaped like real `_run_window`."""
    return {
        'trades': [],
        'declared_stops': {},
        'orders': 0,
        'slippage_realised_bps': None,
        'slippage_realised_cost_bps': None,
        'slippage_realised_buy_bps': None,
        'slippage_realised_sell_bps': None,
        'slippage_predicted_cost_bps': None,
        'slippage_predict_vs_realised_gap_bps': None,
        'slippage_n_samples': 0,
        'slippage_n_excluded': 0,
        'slippage_n_uncalibrated_predict': 0,
        'slippage_n_predicted_samples': 0,
        'n_limit_orders_submitted': 0,
        'n_limit_filled_full': 0,
        'n_limit_filled_partial': 0,
        'n_limit_filled_zero': 0,
        'n_limit_marketable_taker': 0,
        'n_passive_limits': 0,
        'maker_fill_efficiency_p50': None,
        'maker_fill_efficiency_mean': None,
        'market_impact_realised_bps': None,
        'market_impact_n_samples': 0,
        'market_impact_n_flagged': 0,
        'market_impact_n_uncalibrated': 0,
        'market_impact_n_rejected': 0,
        'atr_k': '0.5',
        'atr_window_seconds': 900,
        'n_atr_rejected': 0,
        'n_atr_uncalibrated': 0,
        'event_spine_jsonl': '/tmp/_fake_spine.jsonl',
        'event_spine_n_events': 0,
        'book_gap_max_seconds': 0.0,
        'book_gap_n_observed': 0,
        'book_gap_p95_seconds': 0.0,
        'runtime_predictions': runtime_predictions,
    }


def test_sweep_run_raises_when_runtime_predictions_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subprocess returns empty `runtime_predictions` → ParityViolation.

    Codex P1(c): the unconditional `assert_signals_parity` call at
    sweep-level had no test. This test plus a mutation that
    re-introduces the `if runtime_preds_raw:` guard verifies the
    call IS unconditional.

    Mutation: re-introducing
        `if isinstance(runtime_preds_raw, list) and runtime_preds_raw:`
        before `assert_signals_parity` makes empty captures
        silently pass; this `pytest.raises` would fail.
    """
    decoder_id = '7'
    exp_dir = tmp_path / 'exp'
    exp_dir.mkdir()
    picks = [(7, Decimal('0.5'), exp_dir, 7)]
    # Build a 2-tick table so the helper can run if it gets data.
    base = datetime(2024, 12, 1, tzinfo=UTC)
    ticks = [base + timedelta(hours=i) for i in range(2)]
    signals = {decoder_id: _build_table_for_ticks(decoder_id, ticks)}
    _install_sweep_stubs(
        monkeypatch,
        picks=picks, signals_per_decoder=signals,
        subprocess_result=_result_template(runtime_predictions=[]),
    )

    exp_code = tmp_path / 'exp.py'
    exp_code.write_text(
        'from limen.sfd.foundational_sfd import logreg_binary as _b\n'
        'params = _b.params\nmanifest = _b.manifest\n',
        encoding='utf-8',
    )
    args = _build_args(
        replay_period_start='2024-12-01', replay_period_end='2024-12-01',
    )
    args.exp_code = exp_code

    with pytest.raises(ParityViolation):
        sweep_module._run(args)


def test_sweep_run_raises_when_runtime_predictions_key_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subprocess result missing `runtime_predictions` key → ParityViolation.

    Defensive test for serialisation drift: if the subprocess
    payload schema changes and the parent expects a key that's
    absent, the parent must NOT silently accept "no predictions
    captured" as success.
    """
    decoder_id = '7'
    exp_dir = tmp_path / 'exp'
    exp_dir.mkdir()
    picks = [(7, Decimal('0.5'), exp_dir, 7)]
    base = datetime(2024, 12, 1, tzinfo=UTC)
    ticks = [base + timedelta(hours=i) for i in range(2)]
    signals = {decoder_id: _build_table_for_ticks(decoder_id, ticks)}
    no_preds_result = _result_template(runtime_predictions=[])
    del no_preds_result['runtime_predictions']
    _install_sweep_stubs(
        monkeypatch,
        picks=picks, signals_per_decoder=signals,
        subprocess_result=no_preds_result,
    )

    exp_code = tmp_path / 'exp.py'
    exp_code.write_text(
        'from limen.sfd.foundational_sfd import logreg_binary as _b\n'
        'params = _b.params\nmanifest = _b.manifest\n',
        encoding='utf-8',
    )
    args = _build_args(
        replay_period_start='2024-12-01', replay_period_end='2024-12-01',
    )
    args.exp_code = exp_code

    with pytest.raises(ParityViolation):
        sweep_module._run(args)


def test_sweep_run_aborts_on_subprocess_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subprocess raises → sweep aborts (no silent continue).

    Codex P1(a): prior `try: run_window_in_subprocess()
    except Exception: continue` silently swallowed child failures
    and let the sweep print "OK n_compared=0". Now re-raises with
    window context.

    Mutation: replacing the new `raise RuntimeError(...) from exc`
    with the original `continue` makes the sweep proceed past the
    failure and exit cleanly; this `pytest.raises` would fail.
    """
    decoder_id = '7'
    exp_dir = tmp_path / 'exp'
    exp_dir.mkdir()
    picks = [(7, Decimal('0.5'), exp_dir, 7)]
    base = datetime(2024, 12, 1, tzinfo=UTC)
    ticks = [base + timedelta(hours=i) for i in range(2)]
    signals = {decoder_id: _build_table_for_ticks(decoder_id, ticks)}
    _install_sweep_stubs(
        monkeypatch,
        picks=picks, signals_per_decoder=signals,
        subprocess_result=RuntimeError,
    )

    exp_code = tmp_path / 'exp.py'
    exp_code.write_text(
        'from limen.sfd.foundational_sfd import logreg_binary as _b\n'
        'params = _b.params\nmanifest = _b.manifest\n',
        encoding='utf-8',
    )
    args = _build_args(
        replay_period_start='2024-12-01', replay_period_end='2024-12-01',
    )
    args.exp_code = exp_code

    with pytest.raises(RuntimeError, match='sweep aborted'):
        sweep_module._run(args)


def test_sweep_run_calls_parity_unconditionally_even_with_empty_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`assert_signals_parity` is invoked even when runtime_predictions is empty.

    Codex (post-auditor-4) P1(c) was emphatic: "the unconditional
    parity call is not mutation-proof. I temporarily restored
    `if runtime_preds_raw:` and the full requested suite still
    passed". This test SPIES on `assert_signals_parity` to verify
    it gets called per-window REGARDLESS of payload shape.

    Mutation: re-introducing
        `if not isinstance(runtime_preds_raw, list) or not runtime_preds_raw:
            continue`
    before the call makes `n_calls == 0`, and this assert fires.
    The post-loop `assert_sweep_signals_parity_ran` would still
    catch the parity-not-run case, but THIS test pins that the
    PER-WINDOW unconditional call is preserved.
    """
    decoder_id = '7'
    exp_dir = tmp_path / 'exp'
    exp_dir.mkdir()
    picks = [(7, Decimal('0.5'), exp_dir, 7)]
    base = datetime(2024, 12, 1, tzinfo=UTC)
    ticks = [base + timedelta(hours=i) for i in range(2)]
    signals = {decoder_id: _build_table_for_ticks(decoder_id, ticks)}
    _install_sweep_stubs(
        monkeypatch,
        picks=picks, signals_per_decoder=signals,
        # Empty predictions: prior conditional guard would have
        # `continue`d before the parity call.
        subprocess_result=_result_template(runtime_predictions=[]),
    )

    n_calls = {'value': 0}

    def _spy_assert(
        *, decoder_id: str, table: SignalsTable,
        runtime_predictions: list[dict[str, object]],
        expected_ticks: list[datetime],
    ) -> int:
        del decoder_id, table, runtime_predictions, expected_ticks
        n_calls['value'] += 1
        return 0

    monkeypatch.setattr(sweep_module, 'assert_signals_parity', _spy_assert)

    exp_code = tmp_path / 'exp.py'
    exp_code.write_text(
        'from limen.sfd.foundational_sfd import logreg_binary as _b\n'
        'params = _b.params\nmanifest = _b.manifest\n',
        encoding='utf-8',
    )
    args = _build_args(
        replay_period_start='2024-12-01', replay_period_end='2024-12-01',
    )
    args.exp_code = exp_code

    # Spy returns 0 → post-loop `assert_sweep_signals_parity_ran`
    # raises. Catch it; the test's interest is the per-window call.
    with pytest.raises(ParityViolation):
        sweep_module._run(args)

    assert n_calls['value'] == 1, (
        f'`assert_signals_parity` must be called UNCONDITIONALLY '
        f'per-window even when runtime_predictions=[]; got '
        f'{n_calls["value"]} calls. Re-introducing the prior '
        f'`if not runtime_preds_raw: continue` guard makes this '
        f'fail with 0 calls.'
    )


def test_sweep_run_passes_per_window_expected_ticks_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex (post-auditor-4 round-3) P1: each `assert_signals_parity`
    call receives ONLY the per-window slice, not the whole sweep grid.

    Mutation: reverting the per-window slice in `_run` to pass the
    full `tick_timestamps` makes day 1's call receive day 2's tick
    (and vice versa). This test SPIES on `assert_signals_parity` and
    asserts call N's `expected_ticks` matches the Nth window only.

    Boundary check: the slice must be `(window_start, window_end]`
    matching `_runtime_tick_timestamps`. With 2 days x interval=3600
    + hours 00:00-01:00, each day has EXACTLY one tick at 01:00.
    """
    decoder_id = '7'
    exp_dir = tmp_path / 'exp'
    exp_dir.mkdir()
    picks = [(7, Decimal('0.5'), exp_dir, 7)]
    # Build a SignalsTable spanning both days so the parity helper
    # has data to look up. We won't actually invoke it (spy), but
    # the helper signature requires a table.
    base_d1 = datetime(2024, 12, 1, 1, 0, tzinfo=UTC)
    base_d2 = datetime(2024, 12, 2, 1, 0, tzinfo=UTC)
    table_ticks = [base_d1, base_d2]
    signals = {decoder_id: _build_table_for_ticks(decoder_id, table_ticks)}

    captured_calls: list[list[datetime]] = []

    def _spy_assert(
        *, decoder_id: str, table: SignalsTable,
        runtime_predictions: list[dict[str, object]],
        expected_ticks: list[datetime],
    ) -> int:
        del decoder_id, table, runtime_predictions
        captured_calls.append(list(expected_ticks))
        # Return non-zero so post-loop guard doesn't fire.
        return 1

    _install_sweep_stubs(
        monkeypatch,
        picks=picks, signals_per_decoder=signals,
        # Shape doesn't matter — spy short-circuits.
        subprocess_result=_result_template(runtime_predictions=[]),
    )
    monkeypatch.setattr(sweep_module, 'assert_signals_parity', _spy_assert)

    exp_code = tmp_path / 'exp.py'
    exp_code.write_text(
        'from limen.sfd.foundational_sfd import logreg_binary as _b\n'
        'params = _b.params\nmanifest = _b.manifest\n',
        encoding='utf-8',
    )
    args = _build_args(
        replay_period_start='2024-12-01', replay_period_end='2024-12-02',
        cpcv_n_groups=2, cpcv_n_test_groups=1,
    )
    args.exp_code = exp_code
    args.trading_hours_start = '00:00'
    args.trading_hours_end = '01:00'

    sweep_module._run(args)

    # 2 days x 1 decoder = 2 windows.
    assert len(captured_calls) == 2, (
        f'expected 2 per-window calls; got {len(captured_calls)}'
    )
    # Each call's expected_ticks must contain ONLY that window's ticks.
    day1_ticks = captured_calls[0]
    day2_ticks = captured_calls[1]
    assert day1_ticks == [base_d1], (
        f'day-1 window must receive ONLY day-1 ticks; got '
        f'{[t.isoformat() for t in day1_ticks]}. A whole-sweep slice '
        f'mutation would include day-2 ticks too.'
    )
    assert day2_ticks == [base_d2], (
        f'day-2 window must receive ONLY day-2 ticks; got '
        f'{[t.isoformat() for t in day2_ticks]}. A whole-sweep slice '
        f'mutation would include day-1 ticks too.'
    )


def test_assert_sweep_signals_parity_ran_zero_raises() -> None:
    """Helper raises ParityViolation when total is 0.

    Pins the post-loop guard (defence-in-depth) directly. The
    per-window swallow fix and per-entry strictness are the
    primary defences; this helper is the second line.

    Mutation: changing `if total > 0: return` to `return` (no
    raise) makes this test fail with no exception.
    """
    with pytest.raises(ParityViolation, match='0 comparisons made'):
        sweep_module.assert_sweep_signals_parity_ran(
            0, n_picks=1, n_days=1,
        )


def test_assert_sweep_signals_parity_ran_nonzero_returns_silently() -> None:
    """Helper is a no-op when total > 0 (positive comparisons made)."""
    sweep_module.assert_sweep_signals_parity_ran(
        42, n_picks=2, n_days=21,
    )
