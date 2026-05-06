"""`bts sweep` — run the backtest pipeline over N decoders by M days."""

# Migration of the operator-driven `/tmp/bts_sweep.py` orchestration into
# the package. Each window runs in a fresh Python subprocess (see
# `backtest_simulator.cli._run_window`) so state bleed between windows is
# impossible. Per-run output is the one-line summary defined in
# `backtest_simulator.cli._metrics.print_run`.
from __future__ import annotations

import argparse
import atexit
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from datetime import time as dtime
from decimal import Decimal
from pathlib import Path
from typing import Final, cast

# tqdm removed — sweep emits one log line per phase + one per window
# completion. Progress bars muddied the parallel-batch output.
from backtest_simulator.cli._metrics import (
    STARTING_CAPITAL,
    Trade,
    max_drawdown_pct,
    pair_metrics,
    pair_trades,
    print_run,
)
from backtest_simulator.cli._pipeline import (
    pick_decoders,
    preflight_tunnel,
    seed_price_at,
)

# Re-exported at module scope so test fixtures can `monkeypatch.setattr(
# sweep, 'read_kline_size_from_experiment_dir', …)` and
# `monkeypatch.setattr(sweep, 'run_window_in_subprocess', …)`. The
# Python runtime emits a runpy `RuntimeWarning` because `cli/__init__.py`
# loads sweep.py (which in turn loads `_run_window`) before the
# subprocess does `python -m backtest_simulator.cli._run_window` — the
# warning is cosmetic, not a behavioural issue, and shadowing it would
# break public API contracts for monkey-patching tests.
from backtest_simulator.cli._run_window import (
    read_kline_size_from_experiment_dir as read_kline_size_from_experiment_dir,
)
from backtest_simulator.cli._run_window import (
    run_window_in_subprocess as run_window_in_subprocess,
)
from backtest_simulator.cli._session_manifest import (
    atomic_index_update as _atomic_index_update,
)
from backtest_simulator.cli._session_manifest import (
    finalize_session as _finalize_session,
)
from backtest_simulator.cli._session_manifest import (
    re_session_id_pattern as _re_session_id_pattern,
)
from backtest_simulator.cli._signals_builder import (
    assert_signals_parity,
    build_signals_table_for_decoder,
)
from backtest_simulator.cli._stats import (
    announce_operator_trades_tape,
    compute_sweep_stats,
    cpcv_pbo,
    daily_return_for_run,
    fetch_buy_hold_benchmark,
    make_seed_price_from_parquet,
)
from backtest_simulator.cli._verbosity import add_verbosity_arg, configure
from backtest_simulator.honesty.cpcv import CpcvPaths
from backtest_simulator.launcher.poller import (
    DEFAULT_N_ROWS,
    DEFAULT_START_DATE_LIMIT,
)
from backtest_simulator.sensors.precompute import SignalsTable

_SECOND_WEEK_APRIL_START: Final = datetime(2026, 4, 6, tzinfo=UTC).date()
_SECOND_WEEK_APRIL_END: Final = datetime(2026, 4, 12, tzinfo=UTC).date()

_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)


def assert_sweep_signals_parity_ran(
    sweep_signals_parity_total: int,
    *, n_picks: int, n_days: int,
) -> None:
    """Raise `ParityViolation` if no per-window parity ran across the sweep.

    Codex (post-auditor-4) P1: per-window
    `try: run_window_in_subprocess() except Exception: continue`
    swallowed child-window failures. Even with the per-window
    `assert_signals_parity` mandatory and the per-entry strictness
    in the helper, a sweep where EVERY window failed at the
    subprocess level would print "OK n_compared=0" cheerfully
    (codex's repro produced exactly that with `rc=0`).

    The post-window swallow is now ALSO closed (`raise ... from
    exc`), but this post-loop guard is the belt-and-braces
    second line of defence. Reaching this function with `total
    == 0` means: the sweep DID run, no per-window exception was
    raised, AND yet zero parity comparisons happened. That can
    only mean the parity body was bypassed somehow — a future
    silent regression. Loud, not silent.
    """
    if sweep_signals_parity_total > 0:
        return
    from backtest_simulator.exceptions import ParityViolation
    msg = (
        f'sweep signals parity: 0 comparisons made across the '
        f'WHOLE sweep ({n_picks} decoder(s) x {n_days} day(s) = '
        f'{n_picks * n_days} window(s)). The mandatory '
        f'SignalsTable parity check did NOT run for any window. '
        f'Possible causes: capture hook bypassed, parity body '
        f'unreachable due to a refactor regression. Sweep result '
        f'is unreliable and is rejected.'
    )
    raise ParityViolation(msg)


def _resolve_grid_interval(
    picks: list[tuple[int, Decimal, Path, int]],
) -> int:
    """Pick the single `interval_seconds` for the sweep's parity grid.

    A single `bts sweep` invocation runs against ONE exp_code or ONE
    bundle, so every pick's experiment_dir derives from the same SFD
    manifest and shares one `kline_size`. We assert this — a mismatch
    means picks were assembled from heterogeneous training pools and
    the shared parity grid is not well-defined.
    """
    seen: dict[int, Path] = {}
    for _, _, exp_dir, _ in picks:
        ks = read_kline_size_from_experiment_dir(exp_dir)
        prior = seen.get(ks)
        if prior is None:
            seen[ks] = exp_dir
    if len(seen) != 1:
        msg = (
            f'_resolve_grid_interval: picks declare more than one '
            f'kline_size: {seen}. A single sweep must run against a '
            f'homogeneous decoder pool (one bundle / one exp_code) '
            f'— the parity grid uses ONE cadence.'
        )
        raise ValueError(msg)
    return next(iter(seen))


def _runtime_tick_timestamps(
    *,
    days: list[datetime], hours_start: dtime, hours_end: dtime,
    interval_seconds: int,
) -> list[datetime]:
    """Mirror `launcher/clock.py`'s timer firing schedule per-day.

    For each `day` in `days`, find every epoch-aligned
    `interval_seconds` boundary that falls in
    `(window_start, window_end]`. The "next boundary AFTER
    window_start" semantics match `clock.py:99-104` —
    `next_boundary = (elapsed_whole // interval_seconds + 1) *
    interval_seconds`. When `window_start` itself sits exactly on
    a boundary, the FIRST tick is one interval later (matching
    runtime: a strategy started at exactly 08:00:00 with
    interval_seconds=3600 fires its first signal at 09:00:00).
    """
    ticks: list[datetime] = []
    for day in days:
        window_start = datetime.combine(day.date(), hours_start, tzinfo=UTC)
        window_end = datetime.combine(day.date(), hours_end, tzinfo=UTC)
        elapsed = int((window_start - _EPOCH).total_seconds())
        next_boundary = (elapsed // interval_seconds + 1) * interval_seconds
        t = _EPOCH + timedelta(seconds=next_boundary)
        while t <= window_end:
            ticks.append(t)
            t += timedelta(seconds=interval_seconds)
    return ticks


def register(add_parser: Callable[[str, str], argparse.ArgumentParser]) -> None:
    p = add_parser(
        'sweep',
        'Run the backtest pipeline over N decoders x M days.',
    )
    # --exp-code is REQUIRED: bts has no fallback SFD. The
    # operator's file must be a self-contained UEL-compliant
    # Python file with module-level `params()` and `manifest()`
    # callables. Operator convention is to define an SFD class
    # (e.g. `class Round3SFD: @staticmethod def params(): ...`)
    # and expose its static methods at module level via
    # `params = Round3SFD.params; manifest = Round3SFD.manifest`.
    # Any uel.run boilerplate must be guarded by
    # `if __name__ == '__main__':` so importing the file has no
    # side effects — bts drives uel itself with bts-controlled
    # n_permutations + experiment_name.
    exp_group = p.add_mutually_exclusive_group(required=True)
    exp_group.add_argument('--exp-code', type=Path,
                   help=(
                       'Path to a UEL-compliant Python file with '
                       'module-level params() and manifest() '
                       'callables. Mutually exclusive with --bundle. '
                       'bts has no fallback SFD; this file is the '
                       'source of truth for training and retraining '
                       'picked decoders.'
                   ))
    exp_group.add_argument('--bundle', type=Path,
                   help=(
                       'Path to a Limen-exported bundle zip '
                       '(<name>__rNNNN.zip) containing sibling '
                       '<name>.py + <name>.json + <name>.csv. '
                       'Mutually exclusive with --exp-code. bts '
                       'applies the JSON `data_source` override and '
                       '`uel_run.n_permutations` (when --n-permutations '
                       'is not given), and uses the bundled CSV as '
                       'the filter pool.'
                   ))
    p.add_argument('--n-decoders', type=int, default=1)
    p.add_argument('--n-permutations', type=int, default=None)
    p.add_argument('--trading-hours-start', type=str, default=None)
    p.add_argument('--trading-hours-end', type=str, default=None)
    p.add_argument('--replay-period-start', type=str, default=None)
    p.add_argument('--replay-period-end', type=str, default=None)
    p.add_argument('--trades-q-range', type=str, default=None)
    p.add_argument('--tp-min-q', type=float, default=None)
    p.add_argument('--fpr-max-q', type=float, default=None)
    p.add_argument('--input-from-file', type=str, default=None)
    p.add_argument('--kelly-min-q', type=float, default=None)
    p.add_argument('--trade-count-min-q', type=float, default=None)
    p.add_argument('--net-return-min-q', type=float, default=None)
    p.add_argument('--maker', action='store_true', default=False,
                   help=(
                       'Strategy emits LIMIT orders at the estimated '
                       'price (passive maker post). Routes through '
                       'MakerFillModel for realistic queue + partial '
                       'fills. Default: MARKET (taker).'
                   ))
    p.add_argument('--strict-impact', action='store_true', default=False,
                   help=(
                       'Reject ENTER orders (BUY for the long-only '
                       'template) the MarketImpactModel flags as '
                       'exceeding 10%% of concurrent-bucket volume. '
                       'SELL exits are measured but never rejected. '
                       'Default: record telemetry only (observability '
                       'mode).'
                   ))
    p.add_argument('--max-allocation-per-trade-pct', type=str, default='0.4',
                   help=(
                       'Per-trade allocation cap as a decimal fraction. '
                       'Default 0.4 (40%%) accommodates Kelly fractions on '
                       'r0014-class bundles whose backtest_mean_kelly_pct is '
                       '~18%% post-Kelly fraction; lower values cause the '
                       'CapitalController to deny entries with '
                       'CAPITAL_RESERVATION_DENIED. Requires '
                       'vaquum-nexus >= 0.35.0.'
                   ))
    p.add_argument('--predict-lookback', type=int, default=None,
                   help=(
                       'Number of trailing prepared rows fed into '
                       'sensor.predict per tick. Default 1 (single-row '
                       'predict) preserves prior behaviour. >1 enables '
                       'stateful predictors to evolve across rows. '
                       'Requires vaquum-nexus >= 0.36.0.'
                   ))
    p.add_argument('--session-id', type=str, required=True,
                   help=(
                       'Unique identifier for this sweep session. '
                       'Outputs (sweep.log, sweep_per_window.csv, '
                       'sweep_per_tick.csv) are written to '
                       '~/sweep/sessions/<session-id>/ . Errors if the '
                       'directory already exists, so each session keeps '
                       'an immutable record. The dashboard at '
                       'http://127.0.0.1:8910/ lists sessions in a '
                       'dropdown for switching between live + completed runs.'
                   ))
    # Slice #17 Task 17 (CPCV portion). Operator-controllable
    # CSCV partitioning. Defaults C(4,2)=6 paths so the math runs
    # on every default sweep that has enough clean days; otherwise
    # the line skips with reason. purge/embargo default 0 so the
    # operator opts in to label-leakage protection — non-zero
    # values drop train days adjacent to test-group boundaries.
    p.add_argument('--cpcv-n-groups', type=int, default=4,
                   help=(
                       'CPCV (CSCV) total group count for day-aligned '
                       'partitioning. C(n_groups, n_test_groups) paths '
                       'are evaluated. Default: 4.'
                   ))
    p.add_argument('--cpcv-n-test-groups', type=int, default=2,
                   help=(
                       'CPCV test-group count per path. Default: 2 '
                       '(C(4,2)=6 paths).'
                   ))
    p.add_argument('--cpcv-purge-seconds', type=int, default=0,
                   help=(
                       'CPCV purge window in seconds — drop train '
                       'days within this many seconds of any test-'
                       'group boundary (both sides). Default: 0.'
                   ))
    p.add_argument('--cpcv-embargo-seconds', type=int, default=0,
                   help=(
                       'CPCV embargo in seconds — drop train days '
                       'within this many seconds AFTER each test '
                       'block (Lopez de Prado direction). Default: 0.'
                   ))
    p.add_argument('--trades-tape', type=Path, default=None, help='Pre-captured trades parquet covering the replay window. When set, skips CH preflight + prefetch + seed-price lookups.')
    add_verbosity_arg(p)
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    configure(args.verbose)
    # Session-scoped outputs first — the dashboard's `index.json` poll
    # needs to see this session within 2s of `bts sweep` starting,
    # before the heavy filter-pool training / SignalsTable build runs.
    # Layout:
    #   ~/sweep/sessions/<id>/sweep.log            (caller redirects)
    #   ~/sweep/sessions/<id>/sweep_per_window.csv (live per-window)
    #   ~/sweep/sessions/<id>/sweep_per_tick.csv   (live per-tick preds)
    #   ~/sweep/sessions/index.json                (dashboard manifest)
    # `--session-id` is registered as `required=True` via argparse, so a
    # real CLI invocation always carries it. Unit tests in `tests/cli/`
    # construct `argparse.Namespace` directly (bypassing argparse) and
    # may omit it; we tolerate that by routing those callers to a
    # tempdir and skipping the manifest. Operator-facing CLI behaviour
    # is unchanged.
    _session_id = getattr(args, 'session_id', None)
    if _session_id is None:
        import tempfile as _tmp
        session_dir = Path(_tmp.mkdtemp(prefix='_bts_test_sweep_'))
        _index_path: Path | None = None
    else:
        sessions_root = Path.home() / 'sweep' / 'sessions'
        sessions_root.mkdir(parents=True, exist_ok=True)
        # Path-traversal guard: session-id must be a single safe segment
        # so `sessions_root / session_id` cannot escape
        # `~/sweep/sessions/`. `..`, absolute paths, embedded slashes,
        # NUL bytes and shell metacharacters all rejected. Allowed:
        # ASCII letters, digits, dash, underscore, dot (no leading dot,
        # no `..`).
        _SESSION_ID_RE = _re_session_id_pattern()
        if not _SESSION_ID_RE.fullmatch(_session_id) or _session_id == '..':
            sys.stderr.write(
                f'bts sweep: --session-id {_session_id!r} is not a '
                f'safe filesystem segment. Use letters / digits / dash / '
                f'underscore / dot (no leading dot, no slashes, no `..`).\n',
            )
            return 2
        session_dir = sessions_root / _session_id
        # Reject any pre-existing session_dir, even if
        # `sweep_per_window.csv` is missing (Copilot P1: prior guard
        # let `~/sweep/sessions/<id>/` with leftover artefacts
        # overwrite a partially-cleaned run).
        if session_dir.exists():
            sys.stderr.write(
                f'bts sweep: --session-id {_session_id!r} already has '
                f'directory at {session_dir}. Pick a unique id or '
                f'remove the directory to retry.\n',
            )
            return 2
        session_dir.mkdir(parents=True, exist_ok=False)
        _index_path = sessions_root / 'index.json'
        _bundle_label = (
            Path(args.bundle).name if getattr(args, 'bundle', None)
            else (Path(args.exp_code).name if getattr(args, 'exp_code', None) else '?')
        )
        _now_iso = datetime.now(UTC).isoformat()
        _new_entry: dict[str, object] = {
            'id': _session_id,
            'started_at': _now_iso,
            'ended_at': None,
            'bundle': _bundle_label,
            'n_decoders': int(getattr(args, 'n_decoders', 0) or 0),
            'replay_period': [
                getattr(args, 'replay_period_start', None),
                getattr(args, 'replay_period_end', None),
            ],
        }
        _sid_for_closure = _session_id

        def _append_session(sessions: list[dict[str, object]]) -> None:
            # De-duplicate by `id` — if a prior partial run left a
            # stale entry for this session-id (the directory guard
            # above is belt-and-braces; operators occasionally rm -rf
            # the dir but leave the manifest entry), replace in place.
            sessions[:] = [
                s for s in sessions if s.get('id') != _sid_for_closure
            ]
            sessions.append(_new_entry)
        _atomic_index_update(_index_path, _append_session)
        # Stamp `ended_at` on EVERY exit path — clean return,
        # ParityViolation, RuntimeError("sweep aborted: …"), sys.exit,
        # Ctrl-C — so the dashboard never shows a session stuck
        # `live` after the process is gone. atexit fires even on
        # raised exceptions; only `os._exit` would bypass it (we use
        # that in subprocess children, never the parent).
        atexit.register(_finalize_session, _index_path, _session_id)
        print(
            f'[{0.0:7.2f}s] session {_session_id} → {session_dir}',
            flush=True,
        )
    # Silence noisy third-party INFO logs so the sweep log stays uniform
    # `[X.XXs] message` lines. `limen.experiment.trainer.trainer` emits
    # `INFO ... Trained sensor for permutation 0` per training call —
    # for an N-decoder filter pool that's N raw lines without our
    # duration prefix. Operators can re-enable by passing `-vv`
    # (configure() honours verbosity for our own logs); Limen / Praxis /
    # Nexus stay at WARNING regardless.
    import logging as _logging
    for _name in (
        'limen', 'limen.experiment.trainer.trainer',
        'praxis', 'nexus',
    ):
        _logging.getLogger(_name).setLevel(_logging.WARNING)
    # Install the parquet cache patch on `HistoricalData.get_spot_klines`
    # FOR THE PARENT before `pick_decoders` runs filter-pool training.
    # Without this, every `train_single_decoder(...)` invocation calls
    # Limen's pipeline → `manifest.fetch_data()` → `get_spot_klines` →
    # full HuggingFace stream (~40 s each). With N decoders that's
    # N x ~40 s of pure HF refetching just for the filter pool. The
    # patch is idempotent (guarded by `HistoricalData._bts_cache_installed`)
    # so it composes with the same install in the per-window
    # subprocesses (`_run_window._child_main`). The 48h freshness check
    # downstream (in the SignalsTable build phase) refreshes the
    # parquet if stale; once the parquet is on disk every parent /
    # subprocess code path reads it via this patch.
    from backtest_simulator._limen_cache import install_cache
    install_cache()
    from backtest_simulator.cli._pipeline import WORK_DIR
    from backtest_simulator.pipeline.bundle import materialize_bundle_on_args
    materialize_bundle_on_args(args, WORK_DIR)
    if (args.trading_hours_start is None) != (args.trading_hours_end is None):
        sys.stderr.write(
            'bts sweep: --trading-hours-start and --trading-hours-end '
            'must be given together (either both or neither)\n',
        )
        return 2
    if (args.replay_period_start is None) != (args.replay_period_end is None):
        sys.stderr.write(
            'bts sweep: --replay-period-start and --replay-period-end '
            'must be given together (either both or neither)\n',
        )
        return 2
    if args.n_decoders < 1:
        sys.stderr.write(f'bts sweep: --n-decoders must be >= 1, got {args.n_decoders}\n')
        return 2
    hours_start, hours_end = _resolve_hours(args)
    days = _resolve_days(args)
    # `--trades-tape` short-circuits CH; default flow preflights as before.
    _operator_tape: Path | None = Path(args.trades_tape).expanduser().resolve() if args.trades_tape else None
    if _operator_tape is None:
        preflight_tunnel()
    exp_code_path = Path(args.exp_code).expanduser().resolve()
    if not exp_code_path.is_file():
        sys.stderr.write(
            f'bts sweep: --exp-code file not found: {exp_code_path}\n',
        )
        return 2
    trades_q_range: tuple[float, float] | None = None
    if args.trades_q_range is not None:
        lo_str, hi_str = args.trades_q_range.split(',')
        trades_q_range = (float(lo_str), float(hi_str))
    picks, candidate_pool_size = pick_decoders(
        args.n_decoders,
        exp_code_path=exp_code_path,
        n_permutations=args.n_permutations,
        trades_q_range=trades_q_range,
        tp_min_q=args.tp_min_q,
        fpr_max_q=args.fpr_max_q,
        kelly_min_q=args.kelly_min_q,
        trade_count_min_q=args.trade_count_min_q,
        net_return_min_q=args.net_return_min_q,
        input_from_file=args.input_from_file,
    )
    if not picks:
        # `pick_decoders` returned 0 — every candidate failed the
        # `--*-min-q` / `--trades-q-range` filters, or `--n-decoders`
        # is 0, or `--input-from-file` had no usable rows. Without
        # picks the parallel batch's `n_workers = min(0, …, 8) == 0`
        # would crash `ThreadPoolExecutor` with `max_workers must be
        # greater than 0` straight out of stdlib (bit-mis P1). Fail
        # loudly instead with the cause named.
        sys.stderr.write(
            f'bts sweep: pick_decoders returned 0 picks from a '
            f'candidate pool of {candidate_pool_size}. Either relax '
            f'the filter quantiles (--tp-min-q / --fpr-max-q / '
            f'--kelly-min-q / --trade-count-min-q / --net-return-min-q '
            f'/ --trades-q-range), supply more candidates via '
            f'--n-permutations or --input-from-file, or raise '
            f'--n-decoders. There is no work to dispatch.\n',
        )
        return 2
    # Header line removed — the same numbers (decoders x days = runs) are
    # already implicit in the per-window log lines and the `replay
    # launching … window(s) on … worker(s)` line right before the batch.
    total_runs = len(picks) * len(days)
    # Slice #17 Task 16: build SignalsTable per picked decoder via
    # per-tick runtime replay (Nexus's exact recipe). Tick instants
    # match `launcher/clock.py`'s epoch-aligned next-boundary timer
    # firing schedule (codex round-4 P0); iterating klines instead
    # of ticks over-emits when interval_seconds < kline_size, and
    # always over-emits across non-trading hours.
    #
    # Cadence anchor: the runtime PredictLoop's Timer fires every
    # `kline_size` seconds (set in `_run_window.py` from the SAME
    # `read_kline_size_from_experiment_dir`). The parity grid must
    # use the SAME cadence — otherwise different-kline_size bundles
    # (e.g. 7200 in the LogReg-Placeholder bundle vs 3600 in earlier
    # exp-codes) raise ParityViolation on every fire.
    interval_seconds = _resolve_grid_interval(picks)
    replay_start = datetime.combine(days[0].date(), hours_start, tzinfo=UTC)
    replay_end = datetime.combine(days[-1].date(), hours_end, tzinfo=UTC)
    tick_timestamps = _runtime_tick_timestamps(
        days=days, hours_start=hours_start, hours_end=hours_end,
        interval_seconds=interval_seconds,
    )
    raw_lookback_for_signals = getattr(args, 'predict_lookback', None)
    signals_per_decoder = _build_and_save_signals_tables(
        picks, tick_timestamps=tick_timestamps,
        replay_start=replay_start, replay_end=replay_end,
        predict_lookback=(
            None if raw_lookback_for_signals is None
            else int(raw_lookback_for_signals)
        ),
    )
    # `_print_sweep_signals_summary` removed — its `n_decoders / expected_bars
    # / coverage` line just restated `len(picks)` and `len(tick_timestamps)`,
    # both already known. Coverage is a downstream parity concern handled
    # by `assert_signals_parity` per-window. `signals_per_decoder` is still
    # consumed by that check below.
    t_total = time.perf_counter()
    # Accumulators for the sweep-level slippage summary printed
    # after all runs finish. The per-run cost_bps is the
    # side-normalized mean for that run; for the sweep aggregate
    # we want to weight each run by its sample count so a run with
    # 100 fills doesn't get the same vote as one with 2.
    sweep_slip_costs_weighted: list[tuple[Decimal, int]] = []
    sweep_slip_gaps_weighted: list[tuple[Decimal, int]] = []
    sweep_n_samples_total = 0
    sweep_n_excluded_total = 0
    sweep_n_uncal_total = 0
    sweep_runs_with_slip = 0
    # Maker-fill telemetry across the sweep. Counts are zero when
    # no run uses --maker; non-zero when LIMIT orders engaged the
    # maker engine.
    sweep_n_limit_total = 0
    sweep_n_limit_full = 0
    sweep_n_limit_partial = 0
    sweep_n_limit_zero = 0
    sweep_n_limit_taker = 0
    sweep_efficiencies_weighted: list[tuple[Decimal, int]] = []
    # Sweep-level market-impact accumulators. The per-run
    # `realised_bps` is a mean across that run's calibrated
    # samples; the cross-run aggregate is a sample-weighted
    # mean (heavier runs vote more — same shape as the
    # sweep-level slippage cost aggregate). `n_samples` /
    # `n_flagged` / `n_uncalibrated` are simple totals.
    sweep_impact_bps_weighted: list[tuple[Decimal, int]] = []
    sweep_impact_n_samples_total = 0
    sweep_impact_n_flagged_total = 0
    sweep_impact_n_uncalibrated_total = 0
    sweep_impact_n_rejected_total = 0
    # Slice #17 Task 11 — book-gap aggregates. max-of-max across
    # runs (the worst stop-cross latency the sweep observed) plus
    # total stops (denominator). p95 is intentionally NOT
    # aggregated from per-run p95s — that would require carrying
    # raw samples or a mergeable histogram (codex round 1).
    sweep_book_gap_max_seconds = 0.0
    sweep_book_gap_total_stops = 0
    # Auditor (post-v2.0.2) "make it real": running tally of how
    # many runtime-vs-SignalsTable per-tick comparisons succeeded
    # across the whole sweep. The actual mismatch detection is
    # `assert_signals_parity` raising `ParityViolation` per-window;
    # this counter gives the operator a positive indicator that
    # the parity check WAS exercised (vs zero comparisons silently
    # because the runtime never produced predictions).
    sweep_signals_parity_total = 0
    # Slice #17 Task 17 (DSR + PBO + SPA portion). Per-decoder
    # daily-return scalars feed `compute_sweep_stats` after the
    # sweep finishes. Stored per (decoder, day_idx) so day-aligned
    # stats can drop only the days where ANY decoder had trailing
    # inventory (codex round 2 P1: misaligned per-decoder lists
    # let DSR/PBO truncate-by-position over different dates,
    # producing clean-looking but wrong stats). CPCV is deferred
    # until Task 16's path-aware `SignalsTable.lookup` is wired
    # into the live predict path; without it CPCV math would be
    # decorative.
    decoder_day_returns: dict[str, dict[int, float]] = {}
    n_runs_with_trailing_inventory = 0
    # Per-window CSV: one row per (perm, day) appended live so the
    # dashboard streams updates while the sweep runs. `session_dir` was
    # set up at the very top of `_run` (before filter-pool training)
    # so the dashboard's `index.json` poll picks the session up
    # immediately; we just reuse it here.
    import csv as _csv
    csv_dir = session_dir
    csv_path = csv_dir / 'sweep_per_window.csv'
    csv_fp = csv_path.open('w', newline='', encoding='utf-8')
    # atexit close so an exception inside the parallel batch / post-
    # processing never leaves the CSV file handle dangling (Copilot
    # P1). The success path also closes explicitly (~line 1120) so
    # the operator's `tail` of the file isn't blocked on process
    # teardown; the atexit close becomes a no-op on the second
    # invocation. Per-row `csv_fp.flush()` calls keep buffers on
    # disk, so even a hard-kill leaves the rows that were written
    # at the point of failure.
    atexit.register(csv_fp.close)
    csv_writer = _csv.writer(csv_fp)
    csv_writer.writerow([
        'display_id', 'date', 'capital_alloc_pct', 'profit_pct',
        'r_mean', 'max_drawdown_pct', 'n_trades',
        'prob_min', 'prob_max',
    ])
    csv_fp.flush()
    # Per-tick CSV: every captured kline-tick prediction across the
    # sweep. One row per (perm, day, timestamp). Diagnostic for
    # whether `_probs` actually varies tick-to-tick within a window.
    tick_csv_path = csv_dir / 'sweep_per_tick.csv'
    tick_csv_fp = tick_csv_path.open('w', newline='', encoding='utf-8')
    atexit.register(tick_csv_fp.close)
    tick_csv_writer = _csv.writer(tick_csv_fp)
    tick_csv_writer.writerow([
        'display_id', 'date', 'timestamp', 'pred', 'prob',
    ])
    tick_csv_fp.flush()
    # Parallelize per-window subprocess spawns. Each window pays a
    # ~5s boot for Praxis Trading + Nexus StartupSequencer + Trainer;
    # running them concurrently amortises wall-time over CPU/IO. Each
    # subprocess is independent (own work_dir, own EventSpine sqlite,
    # own venue adapter), so there's no shared mutable state to
    # serialise on. ClickHouse is reached over a shared SSH tunnel so
    # we cap concurrency at `min(cpu/2, len, 8)` — enough to amortise
    # boot, low enough to avoid swamping the tunnel with simultaneous
    # 30-min trade slice queries during slippage calibration.
    import concurrent.futures as _cf
    import os as _os

    # ONE ClickHouse fetch for the whole sweep — covers slippage
    # calibration (30-min pre-window slice), per-submit market-impact
    # bucket fetches (1-min pre-submit slice), and the per-submit fill
    # walk (capped at 600s). Subprocesses receive the parquet path and
    # wrap it in `InMemoryTradesFeed`, eliminating per-submit network
    # round-trips that were costing 3-6s each over the SSH tunnel.
    from backtest_simulator.feed.clickhouse import (
        ClickHouseConfig as _ChCfg,
    )
    from backtest_simulator.feed.clickhouse import (
        prefetch_sweep_trades as _prefetch_trades,
    )
    _t_trades = time.perf_counter()
    _trade_fetch_start = replay_start - timedelta(minutes=30)
    _trade_fetch_end = replay_end + timedelta(seconds=600)
    _trades_parquet_path: str | None = None
    if _operator_tape is not None:
        try:
            _trades_parquet_path = announce_operator_trades_tape(_operator_tape, _t_trades)
        except FileNotFoundError as exc:
            sys.stderr.write(f'{exc}\n')
            return 2
    else:
        # `_ChCfg.from_env()` raises `RuntimeError` on any missing CH
        # var; catch it narrowly so unit-test contexts that mock
        # `run_window_in_subprocess` reach the parity-check assertions
        # instead of crashing on env init.
        try:
            _ch_config = _ChCfg.from_env()
        except RuntimeError as _ch_exc:
            sys.stderr.write(
                f'bts sweep: ClickHouse env not configured ({_ch_exc}); '
                f'skipping trades prefetch (subprocesses will fetch '
                f'per-window or fail loudly if real ClickHouse access '
                f'is needed).\n',
            )
            _trades_parquet_path = None
        else:
            _trades_parquet = _prefetch_trades(
                config=_ch_config,
                symbol='BTCUSDT',
                start=_trade_fetch_start,
                end=_trade_fetch_end,
                cache_dir=Path.home() / '.cache' / 'backtest_simulator' / 'trades',
            )
            print(
                f'[{time.perf_counter()-_t_trades:7.2f}s] '
                f'trades tape ready: {_trades_parquet.name}',
                flush=True,
            )
            _trades_parquet_path = str(_trades_parquet)
    _raw_max_alloc = getattr(args, 'max_allocation_per_trade_pct', None)
    _raw_lookback = getattr(args, 'predict_lookback', None)
    # Captured by `_run_one_window` below as concrete typed values so
    # `run_window_in_subprocess` is called with named arguments and
    # pyright can check each one. Avoids the `dict[str, object]`
    # unpacking that the typing-budget ratchet rejected.
    _maker_preference = bool(getattr(args, 'maker', False))
    _strict_impact = bool(getattr(args, 'strict_impact', False))
    _max_alloc_decimal: Decimal | None = (
        None if _raw_max_alloc is None else Decimal(str(_raw_max_alloc))
    )
    _predict_lookback_int: int | None = (
        None if _raw_lookback is None else int(_raw_lookback)
    )
    # Typed spec for one (perm, day) window. NamedTuple gives every
    # downstream unpack site full pyright visibility — was previously
    # `tuple[...]` losing types on `_p, _k, _ed, …` destructuring.
    from typing import NamedTuple as _NamedTuple
    class _WindowSpec(_NamedTuple):
        perm_id: int
        kelly: Decimal
        exp_dir: Path
        display_id: int
        day_idx: int
        day: datetime
        window_start: datetime
        window_end: datetime

    window_specs: list[_WindowSpec] = []
    for perm_id, kelly, exp_dir, display_id in picks:
        decoder_day_returns[str(display_id)] = {}
        for day_idx, day in enumerate(days):
            window_start = datetime.combine(day.date(), hours_start, tzinfo=UTC)
            window_end = datetime.combine(day.date(), hours_end, tzinfo=UTC)
            window_specs.append(_WindowSpec(
                perm_id, kelly, exp_dir, display_id,
                day_idx, day, window_start, window_end,
            ))

    def _run_one_window(spec: _WindowSpec) -> tuple[object, float]:
        (_perm_id, _kelly, _exp_dir, _display_id,
         _day_idx, _day, _ws, _we) = spec
        # Measure WORK time only — the wall-clock from when this thread
        # actually picks up the spec to when the subprocess returns.
        # Capturing at submission time would include queue-wait time
        # and inflate later windows' "duration" with the wait of every
        # window ahead of it in the pool's queue.
        _t_work_start = time.perf_counter()
        _result = run_window_in_subprocess(
            _perm_id, _kelly, _ws, _we, _exp_dir,
            display_id=_display_id,
            maker_preference=_maker_preference,
            strict_impact=_strict_impact,
            trades_parquet_path=_trades_parquet_path,
            max_allocation_per_trade_pct=_max_alloc_decimal,
            predict_lookback=_predict_lookback_int,
        )
        return _result, time.perf_counter() - _t_work_start

    n_workers = min(
        len(window_specs),
        max(1, (_os.cpu_count() or 4) // 2),
        8,
    )
    print(
        f'[{0.0:7.2f}s] replay launching {len(window_specs)} window(s) '
        f'on {n_workers} worker(s)',
        flush=True,
    )
    _t_par = time.perf_counter()
    parallel_results: dict[tuple[int, int, str], object] = {}

    def _write_csv_row_now(_did: int, _label: str, _res: object) -> None:
        """Compute + write the per-window CSV row as soon as the window
        completes, so the dashboard streams updates instead of waiting
        for the whole 140-window batch to finish before any row appears.
        Computation mirrors the ordered post-processing block below
        (kept as the single source of truth for sweep-wide stats and
        the per-line `print_run`)."""
        if not isinstance(_res, dict):
            return
        from typing import cast as _cast2

        from backtest_simulator.cli._run_window import (
            WindowResult as _WR2,
        )
        # Use the symbol (not the string form) so vulture sees the
        # import as live; pyright + cast both accept the bare class.
        _r = _cast2(_WR2, _res)
        _trades_raw = _r.get('trades', [])
        _stops_raw = _r.get('declared_stops', {})
        _trades_local = [Trade(*row) for row in _trades_raw]
        _declared_local = {
            k: Decimal(v) for k, v in _stops_raw.items()
        }
        _pairs, _trailing = pair_trades(_trades_local)
        _net_pnls: list[Decimal] = []
        _r_mults: list[Decimal] = []
        _ret_pcts: list[Decimal] = []
        for _buy, _sell in _pairs:
            _declared = _declared_local.get(_buy.client_order_id)
            _net, _ret, _rmult = pair_metrics((_buy, _sell), _declared)
            _net_pnls.append(_net)
            _ret_pcts.append(_ret)
            if _rmult is not None:
                _r_mults.append(_rmult)
        _total_pct = sum(_ret_pcts, Decimal('0')) if _ret_pcts else None
        _r_mean = (
            sum(_r_mults, Decimal('0')) / len(_r_mults)
            if _r_mults else None
        )
        _max_dd = max_drawdown_pct(_net_pnls, STARTING_CAPITAL)
        _buy_notional = sum(
            (_b.qty * _b.price for _b, _ in _pairs), Decimal('0'),
        )
        _cap_alloc = (
            _buy_notional / STARTING_CAPITAL * Decimal('100')
            if _buy_notional > 0 else Decimal('0')
        )
        _runtime_preds = _r.get('runtime_predictions', [])
        _all_probs: list[float] = []
        for _e in _runtime_preds:
            _prob = _e.get('prob')
            if isinstance(_prob, (int, float)):
                _all_probs.append(float(_prob))
        _pmin = min(_all_probs) if _all_probs else None
        _pmax = max(_all_probs) if _all_probs else None
        csv_writer.writerow([
            _did,
            _label,
            f'{_cap_alloc:.4f}',
            '' if _total_pct is None else f'{_total_pct:.4f}',
            '' if _r_mean is None else f'{_r_mean:.4f}',
            f'{_max_dd:.4f}',
            len(_pairs),
            '' if _pmin is None else f'{_pmin:.6f}',
            '' if _pmax is None else f'{_pmax:.6f}',
        ])
        csv_fp.flush()
        for _entry in _runtime_preds:
            tick_csv_writer.writerow([
                _did,
                _label,
                _entry.get('timestamp', ''),
                _entry.get('pred', ''),
                _entry.get('prob', '') if _entry.get('prob') is not None else '',
            ])
        tick_csv_fp.flush()

    with _cf.ThreadPoolExecutor(max_workers=n_workers) as _ex:
        _futures = {
            _ex.submit(_run_one_window, spec): spec for spec in window_specs
        }
        for _fut in _cf.as_completed(_futures):
            _spec = _futures[_fut]
            (_p, _k, _ed, _did, _di, _d, _ws, _we) = _spec
            _label = _d.date().isoformat()
            try:
                _res, _t_window = _fut.result()
            except Exception as _exc:
                msg = (
                    f'sweep aborted: perm {_did} on '
                    f'{_label} failed: {_exc!r}. '
                    f'Per-window failures are not silently '
                    f'continued — every picked decoder x day must '
                    f'produce a parity-validated run.'
                )
                raise RuntimeError(msg) from _exc
            parallel_results[(_p, _did, _label)] = _res
            _write_csv_row_now(_did, _label, _res)
            _n_done = len(parallel_results)
            _n_trades = 0
            if isinstance(_res, dict):
                _trades_field = cast('dict[str, object]', _res).get('trades')
                if isinstance(_trades_field, list):
                    _n_trades = len(cast('list[object]', _trades_field))
            print(
                f'[{_t_window:7.2f}s] '
                f'perm {_did:<6} {_label}  done  '
                f'({_n_done}/{len(window_specs)}, trades={_n_trades})',
                flush=True,
            )
    print(
        f'[{time.perf_counter()-_t_par:7.2f}s] '
        f'replay batch done ({len(window_specs)} window(s))',
        flush=True,
    )

    # Now iterate windows in stable per-(perm,day) order for
    # post-processing — same logic as before, just lifting the result
    # from `parallel_results` instead of calling `run_window_in_subprocess`.
    for perm_id, kelly, exp_dir, display_id in picks:
        for day_idx, day in enumerate(days):
            window_start = datetime.combine(day.date(), hours_start, tzinfo=UTC)
            window_end = datetime.combine(day.date(), hours_end, tzinfo=UTC)
            day_label = day.date().isoformat()
            result_obj = parallel_results[(perm_id, display_id, day_label)]
            from typing import cast as _cast

            from backtest_simulator.cli._run_window import (
                WindowResult as _WR,
            )
            # Use the symbol (not the string form) so vulture sees the
            # import as live; pyright + cast both accept the bare class.
            result = _cast(_WR, result_obj)
            trades_raw = result['trades']
            stops_raw = result['declared_stops']
            trades = [Trade(*row) for row in trades_raw]
            declared_stops = {k: Decimal(v) for k, v in stops_raw.items()}
            slip_raw = result.get('slippage_realised_cost_bps')
            slip_cost = (
                None if slip_raw is None else Decimal(str(slip_raw))
            )
            slip_n = int(result.get('slippage_n_samples', 0))
            slip_excl = int(result.get('slippage_n_excluded', 0))
            slip_uncal = int(result.get(
                'slippage_n_uncalibrated_predict', 0,
            ))
            slip_predicted_n = int(result.get(
                'slippage_n_predicted_samples', 0,
            ))
            slip_gap_raw = result.get('slippage_predict_vs_realised_gap_bps')
            slip_gap = (
                None if slip_gap_raw is None else Decimal(str(slip_gap_raw))
            )
            n_limit = int(result.get('n_limit_orders_submitted', 0))
            n_limit_full = int(result.get('n_limit_filled_full', 0))
            n_limit_partial = int(result.get('n_limit_filled_partial', 0))
            n_limit_zero = int(result.get('n_limit_filled_zero', 0))
            n_limit_taker = int(result.get('n_limit_marketable_taker', 0))
            n_passive = int(result.get('n_passive_limits', 0))
            eff_raw = result.get('maker_fill_efficiency_p50')
            eff_p50 = (
                None if eff_raw is None else Decimal(str(eff_raw))
            )
            eff_mean_raw = result.get('maker_fill_efficiency_mean')
            eff_mean = (
                None if eff_mean_raw is None else Decimal(str(eff_mean_raw))
            )
            impact_raw = result.get('market_impact_realised_bps')
            impact_bps = (
                None if impact_raw is None else Decimal(str(impact_raw))
            )
            impact_n = int(result.get('market_impact_n_samples', 0))
            impact_flagged = int(result.get('market_impact_n_flagged', 0))
            impact_uncal = int(result.get(
                'market_impact_n_uncalibrated', 0,
            ))
            impact_rejected = int(result.get(
                'market_impact_n_rejected', 0,
            ))
            # Auditor (post-v2.0.3) "parity must not silently skip":
            # the prior round's `if runtime_preds_raw:` guard let a
            # broken capture hook or missing payload reach the
            # final "no comparisons made" print and exit cleanly,
            # leaving the mandatory SignalsTable path UNVALIDATED.
            # Now the call ALWAYS runs; the helper raises
            # `ParityViolation` when it produces 0 comparisons,
            # making the silent-skip path impossible. The runtime
            # predictions list comes from `_run_window`'s wrapped
            # `produce_signal` hook; if it's empty or non-list,
            # that's itself a capture-side bug surfaced as a
            # ParityViolation (the Five Principles: bts work that
            # exists must do real work or fail loudly).
            decoder_key = str(display_id)
            table = signals_per_decoder.get(decoder_key)
            if table is None:
                from backtest_simulator.exceptions import (
                    ParityViolation as _ParityViolation,
                )
                msg_no_table = (
                    f'sweep signals parity: decoder '
                    f'{decoder_key!r} has no SignalsTable in '
                    f'signals_per_decoder; the build path skipped '
                    f'this decoder. Mandatory parity reference is '
                    f'absent — auditor post-v2.0.2 contract requires '
                    f'a table for every picked decoder.'
                )
                raise _ParityViolation(msg_no_table)
            runtime_preds_list = result.get('runtime_predictions', [])
            # Codex (post-auditor-4 round-3): pass PER-WINDOW
            # expected ticks, NOT the whole sweep grid. A first-
            # window check must reject a captured second-day tick
            # (and vice versa). The slice matches
            # `_runtime_tick_timestamps`'s `(window_start,
            # window_end]` semantics — INCLUSIVE at end (the
            # PredictLoop fires AT window_end if it sits on a
            # boundary; that's the LAST tick of THIS window, not
            # the first of the next).
            window_expected_ticks = [
                t for t in tick_timestamps
                if window_start < t <= window_end
            ]
            n_signals_compared = assert_signals_parity(
                decoder_id=decoder_key,
                table=table,
                runtime_predictions=runtime_preds_list,
                expected_ticks=window_expected_ticks,
                interval_seconds=interval_seconds,
            )
            sweep_signals_parity_total += n_signals_compared
            print_run(
                display_id, day_label, trades, declared_stops,
                n_intents=int(result.get('n_intents', 0)),
                n_fills=int(result.get('n_fills', 0)),
                n_pending=int(result.get('n_pending', 0)),
                n_rejects=int(result.get('n_rejects', 0)),
                slippage_cost_bps=slip_cost,
                slippage_n_samples=slip_n,
                slippage_n_excluded=slip_excl,
                slippage_predict_vs_realised_gap_bps=slip_gap,
                slippage_n_uncalibrated_predict=slip_uncal,
                slippage_n_predicted_samples=slip_predicted_n,
                n_limit_orders_submitted=n_limit,
                n_limit_filled_full=n_limit_full,
                n_limit_filled_partial=n_limit_partial,
                n_limit_filled_zero=n_limit_zero,
                n_limit_marketable_taker=n_limit_taker,
                maker_fill_efficiency_p50=eff_p50,
                market_impact_realised_bps=impact_bps,
                market_impact_n_samples=impact_n,
                market_impact_n_flagged=impact_flagged,
                market_impact_n_uncalibrated=impact_uncal,
            )
            # Book-gap aggregation: max-of-max + total stops.
            run_book_gap_max = float(result.get('book_gap_max_seconds', 0.0))
            run_book_gap_n = int(result.get('book_gap_n_observed', 0))
            if run_book_gap_max > sweep_book_gap_max_seconds:
                sweep_book_gap_max_seconds = run_book_gap_max
            sweep_book_gap_total_stops += run_book_gap_n
            # Accumulate per (decoder, day_idx) daily return for
            # Task 17 post-sweep stats. None means trailing
            # inventory at window close — don't pretend unrealised
            # PnL is a 0 return; codex round 1 P1. Storing per
            # day_idx keeps decoders alignable later — if ANY
            # decoder has trailing on day X, day X is dropped from
            # all decoders so DSR/PBO/SPA see same-date returns
            # (codex round 2 P1).
            day_return = daily_return_for_run(trades, declared_stops)
            if day_return is None:
                n_runs_with_trailing_inventory += 1
            else:
                decoder_day_returns[str(display_id)][day_idx] = day_return
            # Per-window + per-tick CSV writes happen live inside the
            # parallel-batch `as_completed` loop above
            # (`_write_csv_row_now`) so the dashboard streams rows as
            # windows finish, instead of waiting for the whole batch.
            # The duplicated `pair_trades` / `pair_metrics` /
            # `max_drawdown_pct` computations that used to live here
            # were dead — the live writer is the single source of
            # truth for sweep CSV output.
            sweep_n_limit_total += n_limit
            sweep_n_limit_full += n_limit_full
            sweep_n_limit_partial += n_limit_partial
            sweep_n_limit_zero += n_limit_zero
            sweep_n_limit_taker += n_limit_taker
            # Weight by `n_passive` (count of passive LIMITs in
            # this run), NOT `n_limit` (which includes marketable
            # takers — runs with mostly takers would over-weight
            # their unrelated p50 in the cross-run mean). The
            # per-run efficiency was computed only across passive
            # LIMITs, so the denominator must match. Codex round
            # 4 P2 caught the prior n_limit weighting.
            if eff_mean is not None and n_passive > 0:
                sweep_efficiencies_weighted.append((eff_mean, n_passive))
            # Market-impact aggregation: weight per-run mean by
            # sample count so a 200-order run doesn't get drowned
            # by a 2-order run. Uncalibrated submits are tracked
            # separately so the operator can spot calibration
            # gaps in the sweep summary.
            if impact_bps is not None and impact_n > 0:
                sweep_impact_bps_weighted.append((impact_bps, impact_n))
            sweep_impact_n_samples_total += impact_n
            sweep_impact_n_flagged_total += impact_flagged
            sweep_impact_n_uncalibrated_total += impact_uncal
            sweep_impact_n_rejected_total += impact_rejected
            if slip_cost is not None:
                sweep_runs_with_slip += 1
                sweep_n_samples_total += slip_n
                sweep_n_excluded_total += slip_excl
                sweep_n_uncal_total += slip_uncal
                if slip_n > 0:
                    sweep_slip_costs_weighted.append((slip_cost, slip_n))
                # Gap is the mean over fills where BOTH realised and
                # predicted are available. Use the adapter's own
                # n_predicted_samples (count of fills with non-None
                # prediction) as the weight, so a run with many
                # uncalibrated_predict fills doesn't get over-weighted.
                if slip_gap is not None and slip_predicted_n > 0:
                    sweep_slip_gaps_weighted.append(
                        (slip_gap, slip_predicted_n),
                    )
    t_wall = time.perf_counter() - t_total
    csv_fp.close()
    tick_csv_fp.close()
    print(f'\ndone   {total_runs} run(s) in {t_wall:.1f}s')
    print(f'sweep csv        {csv_path}')
    print(f'sweep tick csv   {tick_csv_path}')
    # Honesty contract: every BUY filled inside a window MUST close
    # before the window ends. Open inventory at window close means
    # the day's PnL is unrealized and the day gets silently dropped
    # from DSR/SPA/PBO via `daily_return_for_run` returning None.
    # Stats compiled on the surviving subset are not honest — they
    # represent the leftover after the most informative days were
    # excluded. If force-flatten failed for ANY window, the operator
    # cannot trust the sweep result, so we abort loud rather than
    # print a numeric verdict on a partial sample.
    if n_runs_with_trailing_inventory > 0:
        from backtest_simulator.exceptions import (
            ParityViolation as _ParityViolation,
        )
        msg_open_inventory = (
            f'sweep aborted: {n_runs_with_trailing_inventory} '
            f'window(s) ended with open inventory at window close. '
            f'Honest invariant violated: every BUY filled inside a '
            f'window must close before the window ends (force-flatten '
            f'or natural SELL signal). Stats computed on the '
            f'surviving subset would silently exclude the strategy\'s '
            f'most active days — refusing to ship a misleading verdict.'
        )
        raise _ParityViolation(msg_open_inventory)
    _print_sweep_slippage_summary(
        sweep_slip_costs_weighted,
        sweep_slip_gaps_weighted,
        sweep_n_samples_total,
        sweep_n_excluded_total,
        sweep_n_uncal_total,
        sweep_runs_with_slip,
        total_runs,
    )
    _print_sweep_maker_summary(
        sweep_n_limit_total,
        sweep_n_limit_full,
        sweep_n_limit_partial,
        sweep_n_limit_zero,
        sweep_n_limit_taker,
        sweep_efficiencies_weighted,
    )
    _print_sweep_impact_summary(
        sweep_impact_bps_weighted,
        sweep_impact_n_samples_total,
        sweep_impact_n_flagged_total,
        sweep_impact_n_uncalibrated_total,
        sweep_impact_n_rejected_total,
        total_runs,
    )
    _print_sweep_book_gap_summary(
        max_seconds=sweep_book_gap_max_seconds,
        total_stops=sweep_book_gap_total_stops,
    )
    # Auditor (post-v2.0.3) + codex (post-auditor-4): require AT
    # LEAST ONE window's parity to have actually run across the
    # whole sweep. Closes the silent-skip path through the
    # per-window `try: run_window_in_subprocess() except
    # Exception: continue` block above (subprocess failure for
    # every window would otherwise print "OK n_compared=0").
    assert_sweep_signals_parity_ran(
        sweep_signals_parity_total,
        n_picks=len(picks), n_days=len(days),
    )
    print(
        f'sweep signals parity  '
        f'OK n_compared={sweep_signals_parity_total} '
        f'(runtime Sensor.predict matches SignalsTable per-tick)',
    )
    # Align decoders by day: drop any day where ANY decoder had
    # trailing inventory. This guarantees DSR/PBO/SPA see same-
    # date returns for every decoder (codex round 2 P1).
    all_decoder_ids = list(decoder_day_returns.keys())
    clean_day_indices = [
        day_idx for day_idx in range(len(days))
        if all(
            day_idx in decoder_day_returns[d] for d in all_decoder_ids
        )
    ]
    per_decoder_returns: dict[str, list[float]] = {
        d: [decoder_day_returns[d][idx] for idx in clean_day_indices]
        for d in all_decoder_ids
    }
    clean_days = [days[i] for i in clean_day_indices]
    # n_search_trials = max(operator's --n-permutations, the
    # actual candidate pool size from `pick_decoders`). The pool
    # size handles cached-mode + --input-from-file where
    # `args.n_permutations` doesn't reflect the real search
    # space (codex round 2 P1).
    n_search_trials = max(
        int(args.n_permutations), int(candidate_pool_size),
    )
    _print_sweep_stats_summary(
        per_decoder_returns, clean_days, hours_start, hours_end,
        n_search_trials=n_search_trials,
        n_runs_with_trailing_inventory=n_runs_with_trailing_inventory,
        seed_price_at=make_seed_price_from_parquet(_operator_tape) if _operator_tape is not None else seed_price_at,
    )
    _print_cpcv_pbo_summary(
        per_decoder_returns=per_decoder_returns,
        days=clean_days,
        n_groups=int(args.cpcv_n_groups),
        n_test_groups=int(args.cpcv_n_test_groups),
        purge_seconds=int(args.cpcv_purge_seconds),
        embargo_seconds=int(args.cpcv_embargo_seconds),
    )
    # Reaches here only on the success path — `_finalize_session`
    # (registered via atexit at session-creation time) handles BOTH
    # success and failure paths so the dashboard never shows a
    # session stuck `live` after the process exits.
    return 0


def _build_and_save_signals_tables(
    picks: list[tuple[int, Decimal, Path, int]],
    *,
    tick_timestamps: list[datetime],
    replay_start: datetime, replay_end: datetime,
    predict_lookback: int | None = None,
) -> dict[str, SignalsTable]:
    """Build + save SignalsTable per picked decoder.

    Slice #17 Task 16. Walks each picked decoder's experiment dir
    via Limen `Trainer`, retrains the Pass-2 (1,0,0) Sensor (the
    runtime predictor), fetches klines via the same source the
    launcher's poller uses (`HistoricalData.get_spot_klines`), and
    runs per-bar replay over the sweep's replay window.

    Trainer init is amortised per `exp_dir` so the ~20 s ClickHouse
    fetch only happens once per experiment, not per decoder.

    Auditor (post-v2.0.2): now returns ONLY the tables (klines were
    a residual return value from the bar-level CPCV that v2.0.2
    replaced with day-level deployed-strategy returns). The tables
    are the sweep-time PARITY REFERENCE for runtime predictions —
    `assert_signals_parity` in `_signals_builder.py` checks them
    against the captured `produce_signal` outputs per window.
    """
    import importlib
    import json as _json

    from limen import HistoricalData, Trainer
    by_exp_dir: dict[Path, list[tuple[int, int]]] = {}
    for perm_id, _, exp_dir, display_id in picks:
        by_exp_dir.setdefault(exp_dir, []).append((perm_id, display_id))
    if not by_exp_dir:
        # Caller (`_run`) guards against `len(picks) == 0` upstream and
        # exits before reaching this helper. Belt-and-braces: an empty
        # mapping here would hit `next(iter(...))` with StopIteration —
        # surface it as a clear ValueError instead so the message names
        # the cause (Copilot P1, defensive narrow that's reachable only
        # via mis-call from a future entry point).
        msg = (
            '_build_and_save_signals_tables: no picks → no SignalsTable '
            'to build. Caller must short-circuit on empty picks.'
        )
        raise ValueError(msg)
    historical = HistoricalData()
    # Mode-2 picks share the same data_source_config (same bundle), so
    # we read the manifest from the FIRST exp_dir's metadata.json
    # WITHOUT instantiating Trainer (Trainer.__init__ pulls data
    # internally), fetch klines ONCE, then hand the same DataFrame to
    # every Trainer via the `data=` kwarg so they skip their internal
    # fetch_data(). Without this, klines would be pulled N+1 times:
    # once per Trainer (one per decoder-group) plus our explicit
    # get_spot_klines call.
    first_exp_dir = next(iter(by_exp_dir))
    _tm = time.perf_counter()
    with (first_exp_dir / 'metadata.json').open('r') as _mf:
        _meta = _json.load(_mf)
    _sfd = importlib.import_module(_meta['sfd_module'])
    manifest_for_cfg = _sfd.manifest()
    print(
        f'[{time.perf_counter()-_tm:7.2f}s] '
        f'manifest loaded (kline_size={int(manifest_for_cfg.data_source_config.params["kline_size"]) if manifest_for_cfg.data_source_config else "?"})',
        flush=True,
    )
    cfg = manifest_for_cfg.data_source_config
    if cfg is None or 'kline_size' not in cfg.params:
        msg = (
            f'sweep signals: manifest at {first_exp_dir} has no '
            f'kline_size in data_source_config; cannot fetch '
            f'klines for SignalsTable build.'
        )
        raise ValueError(msg)
    kline_size = int(cfg.params['kline_size'])
    ds_params = dict(cfg.params)
    ds_params.pop('kline_size', None)
    ds_n_rows_obj = ds_params.pop('n_rows', DEFAULT_N_ROWS)
    ds_start_obj = ds_params.pop(
        'start_date_limit', DEFAULT_START_DATE_LIMIT,
    )
    if ds_params:
        msg = (
            f'sweep signals: manifest at {first_exp_dir} declares '
            f'unsupported data_source.params keys: '
            f'{sorted(ds_params)}. Limen.HistoricalData.get_spot_klines '
            f'accepts only n_rows / kline_size / start_date_limit.'
        )
        raise ValueError(msg)
    if not isinstance(ds_n_rows_obj, int):
        msg = (
            f'sweep signals: manifest at {first_exp_dir} n_rows must be '
            f'int, got {type(ds_n_rows_obj).__name__}={ds_n_rows_obj!r}'
        )
        raise TypeError(msg)
    if not isinstance(ds_start_obj, str):
        msg = (
            f'sweep signals: manifest at {first_exp_dir} start_date_limit '
            f'must be str, got '
            f'{type(ds_start_obj).__name__}={ds_start_obj!r}'
        )
        raise TypeError(msg)
    # Internal data handler with 48h staleness threshold:
    #   1. Local parquet exists AND its max datetime is < 48h behind
    #      `datetime.now(UTC)` → use the local file, skip the fetch.
    #   2. Otherwise (file missing OR last bar > 48h old) → call
    #      `get_spot_klines` and atomically replace the local file.
    # The cache path matches `_limen_cache._cache_path` so the
    # subprocess pollers (which call `install_cache()` and read the
    # same parquet) see the up-to-date data.
    import polars as _pl
    _cp = (
        Path.home() / '.cache' / 'backtest_simulator' / 'limen_klines'
        / f'btcusdt_{kline_size}.parquet'
    )
    _now_utc = datetime.now(UTC)
    _STALE_THRESHOLD_HOURS = 48
    _use_cache = False
    klines_shared = None
    _t_klines = time.perf_counter()
    if _cp.is_file():
        cached = _pl.read_parquet(_cp)
        # `polars.Series.max()` is typed as `PythonLiteral | None`;
        # at runtime on a `Datetime[us, UTC]` column it's a `datetime`
        # (or `None` when empty). Narrow explicitly so the tz-aware
        # comparison below is type-safe.
        _cached_max_obj: object = cached['datetime'].max()
        if isinstance(_cached_max_obj, datetime):
            _cached_max_aware = (
                _cached_max_obj if _cached_max_obj.tzinfo is not None
                else _cached_max_obj.replace(tzinfo=UTC)
            )
            age_h = (_now_utc - _cached_max_aware).total_seconds() / 3600.0
            if age_h < _STALE_THRESHOLD_HOURS:
                klines_shared = cached
                _use_cache = True
                print(
                    f'[{time.perf_counter()-_t_klines:7.2f}s] '
                    f'klines cache HIT (age={age_h:.1f}h, '
                    f'{cached.height} rows)',
                    flush=True,
                )
    if not _use_cache:
        # `install_cache` (registered in `_run` for the filter-pool
        # training pass) wraps `HistoricalData.get_spot_klines` to
        # return the cached parquet whenever the file exists — so a
        # naive refetch call here would re-load the SAME stale data
        # we already rejected. Delete the parquet first; the wrapper
        # then sees a cache miss, fetches fresh from HuggingFace, and
        # writes the new file back to disk under its own atomic
        # `_cp.tmp` → `_cp` move. The subprocesses (which install the
        # same wrapper) read the refreshed parquet on their next call.
        # Copilot P1.
        if _cp.is_file():
            _cp.unlink()
        klines_shared = historical.get_spot_klines(
            n_rows=ds_n_rows_obj,
            kline_size=kline_size,
            start_date_limit=ds_start_obj,
        )
        # The wrapper writes the parquet itself on its miss path; no
        # explicit `klines_shared.write_parquet(...)` call needed.
        # Copilot P1: prior code passed n_rows implicitly None, which
        # bypassed the bundle's declared row cap and risked pulling the
        # full dataset (memory + parity drift vs the runtime poller).
        print(
            f'[{time.perf_counter()-_t_klines:7.2f}s] '
            f'klines refetched from HuggingFace ({klines_shared.height} rows)',
            flush=True,
        )
    if klines_shared is None:
        # Defensive narrow — the cache hit and refetch branches above
        # both assign `klines_shared`, so this is unreachable in
        # practice. Pyright needs the explicit `is None` check to
        # drop the `DataFrame | None` annotation that lingers after
        # the `_use_cache` branching.
        msg = 'sweep signals: klines_shared not populated'
        raise RuntimeError(msg)
    # Normalize the polars `.max()` to a tz-aware UTC datetime before
    # the staleness boundary compare (zero-bang P1: cache-hit path's
    # `_cached_max_aware` already does this; the post-fetch path
    # reused the raw `.max()` and would `TypeError` if Polars returned
    # tz-naive on the refetched parquet).
    _klines_max_dt_raw: object = klines_shared['datetime'].max()
    if not isinstance(_klines_max_dt_raw, datetime):
        msg = (
            f'sweep signals: klines max datetime is not a datetime '
            f'(got {type(_klines_max_dt_raw).__name__}={_klines_max_dt_raw!r}); '
            f'cannot bound the SignalsTable build window.'
        )
        raise RuntimeError(msg)
    _klines_max_dt = (
        _klines_max_dt_raw if _klines_max_dt_raw.tzinfo is not None
        else _klines_max_dt_raw.replace(tzinfo=UTC)
    )
    if _klines_max_dt < replay_end:
        msg = (
            f'sweep signals: klines end at {_klines_max_dt.isoformat()}, '
            f'which is before replay_end '
            f'{replay_end.isoformat()}. The SignalsTable '
            f'would silently produce constant predictions '
            f'for every tick past {_klines_max_dt.isoformat()}. Either '
            f'shorten --replay-period-end or wait for the '
            f'upstream klines dataset to publish through '
            f'the requested window.'
        )
        raise RuntimeError(msg)
    tables: dict[str, SignalsTable] = {}
    _t_build_all = time.perf_counter()
    n_total = sum(len(d) for d in by_exp_dir.values())
    for exp_dir, decoders in by_exp_dir.items():
        # The two heavy steps in the SignalsTable build are SHARED across
        # all decoders that share an `exp_dir`: `Trainer(exp_dir, data=…)`
        # constructs the manifest + scaler factory; `trainer.train([…])`
        # fits N sklearn classifiers in one pass. Their wall time was
        # previously hidden — the per-decoder line started AFTER this
        # block, so 10 x 0.12s did not equal the outer summary's seconds.
        # Logging it here makes the math add up.
        _t_trainer = time.perf_counter()
        trainer = Trainer(exp_dir, data=klines_shared)
        manifest = trainer._manifest
        klines = klines_shared
        sensors = trainer.train([pid for pid, _ in decoders])
        print(
            f'[{time.perf_counter()-_t_trainer:7.2f}s] '
            f'sensors trained ({len(decoders)} decoder(s))',
            flush=True,
        )
        for (perm_id, display_id), sensor in zip(decoders, sensors, strict=True):
            _t_dec = time.perf_counter()
            round_params = dict(
                trainer._round_data[perm_id]['round_params'],
            )
            decoder_id = str(display_id)
            table = build_signals_table_for_decoder(
                manifest=manifest, sensor=sensor, klines=klines,
                tick_timestamps=tick_timestamps,
                round_params=round_params, decoder_id=decoder_id,
                predict_lookback=predict_lookback,
                # Mirror the bundle's `data_source.params['n_rows']`
                # the runtime poller reads (zero-bang on PR #43): without
                # this, replay tails 5000 rows where the runtime tails
                # the bundle's declared count, features diverge for any
                # bundle declaring n_rows > 5000, and assert_signals_parity
                # fires ParityViolation.
                n_rows=ds_n_rows_obj,
            )
            # Load-bearing gates — wired so neither stays decorative.
            table.assert_split_alignment(manifest.split_config)
            table.assert_window_covers(replay_start, replay_end)
            table.save(exp_dir / 'signals_tables')
            tables[decoder_id] = table
            print(
                f'[{time.perf_counter()-_t_dec:7.2f}s] '
                f'SignalsTable built decoder {display_id:<6} '
                f'({len(tick_timestamps)} ticks)',
                flush=True,
            )
    print(
        f'[{time.perf_counter()-_t_build_all:7.2f}s] '
        f'SignalsTable build done ({n_total} decoder(s))',
        flush=True,
    )
    return tables


def _print_cpcv_pbo_summary(
    *,
    per_decoder_returns: dict[str, list[float]],
    days: list[datetime],
    n_groups: int,
    n_test_groups: int,
    purge_seconds: int,
    embargo_seconds: int,
) -> None:
    """Slice #17 Task 17 — day-level CSCV PBO over deployed-strategy returns.

    Lopez de Prado §11 directly: each path picks `n_test_groups`
    of `n_groups` as OOS, the rest as IS. The deployed-strategy
    daily returns (`daily_return_for_run` output, already
    feeding DSR/SPA) are partitioned by group and Sharpe is
    computed per IS / OOS subset.

    Auditor (post-v2.0.1): the prior bar-level implementation
    used `pred * close_to_next_close_return` — a signal-return
    proxy that ignored stops, slippage, impact, maker-fill, and
    trailing-inventory exclusions. `per_decoder_returns` IS the
    deployed strategy path, so the PBO surface ranks what
    `bts sweep` actually trades.

    Skips with reason when the input shape is too thin.
    """
    if not per_decoder_returns or len(per_decoder_returns) < 2:
        print(
            f'sweep cpcv_pbo   skipped: need >=2 decoders, have '
            f'{len(per_decoder_returns)}',
        )
        return
    try:
        paths = CpcvPaths.build(
            n_groups=n_groups, n_test_groups=n_test_groups,
            purge_seconds=purge_seconds, embargo_seconds=embargo_seconds,
        )
    except ValueError as exc:
        print(f'sweep cpcv_pbo   skipped: {exc}')
        return
    result = cpcv_pbo(
        paths=paths,
        per_decoder_returns=per_decoder_returns,
        days=days,
    )
    if result is None:
        print(
            f'sweep cpcv_pbo   skipped: every path lost its IS or OOS '
            f'pool to purge/embargo or predictions are degenerate '
            f'(n_groups={n_groups} test_groups={n_test_groups} '
            f'purge_s={purge_seconds} embargo_s={embargo_seconds})',
        )
        return
    # Codex round-7 follow-up: with n_decoders=2 the OOS rank is
    # binary (1 or 2), so PBO collapses to 0.0 or 1.0. The number
    # is still computed honestly — but the precision shown can
    # mislead. Surface the constraint so the operator knows to
    # widen --n-decoders for a continuous PBO.
    binary_warning = (
        '  WARN: n_decoders=2 -> PBO is structurally binary (0.0 or '
        '1.0); widen --n-decoders for finer resolution'
        if result.n_decoders == 2 else ''
    )
    print(
        f'sweep cpcv_pbo   prob_overfit={result.pbo:.3f}  '
        f'n_paths={result.n_paths}  n_groups={n_groups}  '
        f'n_test_groups={n_test_groups}  '
        f'purge_seconds={result.purge_seconds}  '
        f'embargo_seconds={result.embargo_seconds}  '
        f'skipped_paths={result.n_paths_skipped}'
        f'{binary_warning}',
    )


def _print_sweep_stats_summary(
    per_decoder_returns: dict[str, list[float]],
    days: list[datetime], hours_start: dtime, hours_end: dtime,
    *, n_search_trials: int, n_runs_with_trailing_inventory: int,
    seed_price_at: Callable[[datetime], Decimal],
) -> None:
    """Slice #17 Task 17 — DSR + PBO + SPA summary lines.

    Each line either reports the result OR a `skipped: <reason>`
    string so the operator knows WHY a stat didn't fire. `days`
    here is the day-aligned subset (codex round 2 P1: only days
    where ALL decoders cleanly closed feed the stats — otherwise
    DSR/PBO truncate-by-position over different dates).
    `n_runs_with_trailing_inventory` is the per-(decoder, day)
    pair count of runs with open positions at window close — the
    primary "what's been excluded" signal.
    """
    n_clean_days = len(days)
    if not per_decoder_returns or n_clean_days < 2:
        print(
            f'sweep stats      skipped: '
            f'{n_runs_with_trailing_inventory} run(s) had trailing '
            f'inventory at window close, only {n_clean_days} '
            f'day-aligned observation(s)',
        )
        return
    print(
        f'sweep stats      n_runs_excluded_open_position='
        f'{n_runs_with_trailing_inventory}  clean_days={n_clean_days}',
    )
    benchmark = fetch_buy_hold_benchmark(
        days, hours_start, hours_end,
        seed_price_at=seed_price_at,
    )
    stats = compute_sweep_stats(
        per_decoder_returns, benchmark, n_search_trials=n_search_trials,
    )
    if stats.dsr is None:
        print(
            f'sweep dsr        skipped: constant or pathological '
            f'returns (sharpe_best={stats.best_sharpe})',
        )
    else:
        print(
            f'sweep dsr        sharpe_best={stats.best_sharpe:.3f}  '
            f'decoder={stats.best_decoder}  n_trials={n_search_trials}  '
            f'deflated={stats.dsr.deflated_sharpe:.3f}  '
            f'p_value={stats.dsr.p_value:.3f}',
        )
    if stats.spa is None:
        print('sweep spa        skipped: empty or mismatched candidate set')
    else:
        print(
            f'sweep spa        statistic={stats.spa.statistic:.3f}  '
            f'p_value={stats.spa.p_value:.3f}  '
            f'n_candidates={stats.spa.n_candidates}',
        )
    # Legacy half/half `sweep pbo` line removed (codex round 5 P1):
    # the underlying primitive (`probability_of_backtest_overfitting`)
    # picks the IS winner via deterministic `max()` on tied Sharpes,
    # which fabricates `pbo=0.000` on zero-return splits even when
    # the full per-decoder series differ. `sweep cpcv_pbo` is the
    # honest replacement (printed below) — it skips per-path on IS
    # and OOS rank ties and consumes `CpcvPaths` directly.


def _print_sweep_book_gap_summary(
    *, max_seconds: float, total_stops: int,
) -> None:
    """Slice #17 Task 11 — book-gap one-line sweep summary.

    Surfaces the worst stop-cross-to-trade latency observed across
    all (decoder, day) runs in the sweep. `total_stops` is the
    denominator. When 0, no STOP/TP order fired anywhere in this
    sweep — print a skip line so the operator knows the metric
    didn't have data, not that it observed zero gap (codex round 1
    P1 wording: phrase as "no STOP/TP trigger fills observed",
    not "no risk events").
    """
    if total_stops <= 0:
        print(
            'sweep book_gap   skipped: no STOP/TP trigger fills '
            'observed (default long_on_signal template uses MARKET '
            'orders; this metric activates when strategies emit '
            'STOP/TP orders)',
        )
        return
    print(
        f'sweep book_gap   max_seconds={max_seconds:.3f}  '
        f'total_stops={total_stops}',
    )


def _print_sweep_impact_summary(
    weighted_bps: list[tuple[Decimal, int]],
    n_samples_total: int,
    n_flagged_total: int,
    n_uncalibrated_total: int,
    n_rejected_total: int,
    total_runs: int,
) -> None:
    """Print the sweep-level market-impact summary line.

    Silent when the sweep saw zero impact samples AND zero
    uncalibrated submits (the model was either off or no
    orders fired). Active otherwise — the operator gets:
      - sample-weighted mean impact bps across all calibrated
        submits (heavier runs vote more, matching the
        slippage-cost aggregator's weighting).
      - total samples + flagged-as-too-large counts.
      - uncalibrated count and a WARN suffix when
        `n_uncalibrated / (n_samples + n_uncalibrated) > 10%`
        — the operator's "calibration window is too narrow"
        signal, mirroring the slippage coverage gap WARN.
      - flagged-fraction WARN when >= 5% of submitted orders
        crossed the `threshold_fraction` of bucket volume —
        the operator must down-size before live.
    """
    del total_runs  # reserved for future per-run normalisation
    if n_samples_total == 0 and n_uncalibrated_total == 0:
        return
    if weighted_bps and n_samples_total > 0:
        bps_total = sum(
            (bps * Decimal(n) for bps, n in weighted_bps),
            Decimal('0'),
        )
        bps_weight = sum(
            (Decimal(n) for _, n in weighted_bps),
            Decimal('0'),
        )
        mean_bps = bps_total / bps_weight
        bps_str = (
            f'+{mean_bps.quantize(Decimal("0.01"))}bp'
            if mean_bps >= 0
            else f'{mean_bps.quantize(Decimal("0.01"))}bp'
        )
    else:
        bps_str = 'n/a'
    line = (
        f'sweep impact     mean={bps_str}  n={n_samples_total} '
        f'flagged={n_flagged_total}  rejected={n_rejected_total}  '
        f'uncalibrated={n_uncalibrated_total}'
    )
    denom = n_samples_total + n_uncalibrated_total
    if denom > 0:
        gap_pct = Decimal(n_uncalibrated_total) * Decimal('100') / Decimal(denom)
        if gap_pct >= Decimal('10'):
            line += (
                f'  WARN: {gap_pct.quantize(Decimal("0.1"))}% calibration gap '
                f'— widen the pre-window slice or increase bucket_minutes'
            )
    if n_samples_total > 0:
        flagged_pct = (
            Decimal(n_flagged_total) * Decimal('100') / Decimal(n_samples_total)
        )
        if flagged_pct >= Decimal('5'):
            line += (
                f'  WARN: {flagged_pct.quantize(Decimal("0.1"))}% of orders '
                f'flagged as size > threshold of concurrent volume — '
                f'down-size before live'
            )
    print(line)


def _print_sweep_maker_summary(
    n_total: int,
    n_full: int,
    n_partial: int,
    n_zero: int,
    n_taker: int,
    weighted_efficiencies: list[tuple[Decimal, int]],
) -> None:
    """Sweep-level maker-fill summary.

    Silent when no LIMIT orders ran across the sweep (the
    default MARKET-strategy case). Active under `--maker` mode,
    when the strategy template emits LIMIT orders. The
    `mkt_taker` count separates marketable LIMITs (limit price
    crossed at submit → taker) from passive LIMITs that engaged
    the maker engine. `fill_eff_mean` is the passive-count-
    weighted arithmetic mean of per-run efficiency means — a
    true cross-run mean fraction filled across passive LIMITs.

    Aggregation contract:
      - `weighted_efficiencies` is `[(per_run_eff_mean, n_passive_in_run), ...]`.
      - Sum of (eff * n) / sum of n = passive-count-weighted mean.
      - Empty list → `n/a` (no passive LIMITs ran in the sweep).
    """
    if n_total == 0:
        return
    n_passive = n_total - n_taker
    if weighted_efficiencies:
        eff_total = sum(
            (e * Decimal(n) for e, n in weighted_efficiencies),
            Decimal('0'),
        )
        eff_weight = sum(
            (Decimal(n) for _, n in weighted_efficiencies),
            Decimal('0'),
        )
        mean_eff = eff_total / eff_weight
        eff_str = (
            f'{(mean_eff * Decimal("100")).quantize(Decimal("0.1"))}%'
        )
    else:
        eff_str = 'n/a'
    line = (
        f'sweep maker      n={n_total} LIMIT order(s)  '
        f'full={n_full} partial={n_partial} zero={n_zero}  '
        f'mkt_taker={n_taker}  passive={n_passive}  '
        f'fill_eff_mean={eff_str}'
    )
    if n_passive > 0:
        zero_pct = Decimal(n_zero) * Decimal('100') / Decimal(n_passive)
        if zero_pct >= Decimal('50'):
            line += (
                '  WARN: >=50% of passive LIMITs went unfilled — '
                'limit prices may be too far from touch'
            )
    print(line)


def _print_sweep_slippage_summary(
    weighted_costs: list[tuple[Decimal, int]],
    weighted_gaps: list[tuple[Decimal, int]],
    n_samples_total: int,
    n_excluded_total: int,
    n_uncal_total: int,
    runs_with_slip: int,
    total_runs: int,
) -> None:
    """Print a sweep-level slippage summary line + coverage warning.

    The per-run line gave the operator one number per run. The
    sweep aggregate makes slippage a first-class sweep decision
    metric:
      - sample-weighted mean cost across all measured fills
        (heavier runs vote more — a run with 200 fills shouldn't
        be drowned by a run with 2).
      - total samples and excluded across the sweep.
      - coverage gap warning when excluded / (samples + excluded)
        crosses 10 % — widening the calibration window or
        increasing `dt_seconds` is the operator's response.
      - explicit "all runs slip-off" message when no run had a
        model attached, so the operator can tell "no slippage
        signal" from "slippage signal of zero".
    """
    if runs_with_slip == 0:
        print(
            'sweep slippage  off — no run produced a calibrated '
            f'SlippageModel ({total_runs} run(s))',
        )
        return
    if n_samples_total == 0:
        # Some runs had a model but no fills were measured. If
        # there are any excluded fills, that's a 100% coverage
        # gap — emit the WARN explicitly. Codex flagged the
        # earlier "return early without WARN" regression.
        line = (
            f'sweep slippage  no fills measured across {runs_with_slip} '
            f'slip-on run(s); excluded={n_excluded_total}'
        )
        if n_excluded_total > 0:
            line += (
                '  WARN: 100% coverage gap — every fill landed in '
                'n_excluded; widen calibration window or increase '
                'dt_seconds'
            )
        print(line)
        return
    weighted_total: Decimal = sum(
        (cost * Decimal(n) for cost, n in weighted_costs),
        Decimal('0'),
    )
    weight_total: Decimal = sum(
        (Decimal(n) for _, n in weighted_costs),
        Decimal('0'),
    )
    mean_cost = weighted_total / weight_total
    coverage_denom = n_samples_total + n_excluded_total
    coverage_pct = (
        Decimal(n_excluded_total) / Decimal(coverage_denom) * Decimal('100')
        if coverage_denom > 0 else Decimal('0')
    )
    cost_str = f'{"+" if mean_cost > 0 else ""}{mean_cost.quantize(Decimal("0.01"))}'
    pct_str = f'{coverage_pct.quantize(Decimal("0.1"))}'
    line = (
        f'sweep slippage  cost {cost_str}bp  n={n_samples_total} fills '
        f'across {runs_with_slip} run(s)  excluded={n_excluded_total} '
        f'({pct_str}% coverage gap)'
    )
    if coverage_pct > Decimal('10'):
        line += (
            '  WARN: coverage gap >10% — widen calibration window '
            'or increase dt_seconds'
        )
    print(line)
    # Calibration-loop summary: realised - predicted gap. The
    # gap denominator is the count of fills where BOTH realised
    # AND predicted succeeded — NOT n_samples_total (which
    # includes uncalibrated_predict fills the gap can't cover).
    # Codex / auditor pinned this weighting bug.
    n_predicted_samples_total = sum(
        n for _, n in weighted_gaps
    )
    if n_predicted_samples_total == 0:
        # No successful predictions across the entire sweep.
        # Don't fabricate a "gap = 0" — that would imply
        # "calibration matched reality" when the truth is "no
        # calibration signal at all." Distinct messages for the
        # two states (no slippage model vs model attached but
        # zero successful predictions).
        if n_uncal_total > 0:
            print(
                'sweep calibration  no successful predictions  '
                f'uncalibrated_predict={n_uncal_total} '
                '(buckets do not cover this run\'s qty distribution)'
                '  WARN: cannot evaluate calibration loop',
            )
        # else: no model engaged on the predict side; the slippage
        # summary above already conveyed the off/no-fills state.
        return
    gap_total = sum(
        (g * Decimal(n) for g, n in weighted_gaps),
        Decimal('0'),
    )
    mean_gap = gap_total / Decimal(n_predicted_samples_total)
    gap_str = (
        f'{"+" if mean_gap > 0 else ""}'
        f'{mean_gap.quantize(Decimal("0.01"))}'
    )
    gap_line = (
        f'sweep calibration  predict-vs-realised gap {gap_str}bp '
        f'across {n_predicted_samples_total} predicted fills  '
        f'uncalibrated_predict={n_uncal_total}'
    )
    if abs(mean_gap) > Decimal('1.0'):
        gap_line += (
            '  WARN: gap >1bp — calibration window or qty buckets '
            'do not match this run; recalibrate'
        )
    print(gap_line)


def _resolve_hours(args: argparse.Namespace) -> tuple[dtime, dtime]:
    if args.trading_hours_start is None:
        return dtime(0, 0), dtime(23, 59)
    start = datetime.strptime(args.trading_hours_start, '%H:%M').time()
    end = datetime.strptime(args.trading_hours_end, '%H:%M').time()
    if end <= start:
        msg = f'trading-hours-end ({end}) must be > start ({start}) within one day'
        raise ValueError(msg)
    return start, end


def _resolve_days(args: argparse.Namespace) -> list[datetime]:
    if args.replay_period_start is None:
        start, end = _SECOND_WEEK_APRIL_START, _SECOND_WEEK_APRIL_END
    else:
        start = datetime.strptime(args.replay_period_start, '%Y-%m-%d').date()
        end = datetime.strptime(args.replay_period_end, '%Y-%m-%d').date()
    if end < start:
        msg = f'replay-period-end ({end}) must be >= start ({start})'
        raise ValueError(msg)
    days: list[datetime] = []
    d = start
    while d <= end:
        days.append(datetime.combine(d, dtime(0, 0), tzinfo=UTC))
        d += timedelta(days=1)
    return days
