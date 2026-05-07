"""`bts sweep` — run the backtest pipeline over N decoders by M days."""
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

from backtest_simulator.cli._metrics import (
    STARTING_CAPITAL,
    Trade,
    max_drawdown_pct,
    pair_metrics,
    pair_trades,
    print_run,
)
from backtest_simulator.cli._pipeline import pick_decoders
from backtest_simulator.cli._run_window import (
    read_kline_size_from_experiment_dir as read_kline_size_from_experiment_dir,
)
from backtest_simulator.cli._run_window import run_window_in_subprocess as run_window_in_subprocess
from backtest_simulator.cli._session_manifest import atomic_index_update as _atomic_index_update
from backtest_simulator.cli._session_manifest import finalize_session as _finalize_session
from backtest_simulator.cli._session_manifest import re_session_id_pattern as _re_session_id_pattern
from backtest_simulator.cli._signals_builder import (
    assert_signals_parity,
    build_signals_table_for_decoder,
)
from backtest_simulator.cli._stats import announce_operator_trades_tape, daily_return_for_run
from backtest_simulator.cli._verbosity import add_verbosity_arg, configure
from backtest_simulator.launcher.poller import DEFAULT_N_ROWS, DEFAULT_START_DATE_LIMIT
from backtest_simulator.sensors.precompute import SignalsTable

_SECOND_WEEK_APRIL_START: Final = datetime(2026, 4, 6, tzinfo=UTC).date()
_SECOND_WEEK_APRIL_END: Final = datetime(2026, 4, 12, tzinfo=UTC).date()
_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)

def assert_sweep_signals_parity_ran(sweep_signals_parity_total: int, *, n_picks: int, n_days: int) -> None:
    if sweep_signals_parity_total > 0:
        return
    from backtest_simulator.exceptions import ParityViolation
    msg = f'sweep signals parity: 0 comparisons made across the WHOLE sweep ({n_picks} decoder(s) x {n_days} day(s) = {n_picks * n_days} window(s)). The mandatory SignalsTable parity check did NOT run for any window. Possible causes: capture hook bypassed, parity body unreachable due to a refactor regression. Sweep result is unreliable and is rejected.'
    raise ParityViolation(msg)

def _resolve_grid_interval(picks: list[tuple[int, Decimal, Path, int]]) -> int:
    seen: dict[int, Path] = {}
    for _, _, exp_dir, _ in picks:
        ks = read_kline_size_from_experiment_dir(exp_dir)
        prior = seen.get(ks)
        if prior is None:
            seen[ks] = exp_dir
    if len(seen) != 1:
        msg = (
            f'_resolve_grid_interval: picks declare more than one '
            f'kline_size: {seen}. A sweep grid must run at one cadence; '
            f'mixed-cadence picks would land on the wrong tick boundaries.'
        )
        raise ValueError(msg)
    return next(iter(seen))

def _runtime_tick_timestamps(*, days: list[datetime], hours_start: dtime, hours_end: dtime, interval_seconds: int) -> list[datetime]:
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
    p = add_parser('sweep', 'Run the backtest pipeline over N decoders x M days.')
    exp_group = p.add_mutually_exclusive_group(required=True)
    exp_group.add_argument('--exp-code', type=Path, help='Path to a UEL-compliant Python file with module-level params() and manifest() callables. Mutually exclusive with --bundle. bts has no fallback SFD; this file is the source of truth for training and retraining picked decoders.')
    exp_group.add_argument('--bundle', type=Path, help='Path to a Limen-exported bundle zip (<name>__rNNNN.zip) containing sibling <name>.py + <name>.json + <name>.csv. Mutually exclusive with --exp-code. bts applies the JSON `data_source` override and `uel_run.n_permutations` (when --n-permutations is not given), and uses the bundled CSV as the filter pool.')
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
    p.add_argument('--maker', action='store_true', default=False, help='Strategy emits LIMIT orders at the estimated price (passive maker post). Routes through MakerFillModel for realistic queue + partial fills. Default: MARKET (taker).')
    p.add_argument('--strict-impact', action='store_true', default=False, help='Reject ENTER orders (BUY for the long-only template) the MarketImpactModel flags as exceeding 10%% of concurrent-bucket volume. SELL exits are measured but never rejected. Default: record telemetry only (observability mode).')
    p.add_argument('--max-allocation-per-trade-pct', type=str, default='0.4', help='Per-trade allocation cap as a decimal fraction. Default 0.4 (40%%) accommodates Kelly fractions on r0014-class bundles whose backtest_mean_kelly_pct is ~18%% post-Kelly fraction; lower values cause the CapitalController to deny entries with CAPITAL_RESERVATION_DENIED. Requires vaquum-nexus >= 0.35.0.')
    p.add_argument('--predict-lookback', type=int, default=None, help='Number of trailing prepared rows fed into sensor.predict per tick. Default 1 (single-row predict) preserves prior behaviour. >1 enables stateful predictors to evolve across rows. Requires vaquum-nexus >= 0.36.0.')
    p.add_argument('--session-id', type=str, required=True, help='Unique identifier for this sweep session. Outputs (sweep.log, sweep_per_window.csv, sweep_per_tick.csv) are written to ~/sweep/sessions/<session-id>/ . Errors if the directory already exists, so each session keeps an immutable record. The dashboard at http://127.0.0.1:8910/ lists sessions in a dropdown for switching between live + completed runs.')
    p.add_argument('--cpcv-n-groups', type=int, default=4, help='CPCV (CSCV) total group count for day-aligned partitioning. C(n_groups, n_test_groups) paths are evaluated. Default: 4.')
    p.add_argument('--cpcv-n-test-groups', type=int, default=2, help='CPCV test-group count per path. Default: 2 (C(4,2)=6 paths).')
    p.add_argument('--cpcv-purge-seconds', type=int, default=0, help='CPCV purge window in seconds — drop train days within this many seconds of any test-group boundary (both sides). Default: 0.')
    p.add_argument('--cpcv-embargo-seconds', type=int, default=0, help='CPCV embargo in seconds — drop train days within this many seconds AFTER each test block (Lopez de Prado direction). Default: 0.')
    p.add_argument('--trades-tape', type=Path, default=None, help='Pre-captured trades parquet covering the replay window. When set, skips CH preflight + prefetch + seed-price lookups.')
    add_verbosity_arg(p)
    p.set_defaults(func=_run)

def _run(args: argparse.Namespace) -> int:
    configure(args.verbose)
    _session_id = getattr(args, 'session_id', None)
    if _session_id is None:
        import tempfile as _tmp
        session_dir = Path(_tmp.mkdtemp(prefix='_bts_test_sweep_'))
        _index_path: Path | None = None
    else:
        sessions_root = Path.home() / 'sweep' / 'sessions'
        sessions_root.mkdir(parents=True, exist_ok=True)
        _SESSION_ID_RE = _re_session_id_pattern()
        if not _SESSION_ID_RE.fullmatch(_session_id) or _session_id == '..':
            sys.stderr.write(
                f'bts sweep: --session-id {_session_id!r} is not a valid '
                f'session identifier (allowed: alphanumeric, dash, '
                f'underscore; not "..").\n',
            )
            return 2
        session_dir = sessions_root / _session_id
        session_dir.mkdir(parents=True, exist_ok=False)
        _index_path = sessions_root / 'index.json'
        _bundle_label = Path(args.bundle).name if getattr(args, 'bundle', None) else Path(args.exp_code).name if getattr(args, 'exp_code', None) else '?'
        _now_iso = datetime.now(UTC).isoformat()
        _new_entry: dict[str, object] = {'id': _session_id, 'started_at': _now_iso, 'ended_at': None, 'bundle': _bundle_label, 'n_decoders': int(getattr(args, 'n_decoders', 0) or 0), 'replay_period': [getattr(args, 'replay_period_start', None), getattr(args, 'replay_period_end', None)]}
        _sid_for_closure = _session_id

        def _append_session(sessions: list[dict[str, object]]) -> None:
            sessions[:] = [s for s in sessions if s.get('id') != _sid_for_closure]
            sessions.append(_new_entry)
        _atomic_index_update(_index_path, _append_session)
        atexit.register(_finalize_session, _index_path, _session_id)
        print(f'[{0.0:7.2f}s] session {_session_id} → {session_dir}', flush=True)
    import logging as _logging
    for _name in ('limen', 'limen.experiment.trainer.trainer', 'praxis', 'nexus'):
        _logging.getLogger(_name).setLevel(_logging.WARNING)
    from backtest_simulator._limen_cache import install_cache
    install_cache()
    from backtest_simulator.cli._pipeline import WORK_DIR
    from backtest_simulator.pipeline.bundle import materialize_bundle_on_args
    materialize_bundle_on_args(args, WORK_DIR)
    hours_start, hours_end = _resolve_hours(args)
    days = _resolve_days(args)
    _operator_tape: Path | None = Path(args.trades_tape).expanduser().resolve() if args.trades_tape else None
    exp_code_path = Path(args.exp_code).expanduser().resolve()
    trades_q_range: tuple[float, float] | None = None
    picks, _candidate_pool_size = pick_decoders(args.n_decoders, exp_code_path=exp_code_path, n_permutations=args.n_permutations, trades_q_range=trades_q_range, tp_min_q=args.tp_min_q, fpr_max_q=args.fpr_max_q, kelly_min_q=args.kelly_min_q, trade_count_min_q=args.trade_count_min_q, net_return_min_q=args.net_return_min_q, input_from_file=args.input_from_file)
    total_runs = len(picks) * len(days)
    interval_seconds = _resolve_grid_interval(picks)
    replay_start = datetime.combine(days[0].date(), hours_start, tzinfo=UTC)
    replay_end = datetime.combine(days[-1].date(), hours_end, tzinfo=UTC)
    tick_timestamps = _runtime_tick_timestamps(days=days, hours_start=hours_start, hours_end=hours_end, interval_seconds=interval_seconds)
    raw_lookback_for_signals = getattr(args, 'predict_lookback', None)
    signals_per_decoder = _build_and_save_signals_tables(picks, tick_timestamps=tick_timestamps, replay_start=replay_start, replay_end=replay_end, predict_lookback=None if raw_lookback_for_signals is None else int(raw_lookback_for_signals))
    t_total = time.perf_counter()
    sweep_slip_costs_weighted: list[tuple[Decimal, int]] = []
    sweep_slip_gaps_weighted: list[tuple[Decimal, int]] = []
    sweep_n_samples_total = 0
    sweep_n_excluded_total = 0
    sweep_n_uncal_total = 0
    sweep_runs_with_slip = 0
    sweep_n_limit_total = 0
    sweep_n_limit_full = 0
    sweep_n_limit_partial = 0
    sweep_n_limit_zero = 0
    sweep_n_limit_taker = 0
    sweep_impact_bps_weighted: list[tuple[Decimal, int]] = []
    sweep_impact_n_samples_total = 0
    sweep_impact_n_flagged_total = 0
    sweep_impact_n_uncalibrated_total = 0
    sweep_impact_n_rejected_total = 0
    sweep_book_gap_total_stops = 0
    sweep_signals_parity_total = 0
    decoder_day_returns: dict[str, dict[int, float]] = {}
    n_runs_with_trailing_inventory = 0
    import csv as _csv
    csv_dir = session_dir
    csv_path = csv_dir / 'sweep_per_window.csv'
    csv_fp = csv_path.open('w', newline='', encoding='utf-8')
    atexit.register(csv_fp.close)
    csv_writer = _csv.writer(csv_fp)
    csv_writer.writerow(['display_id', 'date', 'capital_alloc_pct', 'profit_pct', 'r_mean', 'max_drawdown_pct', 'n_trades', 'prob_min', 'prob_max'])
    csv_fp.flush()
    tick_csv_path = csv_dir / 'sweep_per_tick.csv'
    tick_csv_fp = tick_csv_path.open('w', newline='', encoding='utf-8')
    atexit.register(tick_csv_fp.close)
    tick_csv_writer = _csv.writer(tick_csv_fp)
    tick_csv_writer.writerow(['display_id', 'date', 'timestamp', 'pred', 'prob'])
    tick_csv_fp.flush()
    import concurrent.futures as _cf
    import os as _os
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
    _raw_max_alloc = getattr(args, 'max_allocation_per_trade_pct', None)
    _raw_lookback = getattr(args, 'predict_lookback', None)
    _maker_preference = bool(getattr(args, 'maker', False))
    _strict_impact = bool(getattr(args, 'strict_impact', False))
    _max_alloc_decimal: Decimal | None = None if _raw_max_alloc is None else Decimal(str(_raw_max_alloc))
    _predict_lookback_int: int | None = None if _raw_lookback is None else int(_raw_lookback)
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
            window_specs.append(_WindowSpec(perm_id, kelly, exp_dir, display_id, day_idx, day, window_start, window_end))

    def _run_one_window(spec: _WindowSpec) -> tuple[object, float]:
        _perm_id, _kelly, _exp_dir, _display_id, _day_idx, _day, _ws, _we = spec
        _t_work_start = time.perf_counter()
        _result = run_window_in_subprocess(_perm_id, _kelly, _ws, _we, _exp_dir, display_id=_display_id, maker_preference=_maker_preference, strict_impact=_strict_impact, trades_parquet_path=_trades_parquet_path, max_allocation_per_trade_pct=_max_alloc_decimal, predict_lookback=_predict_lookback_int)
        return (_result, time.perf_counter() - _t_work_start)
    n_workers = min(len(window_specs), max(1, (_os.cpu_count() or 4) // 2), 8)
    print(f'[{0.0:7.2f}s] replay launching {len(window_specs)} window(s) on {n_workers} worker(s)', flush=True)
    _t_par = time.perf_counter()
    parallel_results: dict[tuple[int, int, str], object] = {}

    def _write_csv_row_now(_did: int, _label: str, _res: object) -> None:
        from typing import cast as _cast2

        from backtest_simulator.cli._run_window import WindowResult as _WR2
        _r = _cast2(_WR2, _res)
        _trades_raw = _r.get('trades', [])
        _stops_raw = _r.get('declared_stops', {})
        _trades_local = [Trade(*row) for row in _trades_raw]
        _declared_local = {k: Decimal(v) for k, v in _stops_raw.items()}
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
        _r_mean = sum(_r_mults, Decimal('0')) / len(_r_mults) if _r_mults else None
        _max_dd = max_drawdown_pct(_net_pnls, STARTING_CAPITAL)
        _buy_notional = sum((_b.qty * _b.price for _b, _ in _pairs), Decimal('0'))
        _cap_alloc = _buy_notional / STARTING_CAPITAL * Decimal('100') if _buy_notional > 0 else Decimal('0')
        _runtime_preds = _r.get('runtime_predictions', [])
        _all_probs: list[float] = []
        for _e in _runtime_preds:
            _prob = _e.get('prob')
            if isinstance(_prob, (int, float)):
                _all_probs.append(float(_prob))
        _pmin = min(_all_probs) if _all_probs else None
        _pmax = max(_all_probs) if _all_probs else None
        csv_writer.writerow([_did, _label, f'{_cap_alloc:.4f}', '' if _total_pct is None else f'{_total_pct:.4f}', '' if _r_mean is None else f'{_r_mean:.4f}', f'{_max_dd:.4f}', len(_pairs), '' if _pmin is None else f'{_pmin:.6f}', '' if _pmax is None else f'{_pmax:.6f}'])
        csv_fp.flush()
        for _entry in _runtime_preds:
            tick_csv_writer.writerow([_did, _label, _entry.get('timestamp', ''), _entry.get('pred', ''), _entry.get('prob', '') if _entry.get('prob') is not None else ''])
        tick_csv_fp.flush()
    with _cf.ThreadPoolExecutor(max_workers=n_workers) as _ex:
        _futures = {_ex.submit(_run_one_window, spec): spec for spec in window_specs}
        for _fut in _cf.as_completed(_futures):
            _spec = _futures[_fut]
            _p, _k, _ed, _did, _di, _d, _ws, _we = _spec
            _label = _d.date().isoformat()
            try:
                _res, _t_window = _fut.result()
            except Exception as _exc:
                msg = f'sweep aborted: perm {_did} on {_label} failed: {_exc!r}. Per-window failures are not silently continued — every picked decoder x day must produce a parity-validated run.'
                raise RuntimeError(msg) from _exc
            parallel_results[_p, _did, _label] = _res
            _write_csv_row_now(_did, _label, _res)
            _n_done = len(parallel_results)
            _n_trades = 0
            if isinstance(_res, dict):
                _trades_field = cast('dict[str, object]', _res).get('trades')
                if isinstance(_trades_field, list):
                    _n_trades = len(cast('list[object]', _trades_field))
            print(f'[{_t_window:7.2f}s] perm {_did:<6} {_label}  done  ({_n_done}/{len(window_specs)}, trades={_n_trades})', flush=True)
    print(f'[{time.perf_counter() - _t_par:7.2f}s] replay batch done ({len(window_specs)} window(s))', flush=True)
    for perm_id, kelly, exp_dir, display_id in picks:
        for day_idx, day in enumerate(days):
            window_start = datetime.combine(day.date(), hours_start, tzinfo=UTC)
            window_end = datetime.combine(day.date(), hours_end, tzinfo=UTC)
            day_label = day.date().isoformat()
            result_obj = parallel_results[perm_id, display_id, day_label]
            from typing import cast as _cast

            from backtest_simulator.cli._run_window import WindowResult as _WR
            result = _cast(_WR, result_obj)
            trades_raw = result['trades']
            stops_raw = result['declared_stops']
            trades = [Trade(*row) for row in trades_raw]
            declared_stops = {k: Decimal(v) for k, v in stops_raw.items()}
            slip_raw = result.get('slippage_realised_cost_bps')
            slip_cost = None if slip_raw is None else Decimal(str(slip_raw))
            slip_n = int(result.get('slippage_n_samples', 0))
            slip_excl = int(result.get('slippage_n_excluded', 0))
            slip_uncal = int(result.get('slippage_n_uncalibrated_predict', 0))
            slip_predicted_n = int(result.get('slippage_n_predicted_samples', 0))
            slip_gap_raw = result.get('slippage_predict_vs_realised_gap_bps')
            slip_gap = None if slip_gap_raw is None else Decimal(str(slip_gap_raw))
            n_limit = int(result.get('n_limit_orders_submitted', 0))
            n_limit_full = int(result.get('n_limit_filled_full', 0))
            n_limit_partial = int(result.get('n_limit_filled_partial', 0))
            n_limit_zero = int(result.get('n_limit_filled_zero', 0))
            n_limit_taker = int(result.get('n_limit_marketable_taker', 0))
            int(result.get('n_passive_limits', 0))
            eff_raw = result.get('maker_fill_efficiency_p50')
            eff_p50 = None if eff_raw is None else Decimal(str(eff_raw))
            eff_mean_raw = result.get('maker_fill_efficiency_mean')
            None if eff_mean_raw is None else Decimal(str(eff_mean_raw))
            impact_raw = result.get('market_impact_realised_bps')
            impact_bps = None if impact_raw is None else Decimal(str(impact_raw))
            impact_n = int(result.get('market_impact_n_samples', 0))
            impact_flagged = int(result.get('market_impact_n_flagged', 0))
            impact_uncal = int(result.get('market_impact_n_uncalibrated', 0))
            impact_rejected = int(result.get('market_impact_n_rejected', 0))
            decoder_key = str(display_id)
            table = signals_per_decoder.get(decoder_key)
            runtime_preds_list = result.get('runtime_predictions', [])
            window_expected_ticks = [t for t in tick_timestamps if window_start < t <= window_end]
            n_signals_compared = assert_signals_parity(decoder_id=decoder_key, table=table, runtime_predictions=runtime_preds_list, expected_ticks=window_expected_ticks, interval_seconds=interval_seconds)
            sweep_signals_parity_total += n_signals_compared
            print_run(display_id, day_label, trades, declared_stops, n_intents=int(result.get('n_intents', 0)), n_fills=int(result.get('n_fills', 0)), n_pending=int(result.get('n_pending', 0)), n_rejects=int(result.get('n_rejects', 0)), slippage_cost_bps=slip_cost, slippage_n_samples=slip_n, slippage_n_excluded=slip_excl, slippage_predict_vs_realised_gap_bps=slip_gap, slippage_n_uncalibrated_predict=slip_uncal, slippage_n_predicted_samples=slip_predicted_n, n_limit_orders_submitted=n_limit, n_limit_filled_full=n_limit_full, n_limit_filled_partial=n_limit_partial, n_limit_filled_zero=n_limit_zero, n_limit_marketable_taker=n_limit_taker, maker_fill_efficiency_p50=eff_p50, market_impact_realised_bps=impact_bps, market_impact_n_samples=impact_n, market_impact_n_flagged=impact_flagged, market_impact_n_uncalibrated=impact_uncal)
            float(result.get('book_gap_max_seconds', 0.0))
            run_book_gap_n = int(result.get('book_gap_n_observed', 0))
            sweep_book_gap_total_stops += run_book_gap_n
            day_return = daily_return_for_run(trades, declared_stops)
            if day_return is None:
                n_runs_with_trailing_inventory += 1
            else:
                decoder_day_returns[str(display_id)][day_idx] = day_return
            sweep_n_limit_total += n_limit
            sweep_n_limit_full += n_limit_full
            sweep_n_limit_partial += n_limit_partial
            sweep_n_limit_zero += n_limit_zero
            sweep_n_limit_taker += n_limit_taker
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
                if slip_gap is not None and slip_predicted_n > 0:
                    sweep_slip_gaps_weighted.append((slip_gap, slip_predicted_n))
    t_wall = time.perf_counter() - t_total
    csv_fp.close()
    tick_csv_fp.close()
    print(f'\ndone   {total_runs} run(s) in {t_wall:.1f}s')
    print(f'sweep csv        {csv_path}')
    print(f'sweep tick csv   {tick_csv_path}')
    assert_sweep_signals_parity_ran(sweep_signals_parity_total, n_picks=len(picks), n_days=len(days))
    print(f'sweep signals parity  OK n_compared={sweep_signals_parity_total} (runtime Sensor.predict matches SignalsTable per-tick)')
    return 0

def _build_and_save_signals_tables(picks: list[tuple[int, Decimal, Path, int]], *, tick_timestamps: list[datetime], replay_start: datetime, replay_end: datetime, predict_lookback: int | None=None) -> dict[str, SignalsTable]:
    import importlib
    import json as _json

    from limen import HistoricalData, Trainer
    by_exp_dir: dict[Path, list[tuple[int, int]]] = {}
    for perm_id, _, exp_dir, display_id in picks:
        by_exp_dir.setdefault(exp_dir, []).append((perm_id, display_id))
    HistoricalData()
    first_exp_dir = next(iter(by_exp_dir))
    _tm = time.perf_counter()
    with (first_exp_dir / 'metadata.json').open('r') as _mf:
        _meta = _json.load(_mf)
    _sfd = importlib.import_module(_meta['sfd_module'])
    manifest_for_cfg = _sfd.manifest()
    print(f"[{time.perf_counter() - _tm:7.2f}s] manifest loaded (kline_size={(int(manifest_for_cfg.data_source_config.params['kline_size']) if manifest_for_cfg.data_source_config else '?')})", flush=True)
    cfg = manifest_for_cfg.data_source_config
    kline_size = int(cfg.params['kline_size'])
    ds_params = dict(cfg.params)
    ds_params.pop('kline_size', None)
    ds_n_rows_obj = ds_params.pop('n_rows', DEFAULT_N_ROWS)
    ds_params.pop('start_date_limit', DEFAULT_START_DATE_LIMIT)
    import polars as _pl
    _cp = Path.home() / '.cache' / 'backtest_simulator' / 'limen_klines' / f'btcusdt_{kline_size}.parquet'
    _now_utc = datetime.now(UTC)
    _STALE_THRESHOLD_HOURS = 48
    _use_cache = False
    klines_shared = None
    _t_klines = time.perf_counter()
    if _cp.is_file():
        cached = _pl.read_parquet(_cp)
        _cached_max_obj: object = cached['datetime'].max()
        if isinstance(_cached_max_obj, datetime):
            _cached_max_aware = _cached_max_obj if _cached_max_obj.tzinfo is not None else _cached_max_obj.replace(tzinfo=UTC)
            age_h = (_now_utc - _cached_max_aware).total_seconds() / 3600.0
            if age_h < _STALE_THRESHOLD_HOURS:
                klines_shared = cached
                _use_cache = True
                print(f'[{time.perf_counter() - _t_klines:7.2f}s] klines cache HIT (age={age_h:.1f}h, {cached.height} rows)', flush=True)
    _klines_max_dt_raw: object = klines_shared['datetime'].max()
    _klines_max_dt = _klines_max_dt_raw if _klines_max_dt_raw.tzinfo is not None else _klines_max_dt_raw.replace(tzinfo=UTC)
    tables: dict[str, SignalsTable] = {}
    _t_build_all = time.perf_counter()
    n_total = sum(len(d) for d in by_exp_dir.values())
    for exp_dir, decoders in by_exp_dir.items():
        _t_trainer = time.perf_counter()
        trainer = Trainer(exp_dir, data=klines_shared)
        manifest = trainer._manifest
        klines = klines_shared
        sensors = trainer.train([pid for pid, _ in decoders])
        print(f'[{time.perf_counter() - _t_trainer:7.2f}s] sensors trained ({len(decoders)} decoder(s))', flush=True)
        for (perm_id, display_id), sensor in zip(decoders, sensors, strict=True):
            _t_dec = time.perf_counter()
            round_params = dict(trainer._round_data[perm_id]['round_params'])
            decoder_id = str(display_id)
            table = build_signals_table_for_decoder(manifest=manifest, sensor=sensor, klines=klines, tick_timestamps=tick_timestamps, round_params=round_params, decoder_id=decoder_id, predict_lookback=predict_lookback, n_rows=ds_n_rows_obj)
            table.assert_split_alignment(manifest.split_config)
            table.assert_window_covers(replay_start, replay_end)
            table.save(exp_dir / 'signals_tables')
            tables[decoder_id] = table
            print(f'[{time.perf_counter() - _t_dec:7.2f}s] SignalsTable built decoder {display_id:<6} ({len(tick_timestamps)} ticks)', flush=True)
    print(f'[{time.perf_counter() - _t_build_all:7.2f}s] SignalsTable build done ({n_total} decoder(s))', flush=True)
    return tables

def _resolve_hours(args: argparse.Namespace) -> tuple[dtime, dtime]:
    if args.trading_hours_start is None:
        return (dtime(0, 0), dtime(23, 59))
    start = datetime.strptime(args.trading_hours_start, '%H:%M').time()
    end = datetime.strptime(args.trading_hours_end, '%H:%M').time()
    return (start, end)

def _resolve_days(args: argparse.Namespace) -> list[datetime]:
    if args.replay_period_start is None:
        start, end = (_SECOND_WEEK_APRIL_START, _SECOND_WEEK_APRIL_END)
    else:
        start = datetime.strptime(args.replay_period_start, '%Y-%m-%d').date()
        end = datetime.strptime(args.replay_period_end, '%Y-%m-%d').date()
    days: list[datetime] = []
    d = start
    while d <= end:
        days.append(datetime.combine(d, dtime(0, 0), tzinfo=UTC))
        d += timedelta(days=1)
    return days
