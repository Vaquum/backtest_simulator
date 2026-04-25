"""`bts sweep` — run the backtest pipeline over N decoders by M days."""

# Migration of the operator-driven `/tmp/bts_sweep.py` orchestration into
# the package. Each window runs in a fresh Python subprocess (see
# `backtest_simulator.cli._run_window`) so state bleed between windows is
# impossible. Per-run output is the one-line summary defined in
# `backtest_simulator.cli._metrics.print_run`.
from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime, timedelta
from datetime import time as dtime
from decimal import Decimal
from typing import Final

from backtest_simulator.cli._metrics import Trade, print_run
from backtest_simulator.cli._pipeline import (
    ensure_trained,
    pick_decoders,
    preflight_tunnel,
)
from backtest_simulator.cli._run_window import run_window_in_subprocess
from backtest_simulator.cli._verbosity import add_verbosity_arg, configure

_SECOND_WEEK_APRIL_START: Final = datetime(2026, 4, 6, tzinfo=UTC).date()
_SECOND_WEEK_APRIL_END: Final = datetime(2026, 4, 12, tzinfo=UTC).date()


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        'sweep',
        help='Run the backtest pipeline over N decoders x M days.',
    )
    p.add_argument('--n-decoders', type=int, default=1)
    p.add_argument('--n-permutations', type=int, default=30)
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
    add_verbosity_arg(p)
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    configure(args.verbose)
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
    preflight_tunnel()
    if args.input_from_file is None:
        ensure_trained(args.n_permutations)
    trades_q_range: tuple[float, float] | None = None
    if args.trades_q_range is not None:
        lo_str, hi_str = args.trades_q_range.split(',')
        trades_q_range = (float(lo_str), float(hi_str))
    picks = pick_decoders(
        args.n_decoders,
        trades_q_range=trades_q_range,
        tp_min_q=args.tp_min_q,
        fpr_max_q=args.fpr_max_q,
        kelly_min_q=args.kelly_min_q,
        trade_count_min_q=args.trade_count_min_q,
        net_return_min_q=args.net_return_min_q,
        input_from_file=args.input_from_file,
    )
    hours_label = f'{hours_start.strftime("%H:%M")}-{hours_end.strftime("%H:%M")} UTC'
    total_runs = len(picks) * len(days)
    print(
        f'\nbts sweep   {len(picks)} decoder(s) x {len(days)} day(s) = '
        f'{total_runs} run(s)   hours {hours_label}\n',
    )
    t_total = time.perf_counter()
    for perm_id, kelly, exp_dir, display_id in picks:
        for day in days:
            window_start = datetime.combine(day.date(), hours_start, tzinfo=UTC)
            window_end = datetime.combine(day.date(), hours_end, tzinfo=UTC)
            day_label = day.date().isoformat()
            sys.stderr.write(
                f'  ... perm {display_id:<6}  {day_label}  running        \r',
            )
            sys.stderr.flush()
            t0 = time.perf_counter()
            try:
                result = run_window_in_subprocess(
                    perm_id, kelly, window_start, window_end, exp_dir,
                )
            except Exception as exc:  # noqa: BLE001 - surface per-run errors
                dt_ms = int((time.perf_counter() - t0) * 1000)
                print(
                    f'  perm {display_id:<6}  {day_label}  FAILED in {dt_ms}ms: {exc!r}',
                )
                continue
            trades_raw = result['trades']
            stops_raw = result['declared_stops']
            assert isinstance(trades_raw, list)
            assert isinstance(stops_raw, dict)
            trades = [Trade(*row) for row in trades_raw]
            declared_stops = {k: Decimal(str(v)) for k, v in stops_raw.items()}
            print_run(display_id, day_label, trades, declared_stops)
    t_wall = time.perf_counter() - t_total
    print(f'\ndone   {total_runs} run(s) in {t_wall:.1f}s')
    return 0


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
