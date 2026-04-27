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
                    maker_preference=bool(getattr(args, 'maker', False)),
                    strict_impact=bool(getattr(args, 'strict_impact', False)),
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
            print_run(
                display_id, day_label, trades, declared_stops,
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
    print(f'\ndone   {total_runs} run(s) in {t_wall:.1f}s')
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
    return 0


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
