"""`bts run` — run one backtest window for one decoder."""

# Thin wrapper over `backtest_simulator.cli._run_window.run_window_in_subprocess`
# that consumes one decoder + one window and either prints the human-
# readable summary (`--output-format text`) or emits the structured JSON
# report (`--output-format json`). The slippage fields
# (`slippage_realised_bps`, `slippage_realised_cost_bps`,
# `slippage_realised_buy_bps`, `slippage_realised_sell_bps`,
# `slippage_n_samples`, `slippage_n_excluded`) report measurements taken
# against the rolling mid; the SlippageModel does NOT adjust fill_price
# (walk_trades returns realistic taker prices from the historical tape).
# `market_impact_realised_bps`, `market_impact_n_samples`,
# `market_impact_n_flagged`, `market_impact_n_uncalibrated` are
# populated by `MarketImpactModel.evaluate` per order submit. The
# bps figure is measurement-only — `walk_trades` already prices
# against tape so we don't double-charge the impact; the metric
# is the operator-visible "would have moved the book this much
# in live" signal. `book_gap_max_seconds` is populated by its
# own wiring task once it lands.
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from backtest_simulator.cli._metrics import Trade, print_run
from backtest_simulator.cli._pipeline import (
    EXP_DIR,
    ensure_trained,
    pick_decoders,
    preflight_tunnel,
)
from backtest_simulator.cli._run_window import run_window_in_subprocess
from backtest_simulator.cli._verbosity import add_verbosity_arg, configure


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        'run',
        help='Run one backtest window for one decoder.',
    )
    p.add_argument('--window-start', required=True, type=str,
                   help='ISO8601 (e.g. 2026-04-20T08:00:00+00:00).')
    p.add_argument('--window-end', required=True, type=str,
                   help='ISO8601 (must be > --window-start).')
    p.add_argument('--decoder-id', type=int, default=None,
                   help='Permutation id to use; default picks #1 from filter.')
    p.add_argument('--n-decoders', type=int, default=1,
                   help='When --decoder-id is omitted, pick top-N (default 1).')
    p.add_argument('--experiment-dir', type=Path, default=None,
                   help='Override experiment dir (default: package cache).')
    p.add_argument('--input-from-file', type=str, default=None,
                   help='Use a results.csv as the filter pool.')
    p.add_argument('--output-format', choices=['text', 'json'], default='text',
                   help='text (default) or json structured report.')
    p.add_argument('--seed', type=int, default=None,
                   help='Override seed for reproducible runs.')
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
    # Slice #17 Task 29 — ATR R-denominator gameability gate knobs.
    p.add_argument('--atr-k', type=str, default='0.5',
                   help=(
                       'ATR-floor multiplier — stop must be >= '
                       'k * ATR(window) from entry. 0 disables the '
                       'gate. Default: 0.5 (half a local ATR).'
                   ))
    p.add_argument('--atr-window-seconds', type=int, default=900,
                   help=(
                       'ATR window in seconds (mean of per-1-min '
                       'ranges). Default: 900s (14-period ATR).'
                   ))
    # Slice #17 Task 18 — ledger-parity gate. The bts side dumps
    # event_spine.jsonl on every run regardless. With
    # --check-parity-vs, after the run, `assert_ledger_parity` is
    # called against the supplied reference. STRICT default; the
    # CLOCK_NORMALIZED tolerance strips envelope event_seq +
    # timestamp (cross-runtime wall-clock vs frozen difference)
    # but keeps payload bytes intact. Reference must come from a
    # deterministic-action-id source — Praxis core uuid4 generates
    # `command_id` (`launcher.py:524,577`) and bts can also
    # uuid-generate `trade_id` (`action_submitter.py:581`) when
    # the strategy emits None for either; normal run-vs-run STRICT
    # parity is structurally blocked by those non-determinisms
    # until a future task ties IDs to a deterministic counter.
    p.add_argument('--check-parity-vs', type=Path, default=None,
                   help=(
                       'After run, assert event-spine parity vs the '
                       'JSONL reference at PATH. Reference must come '
                       'from a deterministic-action-id source '
                       '(scripted Praxis); normal run-vs-run STRICT '
                       'parity is blocked by Praxis uuid4 generation '
                       'of command_id / trade_id and is tracked as '
                       'a follow-up.'
                   ))
    p.add_argument('--parity-tolerance',
                   choices=['strict', 'clock_normalized'],
                   default='strict',
                   help=(
                       'strict (default): byte-identical compare. '
                       'clock_normalized: strip envelope event_seq + '
                       'timestamp, keep payload bytes (for the '
                       'cross-runtime case where the same scripted '
                       'Praxis runs at different wall-clock times).'
                   ))
    add_verbosity_arg(p)
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    configure(args.verbose)
    window_start = datetime.fromisoformat(args.window_start)
    window_end = datetime.fromisoformat(args.window_end)
    if window_end <= window_start:
        sys.stderr.write(
            f'bts run: --window-end ({window_end}) must be > --window-start ({window_start})\n',
        )
        return 2
    preflight_tunnel()
    perm_id, kelly, exp_dir, display_id = _resolve_decoder(args)
    result = run_window_in_subprocess(
        perm_id, kelly, window_start, window_end, exp_dir,
        maker_preference=bool(getattr(args, 'maker', False)),
        strict_impact=bool(getattr(args, 'strict_impact', False)),
        atr_k=str(getattr(args, 'atr_k', '0.5')),
        atr_window_seconds=int(getattr(args, 'atr_window_seconds', 900)),
    )
    trades_raw = result['trades']
    stops_raw = result['declared_stops']
    assert isinstance(trades_raw, list)
    assert isinstance(stops_raw, dict)
    if args.output_format == 'json':
        report = {
            'decoder_id': display_id,
            'kelly_pct': str(kelly),
            'window_start': window_start.isoformat(),
            'window_end': window_end.isoformat(),
            'orders': result['orders'],
            'trades': trades_raw,
            'declared_stops': stops_raw,
            # Slice #17 Task 11 — book-gap instrumentation.
            # `_walk_stop` records `(t_first_trade - t_cross)` for
            # every STOP/TP trigger; the venue's BookGapInstrument
            # aggregates per run. `n_observed=0` is the honest
            # signal for a run that emitted only MARKET orders.
            'book_gap_max_seconds': result.get('book_gap_max_seconds'),
            'book_gap_n_observed': result.get('book_gap_n_observed', 0),
            'book_gap_p95_seconds': result.get('book_gap_p95_seconds'),
            # Signed mean (directional). Operator should NOT cite
            # this as cost — see `slippage_realised_cost_bps`.
            'slippage_realised_bps': result.get('slippage_realised_bps'),
            # Cost metric: side-normalized mean. Positive = paid
            # spread; negative = price improvement.
            'slippage_realised_cost_bps': result.get(
                'slippage_realised_cost_bps',
            ),
            'slippage_realised_buy_bps': result.get('slippage_realised_buy_bps'),
            'slippage_realised_sell_bps': result.get('slippage_realised_sell_bps'),
            # Calibration loop: model's predicted cost and the
            # realised - predicted gap (calibration error signal).
            'slippage_predicted_cost_bps': result.get(
                'slippage_predicted_cost_bps',
            ),
            'slippage_predict_vs_realised_gap_bps': result.get(
                'slippage_predict_vs_realised_gap_bps',
            ),
            'slippage_n_samples': result.get('slippage_n_samples', 0),
            'slippage_n_excluded': result.get('slippage_n_excluded', 0),
            'slippage_n_uncalibrated_predict': result.get(
                'slippage_n_uncalibrated_predict', 0,
            ),
            'slippage_n_predicted_samples': result.get(
                'slippage_n_predicted_samples', 0,
            ),
            # Maker-fill telemetry. Zero unless --maker engaged
            # the LIMIT-on-signal strategy variant.
            'n_limit_orders_submitted': result.get(
                'n_limit_orders_submitted', 0,
            ),
            'n_limit_filled_full': result.get('n_limit_filled_full', 0),
            'n_limit_filled_partial': result.get('n_limit_filled_partial', 0),
            'n_limit_filled_zero': result.get('n_limit_filled_zero', 0),
            'n_limit_marketable_taker': result.get(
                'n_limit_marketable_taker', 0,
            ),
            'maker_fill_efficiency_p50': result.get(
                'maker_fill_efficiency_p50',
            ),
            # Market impact telemetry. `realised_bps` is the
            # mean of the model's per-order estimated impact
            # (linear interpolation of order_qty against
            # concurrent-bucket volume → bps). `n_samples` is
            # the order count that hit a calibrated bucket;
            # `n_flagged` counts those whose qty exceeded the
            # threshold fraction of bucket volume; and
            # `n_uncalibrated` counts orders whose timestamp
            # had no matching bucket (calibration gap).
            'market_impact_realised_bps': result.get(
                'market_impact_realised_bps',
            ),
            'market_impact_n_samples': result.get(
                'market_impact_n_samples', 0,
            ),
            'market_impact_n_flagged': result.get(
                'market_impact_n_flagged', 0,
            ),
            'market_impact_n_uncalibrated': result.get(
                'market_impact_n_uncalibrated', 0,
            ),
            'market_impact_n_rejected': result.get(
                'market_impact_n_rejected', 0,
            ),
            # ATR R-denominator gameability gate (slice #17 Task 29).
            'n_atr_rejected': result.get('n_atr_rejected', 0),
            'n_atr_uncalibrated': result.get('n_atr_uncalibrated', 0),
            # Slice #17 Task 18 ledger parity. The bts spine is
            # always dumped post-run; the reference comparison
            # fires only when --check-parity-vs is set.
            'event_spine_jsonl': result.get('event_spine_jsonl'),
            'event_spine_n_events': result.get('event_spine_n_events', 0),
        }
        sys.stdout.write(json.dumps(report) + '\n')
        # Codex round-5 P1: JSON mode keeps stdout a single
        # parseable object. Parity status flows via return code
        # + stderr only (no extra stdout lines).
        return _maybe_assert_parity(args, result, emit_human=False)
    trades = [Trade(*row) for row in trades_raw]
    declared_stops = {k: Decimal(str(v)) for k, v in stops_raw.items()}
    slip_raw = result.get('slippage_realised_cost_bps')
    slip_cost = None if slip_raw is None else Decimal(str(slip_raw))
    impact_raw = result.get('market_impact_realised_bps')
    impact_bps = None if impact_raw is None else Decimal(str(impact_raw))
    print_run(
        display_id, window_start.date().isoformat(), trades,
        declared_stops,
        slippage_cost_bps=slip_cost,
        slippage_n_samples=int(result.get('slippage_n_samples', 0)),
        slippage_n_excluded=int(result.get('slippage_n_excluded', 0)),
        market_impact_realised_bps=impact_bps,
        market_impact_n_samples=int(result.get(
            'market_impact_n_samples', 0,
        )),
        market_impact_n_flagged=int(result.get(
            'market_impact_n_flagged', 0,
        )),
        market_impact_n_uncalibrated=int(result.get(
            'market_impact_n_uncalibrated', 0,
        )),
    )
    # Slice #17 Task 11 — book-gap one-liner. Surface ONLY when at
    # least one STOP/TP fired (codex round 1: skip noise on quiet
    # runs). The line lives after print_run's cost-metric line so
    # operators don't have to grep for it.
    book_gap_n = int(result.get('book_gap_n_observed', 0))
    if book_gap_n > 0:
        print(
            f'   book_gap   max={result.get("book_gap_max_seconds", 0.0):.3f}s  '
            f'p95={result.get("book_gap_p95_seconds", 0.0):.3f}s  '
            f'n_stops={book_gap_n}',
        )
    return _maybe_assert_parity(args, result)


def _maybe_assert_parity(
    args: argparse.Namespace, result: dict[str, object],
    *, emit_human: bool = True,
) -> int:
    """Slice #17 Task 18 — call `assert_ledger_parity` if `--check-parity-vs`.

    Returns the CLI exit code: 0 on parity pass (or no check
    requested), 1 on parity violation OR on missing/invalid spine
    artifact when a check WAS requested (codex round 4 P1: the
    parity gate must never silently fail-open).

    `emit_human` controls whether the human-readable status lines
    are written to stdout. Default True (text mode). JSON mode
    passes False so stdout stays a single parseable JSON object;
    parity result still surfaces via return code + stderr (codex
    round-5 P1).
    """
    spine_path_raw = result.get('event_spine_jsonl')
    n_events = int(result.get('event_spine_n_events', 0))
    if not isinstance(spine_path_raw, str):
        if args.check_parity_vs is not None:
            sys.stderr.write(
                f'bts run: --check-parity-vs requested but '
                f'event_spine_jsonl missing from subprocess result '
                f'(got {type(spine_path_raw).__name__}); failing '
                f'parity check loudly.\n',
            )
            return 1
        if emit_human:
            sys.stderr.write(
                f'bts run: event_spine_jsonl missing from subprocess '
                f'result (got {type(spine_path_raw).__name__}); '
                f'no parity check requested.\n',
            )
        return 0
    spine_path = Path(spine_path_raw)
    if args.check_parity_vs is None:
        # Always-dump (codex round 1 #4): emit the path so the
        # operator can chain into other parity tooling. Text mode
        # only — JSON mode keeps stdout pure.
        if emit_human:
            print(
                f'bts run         event_spine_jsonl={spine_path}  '
                f'n_events={n_events}',
            )
        return 0
    reference = Path(args.check_parity_vs)
    from backtest_simulator.exceptions import ParityViolation
    from backtest_simulator.honesty.ledger_parity import (
        ParityTolerance,
        assert_ledger_parity,
    )
    tolerance = ParityTolerance[args.parity_tolerance.upper()]
    try:
        assert_ledger_parity(
            backtest_event_spine=spine_path,
            paper_event_spine=reference,
            tolerance=tolerance,
        )
    except ParityViolation as exc:
        if emit_human:
            print(
                f'bts run         parity FAIL  vs={reference}  '
                f'tolerance={args.parity_tolerance}  n_events={n_events}',
            )
        sys.stderr.write(f'{exc}\n')
        return 1
    if emit_human:
        print(
            f'bts run         parity PASS  vs={reference}  '
            f'tolerance={args.parity_tolerance}  n_events={n_events}',
        )
    return 0


def _resolve_decoder(args: argparse.Namespace) -> tuple[int, Decimal, Path, int]:
    """Pick one decoder from the cache or by explicit id."""
    if args.decoder_id is not None:
        exp_dir = args.experiment_dir or EXP_DIR
        return args.decoder_id, _kelly_for_decoder(exp_dir, args.decoder_id), exp_dir, args.decoder_id
    if args.input_from_file is None:
        ensure_trained(args.n_decoders)
    picks, _ = pick_decoders(
        args.n_decoders,
        input_from_file=args.input_from_file,
    )
    perm_id, kelly, exp_dir, display_id = picks[0]
    return perm_id, kelly, exp_dir, display_id


def _kelly_for_decoder(exp_dir: Path, decoder_id: int) -> Decimal:
    """Look up `backtest_mean_kelly_pct` for a specific permutation_id.

    `bts run --decoder-id N` skips the filter / training pipeline and
    targets one explicit decoder. The kelly value drives the strategy's
    size; defaulting it to 0 (the prior placeholder) made the run
    trivially do nothing. Reading from `results.csv` gives the same
    kelly the sweep would have used.
    """
    import polars as pl
    results_csv = exp_dir / 'results.csv'
    if not results_csv.is_file():
        msg = (
            f'bts run --decoder-id {decoder_id}: no results.csv at '
            f'{results_csv}; run a sweep first to populate the cache, '
            f'or pass --experiment-dir pointing at an existing one.'
        )
        raise RuntimeError(msg)
    df = pl.read_csv(results_csv)
    if 'id' not in df.columns or 'backtest_mean_kelly_pct' not in df.columns:
        msg = (
            f'{results_csv} is missing required columns `id` and / or '
            f'`backtest_mean_kelly_pct`; cannot resolve kelly for '
            f'decoder_id={decoder_id}.'
        )
        raise RuntimeError(msg)
    row = df.filter(pl.col('id') == decoder_id)
    if row.height == 0:
        msg = f'bts run --decoder-id {decoder_id}: id not found in {results_csv}'
        raise RuntimeError(msg)
    raw = row['backtest_mean_kelly_pct'][0]
    return Decimal(str(raw))
