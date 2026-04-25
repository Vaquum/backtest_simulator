"""`bts run` — run one backtest window for one decoder."""

# Thin wrapper over `backtest_simulator.cli._run_window.run_window_in_subprocess`
# that consumes one decoder + one window and either prints the human-
# readable summary (`--output-format text`) or emits the structured JSON
# report (`--output-format json`). `slippage_realised_bps` reflects the
# calibrated SlippageModel applied to every taker fill in this run;
# `book_gap_max_seconds` and `market_impact_realised_bps` are populated
# by their own wiring tasks once they land.
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
            'book_gap_max_seconds': None,
            # Signed mean (directional). Operator should NOT cite
            # this as cost — see `slippage_realised_adverse_bps`.
            'slippage_realised_bps': result.get('slippage_realised_bps'),
            # Cost metric: mean(|bps|). Survives round-trip cancellation.
            'slippage_realised_adverse_bps': result.get(
                'slippage_realised_adverse_bps',
            ),
            'slippage_realised_buy_bps': result.get('slippage_realised_buy_bps'),
            'slippage_realised_sell_bps': result.get('slippage_realised_sell_bps'),
            'slippage_n_samples': result.get('slippage_n_samples', 0),
            'slippage_n_excluded': result.get('slippage_n_excluded', 0),
            'market_impact_realised_bps': None,
        }
        sys.stdout.write(json.dumps(report) + '\n')
        return 0
    trades = [Trade(*row) for row in trades_raw]
    declared_stops = {k: Decimal(str(v)) for k, v in stops_raw.items()}
    print_run(display_id, window_start.date().isoformat(), trades, declared_stops)
    return 0


def _resolve_decoder(args: argparse.Namespace) -> tuple[int, Decimal, Path, int]:
    """Pick one decoder from the cache or by explicit id."""
    if args.decoder_id is not None:
        exp_dir = args.experiment_dir or EXP_DIR
        return args.decoder_id, _kelly_for_decoder(exp_dir, args.decoder_id), exp_dir, args.decoder_id
    if args.input_from_file is None:
        ensure_trained(args.n_decoders)
    picks = pick_decoders(
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
