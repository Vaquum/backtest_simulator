"""Single-window backtest run + subprocess child entry point."""

# `run_window_in_process` builds the manifest, wires the launcher, runs
# the window, and returns picklable trade tuples. `run_window_in_subprocess`
# calls this module as `python -m backtest_simulator.cli._run_window` so
# every window gets a fresh interpreter (necessary on macOS Python 3.12,
# where `multiprocessing.fork` deadlocks inside libdispatch after asyncio
# state has been touched).
#
# The `__main__` entry point reads a JSON payload from stdin, runs the
# window in-process, and writes the JSON result to stdout. Subprocess
# isolation makes between-run state bleed (closed-loop RuntimeError on
# the next submit) impossible — the process exits and a new one starts.
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from backtest_simulator.cli._pipeline import SYMBOL, WORK_DIR, seed_price_at


def _fresh_work_dir(suffix: str) -> Path:
    d = WORK_DIR / suffix
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def run_window_in_process(
    perm_id: int,
    kelly_pct: Decimal,
    window_start: datetime,
    window_end: datetime,
    experiment_dir: Path,
    *,
    maker_preference: bool = False,
    strict_impact: bool = False,
) -> dict[str, object]:
    """In-process single-window run — picklable result for cross-process use."""
    from praxis.launcher import InstanceConfig
    from praxis.trading_config import TradingConfig

    from backtest_simulator.feed.clickhouse import ClickHouseConfig, ClickHouseFeed
    from backtest_simulator.launcher import BacktestLauncher
    from backtest_simulator.pipeline.manifest_builder import (
        AccountSpec,
        ManifestBuilder,
        SensorBinding,
        StrategyParamsSpec,
    )
    from backtest_simulator.venue.fees import FeeSchedule
    from backtest_simulator.venue.filters import BinanceSpotFilters
    from backtest_simulator.venue.simulated import SimulatedVenueAdapter

    seed_price = seed_price_at(window_start)
    suffix = f'{perm_id}_{window_start.date().isoformat()}'
    work = _fresh_work_dir(suffix)
    manifest_dir = work / 'manifest'
    state_dir = work / 'state'
    state_dir.mkdir()

    built = ManifestBuilder(output_dir=manifest_dir).build(
        account=AccountSpec(
            account_id='bts-sweep',
            allocated_capital=Decimal('100000'),
            capital_pool=Decimal('100000'),
        ),
        strategy_id='long_on_signal',
        sensor=SensorBinding(
            experiment_dir=experiment_dir,
            permutation_ids=(perm_id,),
            interval_seconds=3600,
        ),
        strategy_params=StrategyParamsSpec(
            symbol=SYMBOL,
            capital=Decimal('100000'),
            kelly_pct=kelly_pct,
            estimated_price=seed_price,
            stop_bps=Decimal('50'),
            maker_preference=maker_preference,
        ),
    )
    feed = ClickHouseFeed(config=ClickHouseConfig.from_env(), symbol=SYMBOL)
    # Calibrate the slippage model from a pre-window trade slice
    # (strict-causal: end <= window_start, no peek into the run's
    # own fills). 30 minutes is enough density at BTCUSDT scale to
    # populate every (side, qty-bucket) cell; the slice is bounded
    # so the calibration cost is visible in the per-window timing.
    # When the slice is empty (start-of-tape or sparse-volume
    # symbol) `SlippageModel.calibrate` raises and we fall through
    # to slippage_model=None — the operator sees the JSON report's
    # `slippage_realised_bps` come back null and knows the
    # calibration window needs to widen.
    slippage_model = _calibrate_slippage(feed, window_start)
    # MakerFillModel is constructed only when the strategy will
    # use LIMIT orders (`maker_preference=True`). Without LIMIT
    # orders the model would be dead weight. The model's lookback
    # window seeds queue position from the same pre-window tape
    # the slippage model uses.
    maker_fill_model = (
        _calibrate_maker_fill(feed, window_start)
        if maker_preference else None
    )
    # MarketImpactModel: per-submit STRICT-CAUSAL calibration
    # at the venue. Every order submit fetches `[submit_time -
    # bucket_minutes, submit_time)` of pre-submit tape, builds
    # a single-bucket model, and evaluates the order's qty
    # against that bucket. Always-on measurement (the
    # operator gets size-vs-volume telemetry on every sweep).
    # When `strict_impact=True`, the venue REJECTS orders the
    # model flags as exceeding `threshold_fraction` of bucket
    # volume — the auditor's "pre-fill gate" semantic. Codex
    # round 2 P1 caught the prior shape: pre-calibration over
    # `[window_start - 30m, window_end]` was hindsight-
    # informed (each bucket included trades AFTER the matching
    # submit). The per-submit fetch eliminates that lookahead.
    adapter = SimulatedVenueAdapter(
        feed=feed,
        filters=BinanceSpotFilters.binance_spot(SYMBOL),
        fees=FeeSchedule(),
        trade_window_seconds=60,
        slippage_model=slippage_model,
        maker_fill_model=maker_fill_model,
        market_impact_bucket_minutes=1,
        market_impact_threshold_fraction=Decimal('0.1'),
        strict_impact_policy=strict_impact,
    )
    tc = TradingConfig(
        epoch_id=1, venue_rest_url='http://sim', venue_ws_url='ws://sim',
        account_credentials={'bts-sweep': ('k', 's')}, shutdown_timeout=5.0,
    )
    launcher = BacktestLauncher(
        trading_config=tc,
        instances=[InstanceConfig(
            account_id='bts-sweep',
            manifest_path=built.manifest_path,
            strategies_base_path=built.strategies_base_path,
            state_dir=state_dir,
        )],
        venue_adapter=adapter,
        db_path=work / 'event_spine.sqlite',
    )
    launcher.run_window(window_start, window_end)
    account = adapter.history('bts-sweep')
    trade_tuples = [
        (
            t.client_order_id, t.side.name, str(t.qty), str(t.price),
            str(t.fee), t.fee_asset, t.timestamp.isoformat(),
        )
        for t in account.trades
    ]
    declared_stops = {k: str(v) for k, v in launcher._declared_stops.items()}

    def _emit(value: Decimal | None) -> str | None:
        return None if value is None else str(value)

    return {
        'trades': trade_tuples,
        'declared_stops': declared_stops,
        'orders': len(account.orders),
        # Signed mean — directional, NOT a cost metric on its own.
        'slippage_realised_bps': _emit(
            adapter.slippage_realised_aggregate_bps,
        ),
        # Cost metric: side-normalized mean. BUY contributes +bps,
        # SELL contributes -bps. Positive = paid spread on average;
        # negative = price improvement on average. Replaces the
        # earlier mean(|bps|), which counted favorable fills as cost.
        'slippage_realised_cost_bps': _emit(
            adapter.slippage_realised_cost_bps,
        ),
        # Per-side aggregates so the operator can diagnose
        # asymmetric paying.
        'slippage_realised_buy_bps': _emit(
            adapter.slippage_realised_buy_bps,
        ),
        'slippage_realised_sell_bps': _emit(
            adapter.slippage_realised_sell_bps,
        ),
        # Calibration-loop telemetry: predicted cost from the
        # model's `apply()` and the realised - predicted gap.
        # The gap is the calibration error signal; large gaps
        # tell the operator to recalibrate.
        'slippage_predicted_cost_bps': _emit(
            adapter.slippage_predicted_cost_bps,
        ),
        'slippage_predict_vs_realised_gap_bps': _emit(
            adapter.slippage_predict_vs_realised_gap_bps,
        ),
        'slippage_n_samples': adapter.slippage_realised_n_samples,
        # Distinct from "measured zero" — fills excluded because
        # the rolling-mid window had no trades. Operator sees this
        # to know calibration coverage is incomplete.
        'slippage_n_excluded': adapter.slippage_realised_n_excluded,
        # Distinct from `n_excluded`: predict failures (qty
        # outside any calibrated bucket). A high count means
        # the bucket thresholds are wrong for this run's qty
        # distribution.
        'slippage_n_uncalibrated_predict':
            adapter.slippage_n_uncalibrated_predict,
        # The denominator for the gap aggregate: realised
        # samples MINUS uncalibrated predicts. A low value here
        # against a high `n_samples` means the gap is averaged
        # over a thin slice — the calibration coverage is poor.
        'slippage_n_predicted_samples':
            adapter.slippage_n_predicted_samples,
        # Maker-fill telemetry. Zero on every count when the
        # strategy emits MARKET orders only (default). Non-zero
        # when `bts sweep --maker` runs the LIMIT-on-signal
        # variant.
        'n_limit_orders_submitted': adapter.n_limit_orders_submitted,
        'n_limit_filled_full': adapter.n_limit_filled_full,
        'n_limit_filled_partial': adapter.n_limit_filled_partial,
        'n_limit_filled_zero': adapter.n_limit_filled_zero,
        'n_limit_marketable_taker': adapter.n_limit_marketable_taker,
        'n_passive_limits': adapter.n_passive_limits,
        'maker_fill_efficiency_p50': _emit(
            adapter.maker_fill_efficiency_p50,
        ),
        'maker_fill_efficiency_mean': _emit(
            adapter.maker_fill_efficiency_mean,
        ),
        # MarketImpactModel telemetry. `realised_bps` is the mean
        # estimated impact across every submit (BUY + SELL) that
        # hit a calibrated rolling slice. `n_samples` /
        # `n_flagged` / `n_uncalibrated` count those submits,
        # the subset flagged as oversized, and the calibration
        # gaps. `n_rejected` is the strict-impact gate's
        # rejection count — scoped to BUY (entry leg, long-only
        # template) only, so it is always ≤ `n_flagged` and is
        # 0 unless `--strict-impact` is set.
        'market_impact_realised_bps': _emit(
            adapter.market_impact_realised_bps,
        ),
        'market_impact_n_samples': adapter.market_impact_n_samples,
        'market_impact_n_flagged': adapter.market_impact_n_flagged,
        'market_impact_n_uncalibrated':
            adapter.market_impact_n_uncalibrated,
        'market_impact_n_rejected': adapter.market_impact_n_rejected,
    }


def _calibrate_slippage(
    feed: object,
    window_start: datetime,
) -> object | None:
    """Fit a SlippageModel on the 30 minutes of trades before `window_start`.

    Return contract (narrow on purpose):
      - SlippageModel: calibration succeeded — every per-fill
        measurement uses this model's `dt_seconds` and the
        adapter's aggregate properties surface real numbers.
      - None: the trade window was *empty* and only the empty
        window. Adapter falls back to "feature dark" for this
        run — the JSON `slippage_realised_*` fields read None,
        the bts-sweep line shows `slip off`, n_samples / n_excluded
        stay at 0.
      - Anything else (ClickHouse network failure, schema mismatch,
        SlippageModel.calibrate raising on degenerate data,
        LookAheadViolation against `frozen_now`) PROPAGATES as a
        run error. The operator must know if calibration cannot
        run; silently returning None on a tunnel-down would let
        the rest of the run continue with no slippage reporting
        AND no warning, which is exactly the "feature looked
        active but actually wasn't" failure mode the audit
        rejected.
    """
    from datetime import timedelta

    from backtest_simulator.honesty.slippage import SlippageModel
    calibration_end = window_start
    calibration_start = window_start - timedelta(minutes=30)
    # Calibration is strict-causal (end <= window_start <
    # frozen_now) so any LookAheadViolation here is a real bug —
    # let it propagate. ClickHouse connection failures propagate
    # too; the operator must know if the tunnel is down before
    # they trust any sweep output. The only honest fallthrough is
    # the empty-window case, which we check explicitly.
    trades = feed.get_trades(SYMBOL, calibration_start, calibration_end)
    if trades.is_empty():
        return None
    # Schema bridge: the venue-feed contract uses `time, qty`
    # (post-`ClickHouseFeed` rename), but `SlippageModel.calibrate`
    # expects the raw ClickHouse-table schema (`datetime, quantity`).
    # Rename back here so the model keeps its stable contract.
    # Surfaced by `bts sweep` ColumnNotFoundError when the wiring
    # first ran end-to-end.
    trades = trades.rename({'time': 'datetime', 'qty': 'quantity'})
    return SlippageModel.calibrate(
        trades=trades,
        side_buckets=(
            Decimal('0.001'), Decimal('0.01'),
            Decimal('0.1'), Decimal('1.0'),
        ),
        dt_seconds=10,
    )


def _calibrate_maker_fill(
    feed: object,
    window_start: datetime,
) -> object:
    """Build a MakerFillModel from the 30 minutes preceding `window_start`.

    Same schema bridge as slippage. Empty pre-window → RAISE
    (was: silently return None and let the venue fall back to
    the legacy first-crossing/full-fill shortcut, while the CLI
    still advertised maker-engine realism. The operator must
    see uncalibrated maker mode as a loud failure, not as a
    legacy-path silent regression — codex round 3 P2.)

    Other failures propagate as well (tunnel-down, schema
    mismatch). The MakerFillModel.calibrate floor itself raises
    on edge cases (empty tape passed in, non-positive lookback).
    `bts run --maker` / `bts sweep --maker` now fail-loud when
    the calibration window has no trades; the operator widens
    the window or moves to a denser symbol.
    """
    from datetime import timedelta

    from backtest_simulator.honesty.maker_fill import MakerFillModel
    pre = feed.get_trades(
        SYMBOL,
        window_start - timedelta(minutes=30),
        window_start,
    )
    if pre.is_empty():
        msg = (
            f'_calibrate_maker_fill: empty pre-window tape for '
            f'{SYMBOL} [{window_start - timedelta(minutes=30)}, '
            f'{window_start}). MakerFillModel cannot calibrate '
            f'queue position from zero trades; the venue would '
            f'silently fall back to the legacy LIMIT path while '
            f'`--maker` still advertises queue/partial-fill '
            f'realism. Widen the calibration window or run '
            f'`--maker` against a denser-volume window.'
        )
        raise RuntimeError(msg)
    pre = pre.rename({'time': 'datetime', 'qty': 'quantity'})
    return MakerFillModel.calibrate(trades=pre, lookback_minutes=30)


# Market-impact calibration moved into the venue adapter
# (per-submit, strict-causal). The previous run-window
# pre-calibration ran here; codex round 2 P1 caught the
# lookahead and the adapter now does its own bounded fetch
# at each submit. See SimulatedVenueAdapter._record_market_impact_pre_fill.


def run_window_in_subprocess(
    perm_id: int,
    kelly_pct: Decimal,
    window_start: datetime,
    window_end: datetime,
    experiment_dir: Path,
    *,
    maker_preference: bool = False,
    strict_impact: bool = False,
) -> dict[str, object]:
    """Run one window in a fresh Python interpreter for state isolation."""
    payload = json.dumps({
        'perm_id': perm_id,
        'kelly_pct': str(kelly_pct),
        'window_start': window_start.isoformat(),
        'window_end': window_end.isoformat(),
        'experiment_dir': str(experiment_dir),
        'maker_preference': bool(maker_preference),
        'strict_impact': bool(strict_impact),
    })
    proc = subprocess.run(
        [sys.executable, '-m', 'backtest_simulator.cli._run_window'],
        input=payload,
        capture_output=True, text=True, check=False, timeout=120,
    )
    if proc.returncode != 0:
        msg = f'child exit={proc.returncode}; stderr tail: {proc.stderr[-400:]}'
        raise RuntimeError(msg)
    last: str | None = None
    for line in proc.stdout.splitlines():
        s = line.strip()
        if s.startswith('{') and s.endswith('}'):
            last = s
    if last is None:
        msg = f'child produced no JSON result; stdout tail: {proc.stdout[-400:]}'
        raise RuntimeError(msg)
    return json.loads(last)


def _child_main() -> int:
    """Child-process entry: read JSON payload from stdin, run, write JSON result."""
    payload = json.loads(sys.stdin.read())
    result = run_window_in_process(
        int(payload['perm_id']),
        Decimal(payload['kelly_pct']),
        datetime.fromisoformat(payload['window_start']),
        datetime.fromisoformat(payload['window_end']),
        Path(payload['experiment_dir']),
        maker_preference=bool(payload.get('maker_preference', False)),
        strict_impact=bool(payload.get('strict_impact', False)),
    )
    print(json.dumps(result), flush=True)
    return 0


if __name__ == '__main__':  # pragma: no cover - subprocess child entry
    sys.exit(_child_main())
