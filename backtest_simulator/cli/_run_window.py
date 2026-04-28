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
import os
import shutil
import subprocess
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from backtest_simulator.feed.protocol import VenueFeed
    from backtest_simulator.honesty.atr import AtrSanityGate
    from backtest_simulator.honesty.maker_fill import MakerFillModel
    from backtest_simulator.honesty.slippage import SlippageModel

    AtrProvider = Callable[[str, datetime], Decimal | None]

from backtest_simulator.cli._pipeline import (
    SYMBOL,
    WORK_DIR,
    op_sfd_pythonpath,
    seed_price_at,
)

# Tick cadence baked into every per-window manifest. Importers
# (e.g. `_signals_builder.py` via `sweep.py`) pin to this constant
# so SignalsTable build replays at the SAME boundaries Nexus's
# PredictLoop fires at runtime — the byte-equivalence promise
# under "strategy tested, strategy deployed".
RUN_WINDOW_INTERVAL_SECONDS: int = 3600


# Shape of the dict returned by `run_window_in_process` and
# `run_window_in_subprocess`. The window result crosses a JSON
# subprocess boundary, so all "decimal" / "datetime" values are
# strings here — sweep.py re-narrows to Decimal / datetime at use.
# `total=True` so each consumer site reads through the typed shape;
# `runtime_predictions` is the only optional-via-default field.
class WindowResult(TypedDict):
    trades: list[list[str]]
    declared_stops: dict[str, str]
    orders: int
    slippage_realised_bps: str | None
    slippage_realised_cost_bps: str | None
    slippage_realised_buy_bps: str | None
    slippage_realised_sell_bps: str | None
    slippage_predicted_cost_bps: str | None
    slippage_predict_vs_realised_gap_bps: str | None
    slippage_n_samples: int
    slippage_n_excluded: int
    slippage_n_uncalibrated_predict: int
    slippage_n_predicted_samples: int
    n_limit_orders_submitted: int
    n_limit_filled_full: int
    n_limit_filled_partial: int
    n_limit_filled_zero: int
    n_limit_marketable_taker: int
    n_passive_limits: int
    maker_fill_efficiency_p50: str | None
    maker_fill_efficiency_mean: str | None
    market_impact_realised_bps: str | None
    market_impact_n_samples: int
    market_impact_n_flagged: int
    market_impact_n_uncalibrated: int
    market_impact_n_rejected: int
    atr_k: str
    atr_window_seconds: int
    n_atr_rejected: int
    n_atr_uncalibrated: int
    event_spine_jsonl: str
    event_spine_n_events: int
    book_gap_max_seconds: float
    book_gap_n_observed: int
    book_gap_p95_seconds: float
    runtime_predictions: list[dict[str, object]]


def capture_runtime_prediction(
    *, wired: object, signal: object, sink: list[dict[str, object]],
) -> None:
    """Append `(sensor_id, timestamp, pred)` to `sink` if `signal` carries a `_preds` int.

    Auditor (post-v2.0.2) "make it real": this is the per-call
    capture predicate inside `run_window_in_process`'s wrapped
    `produce_signal`. Extracted to module level so the
    MappingProxyType / non-int / missing-attr edge cases are
    testable without spinning up a real `BacktestLauncher`.

    Codex (post-auditor-3 P1): real Nexus emits `Signal.values` as
    a `types.MappingProxyType` (read-only dict view). The earlier
    `isinstance(values_obj, dict)` check silently dropped every
    real-runtime prediction — sweep parity printed "no comparisons
    made" forever. Accept `collections.abc.Mapping` instead.

    Silently no-ops on:
      - `signal.values` is not a `Mapping`
      - `_preds` key is absent
      - `_preds` value is not an `int`
      - `signal.timestamp` is not a `datetime`

    Each silent skip is a real "missing data" signal — the wrapper
    cannot synthesize a valid (t, pred) pair from the absence. The
    sweep counts captures via `len(sink)` per window.
    """
    from collections.abc import Mapping
    from typing import cast
    values_obj = getattr(signal, 'values', None)
    if not isinstance(values_obj, Mapping):
        return
    values = cast('Mapping[str, object]', values_obj)
    pred_val = values.get('_preds')
    if not isinstance(pred_val, int):
        return
    ts_obj = getattr(signal, 'timestamp', None)
    if not isinstance(ts_obj, datetime):
        return
    sink.append({
        'sensor_id': str(getattr(wired, 'sensor_id', '')),
        'timestamp': ts_obj.isoformat(),
        'pred': pred_val,
    })


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
    atr_k: str = '0.5',
    atr_window_seconds: int = 900,
) -> WindowResult:
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
            interval_seconds=RUN_WINDOW_INTERVAL_SECONDS,
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
    # Slice #17 Task 29 — ATR sanity gate, R-denominator floor.
    # Defaults: 900s window (14-period ATR convention) + k=0.5
    # (stop must be ≥ half a local ATR). The strategy template's
    # default stop_bps=50 on $70k BTC ≈ $350 distance, well above
    # floor=0.5*typical-1-min-BTC-ATR; gate fires only on
    # gameably tight stops. `--atr-k 0` disables.
    atr_gate, atr_provider = _build_atr_gate_and_provider(
        feed, k=Decimal(atr_k), window_seconds=atr_window_seconds,
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
        atr_gate=atr_gate,
        atr_provider=atr_provider,
    )
    # Auditor (post-v2.0.2) "make it real": capture per-tick runtime
    # predictions so the sweep can assert SignalsTable parity AGAINST
    # the deployed strategy's actual decisions. The hook wraps Nexus's
    # `produce_signal` (the only path Sensor.predict reaches at
    # runtime) and records (timestamp, _preds) per call. The
    # `signal.timestamp` is `datetime.now(UTC)` evaluated INSIDE
    # `accelerated_clock`, which freezegun pins to the simulated tick
    # boundary — same instant the sweep used to build SignalsTable.
    # CPython list.append is atomic under GIL, safe from PredictLoop's
    # threading.Timer.
    runtime_predictions: list[dict[str, object]] = []
    import nexus.strategy.predict_loop as _predict_loop_mod
    # Nexus's `produce_signal(wired, market_data) -> Signal` is
    # not in `__all__`, so `predict_loop_mod.produce_signal` direct
    # reads trigger `reportPrivateImportUsage`. Use `getattr` /
    # `setattr` for the read + override seam — the contract IS
    # public (Nexus's PredictLoop calls it), just not advertised
    # via `__all__`.
    _real_produce_signal = getattr(_predict_loop_mod, 'produce_signal')
    def _capturing_produce_signal(
        wired: object, market_data: object,
    ) -> object:
        signal = _real_produce_signal(wired, market_data)
        capture_runtime_prediction(
            wired=wired, signal=signal, sink=runtime_predictions,
        )
        return signal

    setattr(_predict_loop_mod, 'produce_signal', _capturing_produce_signal)
    try:
        launcher.run_window(window_start, window_end)
    finally:
        setattr(_predict_loop_mod, 'produce_signal', _real_produce_signal)
    # Slice #17 Task 18: dump the run's EventSpine to JSONL for
    # ledger-parity comparison. Always-dump (codex round 1 #4): the
    # operator gets a comparable artifact regardless of whether
    # `--check-parity-vs` is set. Cost is small (one sqlite scan
    # post-run); operator can chain into other parity tooling.
    from backtest_simulator.honesty.ledger_parity import (
        dump_event_spine_to_jsonl,
    )
    spine_jsonl = work / 'event_spine.jsonl'
    spine_n_events = dump_event_spine_to_jsonl(
        sqlite_path=work / 'event_spine.sqlite',
        jsonl_path=spine_jsonl,
    )
    book_gap = adapter.book_gap_snapshot()
    account = adapter.history('bts-sweep')
    trade_tuples = [
        (
            t.client_order_id, t.side.name, str(t.qty), str(t.price),
            str(t.fee), t.fee_asset, t.timestamp.isoformat(),
        )
        for t in account.trades
    ]
    # `_declared_stops` is the launcher's per-trade honesty record
    # (coid -> declared_stop_price). The underscore is internal-state
    # naming; we read it here from the same package's launcher and
    # the slot has no public accessor (the launcher is the only
    # writer; sweep parent rehydrates via this dict for `R` math).
    _stops: dict[str, Decimal] = getattr(launcher, '_declared_stops')
    declared_stops = {str(k): str(v) for k, v in _stops.items()}

    def _emit(value: Decimal | None) -> str | None:
        return None if value is None else str(value)

    result: WindowResult = {
        'trades': [list(row) for row in trade_tuples],
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
        # Slice #17 Task 29 — ATR R-denominator gameability gate.
        # `n_atr_rejected` counts ENTER+BUY denied for stops
        # tighter than `k * ATR(window)` from entry. Always 0
        # for strategies whose stops are sane vs local
        # volatility; non-zero when the strategy emits 1-bp
        # stops or similar gameability vectors.
        # `n_atr_uncalibrated` counts denials where the ATR
        # provider returned None (empty pre-decision tape) —
        # the operator's "warmup" signal.
        # Auditor: surface the floor that PRODUCED these counts
        # so two sweeps with different `--atr-k` /
        # `--atr-window-seconds` don't show different rejection
        # counts without showing what threshold each one used.
        # The bts artifact must be self-describing for later
        # comparison.
        'atr_k': str(atr_k),
        'atr_window_seconds': int(atr_window_seconds),
        'n_atr_rejected': launcher.n_atr_rejected,
        'n_atr_uncalibrated': launcher.n_atr_uncalibrated,
        'event_spine_jsonl': str(spine_jsonl),
        'event_spine_n_events': spine_n_events,
        # Slice #17 Task 11 — book-gap instrumentation. The venue's
        # BookGapInstrument records the time between the last sub-
        # stop tape tick and the trigger tick on every STOP/TP fill.
        # n_observed=0 means no stop fired in this run (e.g.
        # long_on_signal emits MARKET entries / exits, so default
        # bts runs show zero). Non-zero values surface a separate
        # line on `bts run` text mode and fields on
        # `--output-format json`.
        'book_gap_max_seconds': float(book_gap.max_stop_cross_to_trade_seconds),
        'book_gap_n_observed': int(book_gap.n_stops_observed),
        'book_gap_p95_seconds': float(book_gap.p95_stop_cross_to_trade_seconds),
        # Auditor (post-v2.0.2) "make it real": list of per-tick
        # runtime predictions captured from the wrapped
        # `produce_signal`. Each entry: `{sensor_id, timestamp,
        # pred}`. Sweep parent compares this against the
        # SignalsTable for the matching decoder; mismatch raises
        # `ParityViolation`. Empty when no PredictLoop tick fired
        # in the window (e.g. window shorter than interval_seconds).
        'runtime_predictions': runtime_predictions,
    }
    return result


def _build_atr_gate_and_provider(
    feed: VenueFeed, *, k: Decimal, window_seconds: int,
) -> tuple[AtrSanityGate, AtrProvider]:
    """Construct the ATR sanity gate + per-submit provider.

    Provider closes over `feed`, fetches `[t - window_seconds,
    t)` of strict-causal pre-decision tape, and calls
    `compute_atr_from_tape(period_seconds=60)` — Wilder's true-
    range ATR (per-bucket H-L plus gap-vs-prev-close arms,
    averaged across buckets). With the defaults (window=900s,
    period=60s) the result is 15 buckets — classic 14-period
    ATR shape. Slice #17 Task 29.
    """
    import polars as pl

    from backtest_simulator.honesty.atr import (
        AtrSanityGate,
        compute_atr_from_tape,
    )
    gate = AtrSanityGate(atr_window_seconds=window_seconds, k=k)

    def atr_provider(symbol: str, t: datetime) -> Decimal | None:
        from datetime import timedelta
        raw = feed._get_trades_for_venue(
            symbol, t - timedelta(seconds=window_seconds), t,
            venue_lookahead_seconds=0,
        )
        pre = raw.filter(pl.col('time') < t)
        return compute_atr_from_tape(
            trades_pre_decision=pre, period_seconds=60,
        )
    return gate, atr_provider


def _calibrate_slippage(
    feed: VenueFeed,
    window_start: datetime,
) -> SlippageModel | None:
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
    feed: VenueFeed,
    window_start: datetime,
) -> MakerFillModel:
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
    atr_k: str = '0.5',
    atr_window_seconds: int = 900,
) -> WindowResult:
    """Run one window in a fresh Python interpreter for state isolation."""
    payload = json.dumps({
        'perm_id': perm_id,
        'kelly_pct': str(kelly_pct),
        'window_start': window_start.isoformat(),
        'window_end': window_end.isoformat(),
        'experiment_dir': str(experiment_dir),
        'maker_preference': bool(maker_preference),
        'strict_impact': bool(strict_impact),
        'atr_k': str(atr_k),
        'atr_window_seconds': int(atr_window_seconds),
    })
    # Propagate the bts op-sfd cache dir to the child's
    # PYTHONPATH so Limen's `Trainer.train()` (invoked inside
    # the child via `BacktestLauncher`) can resolve
    # `metadata['sfd_module']` (= `_bts_op_<sha16>`) via plain
    # `importlib.import_module(...)`. Codex round-2 P0: without
    # this, the child's sys.path lacks the snapshot dir and the
    # reimport raises `ModuleNotFoundError`.
    env = os.environ.copy()
    existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = os.pathsep.join(
        p for p in (op_sfd_pythonpath(), existing) if p
    )
    proc = subprocess.run(
        [sys.executable, '-m', 'backtest_simulator.cli._run_window'],
        input=payload,
        capture_output=True, text=True, check=False, timeout=120,
        env=env,
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
    # The child writes a `WindowResult`-shaped dict via `json.dumps`;
    # `json.loads` returns `Any`, so a typed cast at this boundary
    # gives the parent (`sweep.py`) typed access without piling up
    # `reportUnknown*` cascades on every key read.
    return cast('WindowResult', json.loads(last))


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
        atr_k=str(payload.get('atr_k', '0.5')),
        atr_window_seconds=int(payload.get('atr_window_seconds', 900)),
    )
    print(json.dumps(result), flush=True)
    return 0


if __name__ == '__main__':  # pragma: no cover - subprocess child entry
    sys.exit(_child_main())
