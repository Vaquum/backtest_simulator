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
import time as _bts_time
from datetime import datetime, timedelta
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

if TYPE_CHECKING:
    from backtest_simulator.feed.protocol import VenueFeed
    from backtest_simulator.honesty.maker_fill import MakerFillModel
    from backtest_simulator.honesty.slippage import SlippageModel

from backtest_simulator.cli._pipeline import (
    SYMBOL,
    WORK_DIR,
    op_sfd_pythonpath,
    seed_price_at,
)


def _effective_kelly_pct_for_allocation(
    kelly_pct: Decimal,
    max_allocation_per_trade_pct: Decimal | None,
) -> Decimal:
    """Clamp Kelly sizing so CAPITAL sees max, not reject.

    The strategy sizes raw notional as `capital * kelly_pct / 100`.
    bts reserves a small notional buffer before sending to Nexus, so
    cap the baked Kelly at `max / (1 + buffer)` rather than at `max`
    directly. Example: max=0.4 with a 7% buffer -> raw Kelly cap is
    37.383%, producing a reserved notional of 40%.
    """
    if max_allocation_per_trade_pct is None:
        return kelly_pct
    if max_allocation_per_trade_pct <= 0:
        msg = (
            f'max_allocation_per_trade_pct must be positive, '
            f'got {max_allocation_per_trade_pct}'
        )
        raise ValueError(msg)
    from backtest_simulator.launcher.action_submitter import (
        NOTIONAL_RESERVATION_BUFFER,
    )
    max_kelly_pct = (
        max_allocation_per_trade_pct
        / (Decimal('1') + NOTIONAL_RESERVATION_BUFFER)
        * Decimal('100')
    ).quantize(
        Decimal('0.000001'), rounding=ROUND_DOWN,
    )
    return min(kelly_pct, max_kelly_pct)


def read_kline_size_from_experiment_dir(experiment_dir: Path) -> int:
    """Read `kline_size` from `experiment_dir/metadata.json`'s SFD manifest.

    Mirrors `BacktestLauncher._kline_size_from_experiment_dir` so the
    sensor's `interval_seconds` (the runtime PredictLoop Timer cadence)
    and the sweep's `expected_ticks` parity grid both anchor to the
    SAME source of truth — the SFD's `data_source_config.params
    ['kline_size']`. Anything else is drift between "what the bundle
    declared" and "what bts actually ticks at", which is the bug
    operator hit on the LogReg-Placeholder bundle (kline_size=7200,
    bts hardcoded to 3600).
    """
    metadata_path = experiment_dir / 'metadata.json'
    if not metadata_path.is_file():
        msg = (
            f'read_kline_size_from_experiment_dir: metadata.json not '
            f'found at {metadata_path} — cannot derive kline_size for '
            f'the SensorBinding interval. Limen writes this file when '
            f'the experiment dir is created; missing it means the dir '
            f'was created out-of-band or got truncated.'
        )
        raise FileNotFoundError(msg)
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    sfd_module_name = metadata['sfd_module']
    import importlib  # local import — cli hot path is import-light
    sfd = importlib.import_module(sfd_module_name)
    limen_manifest = sfd.manifest()
    return int(limen_manifest.data_source_config.params['kline_size'])


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
    event_spine_jsonl: str
    event_spine_n_events: int
    n_intents: int
    n_submitted: int
    n_fills: int
    n_pending: int
    n_rejects: int
    book_gap_max_seconds: float
    book_gap_n_observed: int
    book_gap_p95_seconds: float
    runtime_predictions: list[dict[str, object]]


def capture_runtime_prediction(
    *, wired: object, signal: object, sink: list[dict[str, object]],
    window_start: datetime | None = None,
    window_end: datetime | None = None,
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

    `window_start` / `window_end`: optional bounds (UTC). When given,
    signals whose `timestamp` falls outside `[window_start, window_end]`
    are dropped. This filters the launcher's grace-drain stragglers
    that fire AFTER `accelerated_clock` releases and so carry wall-
    clock timestamps from the present moment, not the simulated tick
    instant. Without this filter, a sweep that ran at 18:29 wall
    today would record a "runtime tick at 2026-04-29T18:29:34Z" for
    a 2026-04-15 simulated window, breaking parity.

    Silently no-ops on:
      - `signal.values` is not a `Mapping`
      - `_preds` key is absent
      - `_preds` value is not an `int`
      - `signal.timestamp` is not a `datetime`
      - `signal.timestamp` falls outside `[window_start, window_end]`
        (when those bounds are supplied)

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
    if window_start is not None and ts_obj < window_start:
        return
    if window_end is not None and ts_obj > window_end:
        return
    # `_probs` is the raw decoder probability emitted alongside the
    # binary `_preds` label by the LogReg sensor (see strategy
    # template's `signal.values.get('_probs')`). Carrying it through
    # capture so per-window CSV can record the entry-tick prob.
    prob_val = values.get('_probs')
    sink.append({
        'sensor_id': str(getattr(wired, 'sensor_id', '')),
        'timestamp': ts_obj.isoformat(),
        'pred': pred_val,
        'prob': float(prob_val) if isinstance(prob_val, (int, float)) else None,
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
    trades_parquet_path: str | None = None,
    max_allocation_per_trade_pct: Decimal | None = None,
    predict_lookback: int | None = None,
    display_id: int | None = None,
) -> WindowResult:
    """In-process single-window run — picklable result for cross-process use."""
    if predict_lookback is not None and predict_lookback < 1:
        msg = (
            f'predict_lookback must be >= 1, got {predict_lookback}. '
            f'Nexus produce_signal feeds tail(lookback) into predict; '
            f'lookback < 1 yields an empty x_test and IndexError downstream.'
        )
        raise ValueError(msg)
    from praxis.launcher import InstanceConfig
    from praxis.trading_config import TradingConfig

    from backtest_simulator.feed.clickhouse import (
        ClickHouseConfig,
        ClickHouseFeed,
        InMemoryTradesFeed,
    )
    import polars as _pl_mod
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

    # Subprocess runs silently — parent logs one consolidated
    # `perm X day Y - done (Ns)` line per window after `.result()`.
    # Internal sub-phase timing was useful when diagnosing the boot /
    # advance_clock split; with parallel batches the multiplexed
    # stderr was unreadable. Removed.
    _bts_t_start = _bts_time.perf_counter()
    def _bts_lap(label: str, start: float) -> float:  # no-op kept for shape
        return _bts_time.perf_counter()
    _bts_t = _bts_t_start
    seed_price = seed_price_at(window_start)
    # work_dir uniqueness must follow the human-facing pick identifier,
    # not the trainer permutation index. In the file-list filter mode
    # `perm_id` is hardcoded to 0 across every pick because each pick's
    # sub_dir was trained with `permutation_ids=[0]`; reusing it as the
    # work_dir suffix collides every pick onto `0_<date>` and the next
    # pick's `_fresh_work_dir` rmtree's the previous spine. `display_id`
    # is the unique decoder/file id (== `perm_id` in the experiment-dir
    # filter mode where they coincide).
    suffix_id = perm_id if display_id is None else display_id
    suffix = f'{suffix_id}_{window_start.date().isoformat()}'
    work = _fresh_work_dir(suffix)
    manifest_dir = work / 'manifest'
    state_dir = work / 'state'
    state_dir.mkdir()

    # Honor the experiment's declared kline_size as the runtime
    # PredictLoop tick cadence. With kline_size=7200 (2-hour klines)
    # in the bundle and a hardcoded 3600 here, every other Timer tick
    # would re-predict on stale data and the parity check downstream
    # would land on a wrong-cadence grid (codex P0: "different bundles
    # must run properly").
    interval_seconds = read_kline_size_from_experiment_dir(experiment_dir)
    effective_kelly_pct = _effective_kelly_pct_for_allocation(
        kelly_pct, max_allocation_per_trade_pct,
    )
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
            interval_seconds=interval_seconds,
        ),
        strategy_params=StrategyParamsSpec(
            symbol=SYMBOL,
            capital=Decimal('100000'),
            kelly_pct=effective_kelly_pct,
            estimated_price=seed_price,
            stop_bps=Decimal('50'),
            # Force-flatten cutoff = `window_end - kline_size`. The
            # strategy's last in-window signal arrives at this
            # timestamp; on or after it, the strategy emits SELL on
            # any held inventory and refuses new BUYs. Without this
            # cutoff, a BUY late in the window could open a position
            # that has no closing signal before subprocess shutdown,
            # leaving an orphaned position the per-day summary
            # mislabels as `trades 0`.
            force_flatten_after=window_end - timedelta(seconds=interval_seconds),
            maker_preference=maker_preference,
        ),
    )
    _bts_t = _bts_lap('ManifestBuilder.build', _bts_t)
    # If the parent prefetched the sweep's trade tape, use the parquet
    # via `InMemoryTradesFeed` — every trade query is now an in-memory
    # filter, not a ClickHouse round-trip. Falls back to a live
    # `ClickHouseFeed` only when the parquet wasn't passed (e.g. callers
    # outside the sweep flow like `bts run`).
    feed: ClickHouseFeed | InMemoryTradesFeed
    if trades_parquet_path is not None:
        feed = InMemoryTradesFeed(
            _pl_mod.read_parquet(trades_parquet_path),
            symbol=SYMBOL,
        )
    else:
        feed = ClickHouseFeed(
            config=ClickHouseConfig.from_env(), symbol=SYMBOL,
        )
    _bts_t = _bts_lap('ClickHouseFeed init', _bts_t)
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
    _bts_t = _bts_lap('_calibrate_slippage', _bts_t)
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
        # Per-submit fill walk capped at `min(kline_size, 600s)`. The
        # original design fetched a full kline of trade tape per submit
        # (4h on r0014), which over the SSH tunnel cost 3-6s wall and
        # tripped `drain slow` on every order. Real market orders fill
        # in the first few trades — 10 minutes is ample tape to walk a
        # MARKET fill or expire a stuck LIMIT. The kline cap still
        # applies for sub-10m bundles. `window_end_clamp` continues to
        # block any peek past the window end.
        trade_window_seconds=min(interval_seconds, 600),
        window_end_clamp=window_end,
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
    _bts_t = _bts_lap('SimulatedVenueAdapter init', _bts_t)
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
        max_allocation_per_trade_pct=max_allocation_per_trade_pct,
        # Smart kline-aware clock tick: pass the bundle's kline_size
        # so `_advance_clock_until` jumps big between boundaries and
        # crosses each boundary with a 250ms real-time pause. Brings
        # the per-day idle `time.sleep` budget from ~7s to ~1.5s
        # while preserving the parity guarantee (each kline boundary
        # gets a single freezer tick + generous asyncio settle).
        # Earlier attempt to use `kline_size` as the UNIFORM tick
        # collapsed Timer fires (1 of 6 captured); the smart path
        # ticks 1s across the boundary instead of jumping past it.
        clock_tick_seconds=interval_seconds,
    )
    _bts_t = _bts_lap('BacktestLauncher init', _bts_t)
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
    _lookback = predict_lookback
    def _capturing_produce_signal(
        wired: object, market_data: object,
    ) -> object:
        if _lookback is None:
            signal = _real_produce_signal(wired, market_data)
        else:
            signal = _real_produce_signal(
                wired, market_data, lookback=_lookback,
            )
        capture_runtime_prediction(
            wired=wired, signal=signal, sink=runtime_predictions,
            window_start=window_start, window_end=window_end,
        )
        return signal

    setattr(_predict_loop_mod, 'produce_signal', _capturing_produce_signal)
    _bts_t = _bts_lap('predict-hook install', _bts_t)
    try:
        launcher.run_window(window_start, window_end)
    finally:
        setattr(_predict_loop_mod, 'produce_signal', _real_produce_signal)
    _bts_t = _bts_lap('launcher.run_window', _bts_t)
    # Slice #17 Task 18: dump the run's EventSpine to JSONL for
    # ledger-parity comparison. Always-dump (codex round 1 #4): the
    # operator gets a comparable artifact regardless of whether
    # `--check-parity-vs` is set. Cost is small (one sqlite scan
    # post-run); operator can chain into other parity tooling.
    from backtest_simulator.honesty.ledger_parity import (
        count_event_spine_events,
        dump_event_spine_to_jsonl,
    )
    spine_sqlite = work / 'event_spine.sqlite'
    spine_jsonl = work / 'event_spine.jsonl'
    spine_n_events = dump_event_spine_to_jsonl(
        sqlite_path=spine_sqlite,
        jsonl_path=spine_jsonl,
    )
    _bts_t = _bts_lap('dump_event_spine', _bts_t)
    # Per-window event-type counts feed the operator's five-question
    # scan: did my strategy decide to act (intents)? did the venue
    # accept (submitted)? did money move (fills)? what's still hanging
    # (pending)? what got blocked (rejects)? print_run renders these
    # as parenthetical extras after `trades N` so a trader can tell
    # a flat day from a fail-to-fill day at a glance.
    spine_counts = count_event_spine_events(sqlite_path=spine_sqlite)
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
        'event_spine_jsonl': str(spine_jsonl),
        'event_spine_n_events': spine_n_events,
        'n_intents': spine_counts['intents'],
        'n_submitted': spine_counts['submitted'],
        'n_fills': spine_counts['fills'],
        'n_pending': spine_counts['pending'],
        'n_rejects': spine_counts['rejects'],
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
    trades_parquet_path: str | None = None,
    max_allocation_per_trade_pct: Decimal | None = None,
    predict_lookback: int | None = None,
    display_id: int | None = None,
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
        'trades_parquet_path': trades_parquet_path,
        'max_allocation_per_trade_pct': (
            None if max_allocation_per_trade_pct is None
            else str(max_allocation_per_trade_pct)
        ),
        'predict_lookback': (
            None if predict_lookback is None else int(predict_lookback)
        ),
        'display_id': None if display_id is None else int(display_id),
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
    # Pin polars to a single Rust thread per subprocess. The Rust
    # thread pool's GIL acquire on shutdown races against Python
    # interpreter teardown; with N parallel subprocesses each running
    # their own polars pool, the panics cascade. One thread → no pool
    # → no race.
    env['POLARS_MAX_THREADS'] = '1'
    # `stderr=None` lets the child's per-phase timing prints reach the
    # operator's terminal in real time. Without this, `capture_output=True`
    # buffered stderr and the phase prints only surfaced on failure —
    # making "stuck for a minute or two" indistinguishable from "crashed".
    proc = subprocess.run(
        [sys.executable, '-m', 'backtest_simulator.cli._run_window'],
        input=payload,
        stdout=subprocess.PIPE, stderr=None,
        text=True, check=False, timeout=120,
        env=env,
    )
    if proc.returncode != 0:
        msg = f'child exit={proc.returncode}; stdout tail: {proc.stdout[-400:]}'
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
    """Child-process entry: read JSON payload from stdin, run, write JSON result.

    When `BTS_RUN_WINDOW_LOG_LEVEL` is set in the environment, the
    child configures stdlib logging at that level (typical values:
    `INFO`, `DEBUG`). The default leaves logging unconfigured so
    `_log.info(...)` calls in the strategy / launcher / Praxis /
    Nexus chain are silently dropped — that's the production sweep
    path where stderr is captured and only surfaced on failure.
    Set the env var when diagnosing per-window behaviour so the
    on_signal / on_outcome / FORCE FLATTEN log lines reach
    `subprocess.run`'s captured stderr (or, with stderr inherited,
    the operator's terminal).
    """
    import logging as _logging
    # Silence Praxis / Nexus / Limen INFO logs in this subprocess.
    # They go to stdout via structlog's PrintLogger; stdout is a PIPE
    # to the parent. Under parallel batches the per-subprocess INFO
    # spam triggers EPIPE on some writes (parent's reader thread
    # contention), surfacing as BrokenPipeError in `nexus-bts-sweep`.
    # Stdlib level (for `_log.info` calls in our own code) and
    # structlog level (for Praxis / Nexus emitters) both moved to
    # WARNING. The parent already silences these at sweep start; the
    # subprocess re-imports fresh, so we re-apply here.
    for _name in ('limen', 'praxis', 'nexus'):
        _logging.getLogger(_name).setLevel(_logging.WARNING)
    try:
        import structlog as _structlog
        _structlog.configure(
            wrapper_class=_structlog.make_filtering_bound_logger(_logging.WARNING),
        )
    except ImportError:
        pass
    _level_name = os.environ.get('BTS_RUN_WINDOW_LOG_LEVEL')
    if _level_name:
        # Copilot caught: `logging.basicConfig(level=...)` is
        # case-sensitive when given a string — `info` / `debug`
        # raise ValueError and crash the child. Normalise to
        # uppercase before handing to logging; reject anything
        # that is not one of the documented levels with a clear
        # error rather than letting a typo crash the subprocess.
        _level_upper = _level_name.strip().upper()
        _allowed = {'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'}
        if _level_upper not in _allowed:
            msg_level = (
                f'BTS_RUN_WINDOW_LOG_LEVEL={_level_name!r} is not a '
                f'recognised stdlib logging level; expected one of '
                f'{sorted(_allowed)} (case-insensitive).'
            )
            raise ValueError(msg_level)
        _logging.basicConfig(
            level=_level_upper,
            format='%(levelname)s %(name)s %(message)s',
            force=True,
        )
    # Hard rule: one HuggingFace pull per sweep, performed in the parent
    # before subprocesses spawn. The parent writes the fresh klines to
    # `~/.cache/backtest_simulator/limen_klines/btcusdt_{kline_size}.parquet`;
    # `install_cache()` patches `HistoricalData.get_spot_klines` so the
    # launcher's `BacktestMarketDataPoller.start()` reads that parquet
    # rather than re-streaming the full dataset from HuggingFace.
    # Idempotent: the install guard (`HistoricalData._bts_cache_installed`)
    # short-circuits a second call within the same interpreter.
    from backtest_simulator._limen_cache import install_cache
    install_cache()
    payload = json.loads(sys.stdin.read())
    raw_max_alloc = payload.get('max_allocation_per_trade_pct')
    raw_lookback = payload.get('predict_lookback')
    raw_display_id = payload.get('display_id')
    raw_trades_parquet = payload.get('trades_parquet_path')
    result = run_window_in_process(
        int(payload['perm_id']),
        Decimal(payload['kelly_pct']),
        datetime.fromisoformat(payload['window_start']),
        datetime.fromisoformat(payload['window_end']),
        Path(payload['experiment_dir']),
        maker_preference=bool(payload.get('maker_preference', False)),
        strict_impact=bool(payload.get('strict_impact', False)),
        trades_parquet_path=(
            None if raw_trades_parquet is None else str(raw_trades_parquet)
        ),
        max_allocation_per_trade_pct=(
            None if raw_max_alloc is None else Decimal(str(raw_max_alloc))
        ),
        predict_lookback=(
            None if raw_lookback is None else int(raw_lookback)
        ),
        display_id=None if raw_display_id is None else int(raw_display_id),
    )
    print(json.dumps(result), flush=True)
    # Skip Python interpreter teardown to dodge the polars/pyo3 race
    # where Rust worker threads call Python APIs after the interpreter
    # has begun finalising. The result is already serialised to stdout
    # and read by the parent before the child's exit signal lands.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':  # pragma: no cover - subprocess child entry
    sys.exit(_child_main())
