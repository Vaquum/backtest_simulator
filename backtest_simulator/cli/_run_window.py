"""Single-window backtest run + subprocess child entry point."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

if TYPE_CHECKING:
    from backtest_simulator.feed.protocol import VenueFeed
    from backtest_simulator.honesty.maker_fill import MakerFillModel
    from backtest_simulator.honesty.slippage import SlippageModel

from backtest_simulator.cli._pipeline import SYMBOL, WORK_DIR, op_sfd_pythonpath, seed_price_at


def _effective_kelly_pct_for_allocation(kelly_pct: Decimal, max_allocation_per_trade_pct: Decimal | None) -> Decimal:
    if max_allocation_per_trade_pct is None:
        return kelly_pct
    from backtest_simulator.launcher.action_submitter import NOTIONAL_RESERVATION_BUFFER
    max_kelly_pct = (max_allocation_per_trade_pct / (Decimal('1') + NOTIONAL_RESERVATION_BUFFER) * Decimal('100')).quantize(Decimal('0.000001'), rounding=ROUND_DOWN)
    return min(kelly_pct, max_kelly_pct)

def read_kline_size_from_experiment_dir(experiment_dir: Path) -> int:
    metadata_path = experiment_dir / 'metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    sfd_module_name = metadata['sfd_module']
    import importlib
    sfd = importlib.import_module(sfd_module_name)
    limen_manifest = sfd.manifest()
    return int(limen_manifest.data_source_config.params['kline_size'])

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

def capture_runtime_prediction(*, wired: object, signal: object, sink: list[dict[str, object]], window_start: datetime | None=None, window_end: datetime | None=None) -> None:
    del window_start, window_end
    from collections.abc import Mapping
    from typing import cast
    values_obj = getattr(signal, 'values', None)
    values = cast('Mapping[str, object]', values_obj)
    pred_val = values.get('_preds')
    ts_obj = cast('datetime', getattr(signal, 'timestamp', None))
    sink.append({'sensor_id': str(getattr(wired, 'sensor_id', '')), 'timestamp': ts_obj.isoformat(), 'pred': pred_val})

def _fresh_work_dir(suffix: str) -> Path:
    d = WORK_DIR / suffix
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d

def run_window_in_process(perm_id: int, kelly_pct: Decimal, window_start: datetime, window_end: datetime, experiment_dir: Path, *, maker_preference: bool=False, strict_impact: bool=False, trades_parquet_path: str | None=None, max_allocation_per_trade_pct: Decimal | None=None, predict_lookback: int | None=None, display_id: int | None=None) -> WindowResult:
    import polars as _pl_mod
    from praxis.launcher import InstanceConfig
    from praxis.trading_config import TradingConfig

    from backtest_simulator.feed.clickhouse import (
        ClickHouseConfig,
        ClickHouseFeed,
        InMemoryTradesFeed,
    )
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
    if trades_parquet_path is not None:
        from backtest_simulator.cli._stats import make_seed_price_from_parquet
        seed_price = make_seed_price_from_parquet(Path(trades_parquet_path))(window_start)
    else:
        seed_price = seed_price_at(window_start)
    suffix_id = perm_id if display_id is None else display_id
    suffix = f'{suffix_id}_{window_start.date().isoformat()}'
    work = _fresh_work_dir(suffix)
    manifest_dir = work / 'manifest'
    state_dir = work / 'state'
    state_dir.mkdir()
    interval_seconds = read_kline_size_from_experiment_dir(experiment_dir)
    effective_kelly_pct = _effective_kelly_pct_for_allocation(kelly_pct, max_allocation_per_trade_pct)
    built = ManifestBuilder(output_dir=manifest_dir).build(account=AccountSpec(account_id='bts-sweep', allocated_capital=Decimal('100000'), capital_pool=Decimal('100000')), strategy_id='long_on_signal', sensor=SensorBinding(experiment_dir=experiment_dir, permutation_ids=(perm_id,), interval_seconds=interval_seconds), strategy_params=StrategyParamsSpec(symbol=SYMBOL, capital=Decimal('100000'), kelly_pct=effective_kelly_pct, estimated_price=seed_price, stop_bps=Decimal('50'), force_flatten_after=window_end - timedelta(seconds=interval_seconds), maker_preference=maker_preference))
    feed: ClickHouseFeed | InMemoryTradesFeed
    if trades_parquet_path is not None:
        feed = InMemoryTradesFeed(_pl_mod.read_parquet(trades_parquet_path).set_sorted('time'), symbol=SYMBOL)
    else:
        feed = ClickHouseFeed(config=ClickHouseConfig.from_env(), symbol=SYMBOL)
    slippage_model = _calibrate_slippage(feed, window_start)
    maker_fill_model = _calibrate_maker_fill(feed, window_start) if maker_preference else None
    adapter = SimulatedVenueAdapter(feed=feed, filters=BinanceSpotFilters.binance_spot(SYMBOL), fees=FeeSchedule(), trade_window_seconds=min(interval_seconds, 600), window_end_clamp=window_end, slippage_model=slippage_model, maker_fill_model=maker_fill_model, market_impact_bucket_minutes=1, market_impact_threshold_fraction=Decimal('0.1'), strict_impact_policy=strict_impact)
    tc = TradingConfig(epoch_id=1, venue_rest_url='http://sim', venue_ws_url='ws://sim', account_credentials={'bts-sweep': ('k', 's')}, shutdown_timeout=5.0)
    launcher = BacktestLauncher(trading_config=tc, instances=[InstanceConfig(account_id='bts-sweep', manifest_path=built.manifest_path, strategies_base_path=built.strategies_base_path, state_dir=state_dir)], venue_adapter=adapter, db_path=work / 'event_spine.sqlite', max_allocation_per_trade_pct=max_allocation_per_trade_pct, clock_tick_seconds=interval_seconds)
    runtime_predictions: list[dict[str, object]] = []
    import nexus.strategy.predict_loop as _predict_loop_mod
    _real_produce_signal = getattr(_predict_loop_mod, 'produce_signal')
    _lookback = predict_lookback

    def _capturing_produce_signal(wired: object, market_data: object) -> object:
        if _lookback is None:
            signal = _real_produce_signal(wired, market_data)
        else:
            signal = _real_produce_signal(wired, market_data, lookback=_lookback)
        _len_before = len(runtime_predictions)
        capture_runtime_prediction(wired=wired, signal=signal, sink=runtime_predictions, window_start=window_start, window_end=window_end)
        if len(runtime_predictions) > _len_before:
            from collections.abc import Mapping
            from typing import cast as _cast_local
            _values_obj = getattr(signal, 'values', None)
            if isinstance(_values_obj, Mapping):
                _values_typed = _cast_local('Mapping[str, object]', _values_obj)
                _prob: object = _values_typed.get('_probs')
                if isinstance(_prob, (int, float)):
                    runtime_predictions[-1]['prob'] = float(_prob)
        return signal
    setattr(_predict_loop_mod, 'produce_signal', _capturing_produce_signal)
    try:
        launcher.run_window(window_start, window_end)
    finally:
        setattr(_predict_loop_mod, 'produce_signal', _real_produce_signal)
    from backtest_simulator.honesty.ledger_parity import (
        count_event_spine_events,
        dump_event_spine_to_jsonl,
    )
    spine_sqlite = work / 'event_spine.sqlite'
    spine_jsonl = work / 'event_spine.jsonl'
    spine_n_events = dump_event_spine_to_jsonl(sqlite_path=spine_sqlite, jsonl_path=spine_jsonl)
    spine_counts = count_event_spine_events(sqlite_path=spine_sqlite)
    book_gap = adapter.book_gap_snapshot()
    account = adapter.history('bts-sweep')
    trade_tuples = [(t.client_order_id, t.side.name, str(t.qty), str(t.price), str(t.fee), t.fee_asset, t.timestamp.isoformat()) for t in account.trades]
    _stops: dict[str, Decimal] = getattr(launcher, '_declared_stops')
    declared_stops = {str(k): str(v) for k, v in _stops.items()}

    def _emit(value: Decimal | None) -> str | None:
        return None if value is None else str(value)
    result: WindowResult = {'trades': [list(row) for row in trade_tuples], 'declared_stops': declared_stops, 'orders': len(account.orders), 'slippage_realised_bps': _emit(adapter.slippage_realised_aggregate_bps), 'slippage_realised_cost_bps': _emit(adapter.slippage_realised_cost_bps), 'slippage_realised_buy_bps': _emit(adapter.slippage_realised_buy_bps), 'slippage_realised_sell_bps': _emit(adapter.slippage_realised_sell_bps), 'slippage_predicted_cost_bps': _emit(adapter.slippage_predicted_cost_bps), 'slippage_predict_vs_realised_gap_bps': _emit(adapter.slippage_predict_vs_realised_gap_bps), 'slippage_n_samples': adapter.slippage_realised_n_samples, 'slippage_n_excluded': adapter.slippage_realised_n_excluded, 'slippage_n_uncalibrated_predict': adapter.slippage_n_uncalibrated_predict, 'slippage_n_predicted_samples': adapter.slippage_n_predicted_samples, 'n_limit_orders_submitted': adapter.n_limit_orders_submitted, 'n_limit_filled_full': adapter.n_limit_filled_full, 'n_limit_filled_partial': adapter.n_limit_filled_partial, 'n_limit_filled_zero': adapter.n_limit_filled_zero, 'n_limit_marketable_taker': adapter.n_limit_marketable_taker, 'n_passive_limits': adapter.n_passive_limits, 'maker_fill_efficiency_p50': _emit(adapter.maker_fill_efficiency_p50), 'maker_fill_efficiency_mean': _emit(adapter.maker_fill_efficiency_mean), 'market_impact_realised_bps': _emit(adapter.market_impact_realised_bps), 'market_impact_n_samples': adapter.market_impact_n_samples, 'market_impact_n_flagged': adapter.market_impact_n_flagged, 'market_impact_n_uncalibrated': adapter.market_impact_n_uncalibrated, 'market_impact_n_rejected': adapter.market_impact_n_rejected, 'event_spine_jsonl': str(spine_jsonl), 'event_spine_n_events': spine_n_events, 'n_intents': spine_counts['intents'], 'n_submitted': spine_counts['submitted'], 'n_fills': spine_counts['fills'], 'n_pending': spine_counts['pending'], 'n_rejects': spine_counts['rejects'], 'book_gap_max_seconds': float(book_gap.max_stop_cross_to_trade_seconds), 'book_gap_n_observed': int(book_gap.n_stops_observed), 'book_gap_p95_seconds': float(book_gap.p95_stop_cross_to_trade_seconds), 'runtime_predictions': runtime_predictions}
    return result

def _calibrate_slippage(feed: VenueFeed, window_start: datetime) -> SlippageModel | None:
    from datetime import timedelta

    from backtest_simulator.honesty.slippage import SlippageModel
    calibration_end = window_start
    calibration_start = window_start - timedelta(minutes=30)
    trades = feed.get_trades(SYMBOL, calibration_start, calibration_end)
    trades = trades.rename({'time': 'datetime', 'qty': 'quantity'})
    return SlippageModel.calibrate(trades=trades, side_buckets=(Decimal('0.001'), Decimal('0.01'), Decimal('0.1'), Decimal('1.0')), dt_seconds=10)

def _calibrate_maker_fill(feed: VenueFeed, window_start: datetime) -> MakerFillModel:
    from datetime import timedelta

    from backtest_simulator.honesty.maker_fill import MakerFillModel
    pre = feed.get_trades(SYMBOL, window_start - timedelta(minutes=30), window_start)
    pre = pre.rename({'time': 'datetime', 'qty': 'quantity'})
    return MakerFillModel.calibrate(trades=pre, lookback_minutes=30)

def run_window_in_subprocess(perm_id: int, kelly_pct: Decimal, window_start: datetime, window_end: datetime, experiment_dir: Path, *, maker_preference: bool=False, strict_impact: bool=False, trades_parquet_path: str | None=None, max_allocation_per_trade_pct: Decimal | None=None, predict_lookback: int | None=None, display_id: int | None=None) -> WindowResult:
    payload = json.dumps({'perm_id': perm_id, 'kelly_pct': str(kelly_pct), 'window_start': window_start.isoformat(), 'window_end': window_end.isoformat(), 'experiment_dir': str(experiment_dir), 'maker_preference': bool(maker_preference), 'strict_impact': bool(strict_impact), 'trades_parquet_path': trades_parquet_path, 'max_allocation_per_trade_pct': None if max_allocation_per_trade_pct is None else str(max_allocation_per_trade_pct), 'predict_lookback': None if predict_lookback is None else int(predict_lookback), 'display_id': None if display_id is None else int(display_id)})
    env = os.environ.copy()
    existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = os.pathsep.join(p for p in (op_sfd_pythonpath(), existing) if p)
    env['POLARS_MAX_THREADS'] = '1'
    proc = subprocess.run([sys.executable, '-m', 'backtest_simulator.cli._run_window'], input=payload, capture_output=True, text=True, check=False, timeout=120, env=env)
    last: str | None = None
    for line in proc.stdout.splitlines():
        s = line.strip()
        if s.startswith('{') and s.endswith('}'):
            last = s
    if last is None:
        msg = f'run_window subprocess produced no JSON line on stdout. stderr tail: {proc.stderr[-500:]}'
        raise RuntimeError(msg)
    return cast('WindowResult', json.loads(last))

def _child_main() -> int:
    import logging as _logging
    for _name in ('limen', 'praxis', 'nexus'):
        _logging.getLogger(_name).setLevel(_logging.WARNING)
    import structlog as _structlog
    _structlog.configure(wrapper_class=_structlog.make_filtering_bound_logger(_logging.WARNING))
    _level_name = os.environ.get('BTS_RUN_WINDOW_LOG_LEVEL')
    from backtest_simulator._limen_cache import install_cache
    install_cache()
    payload = json.loads(sys.stdin.read())
    raw_max_alloc = payload.get('max_allocation_per_trade_pct')
    raw_lookback = payload.get('predict_lookback')
    raw_display_id = payload.get('display_id')
    raw_trades_parquet = payload.get('trades_parquet_path')
    result = run_window_in_process(int(payload['perm_id']), Decimal(payload['kelly_pct']), datetime.fromisoformat(payload['window_start']), datetime.fromisoformat(payload['window_end']), Path(payload['experiment_dir']), maker_preference=bool(payload.get('maker_preference', False)), strict_impact=bool(payload.get('strict_impact', False)), trades_parquet_path=None if raw_trades_parquet is None else str(raw_trades_parquet), max_allocation_per_trade_pct=None if raw_max_alloc is None else Decimal(str(raw_max_alloc)), predict_lookback=None if raw_lookback is None else int(raw_lookback), display_id=None if raw_display_id is None else int(raw_display_id))
    print(json.dumps(result), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    if os.environ.get('COVERAGE_PROCESS_START'):
        from typing import Protocol

        class _CovInstance(Protocol):
            def save(self) -> None: ...

        class _CovClass(Protocol):
            @classmethod
            def current(cls) -> _CovInstance | None: ...

        import importlib
        _coverage_mod = importlib.import_module('coverage')
        _coverage_cls = cast('type[_CovClass]', getattr(_coverage_mod, 'Coverage'))
        _cov = _coverage_cls.current()
        if _cov is None:
            msg = (
                'COVERAGE_PROCESS_START is set but coverage.Coverage.current() '
                'is None — subprocess auto-init did not register. Likely the '
                '_bts_coverage_subprocess.pth file is not on this Python\'s '
                'site-packages. Without the .pth, this subprocess emits no '
                'coverage data; failing loud rather than silently dropping it.'
            )
            raise RuntimeError(msg)
        _cov.save()
    os._exit(0)
if __name__ == '__main__':
    sys.exit(_child_main())
