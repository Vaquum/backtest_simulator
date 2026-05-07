"""Build a `SignalsTable` from a Limen experiment for one decoder via the runtime predict-recipe (sweep-time parity reference)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol

import numpy as np
import polars as pl

from backtest_simulator.launcher.poller import (
    DEFAULT_N_ROWS as POLLER_N_ROWS,
)
from backtest_simulator.sensors.precompute import (
    PredictionsInput,
    SignalsTable,
)


class _DataSourceConfigProtocol(Protocol):
    params: dict[str, object]

class _ManifestProtocol(Protocol):

    split_config: tuple[int, int, int]
    data_source_config: _DataSourceConfigProtocol | None

    def with_params_override(
        self, *, split_config: tuple[int, int, int],
    ) -> _ManifestProtocol:
        del split_config
        raise NotImplementedError

    def prepare_data(
        self, raw_data: pl.DataFrame, round_params: dict[str, object],
    ) -> dict[str, object]:
        del raw_data, round_params
        raise NotImplementedError

class _SensorProtocol(Protocol):
    def predict(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        del data
        raise NotImplementedError

def build_signals_table_for_decoder(
    *,
    manifest: _ManifestProtocol,
    sensor: _SensorProtocol,
    klines: pl.DataFrame,
    tick_timestamps: list[datetime],
    round_params: dict[str, object],
    decoder_id: str,
    predict_lookback: int | None = None,
    n_rows: int | None = None,
) -> SignalsTable:
    cfg = manifest.data_source_config
    if cfg is None or 'kline_size' not in cfg.params:
        msg = (
            'build_signals_table_for_decoder: manifest has no '
            'kline_size in data_source_config; bar_seconds is unknown. '
            'The experiment file must call set_data_source with '
            'kline_size — that is the cadence the launcher uses.'
        )
        raise ValueError(msg)
    bar_seconds = int(str(cfg.params['kline_size']))
    split_config = (
        int(manifest.split_config[0]),
        int(manifest.split_config[1]),
        int(manifest.split_config[2]),
    )
    shift_raw = round_params.get('shift')
    if isinstance(shift_raw, bool) or not isinstance(shift_raw, int):
        msg = (
            f'build_signals_table_for_decoder: round_params[shift] '
            f'must be int, got {type(shift_raw).__name__}={shift_raw!r}.'
        )
        raise TypeError(msg)
    label_horizon_bars = abs(shift_raw)

    if not tick_timestamps:
        msg = (
            f'build_signals_table_for_decoder: tick_timestamps empty '
            f'for decoder_id={decoder_id}. The sweep must compute '
            f'at least one runtime tick from (days, hours_start, '
            f'hours_end, interval_seconds).'
        )
        raise ValueError(msg)

    if predict_lookback is not None and predict_lookback < 1:
        msg = (
            f'build_signals_table_for_decoder: predict_lookback must be '
            f'>= 1, got {predict_lookback}. predict consumes '
            f'tail(predict_lookback); lookback < 1 yields empty x_test '
            f'and an IndexError on `_preds[-1]`.'
        )
        raise ValueError(msg)
    if n_rows is not None and n_rows < 1:
        msg = (
            f'build_signals_table_for_decoder: n_rows must be >= 1, got '
            f'{n_rows}. Per-tick causal slice tails n_rows; n_rows < 1 '
            f'yields an empty causal frame and the bar is silently skipped.'
        )
        raise ValueError(msg)
    manifest_full = manifest.with_params_override(split_config=(1, 0, 0))
    effective_lookback = 1 if predict_lookback is None else predict_lookback
    effective_n_rows = POLLER_N_ROWS if n_rows is None else n_rows
    timestamps: list[datetime] = []
    preds_list: list[int] = []
    probs_list: list[float] = []
    for raw_tick in tick_timestamps:
        tick = _as_utc_datetime(raw_tick)
        causal = klines.filter(pl.col('datetime') <= tick).tail(effective_n_rows)
        if causal.is_empty():
            continue
        data_dict = manifest_full.prepare_data(causal, round_params)
        x_train_obj = data_dict.get('x_train')
        if not isinstance(x_train_obj, pl.DataFrame) or x_train_obj.is_empty():
            continue
        last_x = x_train_obj.tail(effective_lookback).to_numpy()
        result = sensor.predict({'x_test': last_x})
        preds_list.append(int(result['_preds'][-1]))
        probs_list.append(float(result['_probs'][-1]))
        timestamps.append(tick)

    if not timestamps:
        msg = (
            f'build_signals_table_for_decoder: every replay bar was '
            f'consumed by feature warmup for decoder_id={decoder_id}. '
            f'Either the window is too short or `klines` does not '
            f'carry enough warmup bars before replay_window_start.'
        )
        raise ValueError(msg)

    return SignalsTable.from_predictions(
        decoder_id=decoder_id,
        split_config=split_config,
        predictions=PredictionsInput(
            timestamps=timestamps,
            probs=np.asarray(probs_list, dtype=np.float64),
            preds=np.asarray(preds_list, dtype=np.int64),
            label_horizon_bars=label_horizon_bars,
            bar_seconds=bar_seconds,
        ),
    )

def _as_utc_datetime(ts: object) -> datetime:
    if not isinstance(ts, datetime):
        msg = f'_as_utc_datetime: expected datetime, got {type(ts).__name__}'
        raise TypeError(msg)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts

def _snap_runtime_to_expected(
    *, ts: datetime, expected_ticks: list[datetime],
    interval_seconds: int,
) -> datetime | None:
    matches = [
        e for e in expected_ticks
        if e <= ts < e + timedelta(seconds=interval_seconds)
    ]
    if len(matches) == 1:
        return matches[0]
    return None

def assert_signals_parity(
    *, decoder_id: str,
    table: SignalsTable,
    runtime_predictions: list[dict[str, object]],
    expected_ticks: list[datetime],
    interval_seconds: int,
) -> int:
    from backtest_simulator.exceptions import ParityViolation

    if len(set(expected_ticks)) != len(expected_ticks):
        msg = (
            f'assert_signals_parity: expected_ticks contains '
            f'duplicate timestamps for decoder {decoder_id!r}. '
            f'Caller bug — the per-window scheduled grid must '
            f'have unique timestamps.'
        )
        raise ValueError(msg)
    covered = set(expected_ticks)
    captured: list[datetime] = []
    n_compared = 0
    for entry in runtime_predictions:
        ts_raw = entry['timestamp']
        if not isinstance(ts_raw, str):
            msg = (
                f'SignalsTable parity violation for decoder '
                f'{decoder_id!r}: runtime entry has non-string '
                f'`timestamp` field (got {type(ts_raw).__name__}). '
                f'Capture serialiser produced unverifiable data; '
                f'the parity check cannot proceed.'
            )
            raise ParityViolation(msg)
        ts = datetime.fromisoformat(ts_raw)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        runtime_pred = entry.get('pred')
        if not isinstance(runtime_pred, int):
            msg = (
                f'SignalsTable parity violation for decoder '
                f'{decoder_id!r} at tick {ts.isoformat()}: '
                f'runtime entry has non-int `pred` field (got '
                f'{type(runtime_pred).__name__}={runtime_pred!r}). '
                f'Capture serialiser produced unverifiable data.'
            )
            raise ParityViolation(msg)
        snapped_ts = _snap_runtime_to_expected(
            ts=ts, expected_ticks=expected_ticks,
            interval_seconds=interval_seconds,
        )
        if snapped_ts is None:
            grid_min = min(covered) if covered else None
            grid_max = max(covered) if covered else None
            msg = (
                f'SignalsTable parity violation for decoder '
                f'{decoder_id!r}: runtime tick {ts.isoformat()} '
                f'falls OUTSIDE the scheduled per-window `expected_ticks` '
                f'grid (range: {grid_min}..{grid_max}, '
                f'n_ticks={len(covered)}, interval_seconds='
                f'{interval_seconds}). The PredictLoop fired at an '
                f'instant that does not anchor to any expected tick '
                f'within one interval window — capture is broken, '
                f'the runtime cadence diverged from the planned '
                f'grid, or the sweep passed the wrong grid slice.'
            )
            raise ParityViolation(msg)
        captured.append(snapped_ts)
        row = table.lookup(snapped_ts)
        if row is None:
            msg = (
                f'SignalsTable parity violation for decoder '
                f'{decoder_id!r}: scheduled tick {snapped_ts.isoformat()} '
                f'has no row in the table — build skipped this '
                f'instant (runtime ts={ts.isoformat()}). Sweep-time '
                f'replay incomplete; cannot compare runtime '
                f'pred={runtime_pred}.'
            )
            raise ParityViolation(msg)
        if row.pred != runtime_pred:
            msg = (
                f'SignalsTable parity violation for decoder '
                f'{decoder_id!r} at tick {snapped_ts.isoformat()} '
                f'(runtime ts={ts.isoformat()}): '
                f'runtime Sensor.predict() emitted pred={runtime_pred} '
                f'but the sweep-time SignalsTable says pred={row.pred}. '
                f'The deployed strategy is operating on different '
                f'predictions than the sweep replay. This violates the '
                f'"strategy tested, strategy deployed" promise; either '
                f'the runtime market_data fetch diverged from the '
                f'sweep\'s klines, or the sensor changed state between '
                f'build and run.'
            )
            raise ParityViolation(msg)
        n_compared += 1
    from collections import Counter
    captured_counter = Counter(captured)
    expected_counter = Counter(expected_ticks)
    missing = expected_counter - captured_counter
    duplicates = {
        t: cnt for t, cnt in captured_counter.items() if cnt > 1
    }
    if missing:
        n_missing = sum(missing.values())
        sample = sorted(missing)[:3]
        msg = (
            f'SignalsTable parity violation for decoder '
            f'{decoder_id!r}: {n_missing} expected tick(s) NOT '
            f'captured by runtime (sample: '
            f'{[t.isoformat() for t in sample]}). Capture hook '
            f'skipped scheduled boundaries — partial PredictLoop '
            f'firing or subprocess truncated output. Expected: '
            f'{len(expected_ticks)} tick(s); captured: '
            f'{len(captured)}.'
        )
        raise ParityViolation(msg)
    if duplicates:
        sample_dup = sorted(duplicates.items())[:3]
        msg = (
            f'SignalsTable parity violation for decoder '
            f'{decoder_id!r}: runtime captured duplicate ticks '
            f'(sample: {[(t.isoformat(), n) for t, n in sample_dup]}). '
            f'PredictLoop double-fired or capture-side double-'
            f'emitted; either is a real-runtime bug worth surfacing.'
        )
        raise ParityViolation(msg)
    if n_compared == 0:
        n_runtime = len(runtime_predictions)
        n_covered = len(covered)
        msg_zero = (
            f'SignalsTable parity violation for decoder '
            f'{decoder_id!r}: 0 comparisons made (runtime '
            f'predictions={n_runtime}, expected ticks={n_covered}). '
            f'The parity check did NOT run. Either the capture '
            f'hook is broken, the subprocess payload is missing '
            f'`runtime_predictions`, every captured entry was '
            f'malformed, or `expected_ticks` was empty (caller '
            f'bug — every per-window block must have at least one '
            f'scheduled tick).'
        )
        raise ParityViolation(msg_zero)
    return n_compared
