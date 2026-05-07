"""Build a `SignalsTable` from a Limen experiment for one decoder via the runtime predict-recipe (sweep-time parity reference)."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol, cast

import numpy as np
import polars as pl

from backtest_simulator.launcher.poller import DEFAULT_N_ROWS as POLLER_N_ROWS
from backtest_simulator.sensors.precompute import PredictionsInput, SignalsTable


class _DataSourceConfigProtocol(Protocol):
    params: dict[str, object]

class _ManifestProtocol(Protocol):
    split_config: tuple[int, int, int]
    data_source_config: _DataSourceConfigProtocol | None

    def with_params_override(self, *, split_config: tuple[int, int, int]) -> _ManifestProtocol:
        del split_config
        raise NotImplementedError

    def prepare_data(self, raw_data: pl.DataFrame, round_params: dict[str, object]) -> dict[str, object]:
        del raw_data, round_params
        raise NotImplementedError

class _SensorProtocol(Protocol):

    def predict(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        del data
        raise NotImplementedError

def build_signals_table_for_decoder(*, manifest: _ManifestProtocol, sensor: _SensorProtocol, klines: pl.DataFrame, tick_timestamps: list[datetime], round_params: dict[str, object], decoder_id: str, predict_lookback: int | None=None, n_rows: int | None=None) -> SignalsTable:
    cfg = manifest.data_source_config
    assert cfg is not None
    bar_seconds = int(str(cfg.params['kline_size']))
    split_config = (int(manifest.split_config[0]), int(manifest.split_config[1]), int(manifest.split_config[2]))
    shift_raw = round_params.get('shift')
    label_horizon_bars = abs(cast('int', shift_raw))
    manifest_full = manifest.with_params_override(split_config=(1, 0, 0))
    effective_lookback = 1 if predict_lookback is None else predict_lookback
    effective_n_rows = POLLER_N_ROWS if n_rows is None else n_rows
    timestamps: list[datetime] = []
    preds_list: list[int] = []
    probs_list: list[float] = []
    for raw_tick in tick_timestamps:
        tick = _as_utc_datetime(raw_tick)
        causal = klines.filter(pl.col('datetime') <= tick).tail(effective_n_rows)
        data_dict = manifest_full.prepare_data(causal, round_params)
        x_train_obj = cast('pl.DataFrame', data_dict.get('x_train'))
        last_x = x_train_obj.tail(effective_lookback).to_numpy()
        result = sensor.predict({'x_test': last_x})
        preds_list.append(int(result['_preds'][-1]))
        probs_list.append(float(result['_probs'][-1]))
        timestamps.append(tick)
    return SignalsTable.from_predictions(decoder_id=decoder_id, split_config=split_config, predictions=PredictionsInput(timestamps=timestamps, probs=np.asarray(probs_list, dtype=np.float64), preds=np.asarray(preds_list, dtype=np.int64), label_horizon_bars=label_horizon_bars, bar_seconds=bar_seconds))

def _as_utc_datetime(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)

def _snap_runtime_to_expected(*, ts: datetime, expected_ticks: list[datetime], interval_seconds: int) -> datetime | None:
    matches = [e for e in expected_ticks if e <= ts < e + timedelta(seconds=interval_seconds)]
    if len(matches) == 1:
        return matches[0]
    return None

def assert_signals_parity(*, decoder_id: str, table: SignalsTable, runtime_predictions: list[dict[str, object]], expected_ticks: list[datetime], interval_seconds: int) -> int:
    del decoder_id
    set(expected_ticks)
    captured: list[datetime] = []
    n_compared = 0
    for entry in runtime_predictions:
        ts_raw = cast('str', entry['timestamp'])
        ts = datetime.fromisoformat(ts_raw)
        entry.get('pred')
        snapped_ts = _snap_runtime_to_expected(ts=ts, expected_ticks=expected_ticks, interval_seconds=interval_seconds)
        assert snapped_ts is not None
        captured.append(snapped_ts)
        table.lookup(snapped_ts)
        n_compared += 1
    from collections import Counter
    captured_counter = Counter(captured)
    expected_counter = Counter(expected_ticks)
    expected_counter - captured_counter
    {t: cnt for t, cnt in captured_counter.items() if cnt > 1}
    return n_compared
