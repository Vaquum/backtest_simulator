"""SignalsTable — precompute + lookup with per-decoder split-alignment gate."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from backtest_simulator.exceptions import LookAheadViolation
from backtest_simulator.feed.lookahead import frozen_now


def _to_int(value: object) -> int:
    if isinstance(value, bool):
        msg = f'_to_int: expected int, got bool {value!r}'
        raise TypeError(msg)
    if isinstance(value, int):
        return value
    if isinstance(value, (str, float)):
        return int(value)
    msg = f'_to_int: cannot narrow {type(value).__name__} to int'
    raise TypeError(msg)

@dataclass(frozen=True)
class SignalRow:
    timestamp: datetime
    prob: float
    pred: int
    label_t0: datetime
    label_t1: datetime

@dataclass(frozen=True)
class PredictionsInput:
    timestamps: list[datetime]
    probs: np.ndarray
    preds: np.ndarray
    label_horizon_bars: int
    bar_seconds: int

@dataclass
class SignalsTable:
    decoder_id: str
    split_config: tuple[int, int, int]
    bar_seconds: int
    label_horizon_bars: int
    _frame: pl.DataFrame = field(repr=False)

    @classmethod
    def from_predictions(cls, *, decoder_id: str, split_config: tuple[int, int, int], predictions: PredictionsInput) -> SignalsTable:
        from datetime import timedelta
        rows: list[tuple[datetime, float, int, datetime, datetime]] = []
        for i, ts in enumerate(predictions.timestamps):
            label_t0 = ts
            label_t1 = ts + timedelta(seconds=predictions.bar_seconds * predictions.label_horizon_bars)
            rows.append((ts, float(predictions.probs[i]), int(predictions.preds[i]), label_t0, label_t1))
        utc = pl.Datetime(time_zone='UTC')
        frame = pl.DataFrame(rows, schema={'timestamp': utc, 'prob': pl.Float64, 'pred': pl.Int64, 'label_t0': utc, 'label_t1': utc}, orient='row')
        return cls(decoder_id=decoder_id, split_config=split_config, bar_seconds=int(predictions.bar_seconds), label_horizon_bars=int(predictions.label_horizon_bars), _frame=frame)

    def assert_split_alignment(self, eval_split: tuple[int, int, int]) -> None:
        if tuple(eval_split) != tuple(self.split_config):
            msg = f'split-alignment gate: decoder {self.decoder_id} was trained with split={self.split_config} but sweep is evaluating with split={eval_split}. Re-train with the same split.'
            raise LookAheadViolation(msg)

    @staticmethod
    def _lookup_validate_args(t: datetime, purge_seconds: int, embargo_seconds: int) -> None:
        if t.tzinfo is None or t.utcoffset() is None:
            msg = f'SignalsTable.lookup requires a tz-aware datetime with a concrete UTC offset; got {t!r} (tzinfo={t.tzinfo!r}, utcoffset={t.utcoffset()!r}). Strategies must construct timestamps with an explicit tz (e.g. UTC) so the no-look-ahead gate compares apples to apples.'
            raise ValueError(msg)
        if purge_seconds < 0:
            msg = f'SignalsTable.lookup: purge_seconds must be non-negative, got {purge_seconds}.'
            raise ValueError(msg)
        if embargo_seconds < 0:
            msg = f'SignalsTable.lookup: embargo_seconds must be non-negative, got {embargo_seconds}.'
            raise ValueError(msg)

    def _t_in_allowed_groups(self, t: datetime, allowed_groups: tuple[int, ...], n_groups: int) -> bool:
        if self._frame.is_empty():
            return False
        first_ts = self._frame['timestamp'].min()
        last_ts = self._frame['timestamp'].max()
        if not isinstance(first_ts, datetime) or not isinstance(last_ts, datetime):
            msg = f'SignalsTable._t_in_allowed_groups: timestamp min/max returned non-datetime (first={type(first_ts).__name__}); schema corrupt.'
            raise TypeError(msg)
        span_seconds = (last_ts - first_ts).total_seconds()
        if span_seconds <= 0:
            return 0 in allowed_groups
        position = (t - first_ts).total_seconds() / span_seconds
        group_id = min(max(int(position * n_groups), 0), n_groups - 1)
        return group_id in allowed_groups

    def lookup(self, t: datetime, *, allowed_groups: tuple[int, ...] | None=None, n_groups: int=1, purge_seconds: int=0, embargo_seconds: int=0) -> SignalRow | None:
        self._lookup_validate_args(t, purge_seconds, embargo_seconds)
        if allowed_groups is not None and n_groups < 2:
            msg = f'SignalsTable.lookup: group filtering requires n_groups >= 2, got {n_groups}. Single-block partitioning has no train/test separation.'
            raise ValueError(msg)
        now = frozen_now()
        if t > now:
            msg = f'SignalsTable.lookup(t={t}) requested data past frozen_now()={now} for decoder {self.decoder_id}'
            raise LookAheadViolation(msg)
        if allowed_groups is not None and (not self._t_in_allowed_groups(t, allowed_groups, n_groups)):
            return None
        from datetime import timedelta
        cutoff = t - timedelta(seconds=embargo_seconds)
        sliced = self._frame.filter(pl.col('timestamp') <= cutoff)
        if purge_seconds > 0:
            purge_floor = t - timedelta(seconds=purge_seconds)
            sliced = sliced.filter(pl.col('label_t1') <= purge_floor)
        sliced = sliced.sort('timestamp').tail(1)
        if sliced.is_empty():
            return None
        row = sliced.row(0, named=True)
        return SignalRow(timestamp=row['timestamp'], prob=float(row['prob']), pred=int(row['pred']), label_t0=row['label_t0'], label_t1=row['label_t1'])

    def assert_window_covers(self, window_start: datetime, window_end: datetime) -> None:
        if self._frame.is_empty():
            msg = f'SignalsTable.assert_window_covers: table for {self.decoder_id} is empty.'
            raise LookAheadViolation(msg)
        first_ts = self._frame['timestamp'].min()
        last_ts = self._frame['timestamp'].max()
        if not isinstance(first_ts, datetime) or not isinstance(last_ts, datetime):
            msg = f'SignalsTable.assert_window_covers: timestamp min/max returned non-datetime for {self.decoder_id}.'
            raise TypeError(msg)
        from datetime import timedelta
        max_staleness = timedelta(seconds=self.bar_seconds * self.label_horizon_bars)
        one_bar = timedelta(seconds=self.bar_seconds)
        if window_start + one_bar < first_ts:
            msg = f'SignalsTable.assert_window_covers: window_start={window_start} precedes table coverage start={first_ts} for {self.decoder_id} by more than one bar ({self.bar_seconds}s). Rebuild the table with klines that cover the replay window.'
            raise LookAheadViolation(msg)
        if window_end > last_ts + max_staleness:
            msg = f'SignalsTable.assert_window_covers: window_end={window_end} exceeds table coverage end={last_ts} + max_staleness={max_staleness} for {self.decoder_id}. Rebuild with klines that cover the full replay window.'
            raise LookAheadViolation(msg)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        parquet_path = directory / f'{self.decoder_id}.parquet'
        metadata_path = directory / f'{self.decoder_id}.meta.json'
        self._frame.write_parquet(parquet_path)
        metadata_path.write_text(json.dumps({'decoder_id': self.decoder_id, 'split_config': list(self.split_config), 'bar_seconds': int(self.bar_seconds), 'label_horizon_bars': int(self.label_horizon_bars)}), encoding='utf-8')
        return parquet_path

    @classmethod
    def load(cls, directory: Path, decoder_id: str) -> SignalsTable:
        metadata_path = directory / f'{decoder_id}.meta.json'
        metadata: dict[str, object] = json.loads(metadata_path.read_text(encoding='utf-8'))
        frame = pl.read_parquet(directory / f'{decoder_id}.parquet')
        split_config_raw: object = metadata['split_config']
        if not isinstance(split_config_raw, list):
            msg = f'SignalsTable.load: expected list for split_config in {metadata_path}, got {type(split_config_raw).__name__}'
            raise ValueError(msg)
        from typing import cast
        split_config_typed = list(cast('list[object]', split_config_raw))
        if len(split_config_typed) != 3:
            msg = f'SignalsTable.load: expected 3-element list for split_config in {metadata_path}, got {len(split_config_typed)} elements'
            raise ValueError(msg)
        if 'bar_seconds' not in metadata or 'label_horizon_bars' not in metadata:
            msg = f'SignalsTable.load: metadata at {metadata_path} is missing bar_seconds or label_horizon_bars. The stale-row guard cannot run without them. Rebuild the table with the current builder.'
            raise ValueError(msg)
        return cls(decoder_id=decoder_id, split_config=(_to_int(split_config_typed[0]), _to_int(split_config_typed[1]), _to_int(split_config_typed[2])), bar_seconds=_to_int(metadata['bar_seconds']), label_horizon_bars=_to_int(metadata['label_horizon_bars']), _frame=frame)
