"""SignalsTable — precompute + lookup with per-decoder split-alignment gate."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from backtest_simulator.exceptions import LookAheadViolation


def _to_int(value: object) -> int:
    """Narrow `object` to `int` with a clear error on type mismatch."""
    if isinstance(value, bool):
        # bool is a subclass of int but split_config values are
        # genuine ints, not flags.
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
    """One signal value at a timestamp, carrying the decoder's label span."""

    timestamp: datetime
    prob: float
    pred: int
    label_t0: datetime
    label_t1: datetime


@dataclass(frozen=True)
class PredictionsInput:
    """Per-decoder inputs for building a SignalsTable.

    Bundles the five prediction/label fields so the builder takes one
    argument for them instead of five.
    """

    timestamps: list[datetime]
    probs: np.ndarray
    preds: np.ndarray
    label_horizon_bars: int
    bar_seconds: int


@dataclass
class SignalsTable:
    """Precomputed per-bar signals keyed by decoder_id + split_config.

    The lookup raises `LookAheadViolation` if:
      - the requested timestamp is before the earliest precomputed row
        (nothing to return), or
      - the decoder's recorded split_config does not byte-equal the
        split_config the caller says the sweep is using (per-decoder
        split-alignment gate, SPEC §9.3.1).
    """

    decoder_id: str
    split_config: tuple[int, int, int]
    frame: pl.DataFrame  # columns: timestamp, prob, pred, label_t0, label_t1

    @classmethod
    def from_predictions(
        cls,
        *,
        decoder_id: str,
        split_config: tuple[int, int, int],
        predictions: PredictionsInput,
    ) -> SignalsTable:
        from datetime import timedelta
        rows: list[tuple[datetime, float, int, datetime, datetime]] = []
        for i, ts in enumerate(predictions.timestamps):
            label_t0 = ts
            label_t1 = ts + timedelta(
                seconds=predictions.bar_seconds * predictions.label_horizon_bars,
            )
            rows.append((
                ts, float(predictions.probs[i]), int(predictions.preds[i]),
                label_t0, label_t1,
            ))
        utc = pl.Datetime(time_zone='UTC')
        frame = pl.DataFrame(
            rows,
            schema={'timestamp': utc, 'prob': pl.Float64, 'pred': pl.Int64, 'label_t0': utc, 'label_t1': utc},
            orient='row',
        )
        return cls(decoder_id=decoder_id, split_config=split_config, frame=frame)

    def assert_split_alignment(self, eval_split: tuple[int, int, int]) -> None:
        if tuple(eval_split) != tuple(self.split_config):
            msg = (
                f'split-alignment gate: decoder {self.decoder_id} was trained '
                f'with split={self.split_config} but sweep is evaluating with '
                f'split={eval_split}. Re-train with the same split.'
            )
            raise LookAheadViolation(msg)

    def lookup(self, t: datetime) -> SignalRow | None:
        """Return the greatest row with `timestamp <= t`, or None if none."""
        sliced = self.frame.filter(pl.col('timestamp') <= t).sort('timestamp').tail(1)
        if sliced.is_empty():
            return None
        row = sliced.row(0, named=True)
        return SignalRow(
            timestamp=row['timestamp'], prob=float(row['prob']),
            pred=int(row['pred']),
            label_t0=row['label_t0'], label_t1=row['label_t1'],
        )

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        parquet_path = directory / f'{self.decoder_id}.parquet'
        metadata_path = directory / f'{self.decoder_id}.meta.json'
        self.frame.write_parquet(parquet_path)
        metadata_path.write_text(json.dumps({
            'decoder_id': self.decoder_id,
            'split_config': list(self.split_config),
        }), encoding='utf-8')
        return parquet_path

    @classmethod
    def load(cls, directory: Path, decoder_id: str) -> SignalsTable:
        metadata_path = directory / f'{decoder_id}.meta.json'
        metadata: dict[str, object] = json.loads(metadata_path.read_text(encoding='utf-8'))
        frame = pl.read_parquet(directory / f'{decoder_id}.parquet')
        split_config_raw: object = metadata['split_config']
        if not isinstance(split_config_raw, list):
            msg = (
                f'SignalsTable.load: expected list for split_config '
                f'in {metadata_path}, got {type(split_config_raw).__name__}'
            )
            raise ValueError(msg)
        # `list` after isinstance is `list[Unknown]`; widen each cell
        # via _to_int() which validates and narrows from object → int.
        # cast() to widen list[Unknown] → list[object] for pyright;
        # runtime-equivalent.
        from typing import cast
        split_config_typed = list(cast('list[object]', split_config_raw))
        if len(split_config_typed) != 3:
            msg = (
                f'SignalsTable.load: expected 3-element list for split_config '
                f'in {metadata_path}, got {len(split_config_typed)} elements'
            )
            raise ValueError(msg)
        return cls(
            decoder_id=decoder_id,
            split_config=(
                _to_int(split_config_typed[0]),
                _to_int(split_config_typed[1]),
                _to_int(split_config_typed[2]),
            ),
            frame=frame,
        )
