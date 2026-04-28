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

    `bar_seconds` and `label_horizon_bars` are persisted because the
    lookup's stale-row guard (slice #17 Task 16, codex round 1 P0)
    needs them: if the latest causal row is older than
    `bar_seconds * label_horizon_bars`, the lookup returns None
    rather than silently feeding the strategy a row whose label
    horizon expired before `t`.
    """

    decoder_id: str
    split_config: tuple[int, int, int]
    bar_seconds: int
    label_horizon_bars: int
    # `_frame` holds future label rows; access through `lookup(t)` only.
    # Reaching past the underscore (`signals._frame.filter(...)`) is a
    # documented bypass — `tests/honesty/test_prescient_strategy.py`
    # introspection test fires if a strategy does so.
    # `repr=False` prevents the default dataclass repr from dumping the
    # full timestamp/label/prob columns when the table is logged or
    # printed (codex round 3): a strategy that prints its decoder
    # would otherwise have free access to every future row in plain
    # text via `repr(signals)`.
    _frame: pl.DataFrame = field(repr=False)

    @property
    def n_bars(self) -> int:
        """Number of precomputed rows (== covered ticks).

        Public read-only counter. The internal `_frame` is intentionally
        private (it carries every label_t0/label_t1 future row); callers
        that only need the count should use this.
        """
        return int(self._frame.height)

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
        return cls(
            decoder_id=decoder_id, split_config=split_config,
            bar_seconds=int(predictions.bar_seconds),
            label_horizon_bars=int(predictions.label_horizon_bars),
            _frame=frame,
        )

    def assert_split_alignment(self, eval_split: tuple[int, int, int]) -> None:
        if tuple(eval_split) != tuple(self.split_config):
            msg = (
                f'split-alignment gate: decoder {self.decoder_id} was trained '
                f'with split={self.split_config} but sweep is evaluating with '
                f'split={eval_split}. Re-train with the same split.'
            )
            raise LookAheadViolation(msg)

    @staticmethod
    def _lookup_validate_args(
        t: datetime, purge_seconds: int, embargo_seconds: int,
    ) -> None:
        """Reject non-UTC `t` and negative purge/embargo values."""
        if t.tzinfo is None or t.utcoffset() is None:
            msg = (
                f'SignalsTable.lookup requires a tz-aware datetime with '
                f'a concrete UTC offset; got {t!r} (tzinfo={t.tzinfo!r}, '
                f'utcoffset={t.utcoffset()!r}). Strategies must construct '
                f'timestamps with an explicit tz (e.g. UTC) so the '
                f'no-look-ahead gate compares apples to apples.'
            )
            raise ValueError(msg)
        if purge_seconds < 0:
            msg = (
                f'SignalsTable.lookup: purge_seconds must be non-negative, '
                f'got {purge_seconds}.'
            )
            raise ValueError(msg)
        if embargo_seconds < 0:
            msg = (
                f'SignalsTable.lookup: embargo_seconds must be non-negative, '
                f'got {embargo_seconds}.'
            )
            raise ValueError(msg)

    def _t_in_allowed_groups(
        self, t: datetime,
        allowed_groups: tuple[int, ...], n_groups: int,
    ) -> bool:
        """Map t -> group_id (Lopez de Prado §11 CSCV) and check membership."""
        if self._frame.is_empty():
            return False
        first_ts = self._frame['timestamp'].min()
        last_ts = self._frame['timestamp'].max()
        span_seconds = (last_ts - first_ts).total_seconds()
        if span_seconds <= 0:
            # Single-row table: every bar is in group 0.
            return 0 in allowed_groups
        position = (t - first_ts).total_seconds() / span_seconds
        # Clamp to [0, n_groups-1]: bars at exactly last_ts otherwise
        # compute group_id == n_groups (out of range).
        group_id = min(max(int(position * n_groups), 0), n_groups - 1)
        return group_id in allowed_groups

    def lookup(
        self,
        t: datetime,
        *,
        allowed_groups: tuple[int, ...] | None = None,
        n_groups: int = 1,
        purge_seconds: int = 0,
        embargo_seconds: int = 0,
    ) -> SignalRow | None:
        """Return the greatest row with `timestamp <= t`, or None if none.

        The single causal accessor for strategies. Naive datetimes are
        rejected loudly — a tz-naive `t` would slip past the
        `t > frozen_now()` guard (frozen_now is tz-aware, so `<` between
        the two is implementation-defined) and then crash the Polars
        filter against the UTC frame schema. Both failure modes hide
        a real lookahead leak from the operator's eye.

        Args:
          t: query timestamp (tz-aware, UTC offset concrete).
          allowed_groups: CPCV group filter. When supplied, `t`'s
            group_id is computed from the table's coverage span
            partitioned into `n_groups` equal-time blocks; the
            lookup returns the row ONLY if `group_id ∈
            allowed_groups`. Bars in any other group return None.
            CSCV callers pass `path.test_groups` to get OOS bars,
            `path.train_groups` to get IS bars (Lopez de Prado §11).
            When `None`, no group filtering — the lookup returns
            the latest causal row regardless.
          n_groups: total partition count for group filtering.
            Required >= 2 when `allowed_groups` is supplied.
            Ignored otherwise.
          purge_seconds: exclude any row whose label horizon
            `[label_t0, label_t1]` extends past `t - purge_seconds`.
            A signal whose label resolves inside the embargo zone
            leaks the test boundary forward. Default 0 (off).
          embargo_seconds: shift the causal cutoff back by this
            many seconds. The returned row satisfies
            `timestamp <= t - embargo_seconds`. Default 0 (off).

        Raises:
          ValueError: if `t.tzinfo is None`, if `purge_seconds < 0`,
            if `embargo_seconds < 0`, or if `allowed_groups` is
            supplied with `n_groups < 2` (single-block partitioning
            has no train/test separation).
          LookAheadViolation: if `t > frozen_now()`. A cheating
            strategy passing `t = current_bar + 1bar` would
            otherwise read a label the live system has not yet
            computed.
        """
        self._lookup_validate_args(t, purge_seconds, embargo_seconds)
        if allowed_groups is not None and n_groups < 2:
            msg = (
                f'SignalsTable.lookup: group filtering requires '
                f'n_groups >= 2, got {n_groups}. Single-block '
                f'partitioning has no train/test separation.'
            )
            raise ValueError(msg)
        now = frozen_now()
        if t > now:
            msg = (
                f'SignalsTable.lookup(t={t}) requested data past '
                f'frozen_now()={now} for decoder {self.decoder_id}'
            )
            raise LookAheadViolation(msg)
        if allowed_groups is not None and not self._t_in_allowed_groups(
            t, allowed_groups, n_groups,
        ):
            return None
        from datetime import timedelta
        cutoff = t - timedelta(seconds=embargo_seconds)
        sliced = self._frame.filter(pl.col('timestamp') <= cutoff)
        if purge_seconds > 0:
            # Drop rows whose label horizon `label_t1` extends into
            # the purge zone `(t - purge_seconds, t]` — those labels
            # leak the post-cutoff window into the lookup answer.
            purge_floor = t - timedelta(seconds=purge_seconds)
            sliced = sliced.filter(pl.col('label_t1') <= purge_floor)
        sliced = sliced.sort('timestamp').tail(1)
        if sliced.is_empty():
            return None
        row = sliced.row(0, named=True)
        # Multi-year staleness drift (codex round 1 P0 — a 2020 table
        # silently feeding a 2026 sweep) is caught at sweep startup
        # by `assert_window_covers`, not here. A per-row staleness
        # guard at lookup conflicts with `purge_seconds`, which
        # intentionally returns older-than-cutoff rows; the right
        # layer is the up-front window check.
        return SignalRow(
            timestamp=row['timestamp'], prob=float(row['prob']),
            pred=int(row['pred']),
            label_t0=row['label_t0'], label_t1=row['label_t1'],
        )

    def assert_window_covers(
        self, window_start: datetime, window_end: datetime,
    ) -> None:
        """Fail loud if the requested replay window falls outside coverage.

        Catches the operator pointing the sweep at a window the
        SignalsTable wasn't built for — e.g. table covers
        2026-04-08 → 2026-04-12 but sweep replays 2026-04-15
        → 2026-04-19. Without this check, every `lookup(t)` would
        silently return None and the strategy would be flat for
        the entire sweep — a no-op dressed up as honest skip.
        """
        if self._frame.is_empty():
            msg = (
                f'SignalsTable.assert_window_covers: table for '
                f'{self.decoder_id} is empty.'
            )
            raise LookAheadViolation(msg)
        first_ts = self._frame['timestamp'].min()
        last_ts = self._frame['timestamp'].max()
        from datetime import timedelta
        max_staleness = timedelta(
            seconds=self.bar_seconds * self.label_horizon_bars,
        )
        # Allow up to one bar of pre-window slack: runtime's
        # PredictLoop timer fires at "next boundary AFTER
        # window_start", so the first SignalsTable row sits one
        # interval AFTER the operator's window_start. The check
        # only fires when the gap is larger than a single bar —
        # that's the multi-year-drift case codex round-1 P0
        # flagged.
        one_bar = timedelta(seconds=self.bar_seconds)
        if window_start + one_bar < first_ts:
            msg = (
                f'SignalsTable.assert_window_covers: window_start='
                f'{window_start} precedes table coverage start='
                f'{first_ts} for {self.decoder_id} by more than one '
                f'bar ({self.bar_seconds}s). Rebuild the table with '
                f'klines that cover the replay window.'
            )
            raise LookAheadViolation(msg)
        if window_end > last_ts + max_staleness:
            msg = (
                f'SignalsTable.assert_window_covers: window_end='
                f'{window_end} exceeds table coverage end='
                f'{last_ts} + max_staleness={max_staleness} for '
                f'{self.decoder_id}. Rebuild with klines that cover '
                f'the full replay window.'
            )
            raise LookAheadViolation(msg)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        parquet_path = directory / f'{self.decoder_id}.parquet'
        metadata_path = directory / f'{self.decoder_id}.meta.json'
        self._frame.write_parquet(parquet_path)
        metadata_path.write_text(json.dumps({
            'decoder_id': self.decoder_id,
            'split_config': list(self.split_config),
            'bar_seconds': int(self.bar_seconds),
            'label_horizon_bars': int(self.label_horizon_bars),
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
        if 'bar_seconds' not in metadata or 'label_horizon_bars' not in metadata:
            msg = (
                f'SignalsTable.load: metadata at {metadata_path} is '
                f'missing bar_seconds or label_horizon_bars. The '
                f'stale-row guard cannot run without them. Rebuild '
                f'the table with the current builder.'
            )
            raise ValueError(msg)
        return cls(
            decoder_id=decoder_id,
            split_config=(
                _to_int(split_config_typed[0]),
                _to_int(split_config_typed[1]),
                _to_int(split_config_typed[2]),
            ),
            bar_seconds=_to_int(metadata['bar_seconds']),
            label_horizon_bars=_to_int(metadata['label_horizon_bars']),
            _frame=frame,
        )
