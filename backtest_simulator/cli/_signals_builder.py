"""Build a `SignalsTable` from a Limen experiment for one decoder.

Slice #17 Task 16. Builds a precomputed cache of EXACTLY the
predictions Nexus's `produce_signal` would emit at runtime — same
recipe (`manifest_full = manifest.with_params_override(split_config=
(1,0,0))` → `prepare_data(causal_window, round_params)` →
`sensor.predict({'x_test': x_train.tail(1)})`), batched across the
sweep's replay window. Strategy code is unchanged ("strategy tested,
strategy deployed"); SignalsTable is the sweep-time analytics
artifact that pins the lookup contract and feeds CPCV PBO.

Every field is real, sourced from the manifest or the runtime Sensor
— no placeholders, no hardcoded windows. Heavy by design (~50 ms per
bar of `prepare_data`, dominated by feature transforms); the cost
scales with the OPERATOR'S `--replay-period` window (sweep-window
bars + feature warmup), not with the experiment's training span.
"""
from __future__ import annotations

from datetime import UTC, datetime
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
    """Subset of the Limen manifest API we exercise here.

    Limen leaves these untyped at runtime; defining the subset
    lets pyright check our usage without importing private
    Limen internals or pulling in `Any`.
    """

    split_config: tuple[int, int, int]
    data_source_config: _DataSourceConfigProtocol | None

    def with_params_override(
        self, *, split_config: tuple[int, int, int],
    ) -> _ManifestProtocol:
        # Protocol stub — implementations supply the body. `del` keeps
        # vulture from flagging the Protocol-required parameters as unused.
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
) -> SignalsTable:
    """Build SignalsTable for one decoder via per-bar runtime replay.

    Args:
        manifest: Limen manifest from the experiment's exp.py. Used
            for split_config (label) + kline_size (bar cadence) +
            the feature-prep recipe (manifest_full).
        sensor: Trained Pass-2 (1,0,0) Sensor returned by
            `Trainer.train([perm_id])[0]`. The SAME object Nexus's
            launcher would wire as the runtime predictor.
        klines: Pre-fetched klines covering at least
            `[earliest_tick - feature_warmup, latest_tick]` at the
            launcher's cadence (`manifest.data_source_config.params
            ['kline_size']`). Caller (sweep) fetches via
            `HistoricalData.get_spot_klines` with `start_date_limit
            = DEFAULT_START_DATE_LIMIT` so the data source matches
            `BacktestMarketDataPoller`'s fetch path byte-for-byte.
        tick_timestamps: The exact timestamps the runtime
            `praxis.PredictLoop` timer would fire at. Computed by
            the sweep from `(days, hours_start, hours_end,
            interval_seconds)` using the same epoch-aligned "next
            boundary after window_start" semantics as
            `launcher/clock.py`. The builder iterates these
            verbatim — it does NOT iterate the kline grid (codex
            round 4 P0: per-day tick schedule != continuous kline
            stream).
        round_params: Hyperparameters for this perm_id (from
            `round_data.jsonl`'s `round_params` field). Drives
            label_horizon_bars (from `shift`) and is passed to
            `prepare_data` so feature transforms get the same
            params they had at training time.
        decoder_id: Identifier carried on the SignalsTable for
            cross-decoder bookkeeping (typically `str(perm_id)`).

    Returns:
        SignalsTable with one row per replay-window bar. `lookup(t)`
        returns the prediction the strategy would have seen at `t`
        if Nexus had emitted it from per-bar predict on the same
        data — which is what the strategy DOES see, byte-for-byte.

    Raises:
        ValueError: if manifest is missing kline_size or split_config;
            if replay_window contains no bars; if every bar gets
            consumed by feature warmup (replay_window too short or
            warmup bars too few in `klines`).
        TypeError: if `round_params['shift']` is not an int (logreg
            requires it).
    """
    cfg = manifest.data_source_config
    if cfg is None or 'kline_size' not in cfg.params:
        msg = (
            'build_signals_table_for_decoder: manifest has no '
            'kline_size in data_source_config; bar_seconds is unknown. '
            'The experiment file must call set_data_source with '
            'kline_size — that is the cadence the launcher uses.'
        )
        raise ValueError(msg)
    bar_seconds = int(cfg.params['kline_size'])
    split_config = (
        int(manifest.split_config[0]),
        int(manifest.split_config[1]),
        int(manifest.split_config[2]),
    )
    # `shift` shifts the target column by N bars; |shift| is the
    # label horizon (purge math in lookup needs it). bool is a
    # subclass of int — reject explicitly so a `True/False` shift
    # doesn't slip through and break label_t1 silently.
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

    # Per-tick replay — Nexus's exact recipe, batched. Match
    # `BacktestMarketDataPoller.get_market_data` byte-for-byte at
    # each tick: causal slice + `tail(POLLER_N_ROWS)` (codex round-3
    # P0). Iterate at the runtime tick instants (codex round-4 P0).
    manifest_full = manifest.with_params_override(split_config=(1, 0, 0))
    timestamps: list[datetime] = []
    preds_list: list[int] = []
    probs_list: list[float] = []
    for raw_tick in tick_timestamps:
        tick = _as_utc_datetime(raw_tick)
        causal = klines.filter(pl.col('datetime') <= tick).tail(POLLER_N_ROWS)
        if causal.is_empty():
            # No klines yet at this tick — runtime poller would
            # return an empty frame and `produce_signal` would emit
            # an empty `signal.values`. Skip honestly.
            continue
        data_dict = manifest_full.prepare_data(causal, round_params)
        x_train_obj = data_dict.get('x_train')
        # Limen's prep returns x_train as a polars DataFrame — narrow
        # explicitly so the .is_empty() / .tail() / .to_numpy() chain
        # is type-safe even though the dict is dict[str, object].
        if not isinstance(x_train_obj, pl.DataFrame) or x_train_obj.is_empty():
            # Feature warmup hasn't filled — Nexus's runtime would
            # also emit no Signal here. Skip honestly.
            continue
        last_x = x_train_obj.tail(1).to_numpy()
        result = sensor.predict({'x_test': last_x})
        preds_list.append(int(result['_preds'][0]))
        probs_list.append(float(result['_probs'][0]))
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
    """Normalise polars datetime to a tz-aware UTC `datetime`."""
    if not isinstance(ts, datetime):
        msg = f'_as_utc_datetime: expected datetime, got {type(ts).__name__}'
        raise TypeError(msg)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts
