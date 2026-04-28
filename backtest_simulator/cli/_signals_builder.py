"""Build a `SignalsTable` from a Limen experiment for one decoder via the runtime predict-recipe (sweep-time parity reference)."""

# Slice #17 Task 16. Builds a precomputed cache of EXACTLY the
# predictions Nexus's `produce_signal` would emit at runtime — same
# recipe (`manifest_full = manifest.with_params_override(split_config=
# (1,0,0))` → `prepare_data(causal_window, round_params)` →
# `sensor.predict({'x_test': x_train.tail(1)})`), batched across the
# sweep's replay window. Strategy code is unchanged ("strategy tested,
# strategy deployed").
#
# Auditor (post-v2.0.2) "make it real": SignalsTable is now the
# sweep-time PARITY REFERENCE. After every per-window run, the
# deployed strategy's per-tick predictions (captured via Limen
# `Sensor.predict` calls inside the BacktestLauncher) are compared
# against this table; a mismatch raises `ParityViolation`. The
# table feeds operator-side analysis directly because the runtime
# CONFIRMED it matches the deployed predictions tick-by-tick.
#
# Every field is real, sourced from the manifest or the runtime Sensor
# — no placeholders, no hardcoded windows. Heavy by design (~50 ms per
# bar of `prepare_data`, dominated by feature transforms); the cost
# scales with the OPERATOR's `--replay-period` window (sweep-window
# bars + feature warmup), not with the experiment's training span.
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


def assert_signals_parity(
    *, decoder_id: str,
    table: SignalsTable,
    runtime_predictions: list[dict[str, object]],
    tick_timestamps: list[datetime],
) -> int:
    """Verify SignalsTable matches per-tick runtime predictions.

    Auditor (post-v2.0.2) "make it real": SignalsTable was being
    built every sweep without being consumed by any decision metric
    (CPCV moved to deployed-strategy daily returns in v2.0.2). Per
    the Five Principles, sweep-time work that nothing reads is
    ornamentation. The fix turns the table into the SWEEP-TIME
    PARITY REFERENCE: after each per-window run captures Limen
    `Sensor.predict` outputs via the `produce_signal` hook in
    `_run_window`, this function compares them against the
    SignalsTable for the same decoder.

    The comparison is per-tick: for every captured `(timestamp,
    pred)` whose timestamp is IN the table's `tick_timestamps`
    grid, `SignalsTable.lookup(timestamp).pred` must equal the
    captured `pred`. Any mismatch raises `ParityViolation` — the
    deployed strategy is operating on different predictions than
    the sweep-time replay says it should be. That divergence
    violates "strategy tested, strategy deployed" and the
    operator must see it loudly.

    Captured ticks NOT in `tick_timestamps` (e.g. PredictLoop
    timer ticks that fired between scheduled boundaries, or
    post-window ticks if the launcher kept ticking past
    window_end) are silently skipped. `SignalsTable.lookup(t)`
    forward-fills past the last covered row by contract, so we
    cannot rely on `lookup(...) is None` alone to distinguish
    "covered" from "post-window"; the explicit allow-list via
    `tick_timestamps` is the precise gate.

    Returns: the count of TICKS THAT WERE SUCCESSFULLY COMPARED
    (timestamp ∈ tick_timestamps AND pred matched). The sweep
    prints this so the operator sees how many predictions the
    parity check actually validated.
    """
    from backtest_simulator.exceptions import ParityViolation

    covered = set(tick_timestamps)
    n_compared = 0
    for entry in runtime_predictions:
        ts_raw = entry['timestamp']
        if not isinstance(ts_raw, str):
            continue
        ts = datetime.fromisoformat(ts_raw)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        runtime_pred = entry.get('pred')
        if not isinstance(runtime_pred, int):
            continue
        if ts not in covered:
            # Tick is outside the SignalsTable's build-time grid:
            # the table never claimed coverage at this exact
            # instant. Silent — neither match nor violation.
            continue
        row = table.lookup(ts)
        if row is None:
            # In-grid timestamp with no row is a build-side bug
            # (the table failed to write the row it scheduled).
            # Loud: the parity check cannot proceed without the
            # row to compare against.
            msg = (
                f'SignalsTable parity violation for decoder '
                f'{decoder_id!r}: scheduled tick {ts.isoformat()} '
                f'has no row in the table — build skipped this '
                f'instant. Sweep-time replay incomplete; cannot '
                f'compare runtime pred={runtime_pred}.'
            )
            raise ParityViolation(msg)
        if row.pred != runtime_pred:
            msg = (
                f'SignalsTable parity violation for decoder '
                f'{decoder_id!r} at tick {ts.isoformat()}: '
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
    return n_compared
