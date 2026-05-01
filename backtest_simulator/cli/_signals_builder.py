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
    predict_lookback: int | None = None,
    n_rows: int | None = None,
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
    bar_seconds = int(str(cfg.params['kline_size']))
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
    # each tick: causal slice + `tail(n_rows)` where `n_rows`
    # comes from the bundle's `data_source.params` (falls back to
    # POLLER_N_ROWS if the bundle didn't declare it). Iterate at
    # the runtime tick instants (codex round-4 P0).
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
        last_x = x_train_obj.tail(effective_lookback).to_numpy()
        result = sensor.predict({'x_test': last_x})
        # Mirror Nexus's `_extract_values`: take the last element of
        # the predict output array, which corresponds to the current
        # tick. With effective_lookback==1 this reduces to the legacy
        # single-row behaviour.
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
    """Normalise polars datetime to a tz-aware UTC `datetime`."""
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
    """Find the unique `expected_tick e` such that `e <= ts < e + interval_seconds`.

    The PredictLoop's Timer fires at an exact `expected_tick` boundary,
    but the strategy's signal carries `signal.timestamp = datetime.now
    (UTC)` at signal-construction time — which lands AFTER the boundary
    by the launcher's main-loop callback drift (documented in
    `launcher.py:_advance_clock_until` as "~10 min per main tick").
    Each runtime tick belongs to the unique expected_tick whose half-
    open `[e, e + interval_seconds)` window contains it.

    Returns the matching `e` on success; `None` if no expected_tick is
    within `interval_seconds` of `ts` (drift > one interval is a real
    cadence divergence, not just callback jitter).
    """
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

    Auditor (post-v2.0.3): the prior round let `runtime_predictions=
    []` reach the helper as a no-op (return 0). The sweep summary
    then printed "no comparisons made" cheerfully — the mandatory
    parity check could silently NOT RUN if the capture hook broke
    or the subprocess result was missing the `runtime_predictions`
    payload. Now ZERO comparisons is a `ParityViolation`.

    Codex (post-auditor-4) P1: the prior round ALSO silently
    skipped malformed entries (non-string timestamp, non-int pred)
    and out-of-grid timestamps. Codex repro:
    `runtime=[valid match, out-of-grid pred=99]` returned `1` with
    no violation, so a partial capture-hook failure (one bad row
    among many good) was silently OK. Fix: ANY skipped entry
    raises. The operator either gets a clean run (every entry
    comparable + matched) or a loud violation naming what went
    wrong.

    Returns: the count of TICKS THAT WERE SUCCESSFULLY COMPARED.
    Equal to `len(runtime_predictions)` on success — every entry
    matched. Anything less raises.
    """
    from backtest_simulator.exceptions import ParityViolation

    # Codex (post-auditor-4 round-2) P1: parity must be a TWO-WAY
    # multiset match. Every expected tick MUST appear exactly once
    # in `runtime_predictions`. Missing ticks (capture skipped a
    # boundary), duplicate ticks (PredictLoop double-fired), and
    # extra/out-of-grid ticks all raise. Operator-side caller passes
    # PER-WINDOW expected ticks, NOT the whole sweep grid.
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
            # Codex (post-auditor-4) P1: non-string timestamp is
            # a capture-side serialiser bug — the entry can't be
            # compared against the table. Loud, not silent.
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
        # Snap the runtime tick down to its anchoring expected_tick.
        # Why: the PredictLoop's Timer fires at `target` (an exact
        # `expected_tick` boundary), but the strategy's signal then
        # carries `signal.timestamp = datetime.now(UTC)` at the moment
        # of signal-construction — which lands AFTER `target` by the
        # launcher's main-loop callback drift (the documented
        # "~10 min per main tick" in `launcher.py:_advance_clock_until`).
        # The runtime tick belongs to the unique `expected_tick e`
        # such that `e <= ts < e + interval_seconds`. Drifts BEYOND
        # one interval are real bugs and still raise.
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
            # In-grid timestamp with no row is a build-side bug
            # (the table failed to write the row it scheduled).
            # Loud: the parity check cannot proceed without the
            # row to compare against.
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
    # Codex (post-auditor-4 round-2) P1: two-way multiset check.
    # Every expected tick MUST appear in `captured` exactly once.
    # Missing → capture-side skipped a scheduled boundary.
    # Duplicate → PredictLoop double-fired (or capture-side
    # double-emitted). Both are honesty violations — neither is
    # a "valid" PredictLoop output the sweep should accept.
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
        # Belt-and-braces: even after the multiset check, if
        # neither runtime_predictions nor expected_ticks contained
        # comparable entries, the parity body did not run. The
        # caller (sweep) should never pass empty expected_ticks;
        # but if it does, this raise surfaces it.
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
