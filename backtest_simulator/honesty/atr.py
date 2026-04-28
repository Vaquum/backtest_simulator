"""ATR-based stop-distance gate — rejects R-denominator gameability."""
from __future__ import annotations

# Slice #17 Task 10: every ENTER-with-stop must satisfy
# `|entry - declared_stop| >= k * ATR(window)`. A strategy that
# declares `stop = entry * 0.9999` (1 bp distance) drives R to zero
# and inflates every realised return measured in R units; the
# operator sees "10R win!" on a 0.1% return. The gate compares
# declared stop distance against `k` of one local ATR and returns a
# loud allowed/reason pair.
#
# ATR is supplied externally — the gate does no IO, no clock reads,
# no feed lookups. Production wiring computes ATR from the feed's
# trade tape over `atr_window_seconds` ending at the action's
# decision time; tests inject ATR directly. Keeping the gate
# state-free makes it cheap to call inline in `validation_pipeline`
# without threading a feed handle through.
from dataclasses import dataclass
from decimal import Decimal

import polars as pl


def compute_atr_from_tape(
    *, trades_pre_decision: pl.DataFrame, period_seconds: int = 60,
) -> Decimal | None:
    """ATR = mean of true range per period over the supplied tape.

    `trades_pre_decision` must carry `time` (Datetime) and `price`
    columns. Caller owns the strict-causal contract (slice ends
    BEFORE decision time). `period_seconds` controls the bucket
    width — 60 = 1-minute periods, the classic ATR convention.

    True range per bucket (Wilder's ATR):
      TR_i = max(H_i - L_i, |H_i - C_{i-1}|, |L_i - C_{i-1}|)

    where `C_{i-1}` is the previous bucket's last (close) price.
    The first bucket has no `prev_close`; TR_0 = H_0 - L_0. ATR
    = mean(TR_i). Without the `|H - prev_close|` / `|L -
    prev_close|` arms, a tape that gaps BETWEEN buckets but
    stays tight WITHIN each bucket would understate volatility,
    which directly weakens the R-denominator floor — auditor
    round 2 P1 caught this on the first ship. Mutation proof:
    `test_compute_atr_from_tape_gap_between_buckets` exercises a
    tape with intra-bucket range=1 and bucket-to-bucket gap=10;
    the old (range-only) impl returned 1, true range returns
    >=5 for an honest floor.

    Returns `None` for empty / no-bucket cases — the
    "uncalibrated" signal. `Decimal('0')` for a flat tape with
    no movement (gate's own `atr_zero` path rejects).
    """
    if trades_pre_decision.is_empty():
        return None
    sorted_trades = trades_pre_decision.sort('time')
    bucketed = sorted_trades.with_columns(
        pl.col('time').dt.truncate(f'{period_seconds}s').alias('_bucket'),
    )
    agg = bucketed.group_by('_bucket', maintain_order=True).agg(
        pl.col('price').max().alias('_high'),
        pl.col('price').min().alias('_low'),
        pl.col('price').last().alias('_close'),
    ).sort('_bucket')
    if agg.is_empty():
        return None
    true_ranges: list[Decimal] = []
    prev_close: Decimal | None = None
    for row in agg.iter_rows(named=True):
        # Auditor (post-v2.0.3): renamed `l` -> `low` to satisfy
        # ruff E741 ambiguous-variable rule (lowercase L vs digit
        # 1). Same for the H/L/C trio kept side-by-side for
        # readability of the Wilder TR formula.
        high = Decimal(str(row['_high']))
        low = Decimal(str(row['_low']))
        close = Decimal(str(row['_close']))
        if prev_close is None:
            tr = high - low
        else:
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
        true_ranges.append(tr)
        prev_close = close
    if not true_ranges:
        return None
    return sum(true_ranges, Decimal('0')) / Decimal(str(len(true_ranges)))


@dataclass(frozen=True)
class AtrSanityDecision:
    """Outcome of `AtrSanityGate.evaluate` — allowed/rejected with context.

    `allowed=True` means the declared stop is at least `k * atr` away
    from entry. `allowed=False` carries `reason` (string identifier,
    not free-form prose) and the two distances the caller can
    surface to the operator.
    """

    allowed: bool
    reason: str | None
    stop_distance: Decimal
    min_required_distance: Decimal


class AtrSanityGate:
    """Reject ENTER-with-stop where `|entry - stop| < k * atr`.

    Parameters
    ----------
    atr_window_seconds:
        How many seconds of trade tape the caller's ATR computation
        looked back over. The gate stores it for diagnostics only —
        the gate itself takes ATR as a `Decimal` argument and does
        not recompute.
    k:
        Multiplier on ATR that defines the minimum acceptable stop
        distance. `k=0.5` means "stop must be at least half a local
        ATR away from entry". Lower `k` admits tighter stops; `k=0`
        disables the gate entirely (stop_distance >= 0 always
        holds).
    """

    def __init__(self, *, atr_window_seconds: int, k: Decimal) -> None:
        if k < Decimal('0'):
            # `k < 0` would make `min_required = k * atr` negative for
            # any positive ATR, so `stop_distance >= negative` admits
            # every stop and the honesty contract evaporates. Reject
            # at construction so the misconfiguration is surfaced at
            # the operator's wiring site rather than silently
            # disabling the gate at every evaluate call. Codex Task 10
            # round 2 pinned this gap.
            msg = (
                f'AtrSanityGate: k must be non-negative '
                f'(got {k}). k=0 disables the gate; k<0 inverts the '
                f'threshold and silently admits every stop, which is '
                f'the very gameability vector the gate exists to close.'
            )
            raise ValueError(msg)
        self._atr_window_seconds = atr_window_seconds
        self._k = k

    @property
    def atr_window_seconds(self) -> int:
        return self._atr_window_seconds

    @property
    def k(self) -> Decimal:
        return self._k

    def evaluate(
        self,
        *,
        entry_price: Decimal,
        stop_price: Decimal,
        atr: Decimal,
    ) -> AtrSanityDecision:
        stop_distance = abs(entry_price - stop_price)
        if self._k == Decimal('0'):
            # Gate is disabled — never deny, even on flat/missing/
            # signalling-NaN/Infinity ATR. The operator who
            # explicitly turned the gate off has accepted full
            # responsibility for stop sizing; the gate's job is to
            # step out of the way without imposing ATR validation
            # as a side-effect. The short-circuit MUST come BEFORE
            # `self._k * atr` so a signalling-NaN/Infinity ATR does
            # not raise InvalidOperation and crash the disabled
            # path. Codex Task 10 rounds 3 + 4 pinned both layers
            # of this contract.
            return AtrSanityDecision(
                allowed=True,
                reason=None,
                stop_distance=stop_distance,
                min_required_distance=Decimal('0'),
            )
        min_required = self._k * atr
        if atr <= Decimal('0'):
            # ATR <= 0 covers two failure modes:
            #   - atr == 0: flat tape / empty window.
            #   - atr  < 0: bad upstream ATR feed (a bug, but the gate
            #     must not silently disable itself when the threshold
            #     `k * atr` goes negative — every stop_distance >= 0
            #     would pass and the honesty contract evaporates).
            # Reject loudly so the caller fixes the input rather than
            # quietly running with a degenerate or inverted threshold.
            # Codex Task 10 round 1 pinned the negative-ATR vector.
            reason = 'atr_zero' if atr == Decimal('0') else 'atr_negative'
            return AtrSanityDecision(
                allowed=False,
                reason=reason,
                stop_distance=stop_distance,
                min_required_distance=min_required,
            )
        if stop_distance < min_required:
            return AtrSanityDecision(
                allowed=False,
                reason='stop_tighter_than_min_atr_fraction',
                stop_distance=stop_distance,
                min_required_distance=min_required,
            )
        return AtrSanityDecision(
            allowed=True,
            reason=None,
            stop_distance=stop_distance,
            min_required_distance=min_required,
        )
