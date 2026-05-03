"""FillModel — strict-live-reality tape matching; fills at actual tape price, never at declared stop."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

import polars as pl

from backtest_simulator.venue.filters import BinanceSpotFilters
from backtest_simulator.venue.types import FillModelConfig, FillResult, PendingOrder

if TYPE_CHECKING:
    from backtest_simulator.honesty.book_gap import BookGapInstrument
    from backtest_simulator.honesty.maker_fill import MakerFillModel


@dataclass(frozen=True)
class WalkContext:
    """Optional auxiliary inputs for `walk_trades`.

    Bundles the three optional kw-only parameters (maker model,
    pre-submit trades for queue calibration, book-gap instrument)
    so `walk_trades` keeps a 5-argument surface — ruff's PLR0913
    cap. Every field defaults to None so callers that don't need
    a specific feature can omit it cleanly.
    """

    maker_model: MakerFillModel | None = None
    trades_pre_submit: pl.DataFrame | None = None
    book_gap_instrument: BookGapInstrument | None = None

# Governing principle (from PR #15 body): backtest ≡ paper ≡ live. One
# methodology. When the three environments disagree, the one that matches
# real market reality wins. Fills reflect actual tape execution, not
# strategy intent — the declared stop is a *measurement unit* for the R
# denominator, never a *promise* about where fills land.
#
# Concretely:
#   - MARKET entry walks the tape. If the tape breaches the order's
#     attached `stop_price` mid-walk, the walk HALTS. Residual qty is
#     never booked at the declared stop (that would be a phantom fill
#     at a price the tape did not offer). Partial fill is returned as-is.
#   - STOP_LOSS / STOP_LOSS_LIMIT / TAKE_PROFIT orders fill at the FIRST
#     tape tick at or past `stop_price` — at the tick's ACTUAL price,
#     not at the declared stop. A gapping tape produces gap slippage
#     (fill worse than stop), as it does in live trading.
#
# R denominator stays clean (|entry - declared_stop| * qty is definitional
# and unaffected). R numerator reflects real market execution: a strategy's
# R distribution is meaningful only if stopped trades are allowed to land
# worse than -1R on gapping markets — which is the empirical truth.


def walk_trades(
    order: PendingOrder,
    trades: pl.DataFrame,
    config: FillModelConfig,
    filters: BinanceSpotFilters,
    context: WalkContext | None = None,
) -> list[FillResult]:
    """Match `order` against the historical `trades` stream — strict-live-reality.

    MARKET walks trades from `submit_time + submit_latency` until qty is
    filled OR the tape breaches the order's attached `stop_price`. On
    breach the walk HALTS and only the pre-breach qty fills at its
    actual tape prices; no residual is booked at the declared stop.

    LIMIT — when `maker_model` is supplied, the maker engine walks the
    queue against post-submit aggressors and emits fills as queue
    position depletes. The optional `trades_pre_submit` slice (from
    the same widened venue fetch) seeds the maker's initial queue
    position from `[submit_time - lookback_minutes, submit_time)` —
    the venue derives this slice fresh per submit so an order
    submitted hours after `window_start` still has a live lookback.
    When `maker_model` is None, the legacy O(1) "first crossing tick
    = full fill at limit price" path runs.

    STOP_LOSS / STOP_LOSS_LIMIT / TAKE_PROFIT: the tape triggers the
    stop on the FIRST tick at or past `stop_price`. The fill lands at
    that tick's actual price, which may be at stop or worse (gap
    slippage) — matching live execution.
    """
    submit_ts = order.submit_time + timedelta(milliseconds=config.submit_latency_ms)
    window = trades.filter(pl.col('time') >= submit_ts).sort('time')
    if window.is_empty():
        return []
    ctx = context if context is not None else WalkContext()
    if order.order_type == 'MARKET':
        return _walk_market(order, window, filters)
    if order.order_type == 'LIMIT':
        return _walk_limit(
            order, window, filters,
            maker_model=ctx.maker_model,
            trades_pre_submit=ctx.trades_pre_submit,
        )
    if order.order_type in {'STOP_LOSS', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT'}:
        return _walk_stop(order, window, filters, ctx.book_gap_instrument)
    return []


def _walk_market(order: PendingOrder, window: pl.DataFrame, filters: BinanceSpotFilters) -> list[FillResult]:
    # The tape walk aggregates into ONE VWAP fill rather than emitting
    # one FillResult per consumed tape tick. Real venues return a
    # handful of fills for a typical MARKET order (one per price level
    # hit on the book), not hundreds; emitting one per tape tick
    # overstates fill granularity AND every FillResult generates a
    # `FillReceived` event_spine append in Praxis (283 appends per
    # order blows past any reasonable drain budget). Aggregating keeps
    # the backtest honest on total cost/qty (VWAP preserves both)
    # while staying realistic on fill-event shape.
    #
    # Declared-stop safety (live-reality semantics): if the order
    # carries a `stop_price`, the walk monitors each tape tick for a
    # breach. A breach means the market has already moved past the
    # strategy's declared stop during the entry window. The walk
    # HALTS on breach — we return only the pre-breach partial fill at
    # its actual tape prices. NO residual is booked at the declared
    # stop; that would be a phantom fill at a price the tape did not
    # offer, which live execution cannot reproduce. The strategy is
    # expected to observe the partial fill and emit its own close
    # action (or the attached protective-stop layer will fire a
    # separate STOP_LOSS SELL at the next tick past stop_price).
    #
    # Effect on R: the R denominator (|entry - declared_stop| * qty)
    # is definitional and unchanged. The R numerator reflects the
    # actual market move, which means a gapping-through-stop trade
    # CAN produce R worse than -1 — and that is correct. The
    # previous "residual at stop" model forced R to exactly -1 on
    # every stopped trade, hiding gap risk from the R distribution.
    stop_price = order.stop_price
    remaining = order.qty
    consumed_qty = Decimal('0')
    consumed_notional = Decimal('0')
    last_time: datetime | None = None
    stop_halted = False
    for row in window.iter_rows(named=True):
        if remaining <= 0:
            break
        px = Decimal(str(row['price']))
        # Stop-breach is a MID-WALK halt that models gap-through-stop
        # during entry consumption: the strategy's declared stop
        # (anchored to the seed price baked at window_start) is
        # crossed AFTER some qty has already filled, so the residual
        # is left unbooked. Pre-fix this branch fired on the FIRST
        # tape tick whenever the seed-anchored stop sat above the
        # current market — i.e., when the price had merely DRIFTED
        # between window_start and submit_time (typical for any
        # multi-hour window). The strategy never entered, the venue
        # silently returned zero fills, and Praxis's
        # TradeOutcomeProduced=PENDING fallback (2-min timeout)
        # surfaced the phantom intent the operator hit on perm-130
        # 04-17 / 04-24. Real-world MARKET BUYs gap-fill at the
        # current tape price even when that price is already past
        # the strategy's stale stop reference; the protective stop
        # only matters as an EXIT trigger after entry. Gate the
        # breach halt on `consumed_qty > 0` so the FIRST tick always
        # enters and only LATER ticks (a true intra-walk dip) halt.
        if stop_price is not None and consumed_qty > 0:
            breaches = (
                (order.side == 'BUY' and px <= stop_price)
                or (order.side == 'SELL' and px >= stop_price)
            )
            if breaches:
                stop_halted = True
                break
        take = min(remaining, Decimal(str(row['qty'])))
        consumed_qty += take
        consumed_notional += take * px
        last_time = _ts(row['time'])
        remaining -= take
    if consumed_qty == 0 or last_time is None:
        return []
    vwap = consumed_notional / consumed_qty
    reason = 'market_stop_halted' if stop_halted else 'market_vwap'
    return [FillResult(
        fill_time=last_time,
        fill_price=filters.round_price(vwap),
        fill_qty=filters.round_qty(consumed_qty),
        is_maker=False, reason=reason,
    )]


def _walk_limit(
    order: PendingOrder,
    window: pl.DataFrame,
    filters: BinanceSpotFilters,
    *,
    maker_model: MakerFillModel | None = None,
    trades_pre_submit: pl.DataFrame | None = None,
) -> list[FillResult]:
    """LIMIT fill path.

    With `maker_model` attached: ALWAYS route through the maker
    engine. The engine's queue-position logic decides whether the
    first crossing aggressor fills us as maker (queue=0 → fill at
    our limit) or doesn't (queue still depleting). Pre-fix, this
    function decided "marketable at submit" from `window.row(0)` —
    but for a passive maker pinned at touch, the very first
    crossing print IS the aggressor that should test our queue, not
    a sign the limit was already through the book. Misrouting
    those to taker booked taker VWAP/fees and zeroed maker
    telemetry on the most common case (touch-pinned LIMIT meeting
    its first SELL aggressor at touch). Codex P2 pinned this.

    Without `maker_model`: the legacy "marketable at first tick →
    taker, otherwise full-fill at limit on first cross" path. This
    is the unrealistic ceiling that the maker engine is wired to
    replace, but is preserved for sites that don't supply the
    model (tests, calls without explicit calibration).

    `trades_pre_submit` is the same-fetch pre-submit slice the venue
    derives from the widened `get_trades_for_venue` query. It
    seeds the maker engine's initial queue position from
    `[submit_time - lookback_minutes, submit_time)`. None falls
    through to the model's stored calibration tape (best-case maker).
    """
    if order.limit_price is None:
        return []
    if maker_model is None:
        return _walk_limit_legacy(order, window, filters)
    # Realistic maker path: rename the venue-feed columns to the
    # MakerFillModel contract (`time→datetime`, `qty→quantity`),
    # walk the queue, convert ImmediateFill list → ONE FillResult.
    # Codex P1 caught the prior shape: emitting one FillResult per
    # ImmediateFill produced multiple `BUY` trade records for a
    # single LIMIT order, and `pair_trades` (cli/_metrics.py)
    # overwrites `open_buy` each BUY before the closing SELL,
    # silently dropping all but the last partial from sweep PnL.
    # `_walk_market` aggregates its multi-tick fills into one
    # VWAP'd FillResult for the same reason — match the shape
    # here so the pairing logic sees one entry per order.
    renamed = window.rename({'time': 'datetime', 'qty': 'quantity'})
    pre = (
        None if trades_pre_submit is None or trades_pre_submit.is_empty()
        else trades_pre_submit.rename({'time': 'datetime', 'qty': 'quantity'})
    )
    immediates = maker_model.evaluate(
        order=order, trades_in_window=renamed,
        trades_pre_submit=pre,
    )
    if not immediates:
        return []
    total_qty = sum((imm.fill_qty for imm in immediates), Decimal('0'))
    if total_qty <= Decimal('0'):
        return []
    total_notional = sum(
        (imm.fill_qty * imm.fill_price for imm in immediates),
        Decimal('0'),
    )
    vwap_price = total_notional / total_qty
    return [FillResult(
        fill_time=immediates[-1].fill_time,
        fill_price=filters.round_price(vwap_price),
        fill_qty=filters.round_qty(total_qty),
        is_maker=True, reason='limit_maker',
    )]


def _walk_limit_legacy(
    order: PendingOrder,
    window: pl.DataFrame,
    filters: BinanceSpotFilters,
) -> list[FillResult]:
    """Legacy LIMIT path: marketable→taker, else first-crossing full-fill."""
    if order.limit_price is None:
        return []
    first_px = Decimal(str(window.row(0, named=True)['price']))
    crossed = (order.side == 'BUY' and first_px <= order.limit_price) or (
        order.side == 'SELL' and first_px >= order.limit_price
    )
    if crossed:
        taker_fills = _walk_market(order, window, filters)
        return [
            FillResult(
                fill_time=f.fill_time, fill_price=f.fill_price,
                fill_qty=f.fill_qty, is_maker=False,
                reason='limit_taker',
            )
            for f in taker_fills
        ]
    for row in window.iter_rows(named=True):
        px = Decimal(str(row['price']))
        cross = (order.side == 'BUY' and px <= order.limit_price) or (
            order.side == 'SELL' and px >= order.limit_price
        )
        if cross:
            return [FillResult(
                fill_time=_ts(row['time']),
                fill_price=filters.round_price(order.limit_price),
                fill_qty=filters.round_qty(order.qty),
                is_maker=True, reason='limit_maker',
            )]
    return []


def _walk_stop(
    order: PendingOrder,
    window: pl.DataFrame,
    filters: BinanceSpotFilters,
    book_gap_instrument: BookGapInstrument | None = None,
) -> list[FillResult]:
    # STOP_LOSS / STOP_LOSS_LIMIT / TAKE_PROFIT: the tape triggers the
    # stop on the first tick at or past `stop_price`. The fill lands
    # at THAT TICK'S ACTUAL PRICE — not at `stop_price` — because
    # that is what live execution does. If the tape gapped through
    # stop (sudden large move), the tick price is materially worse
    # than stop and the strategy realises gap slippage. Modelling the
    # fill at stop price would silently hide that realised slippage
    # from the R distribution.
    #
    # STOP_LOSS_LIMIT semantic: after trigger, the order converts to
    # a LIMIT at `limit_price`. The current implementation treats it
    # the same as STOP_LOSS (fill at first-tick price) — strict LIMIT
    # semantics after trigger are a follow-up refinement. Conservative
    # choice: same fill path, same realism.
    #
    # Slice #17 Task 11 — book-gap instrumentation. When a
    # `book_gap_instrument` is supplied, record the gap between the
    # last sub-stop tape tick (`prev_time`) and the trigger tick
    # (`row_time`). On a first-row trigger (no prior sub-stop tick),
    # `t_cross = t_first_trade` and the recorded gap is 0 — the
    # primitive's `n_stops_observed` still counts that as one
    # observation (codex round 1 P1: dropping zero-gap samples
    # would make `n_stops_observed` mean something other than
    # "number of stop triggers observed").
    if order.stop_price is None:
        return []
    stop = order.stop_price
    prev_time: datetime | None = None
    for row in window.iter_rows(named=True):
        px = Decimal(str(row['price']))
        triggered = (order.side == 'SELL' and px <= stop) or (order.side == 'BUY' and px >= stop)
        if triggered:
            row_time = _ts(row['time'])
            if book_gap_instrument is not None:
                book_gap_instrument.record_stop_cross(
                    t_cross=prev_time if prev_time is not None else row_time,
                    t_first_trade=row_time,
                )
            return [FillResult(
                fill_time=row_time,
                fill_price=filters.round_price(px),
                fill_qty=filters.round_qty(order.qty), is_maker=False, reason='stop_trigger',
            )]
        prev_time = _ts(row['time'])
    return []


def _ts(value: object) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    msg = f'expected datetime in trades stream, got {type(value).__name__}'
    raise TypeError(msg)
