"""FillModel — strict-live-reality tape matching; fills at actual tape price, never at declared stop."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import polars as pl

from backtest_simulator.venue.filters import BinanceSpotFilters
from backtest_simulator.venue.types import FillModelConfig, FillResult, PendingOrder

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
) -> list[FillResult]:
    """Match `order` against the historical `trades` stream — strict-live-reality.

    MARKET walks trades from `submit_time + submit_latency` until qty is
    filled OR the tape breaches the order's attached `stop_price`. On
    breach the walk HALTS and only the pre-breach qty fills at its
    actual tape prices; no residual is booked at the declared stop.

    LIMIT goes taker if crossed at submit, otherwise maker-queued; an
    incoming trade later at or beyond the limit fills the maker leg.

    STOP_LOSS / STOP_LOSS_LIMIT / TAKE_PROFIT: the tape triggers the
    stop on the FIRST tick at or past `stop_price`. The fill lands at
    that tick's actual price, which may be at stop or worse (gap
    slippage) — matching live execution.
    """
    submit_ts = order.submit_time + timedelta(milliseconds=config.submit_latency_ms)
    window = trades.filter(pl.col('time') >= submit_ts).sort('time')
    if window.is_empty():
        return []
    if order.order_type == 'MARKET':
        return _walk_market(order, window, filters)
    if order.order_type == 'LIMIT':
        return _walk_limit(order, window, filters)
    if order.order_type in {'STOP_LOSS', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT'}:
        return _walk_stop(order, window, filters)
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
        if stop_price is not None:
            breaches = (
                (order.side == 'BUY' and px <= stop_price)
                or (order.side == 'SELL' and px >= stop_price)
            )
            if breaches:
                # Stop breached mid-walk. Halt — return only what was
                # filled at pre-breach tape prices. The unfilled
                # residual is released (the caller / capital lifecycle
                # must handle release-on-partial-fill).
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


def _walk_limit(order: PendingOrder, window: pl.DataFrame, filters: BinanceSpotFilters) -> list[FillResult]:
    if order.limit_price is None:
        return []
    first_px = Decimal(str(window.row(0, named=True)['price']))
    crossed = (order.side == 'BUY' and first_px <= order.limit_price) or (
        order.side == 'SELL' and first_px >= order.limit_price
    )
    if crossed:
        return _walk_market(order, window, filters)
    for row in window.iter_rows(named=True):
        px = Decimal(str(row['price']))
        cross = (order.side == 'BUY' and px <= order.limit_price) or (
            order.side == 'SELL' and px >= order.limit_price
        )
        if cross:
            return [FillResult(
                fill_time=_ts(row['time']),
                fill_price=filters.round_price(order.limit_price),
                fill_qty=filters.round_qty(order.qty), is_maker=True, reason='limit_maker',
            )]
    return []


def _walk_stop(order: PendingOrder, window: pl.DataFrame, filters: BinanceSpotFilters) -> list[FillResult]:
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
    if order.stop_price is None:
        return []
    stop = order.stop_price
    for row in window.iter_rows(named=True):
        px = Decimal(str(row['price']))
        triggered = (order.side == 'SELL' and px <= stop) or (order.side == 'BUY' and px >= stop)
        if triggered:
            return [FillResult(
                fill_time=_ts(row['time']),
                fill_price=filters.round_price(px),
                fill_qty=filters.round_qty(order.qty), is_maker=False, reason='stop_trigger',
            )]
    return []


def _ts(value: object) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    msg = f'expected datetime in trades stream, got {type(value).__name__}'
    raise TypeError(msg)
