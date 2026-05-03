"""Per-run metrics + formatting for `bts run` / `bts sweep` output."""

# Pairs BUY -> SELL round trips, computes per-pair (net PnL, return %,
# R multiple), running max-drawdown, profit factor, and the printable
# one-line summary the operator sees per window.
#
# `R multiple` is `(sell-buy)*qty / |buy-stop|*qty` — the strict-live-
# reality measurement: net PnL divided by declared risk (the absolute
# distance from entry to declared stop). `None` when no stop was
# declared on the BUY (defensive — `_check_declared_stop` should have
# rejected it before submission).
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Final

STARTING_CAPITAL: Final[Decimal] = Decimal('100000')


@dataclass(frozen=True, slots=True)
class _Side:
    """Side wrapper with `.name` (matches Praxis's OrderSide.name)."""
    name: str


class Trade:
    """Re-hydrated trade tuple from the subprocess result.

    Lightweight to cross the JSON wire — `_run_window`'s subprocess
    serialises trades to tuples so they survive the asyncio / freezegun
    state stripped at process boundary.
    """

    __slots__ = ('client_order_id', 'fee', 'fee_asset', 'price', 'qty', 'side', 'timestamp')

    def __init__(
        self, coid: str, side: str, qty: str, price: str,
        fee: str, fee_asset: str, ts: str,
    ) -> None:
        self.client_order_id = coid
        self.side = _Side(name=side)
        self.qty = Decimal(qty)
        self.price = Decimal(price)
        self.fee = Decimal(fee)
        self.fee_asset = fee_asset
        self.timestamp = datetime.fromisoformat(ts)


def pair_trades(trades: list[Trade]) -> tuple[list[tuple[Trade, Trade]], list[Trade]]:
    """Pair BUY→SELL round trips. Returns (pairs, trailing_unclosed)."""
    pairs: list[tuple[Trade, Trade]] = []
    open_buy: Trade | None = None
    for t in trades:
        if t.side.name == 'BUY':
            open_buy = t
        elif t.side.name == 'SELL' and open_buy is not None:
            pairs.append((open_buy, t))
            open_buy = None
    return pairs, [open_buy] if open_buy is not None else []


def pair_metrics(
    pair: tuple[Trade, Trade], declared_stop: Decimal | None,
) -> tuple[Decimal, Decimal, Decimal | None]:
    """`(net_pnl, net_return_pct, r_mult|None)` for one BUY→SELL pair.

    `net_return_pct = net / (buy.price * qty) * 100` — net of
    both legs' fees. Coherent accounting basis with PF/R/DD on
    the summary line (zero-bang post-auditor-4 P1).
    """
    buy, sell = pair
    qty = buy.qty
    net = (sell.price - buy.price) * qty - (buy.fee + sell.fee)
    notional = buy.price * qty
    return_pct = Decimal('0') if notional == 0 else net / notional * Decimal('100')
    if declared_stop is None or declared_stop == buy.price:
        return net, return_pct, None
    risk = abs(buy.price - declared_stop) * qty
    return net, return_pct, (None if risk == 0 else net / risk)


def max_drawdown_pct(net_pnls: list[Decimal], capital: Decimal) -> Decimal:
    """Max drawdown as % of starting capital across the equity curve."""
    if not net_pnls:
        return Decimal('0')
    equity = capital
    peak = capital
    max_dd_abs = Decimal('0')
    for pnl in net_pnls:
        equity += pnl
        peak = max(peak, equity)
        dd = peak - equity
        max_dd_abs = max(max_dd_abs, dd)
    return (max_dd_abs / capital) * Decimal('100')


def fmt_dec(value: Decimal, digits: int = 2) -> str:
    sign = '+' if value > 0 else ('' if value == 0 else '-')
    magnitude = abs(value)
    q = Decimal('1').scaleb(-digits)
    return f'{sign}{magnitude.quantize(q)}'


def fmt_price(value: Decimal) -> str:
    return f'{value.quantize(Decimal("0.01"))}'


def print_run(
    perm_id: int, day_label: str,
    trades: list[Trade], declared_stops: dict[str, Decimal],
    *,
    n_intents: int = 0,
    n_fills: int = 0,
    n_pending: int = 0,
    n_rejects: int = 0,
    slippage_cost_bps: Decimal | None = None,
    slippage_n_samples: int = 0,
    slippage_n_excluded: int = 0,
    slippage_predict_vs_realised_gap_bps: Decimal | None = None,
    slippage_n_uncalibrated_predict: int = 0,
    slippage_n_predicted_samples: int = 0,
    n_limit_orders_submitted: int = 0,
    n_limit_filled_full: int = 0,
    n_limit_filled_partial: int = 0,
    n_limit_filled_zero: int = 0,
    n_limit_marketable_taker: int = 0,
    maker_fill_efficiency_p50: Decimal | None = None,
    market_impact_realised_bps: Decimal | None = None,
    market_impact_n_samples: int = 0,
    market_impact_n_flagged: int = 0,
    market_impact_n_uncalibrated: int = 0,
    n_atr_rejected: int = 0,
    n_atr_uncalibrated: int = 0,
) -> None:
    """One-line headline + per-pair detail.

    `slippage_cost_bps` is the side-normalized realised cost from
    `SimulatedVenueAdapter.slippage_realised_cost_bps` — positive
    means the run paid spread on average, negative means it
    captured price improvement, None means no slippage model was
    attached. The headline includes a `slip` column so the
    operator on `bts sweep` (the load-bearing surface) sees what
    the run paid relative to mid.

    `slippage_predict_vs_realised_gap_bps` is the calibration-
    loop signal: realised cost minus predicted cost averaged
    across measured fills. Zero = calibration matches reality;
    large = recalibrate. The headline appends `gap <±bp>` after
    the slip column when the gap is non-None.
    """
    pairs, trailing = pair_trades(trades)
    n_open = len(trailing)
    net_pnls: list[Decimal] = []
    return_pcts: list[Decimal] = []
    r_mults: list[Decimal] = []
    winners: list[Decimal] = []
    losers: list[Decimal] = []
    for buy, sell in pairs:
        declared = declared_stops.get(buy.client_order_id)
        net, ret_pct, r_mult = pair_metrics((buy, sell), declared)
        net_pnls.append(net)
        return_pcts.append(ret_pct)
        if r_mult is not None:
            r_mults.append(r_mult)
        if net > 0:
            winners.append(net)
        elif net < 0:
            losers.append(-net)
    n_trades = len(pairs)
    total_pct = sum(return_pcts, Decimal('0'))
    max_dd_pct = max_drawdown_pct(net_pnls, STARTING_CAPITAL)
    if losers:
        pf = sum(winners, Decimal('0')) / sum(losers, Decimal('0'))
        pf_str = f'{pf.quantize(Decimal("0.01"))}'
    elif winners:
        pf_str = 'inf'
    else:
        pf_str = '—'
    r_mean_str = (
        fmt_dec(sum(r_mults, Decimal('0')) / len(r_mults), 2)
        if r_mults else '—'
    )
    if slippage_cost_bps is None:
        slip_str = 'slip off'
    else:
        slip_str = (
            f'slip {fmt_dec(slippage_cost_bps, 2)}bp '
            f'n={slippage_n_samples}/excl={slippage_n_excluded}'
        )
        # Only show the gap when there's actually a calibration
        # signal — at least one fill where both realised AND
        # predicted succeeded. Otherwise `gap 0.00bp` would
        # falsely imply "calibration matched reality" when the
        # truth is "no signal." Codex / auditor pinned this.
        if slippage_predict_vs_realised_gap_bps is not None and (
            slippage_n_predicted_samples > 0
        ):
            slip_str += (
                f' gap {fmt_dec(slippage_predict_vs_realised_gap_bps, 2)}bp'
                f' (n_predict={slippage_n_predicted_samples}'
            )
            if slippage_n_uncalibrated_predict > 0:
                slip_str += f'/uncal={slippage_n_uncalibrated_predict}'
            slip_str += ')'
        elif slippage_n_uncalibrated_predict > 0:
            # No successful predictions at all — operator must see
            # this distinct from "gap was zero".
            slip_str += (
                f' gap n/a (uncal={slippage_n_uncalibrated_predict})'
            )
    if n_limit_orders_submitted > 0:
        eff_str = (
            'n/a' if maker_fill_efficiency_p50 is None
            else f'{(maker_fill_efficiency_p50 * Decimal("100")).quantize(Decimal("0.1"))}%'
        )
        maker_str = (
            f'  maker n={n_limit_orders_submitted} '
            f'full/part/zero={n_limit_filled_full}/'
            f'{n_limit_filled_partial}/{n_limit_filled_zero}  '
            f'mkt_taker={n_limit_marketable_taker}  '
            f'eff_p50={eff_str}'
        )
    else:
        maker_str = ''
    # Market impact column: only render when the model surfaced
    # at least one calibrated sample OR an uncalibrated submit;
    # otherwise the model wasn't attached / didn't see any
    # orders this run. Format: `imp <bps>bp n=<N>/flagged=<F>`,
    # plus `/uncal=<U>` when there was a calibration gap.
    if (
        market_impact_realised_bps is not None
        and (market_impact_n_samples > 0
             or market_impact_n_uncalibrated > 0)
    ):
        impact_core = (
            f'  imp {fmt_dec(market_impact_realised_bps, 2)}bp '
            f'n={market_impact_n_samples}/flagged={market_impact_n_flagged}'
        )
        if market_impact_n_uncalibrated > 0:
            impact_core += f'/uncal={market_impact_n_uncalibrated}'
        impact_str = impact_core
    else:
        impact_str = ''
    # ATR R-denominator gameability gate (slice #17 Task 29).
    # Only surfaces when at least one ENTER+BUY was denied;
    # silent on healthy runs where the strategy's stops sit
    # comfortably above `k * ATR(window)`.
    if n_atr_rejected > 0 or n_atr_uncalibrated > 0:
        atr_str = f'  atr_rej={n_atr_rejected}/uncal={n_atr_uncalibrated}'
    else:
        atr_str = ''
    # `trades N` counts CLOSED BUY->SELL round trips. A day with
    # order activity that did not close a round trip used to read
    # as `trades 0` and was indistinguishable from a genuinely flat
    # day. The parenthetical extras below answer the trader's
    # five-question scan, ordered by lifecycle stage:
    #   intents  → did my strategy decide to act?
    #   fills    → did money move?
    #   pending  → what's still hanging at window close?
    #   rejects  → what got blocked or expired?
    #   +N open  → unmatched filled inventory (BUY without closing SELL)
    # Only non-zero counts are rendered to keep the headline tight.
    activity_extras: list[str] = []
    if n_intents > 0:
        activity_extras.append(f'intents {n_intents}')
    if n_fills > 0:
        activity_extras.append(f'fills {n_fills}')
    if n_pending > 0:
        activity_extras.append(f'pending {n_pending}')
    if n_rejects > 0:
        activity_extras.append(f'rejects {n_rejects}')
    if n_open > 0:
        activity_extras.append(f'+{n_open} open')
    activity_str = (
        f' ({", ".join(activity_extras)})' if activity_extras else ''
    )
    print(
        f'   perm {perm_id:<4}  {day_label}  '
        f'trades {n_trades}{activity_str}  PF {pf_str:<6}  '
        f'R̄ {r_mean_str:<7}  DD {fmt_dec(-max_dd_pct, 2)}%  '
        f'total {fmt_dec(total_pct, 2)}%  '
        f'{slip_str}'
        f'{maker_str}'
        f'{impact_str}'
        f'{atr_str}',
    )
    for buy, sell in pairs:
        declared = declared_stops.get(buy.client_order_id)
        _net, ret_pct, r_mult = pair_metrics((buy, sell), declared)
        in_tm = buy.timestamp.astimezone(UTC).strftime('%H:%M')
        out_tm = sell.timestamp.astimezone(UTC).strftime('%H:%M')
        r_str = f'R {fmt_dec(r_mult, 2)}' if r_mult is not None else 'R —'
        print(
            f'     {in_tm} → {out_tm}   '
            f'BUY {fmt_price(buy.price):>10} → '
            f'SELL {fmt_price(sell.price):>10}   '
            f'{fmt_dec(ret_pct, 2)}%   {r_str}',
        )
