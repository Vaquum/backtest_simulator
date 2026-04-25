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

from datetime import UTC, datetime
from decimal import Decimal
from typing import Final

STARTING_CAPITAL: Final[Decimal] = Decimal('100000')


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
        self.side = type('_Side', (), {'name': side})()
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
    """Return `(net_pnl, return_pct, r_multiple_or_none)` for one BUY→SELL pair."""
    buy, sell = pair
    qty = buy.qty
    gross = (sell.price - buy.price) * qty
    net = gross - (buy.fee + sell.fee)
    return_pct = (sell.price - buy.price) / buy.price * Decimal('100')
    if declared_stop is None or declared_stop == buy.price:
        r_mult: Decimal | None = None
    else:
        risk = abs(buy.price - declared_stop) * qty
        r_mult = None if risk == 0 else net / risk
    return net, return_pct, r_mult


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
) -> None:
    """One-line headline + per-pair detail."""
    pairs, _trailing = pair_trades(trades)
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
    print(
        f'   perm {perm_id:<4}  {day_label}  '
        f'trades {n_trades:<3}  PF {pf_str:<6}  '
        f'R̄ {r_mean_str:<7}  DD {fmt_dec(-max_dd_pct, 2)}%  '
        f'total {fmt_dec(total_pct, 2)}%',
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
