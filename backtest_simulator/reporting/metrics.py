"""R/trade, profit factor, sum PnL — SPEC §19 #1 option (a): stop-on-entry."""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

from backtest_simulator.exceptions import StopContractViolation


@dataclass(frozen=True)
class TradeRecord:
    """Closed round-trip used by the metrics layer."""

    trade_id: str
    side: str
    entry_price: Decimal
    exit_price: Decimal
    declared_stop: Decimal
    qty: Decimal
    entry_fees: Decimal
    exit_fees: Decimal


@dataclass(frozen=True)
class MetricsSummary:
    """Per-strategy metrics set — headline R and PF, plus support numbers."""

    n_trades: int
    total_fees: Decimal
    r_mean: float
    r_median: float
    profit_factor: float
    sum_pnl_net: Decimal


def r_per_trade(trade: TradeRecord) -> float:
    """R = realized_pnl / (|entry - declared_stop| * qty).

    SPEC §19 #1 option (a): stop-on-entry. No fallback denominator.
    An Action.ENTER that reached this function without a valid
    declared stop is a bug upstream — `NexusRuntime.require_stop` is
    supposed to have raised `StopContractViolation` before the trade
    opened. We re-check here defensively.
    """
    if trade.declared_stop <= 0:
        msg = f'trade {trade.trade_id}: declared_stop must be > 0, got {trade.declared_stop}'
        raise StopContractViolation(msg)
    gross = trade.exit_price - trade.entry_price if trade.side == 'BUY' else trade.entry_price - trade.exit_price
    net = gross * trade.qty - (trade.entry_fees + trade.exit_fees)
    risk = abs(trade.entry_price - trade.declared_stop) * trade.qty
    if risk == 0:
        msg = f'trade {trade.trade_id}: risk_at_entry == 0 (entry_price == declared_stop)'
        raise StopContractViolation(msg)
    return float(net / risk)


def profit_factor(trades: list[TradeRecord]) -> float:
    """Σ(winning net pnl) / |Σ(losing net pnl)|. inf if no losers, 0 if no winners."""
    pnls = [_net_pnl(t) for t in trades]
    winners = sum((p for p in pnls if p > 0), Decimal('0'))
    losers = -sum((p for p in pnls if p < 0), Decimal('0'))
    if losers > 0:
        return float(winners / losers)
    if winners > 0:
        return math.inf
    return 0.0


def summarize(trades: list[TradeRecord]) -> MetricsSummary:
    """Aggregate one sweep-decoder's closed trades into the headline numbers."""
    if not trades:
        return MetricsSummary(
            n_trades=0, total_fees=Decimal('0'), r_mean=0.0, r_median=0.0,
            profit_factor=0.0, sum_pnl_net=Decimal('0'),
        )
    rs = [r_per_trade(t) for t in trades]
    pnls = [_net_pnl(t) for t in trades]
    fees = sum((t.entry_fees + t.exit_fees for t in trades), Decimal('0'))
    sorted_rs = sorted(rs)
    mid = len(sorted_rs) // 2
    median = (sorted_rs[mid - 1] + sorted_rs[mid]) / 2.0 if len(sorted_rs) % 2 == 0 else sorted_rs[mid]
    return MetricsSummary(
        n_trades=len(trades), total_fees=fees,
        r_mean=float(sum(rs) / len(rs)),
        r_median=float(median),
        profit_factor=profit_factor(trades),
        sum_pnl_net=sum(pnls, Decimal('0')),
    )


def _net_pnl(trade: TradeRecord) -> Decimal:
    gross = trade.exit_price - trade.entry_price if trade.side == 'BUY' else trade.entry_price - trade.exit_price
    return gross * trade.qty - (trade.entry_fees + trade.exit_fees)
