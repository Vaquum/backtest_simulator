"""Internal venue-layer types used by the `walk_trades` fill engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True)
class FillResult:
    """A single fill leg produced by `walk_trades`. May be partial."""

    fill_time: datetime
    fill_price: Decimal
    fill_qty: Decimal
    is_maker: bool
    reason: str  # 'market_walk' | 'limit_take' | 'limit_maker' | 'stop_trigger'


@dataclass(frozen=True)
class PendingOrder:
    """Internal open-order representation fed to `walk_trades`."""

    order_id: str
    side: str  # 'BUY' | 'SELL'
    order_type: str  # 'MARKET' | 'LIMIT' | 'STOP_LOSS' | 'STOP_LOSS_LIMIT' | 'TAKE_PROFIT'
    qty: Decimal
    limit_price: Decimal | None
    stop_price: Decimal | None
    time_in_force: str  # 'GTC' | 'IOC' | 'FOK'
    submit_time: datetime
    symbol: str


@dataclass
class FillModelConfig:
    """Submit/cancel/notification latencies + fee-role heuristic cutoff.

    All latency values are explicit parameters (no implicit defaults
    hidden inside the fill engine). HonestyStatus reports the active
    values every run so a strategy's observed latency is always
    reconciled against a declared ms budget.
    """

    submit_latency_ms: int = 50
    cancel_latency_ms: int = 30
    fill_notification_latency_ms: int = 20
    fee_role_by_submit_latency: bool = True
    extras: dict[str, str] = field(default_factory=lambda: dict[str, str]())
