"""Venue-layer dataclasses — kept here so fills/simulated stay lean."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True)
class FillResult:
    """A single fill leg. May be partial; caller aggregates."""

    fill_time: datetime
    fill_price: Decimal
    fill_qty: Decimal
    is_maker: bool
    reason: str  # 'market_walk' | 'limit_take' | 'limit_maker' | 'stop_trigger'


@dataclass(frozen=True)
class PendingOrder:
    """An open order in the synthetic book."""

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
    """Submit/cancel/notification latencies + the fee-role heuristic cutoff."""

    submit_latency_ms: int = 50
    cancel_latency_ms: int = 30
    fill_notification_latency_ms: int = 20
    fee_role_by_submit_latency: bool = True
    extras: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SubmitResult:
    """What the SimulatedVenueAdapter returns for submit_order."""

    order_id: str
    accepted: bool
    reject_reason: str | None
    fills: list[FillResult]
    fees_quote: Decimal


@dataclass
class Account:
    """A registered account on the simulated venue."""

    account_id: str
    api_key: str
    api_secret: str
    ws_sink: Callable[[str, dict[str, object]], Awaitable[None]] | None = None
