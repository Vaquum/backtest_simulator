"""Internal venue-layer types used by the `walk_trades` fill engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True)
class FillResult:

    fill_time: datetime
    fill_price: Decimal
    fill_qty: Decimal
    is_maker: bool
    reason: str

@dataclass(frozen=True)
class PendingOrder:

    order_id: str
    side: str
    order_type: str
    qty: Decimal
    limit_price: Decimal | None
    stop_price: Decimal | None
    time_in_force: str
    submit_time: datetime
    symbol: str

@dataclass
class FillModelConfig:

    submit_latency_ms: int = 50
    cancel_latency_ms: int = 30
    fill_notification_latency_ms: int = 20
    fee_role_by_submit_latency: bool = True
    extras: dict[str, str] = field(default_factory=lambda: dict[str, str]())
