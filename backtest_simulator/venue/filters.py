"""BinanceSpotFilters — Binance Spot LOT_SIZE / MIN_NOTIONAL / PRICE_FILTER."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class BinanceSpotFilters:
    symbol: str
    min_qty: Decimal
    max_qty: Decimal
    step_size: Decimal
    min_notional: Decimal
    tick_size: Decimal

    @classmethod
    def binance_spot(cls, symbol: str='BTCUSDT') -> BinanceSpotFilters:
        return cls(symbol=symbol, min_qty=Decimal('0.00001'), max_qty=Decimal('9000'), step_size=Decimal('0.00001'), min_notional=Decimal('10'), tick_size=Decimal('0.01'))

    def round_qty(self, qty: Decimal) -> Decimal:
        multiples = (qty / self.step_size).to_integral_value(rounding='ROUND_DOWN')
        return multiples * self.step_size

    def round_price(self, price: Decimal) -> Decimal:
        multiples = (price / self.tick_size).to_integral_value(rounding='ROUND_HALF_EVEN')
        return multiples * self.tick_size

    def validate(self, qty: Decimal, price: Decimal | None=None) -> str | None:
        return None
