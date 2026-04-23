"""BinanceSpotFilters — Binance Spot LOT_SIZE / MIN_NOTIONAL / PRICE_FILTER."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class BinanceSpotFilters:
    """Binance Spot exchange filters. Values are conservative defaults for BTCUSDT."""

    symbol: str
    min_qty: Decimal
    max_qty: Decimal
    step_size: Decimal
    min_notional: Decimal
    tick_size: Decimal

    @classmethod
    def binance_spot(cls, symbol: str = 'BTCUSDT') -> BinanceSpotFilters:
        return cls(
            symbol=symbol,
            min_qty=Decimal('0.00001'),
            max_qty=Decimal('9000'),
            step_size=Decimal('0.00001'),
            min_notional=Decimal('10'),
            tick_size=Decimal('0.01'),
        )

    def round_qty(self, qty: Decimal) -> Decimal:
        """Round qty DOWN to step_size multiple."""
        if self.step_size <= 0:
            return qty
        multiples = (qty / self.step_size).to_integral_value(rounding='ROUND_DOWN')
        return multiples * self.step_size

    def round_price(self, price: Decimal) -> Decimal:
        """Round price to tick_size multiple (banker's rounding)."""
        if self.tick_size <= 0:
            return price
        multiples = (price / self.tick_size).to_integral_value(rounding='ROUND_HALF_EVEN')
        return multiples * self.tick_size

    def validate(self, qty: Decimal, price: Decimal) -> str | None:
        """Return rejection reason string, or None if the order is acceptable."""
        if qty < self.min_qty:
            return f'LOT_SIZE: qty {qty} < min_qty {self.min_qty}'
        if qty > self.max_qty:
            return f'LOT_SIZE: qty {qty} > max_qty {self.max_qty}'
        notional = qty * price
        if notional < self.min_notional:
            return f'MIN_NOTIONAL: {notional} < {self.min_notional}'
        return None
