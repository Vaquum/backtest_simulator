"""FeeSchedule — Binance Spot retail defaults + BNB discount + per-symbol overrides."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Final

_DEFAULT_MAKER: Final[Decimal] = Decimal('0.001')
_DEFAULT_TAKER: Final[Decimal] = Decimal('0.001')
_BNB_DISCOUNT: Final[Decimal] = Decimal('0.25')


@dataclass(frozen=True)
class FeeSchedule:
    """Maker/taker schedule with optional BNB discount and per-symbol overrides."""

    maker: Decimal = _DEFAULT_MAKER
    taker: Decimal = _DEFAULT_TAKER
    bnb_discount: bool = False
    per_symbol_maker: dict[str, Decimal] = field(default_factory=dict)
    per_symbol_taker: dict[str, Decimal] = field(default_factory=dict)

    def rate(self, symbol: str, *, is_maker: bool) -> Decimal:
        """Effective fee rate for this symbol + liquidity role."""
        if is_maker:
            base = self.per_symbol_maker.get(symbol, self.maker)
        else:
            base = self.per_symbol_taker.get(symbol, self.taker)
        if self.bnb_discount:
            return base * (Decimal('1') - _BNB_DISCOUNT)
        return base

    def fee(self, symbol: str, notional: Decimal, *, is_maker: bool) -> Decimal:
        """Fee amount in quote currency. notional = qty * fill_price."""
        return notional * self.rate(symbol, is_maker=is_maker)
