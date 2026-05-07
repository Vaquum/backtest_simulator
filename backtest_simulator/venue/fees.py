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
    maker: Decimal = _DEFAULT_MAKER
    taker: Decimal = _DEFAULT_TAKER
    bnb_discount: bool = False
    per_symbol_maker: dict[str, Decimal] = field(default_factory=lambda: dict[str, Decimal]())
    per_symbol_taker: dict[str, Decimal] = field(default_factory=lambda: dict[str, Decimal]())

    def rate(self, symbol: str, *, is_maker: bool) -> Decimal:
        if is_maker:
            base = self.per_symbol_maker.get(symbol, self.maker)
        else:
            base = self.per_symbol_taker.get(symbol, self.taker)
        return base

    def fee(self, symbol: str, notional: Decimal, *, is_maker: bool) -> Decimal:
        return notional * self.rate(symbol, is_maker=is_maker)
