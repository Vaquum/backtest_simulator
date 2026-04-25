"""`ZeroTradeStrategy` — never emits an action; sanity-baseline #1."""

# Slice #17 Task 2: a Nexus `Strategy` whose every callback returns the
# empty action list. Used by `tests/honesty/test_sanity_zero_trade.py`
# to pin the floor of the simulation chain: a strategy that never
# decides to trade should produce zero fills, zero fees, zero
# position_notional, and zero `working_order_notional`. If any of
# those surfaces are non-zero with this strategy installed, the
# simulator is generating phantom trades somewhere in the venue or
# capital lifecycle, and that is itself an honesty violation.
from __future__ import annotations

from nexus.infrastructure.praxis_connector.trade_outcome import TradeOutcome
from nexus.strategy.action import Action
from nexus.strategy.base import Strategy as _StrategyBase
from nexus.strategy.context import StrategyContext
from nexus.strategy.params import StrategyParams
from nexus.strategy.signal import Signal


class ZeroTradeStrategy(_StrategyBase):
    """No-op strategy: every callback returns []."""

    def on_startup(
        self, params: StrategyParams, context: StrategyContext,
    ) -> list[Action]:
        del params, context
        return []

    def on_signal(
        self, signal: Signal, params: StrategyParams, context: StrategyContext,
    ) -> list[Action]:
        del signal, params, context
        return []

    def on_outcome(
        self, outcome: TradeOutcome, params: StrategyParams, context: StrategyContext,
    ) -> list[Action]:
        del outcome, params, context
        return []

    def on_timer(
        self, timer_id: str, params: StrategyParams, context: StrategyContext,
    ) -> list[Action]:
        del timer_id, params, context
        return []

    def on_load(self, data: bytes) -> None:
        del data

    def on_save(self) -> bytes:
        return b''

    def on_shutdown(
        self, params: StrategyParams, context: StrategyContext,
    ) -> list[Action]:
        del params, context
        return []
