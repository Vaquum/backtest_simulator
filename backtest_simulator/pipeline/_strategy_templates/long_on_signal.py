"""Long-on-signal strategy: ENTER when sensor probability crosses threshold, with declared stop."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

from nexus.infrastructure.praxis_connector.trade_outcome import TradeOutcome
from nexus.strategy.action import Action, ActionType
from nexus.strategy.base import Strategy
from nexus.strategy.context import StrategyContext
from nexus.strategy.params import StrategyParams
from nexus.strategy.signal import Signal

_log = logging.getLogger(__name__)


@dataclass
class _Config:
    symbol: str
    side: str
    enter_threshold: float
    stop_bps: Decimal
    qty: Decimal
    prob_key: str


class LongOnSignal(Strategy):
    """ENTER on probability > threshold; Nexus enforces stop via declared_stop_price."""

    def on_startup(self, params: StrategyParams) -> None:
        self._config = _parse_config(params)
        _log.info('LongOnSignal startup', extra={'config': self._config})

    def on_signal(
        self, signal: Signal, params: StrategyParams, context: StrategyContext,
    ) -> list[Action]:
        del params
        prob = signal.values.get(self._config.prob_key)
        if prob is None:
            return []
        if float(prob) < self._config.enter_threshold:
            return []
        if any(p.symbol == self._config.symbol for p in context.positions):
            return []
        reference = Decimal(str(signal.values.get('close', 0)))
        if reference <= 0:
            return []
        stop = reference * (Decimal('1') - self._config.stop_bps / Decimal('10000'))
        return [Action(
            action_type=ActionType.ENTER,
            direction=None, size=self._config.qty,
            execution_mode=None, order_type=None,
            execution_params={
                'declared_stop_price': stop,
                'symbol': self._config.symbol,
                'side': self._config.side,
            },
            deadline=None, trade_id=None, command_id=None,
            maker_preference=None, reference_price=reference,
        )]

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


def _parse_config(params: StrategyParams) -> _Config:
    raw = params.raw
    return _Config(
        symbol=str(raw['symbol']),
        side=str(raw.get('side', 'BUY')),
        enter_threshold=float(raw['enter_threshold']),
        stop_bps=Decimal(str(raw['stop_bps'])),
        qty=Decimal(str(raw['qty'])),
        prob_key=str(raw.get('prob_key', 'probability')),
    )
