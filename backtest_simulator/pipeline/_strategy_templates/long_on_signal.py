"""Long-on-signal strategy template — `__BTS_PARAMS__` is substituted by ManifestBuilder."""
from __future__ import annotations

import json
import logging
from decimal import Decimal

from nexus.core.domain.enums import OrderSide
from nexus.core.domain.order_types import ExecutionMode, OrderType
from nexus.infrastructure.praxis_connector.trade_outcome import TradeOutcome
from nexus.strategy.action import Action, ActionType
from nexus.strategy.base import Strategy as _StrategyBase
from nexus.strategy.context import StrategyContext
from nexus.strategy.params import StrategyParams
from nexus.strategy.signal import Signal

_log = logging.getLogger(__name__)

# Baked-in config. ManifestBuilder substitutes this literal JSON string
# when copying the template; before substitution the strategy would
# still parse cleanly but would exit early at on_startup.
_BAKED_CONFIG: dict[str, object] = json.loads('__BTS_PARAMS__')


class _Config:
    # Plain class (not @dataclass): Nexus's dynamic strategy loader runs
    # `exec_module` without registering the module in `sys.modules`, so
    # `@dataclass` fails at class-definition time when it tries
    # `sys.modules.get(cls.__module__).__dict__`.

    def __init__(
        self, symbol: str, side: str, enter_threshold: float,
        stop_bps: Decimal, qty: Decimal, prob_key: str,
    ) -> None:
        self.symbol = symbol
        self.side = side
        self.enter_threshold = enter_threshold
        self.stop_bps = stop_bps
        self.qty = qty
        self.prob_key = prob_key


class Strategy(_StrategyBase):
    """ENTER on probability > threshold; Nexus enforces stop via declared_stop_price.

    Class name is `Strategy` because Nexus's loader looks up that exact
    module attribute; the base is aliased `_StrategyBase` to avoid collision.
    """

    def on_startup(
        self, params: StrategyParams, context: StrategyContext,
    ) -> list[Action]:
        # Nexus currently constructs `StrategyParams(raw={})` and does not
        # pass per-strategy YAML params through; `_BAKED_CONFIG` above is
        # the mechanism ManifestBuilder uses to carry per-instance config
        # to each strategy file.
        del params, context
        self._config = _Config(
            symbol=str(_BAKED_CONFIG['symbol']),
            side=str(_BAKED_CONFIG.get('side', 'BUY')),
            enter_threshold=float(_BAKED_CONFIG['enter_threshold']),
            stop_bps=Decimal(str(_BAKED_CONFIG['stop_bps'])),
            qty=Decimal(str(_BAKED_CONFIG['qty'])),
            prob_key=str(_BAKED_CONFIG.get('prob_key', 'probability')),
        )
        _log.info('LongOnSignal startup', extra={'symbol': self._config.symbol})
        return []

    def on_signal(
        self, signal: Signal, params: StrategyParams, context: StrategyContext,
    ) -> list[Action]:
        del params
        _log.info('on_signal fired: values=%s', dict(signal.values))
        prob = signal.values.get(self._config.prob_key)
        if prob is None:
            _log.info('signal missing prob_key %r; keys=%s',
                      self._config.prob_key, list(signal.values))
            return []
        if float(prob) < self._config.enter_threshold:
            _log.info('prob %s below threshold %s', prob, self._config.enter_threshold)
            return []
        if any(p.symbol == self._config.symbol for p in context.positions):
            return []
        # Nexus's `produce_signal` doesn't include the current close in
        # `signal.values` (it carries only the model's predict-dict output),
        # so `reference_price` is intentionally None here. The venue
        # adapter fills MARKET orders at the next-trade price from the
        # historical stream regardless of what reference we pass — this
        # keeps the strategy honest without requiring a synthesized
        # close. A richer strategy would read the price from a side
        # channel (context or a per-tick market_data provider).
        side = OrderSide.BUY if self._config.side == 'BUY' else OrderSide.SELL
        return [Action(
            action_type=ActionType.ENTER,
            direction=side, size=self._config.qty,
            execution_mode=ExecutionMode.SINGLE_SHOT,
            order_type=OrderType.MARKET,
            execution_params={
                'symbol': self._config.symbol,
                'stop_bps': str(self._config.stop_bps),
            },
            # Deadline is a future epoch-ms timestamp; 60s from now gives
            # the venue a reasonable window to fill before the order
            # expires. The concrete value is less important than its
            # presence — Nexus's ENTER validator requires a non-null.
            deadline=int((signal.timestamp.timestamp() + 60) * 1000),
            trade_id=None, command_id=None,
            maker_preference=None, reference_price=None,
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

    def on_load(self, data: bytes) -> None:
        del data

    def on_save(self) -> bytes:
        return b''

    def on_shutdown(
        self, params: StrategyParams, context: StrategyContext,
    ) -> list[Action]:
        del params, context
        return []
