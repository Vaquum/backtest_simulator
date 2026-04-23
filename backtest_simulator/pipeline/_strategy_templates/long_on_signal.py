"""Long-on-signal strategy template — `__BTS_PARAMS__` is substituted by ManifestBuilder."""
from __future__ import annotations

import json
import logging
from decimal import Decimal

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

    def on_load(self, blob: bytes) -> None:
        del blob

    def on_save(self) -> bytes:
        return b''

    def on_shutdown(self) -> None:
        pass
