"""`BuyAndHoldStrategy` — buys at the first signal, holds, sells at the last."""

# Slice #17 Task 3: a Nexus `Strategy` whose only decisions are
# (a) ENTER BUY on the first signal it sees and (b) ENTER SELL on the
# `on_shutdown` callback (or on the explicit "last signal" marker
# `_close_position` in the signal values).
#
# Used by tests/honesty/test_sanity_buy_hold.py to pin the canonical
# reference run: a strategy that buys at the open and sells at the
# close must return within +/- 5 bps of `(close - open) / open - fees`.
# Anything wider is a fill / accounting drift in the simulator.
from __future__ import annotations

from decimal import ROUND_DOWN, Decimal

from nexus.core.domain.enums import OrderSide
from nexus.core.domain.order_types import ExecutionMode, OrderType
from nexus.infrastructure.praxis_connector.trade_outcome import TradeOutcome
from nexus.strategy.action import Action, ActionType
from nexus.strategy.base import Strategy as _StrategyBase
from nexus.strategy.context import StrategyContext
from nexus.strategy.params import StrategyParams
from nexus.strategy.signal import Signal


class BuyAndHoldStrategy(_StrategyBase):
    """Buys at the first signal, sells when `_close_position` is signalled."""

    def __init__(
        self,
        strategy_id: str,
        *,
        symbol: str = 'BTCUSDT',
        capital: Decimal = Decimal('100000'),
        kelly_pct: Decimal = Decimal('1'),
        estimated_price: Decimal = Decimal('70000'),
        stop_bps: Decimal = Decimal('500'),
    ) -> None:
        super().__init__(strategy_id)
        self._symbol = symbol
        self._capital = capital
        self._kelly_pct = kelly_pct
        self._estimated_price = estimated_price
        self._stop_bps = stop_bps
        self._long: bool = False
        self._entry_qty: Decimal = Decimal('0')

    def on_startup(
        self, params: StrategyParams, context: StrategyContext,
    ) -> list[Action]:
        del params, context
        return []

    def on_signal(
        self, signal: Signal, params: StrategyParams, context: StrategyContext,
    ) -> list[Action]:
        del params, context
        close = bool(signal.values.get('_close_position', False))
        if close and self._long:
            qty = self._entry_qty
            self._long = False
            self._entry_qty = Decimal('0')
            return [self._build_action(OrderSide.SELL, qty)]
        if not close and not self._long:
            qty_raw = (
                self._capital * self._kelly_pct / Decimal('100')
            ) / self._estimated_price
            qty = qty_raw.quantize(Decimal('0.00001'), rounding=ROUND_DOWN)
            self._long = True
            self._entry_qty = qty
            return [self._build_action(OrderSide.BUY, qty)]
        return []

    def _build_action(self, side: OrderSide, qty: Decimal) -> Action:
        stop_price: Decimal | None = None
        if side == OrderSide.BUY:
            bps = self._stop_bps
            stop_price = self._estimated_price * (
                Decimal('1') - bps / Decimal('10000')
            )
        execution_params: dict[str, object] = {
            'symbol': self._symbol,
            'stop_bps': str(self._stop_bps),
        }
        if stop_price is not None:
            execution_params['stop_price'] = str(stop_price)
        return Action(
            action_type=ActionType.ENTER,
            direction=side,
            size=qty,
            execution_mode=ExecutionMode.SINGLE_SHOT,
            order_type=OrderType.MARKET,
            execution_params=execution_params,
            deadline=60,
            trade_id=None,
            command_id=None,
            maker_preference=None,
            reference_price=self._estimated_price,
        )

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
        if not data:
            return
        import json
        payload = json.loads(data.decode('utf-8'))
        self._long = bool(payload['long'])
        self._entry_qty = Decimal(str(payload['entry_qty']))

    def on_save(self) -> bytes:
        import json
        return json.dumps(
            {'long': self._long, 'entry_qty': str(self._entry_qty)},
        ).encode('utf-8')

    def on_shutdown(
        self, params: StrategyParams, context: StrategyContext,
    ) -> list[Action]:
        del params, context
        return []
