"""`InversePrescientStrategy` — perfect future label, sign-flipped action; the fill-causality probe."""

# Slice #17 Task 7: a Nexus `Strategy` that receives a perfect future
# label via `signal.values['_future_pred']` and DELIBERATELY trades
# the OPPOSITE direction. On a tape whose true direction the label
# captures, this strategy must lose catastrophically — every BUY
# lands at a peak that is about to fall, every SELL exits before a
# rise. If `walk_trades` somehow lets this strategy profit (or even
# break even), the fill model is non-causal: a fill is being awarded
# at a price that respects the strategy's direction independent of
# the order's submit time, which is exactly the regression Task 7
# pins SPEC §9.3's "fill model is causal" guarantee against.
#
# The strategy is the inverse of a "prescient" reference: we don't
# need a prescient implementation in production, only the inverse —
# because if the inverse loses everything on a directional tape, the
# fill model handled the order honestly.
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


class InversePrescientStrategy(_StrategyBase):
    """Trade the OPPOSITE of `signal.values['_future_pred']`.

    The label is supplied by the test fixture as a perfect predictor
    of the next-bar move. This strategy's `on_signal` reads it and
    acts on `1 - label`: BUY when the label says "down", SELL when it
    says "up". Catastrophic loss on a directional tape is the
    expected outcome and the test's pass condition.
    """

    def __init__(
        self,
        strategy_id: str,
        *,
        capital: Decimal = Decimal('100000'),
        kelly_pct: Decimal = Decimal('1'),
        estimated_price: Decimal = Decimal('70000'),
        stop_bps: Decimal = Decimal('50'),
    ) -> None:
        super().__init__(strategy_id)
        self._symbol = 'BTCUSDT'
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
        # The fixture supplies `_future_pred` as a perfect next-bar
        # direction label (1 = up, 0 = down). The strategy *inverts*
        # the label and acts on `1 - label`: a peeking strategy that
        # is also wired up backwards. On a directional tape this is
        # the worst possible policy.
        raw_label = signal.values.get('_future_pred')
        if raw_label is None:
            msg = (
                f'InversePrescientStrategy expected `_future_pred` in '
                f'signal.values; got {signal.values!r}. The test fixture '
                f'must supply a perfect next-bar label.'
            )
            raise ValueError(msg)
        label = int(raw_label)
        inverse = 1 - label
        # inverse == 1 → "up" → strategy goes long (or stays long).
        # inverse == 0 → "down" → strategy goes flat (or stays flat).
        if inverse == 1:
            if self._long:
                return []  # already long, hold through the down move
            qty_raw = (
                self._capital * self._kelly_pct / Decimal('100')
            ) / self._estimated_price
            qty = qty_raw.quantize(Decimal('0.00001'), rounding=ROUND_DOWN)
            self._long = True
            self._entry_qty = qty
            return [self._build_action(OrderSide.BUY, qty)]
        # `inverse` is 0 here: strategy goes flat (or stays flat).
        if not self._long:
            return []  # already flat
        qty = self._entry_qty
        self._long = False
        self._entry_qty = Decimal('0')
        return [self._build_action(OrderSide.SELL, qty)]

    def _build_action(self, side: OrderSide, qty: Decimal) -> Action:
        stop_price: Decimal | None = None
        if side == OrderSide.BUY:
            stop_price = self._estimated_price * (
                Decimal('1') - self._stop_bps / Decimal('10000')
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
