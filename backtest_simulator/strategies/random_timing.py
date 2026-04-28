"""`RandomTimingStrategy` — Bernoulli-flip on each signal; the no-alpha baseline."""

# Slice #17 Task 5: a Nexus `Strategy` that flips long/flat on each
# signal with probability `p_flip` (independently of `_preds`), drawing
# from a `random.Random` seeded at construction. Across many seeds on a
# zero-net-drift tape this MUST produce mean gross return ~ 0 (the
# Bernoulli mask makes BUY/SELL bar-index choice independent of the
# tape's high/low pattern, so positive and negative gross PnL pairs
# cancel in expectation).
#
# Used by tests/honesty/test_sanity_random.py to pin the "no hidden
# alpha" identity. If a regression introduces an *asymmetric* venue-side
# lookahead (e.g. a "best-of-K" scan that gives BUYs the minimum and
# SELLs the maximum of the next K ticks) or a strategy-side cheat that
# observes a future bar's prediction, random timing stops bracketing
# zero in its 95% CI and the simulator is silently generating phantom
# edge. Symmetric one-tick peek is invisible to this baseline by
# construction (BUY/SELL deltas cancel) — Task 3 pins exact-fill-price
# for that path.
from __future__ import annotations

import random
from decimal import ROUND_DOWN, Decimal

from nexus.core.domain.enums import OrderSide
from nexus.core.domain.order_types import ExecutionMode, OrderType
from nexus.infrastructure.praxis_connector.trade_outcome import TradeOutcome
from nexus.strategy.action import Action, ActionType
from nexus.strategy.base import Strategy as _StrategyBase
from nexus.strategy.context import StrategyContext
from nexus.strategy.params import StrategyParams
from nexus.strategy.signal import Signal


class RandomTimingStrategy(_StrategyBase):
    """Flip long/flat with probability `p_flip` per signal; ignore preds."""

    def __init__(
        self,
        strategy_id: str,
        *,
        seed: int,
        capital: Decimal = Decimal('100000'),
        kelly_pct: Decimal = Decimal('1'),
        estimated_price: Decimal = Decimal('70000'),
        stop_bps: Decimal = Decimal('50'),
    ) -> None:
        super().__init__(strategy_id)
        self._seed = seed
        self._rng = random.Random(seed)
        # `p_flip` is fixed at 0.5: the canonical "fair-coin" Bernoulli
        # for an alpha-leakage probe. A configurable rate would invite
        # callers to game the test by tuning a non-fair Bernoulli that
        # happens to bracket zero on the production tape.
        self._symbol = 'BTCUSDT'
        self._capital = capital
        self._kelly_pct = kelly_pct
        self._estimated_price = estimated_price
        self._stop_bps = stop_bps
        self._long: bool = False
        self._entry_qty: Decimal = Decimal('0')
        # `consumed_calls` tracks RNG progress so on_save / on_load can
        # restore deterministically without serialising `random.Random`'s
        # internal tuple.
        self._consumed_calls: int = 0

    def on_startup(
        self, params: StrategyParams, context: StrategyContext,
    ) -> list[Action]:
        del params, context
        return []

    def on_signal(
        self, signal: Signal, params: StrategyParams, context: StrategyContext,
    ) -> list[Action]:
        del signal, params, context
        roll = self._rng.random()
        self._consumed_calls += 1
        # Fair-coin Bernoulli flip: half the time the strategy ignores
        # the signal entirely. The other half it toggles long/flat.
        if roll >= 0.5:
            return []
        if self._long:
            qty = self._entry_qty
            self._long = False
            self._entry_qty = Decimal('0')
            return [self._build_action(OrderSide.SELL, qty)]
        qty_raw = (
            self._capital * self._kelly_pct / Decimal('100')
        ) / self._estimated_price
        qty = qty_raw.quantize(Decimal('0.00001'), rounding=ROUND_DOWN)
        self._long = True
        self._entry_qty = qty
        return [self._build_action(OrderSide.BUY, qty)]

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
        # The RNG stream is reseeded from the saved integer seed and the
        # consumed-call counter. Both `_seed` and `_consumed_calls` MUST
        # be restored — otherwise a follow-up `on_save` writes the
        # constructor seed alongside the loaded consumed-call counter,
        # producing a serialised state that no longer maps back to the
        # current RNG stream position.
        loaded_seed = int(payload['seed'])
        loaded_consumed = int(payload.get('consumed_calls', 0))
        self._seed = loaded_seed
        self._rng = random.Random(loaded_seed)
        for _ in range(loaded_consumed):
            self._rng.random()
        self._consumed_calls = loaded_consumed

    def on_save(self) -> bytes:
        import json
        return json.dumps(
            {
                'long': self._long,
                'entry_qty': str(self._entry_qty),
                'seed': self._seed,
                'consumed_calls': self._consumed_calls,
            },
        ).encode('utf-8')

    def on_shutdown(
        self, params: StrategyParams, context: StrategyContext,
    ) -> list[Action]:
        del params, context
        return []
