"""Preds-based binary-regime, long-only template — `__BTS_PARAMS__` substituted by ManifestBuilder."""
from __future__ import annotations

import json
import logging
from decimal import ROUND_DOWN, Decimal

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
        self, symbol: str, capital: Decimal, kelly_pct: Decimal,
        estimated_price: Decimal, stop_bps: Decimal,
    ) -> None:
        self.symbol = symbol
        self.capital = capital
        self.kelly_pct = kelly_pct
        self.estimated_price = estimated_price
        self.stop_bps = stop_bps


class Strategy(_StrategyBase):
    """Binary regime on `_preds`, long-only, Kelly-sized from baked config.

    State machine:
      preds=1 AND flat  -> ENTER BUY  (size = capital * kelly_pct/100 / est_price)
      preds=0 AND long  -> ENTER SELL (size = self._entry_qty recorded at ENTER)
      preds=1 AND long  -> no-op
      preds=0 AND flat  -> no-op

    Kelly% comes from `backtest_mean_kelly_pct` of the selected decoder
    (baked at manifest-build time). `estimated_price` is the ClickHouse
    seed price at window start — the real fill price comes from the venue
    adapter's next-trade walk, so the qty here is a sizing hint, not a
    price promise.

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
            capital=Decimal(str(_BAKED_CONFIG['capital'])),
            kelly_pct=Decimal(str(_BAKED_CONFIG['kelly_pct'])),
            estimated_price=Decimal(str(_BAKED_CONFIG['estimated_price'])),
            stop_bps=Decimal(str(_BAKED_CONFIG['stop_bps'])),
        )
        # Fresh-start defaults; `on_load` overwrites if persisted state exists.
        self._long: bool = False
        self._entry_qty: Decimal = Decimal('0')
        _log.info(
            'LongOnSignal startup: symbol=%s capital=%s kelly_pct=%s est_price=%s',
            self._config.symbol, self._config.capital,
            self._config.kelly_pct, self._config.estimated_price,
        )
        return []

    def on_signal(
        self, signal: Signal, params: StrategyParams, context: StrategyContext,
    ) -> list[Action]:
        del params, context
        preds_raw = signal.values.get('_preds')
        if preds_raw is None:
            _log.info('signal missing _preds; keys=%s', list(signal.values))
            return []
        preds = int(preds_raw)
        # `was_long` captures the PRIOR state so the log line is unambiguous —
        # readers shouldn't have to infer whether `long=...` is pre- or
        # post-transition. Transitions log their own ENTER BUY / ENTER SELL
        # line below.
        was_long = self._long
        _log.info(
            'on_signal fired: preds=%s probs=%s was_long=%s',
            preds, signal.values.get('_probs'), was_long,
        )
        if preds == 1 and not was_long:
            qty_raw = (
                self._config.capital * self._config.kelly_pct / Decimal('100')
            ) / self._config.estimated_price
            # Round DOWN to Binance's BTCUSDT step_size of 0.00001 so
            # the venue adapter's lot-size filter accepts the qty and
            # the full amount fills (no PARTIALLY_FILLED from a qty
            # that doesn't land on a step boundary). A fractional
            # residue beyond 5 decimals silently turns cmd.qty >
            # filled_qty into a partial status even when the walk
            # consumed everything that was step-legal.
            qty = qty_raw.quantize(Decimal('0.00001'), rounding=ROUND_DOWN)
            self._long = True
            self._entry_qty = qty
            _log.info('ENTER BUY: qty=%s was_long=False -> long=True', qty)
            return [self._build_action(OrderSide.BUY, qty, signal)]
        if preds == 0 and was_long:
            qty = self._entry_qty
            self._long = False
            self._entry_qty = Decimal('0')
            _log.info('ENTER SELL (exit long): qty=%s was_long=True -> long=False', qty)
            return [self._build_action(OrderSide.SELL, qty, signal)]
        return []

    def _build_action(self, side: OrderSide, qty: Decimal, signal: Signal) -> Action:
        del signal
        # Declared-stop enforcement (Part 2): every OPEN-position ENTER
        # must carry a concrete `stop_price`. For long-only we only
        # open on BUY; SELL here is the EXIT leg of an already-open
        # position and doesn't need a new stop (the SELL itself is the
        # risk close). The stop sits `stop_bps` basis points BELOW the
        # reference price for a BUY — e.g. with `stop_bps=50` and
        # `estimated_price=70000`, `stop_price=69650`. The venue
        # fill engine (`venue/fills.py::_walk_market`) halts the entry
        # walk the moment a tick breaches the stop and returns the
        # already-accumulated partial fill; the residual is NOT booked
        # at the declared stop. A separate STOP_* close fills at the
        # breach tick's actual tape price (gap slippage), not at the
        # declared stop, via `_walk_stop`. The declared stop is the
        # measurement unit for R, not a promise about where fills land.
        stop_price: Decimal | None = None
        if side == OrderSide.BUY:
            bps = self._config.stop_bps
            stop_price = self._config.estimated_price * (
                Decimal('1') - bps / Decimal('10000')
            )
        execution_params: dict[str, object] = {
            'symbol': self._config.symbol,
            'stop_bps': str(self._config.stop_bps),
        }
        if stop_price is not None:
            execution_params['stop_price'] = str(stop_price)
        return Action(
            action_type=ActionType.ENTER,
            direction=side, size=qty,
            execution_mode=ExecutionMode.SINGLE_SHOT,
            order_type=OrderType.MARKET,
            execution_params=execution_params,
            # `deadline` is a DURATION in seconds (not an epoch timestamp).
            # Praxis computes the concrete deadline as
            # `cmd.created_at + timedelta(seconds=timeout)`, where
            # `timeout = action.deadline` via Nexus's praxis_outbound.
            # 60s is a reasonable fill window for a backtest MARKET order.
            deadline=60,
            trade_id=None, command_id=None,
            maker_preference=None,
            # `reference_price` is what the action-submitter multiplies
            # by `size` to produce the order's notional for the CAPITAL
            # validator. The backtest uses `estimated_price` baked at
            # manifest-build time (ClickHouse seed price at window
            # start) rather than trying to read live book here — the
            # strategy is notified only of the prediction, not the
            # current tick. Real fills come from the venue adapter's
            # historical trade walk regardless; this price is a sizing
            # hint only.
            reference_price=self._config.estimated_price,
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
        payload = json.loads(data.decode('utf-8'))
        self._long = bool(payload['long'])
        self._entry_qty = Decimal(str(payload['entry_qty']))

    def on_save(self) -> bytes:
        payload = {'long': self._long, 'entry_qty': str(self._entry_qty)}
        return json.dumps(payload).encode('utf-8')

    def on_shutdown(
        self, params: StrategyParams, context: StrategyContext,
    ) -> list[Action]:
        del params, context
        return []
