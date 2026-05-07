"""Preds-based binary-regime, long-only template — `__BTS_PARAMS__` substituted by ManifestBuilder."""
from __future__ import annotations

import json
import logging
from datetime import datetime
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

_BAKED_CONFIG: dict[str, object] = json.loads('__BTS_PARAMS__')

class _Config:

    def __init__(
        self, symbol: str, capital: Decimal, kelly_pct: Decimal,
        estimated_price: Decimal, stop_bps: Decimal,
        force_flatten_after: datetime | None,
        maker_preference: bool = False,
    ) -> None:
        self.symbol = symbol
        self.capital = capital
        self.kelly_pct = kelly_pct
        self.estimated_price = estimated_price
        self.stop_bps = stop_bps
        self.force_flatten_after = force_flatten_after
        self.maker_preference = maker_preference

class Strategy(_StrategyBase):

    def on_startup(
        self, params: StrategyParams, context: StrategyContext,
    ) -> list[Action]:
        del params, context
        force_flatten_raw = _BAKED_CONFIG.get('force_flatten_after')
        force_flatten_after = (
            None if force_flatten_raw is None
            else datetime.fromisoformat(str(force_flatten_raw))
        )
        if (
            force_flatten_after is not None
            and force_flatten_after.utcoffset() is None
        ):
            msg = (
                f'_BAKED_CONFIG[force_flatten_after] must be tz-aware, '
                f'got effectively-naive {force_flatten_after!r}'
            )
            raise ValueError(msg)
        self._config = _Config(
            symbol=str(_BAKED_CONFIG['symbol']),
            capital=Decimal(str(_BAKED_CONFIG['capital'])),
            kelly_pct=Decimal(str(_BAKED_CONFIG['kelly_pct'])),
            estimated_price=Decimal(str(_BAKED_CONFIG['estimated_price'])),
            stop_bps=Decimal(str(_BAKED_CONFIG['stop_bps'])),
            force_flatten_after=force_flatten_after,
            maker_preference=bool(
                _BAKED_CONFIG.get('maker_preference', False),
            ),
        )
        self._long: bool = False
        self._entry_qty: Decimal = Decimal('0')
        self._pending_buy: bool = False
        self._pending_sell: bool = False
        self._must_close_outstanding: bool = False
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
        if (
            self._must_close_outstanding
            and self._long
            and not self._pending_sell
        ):
            self._must_close_outstanding = False
            self._pending_sell = True
            qty = self._entry_qty
            _log.info(
                'CLOSE RETRY: prior SELL did not fully fill; '
                're-emitting at next tick. qty=%s ts=%s',
                qty, signal.timestamp,
            )
            return [self._build_action(OrderSide.SELL, qty, signal)]
        cutoff = self._config.force_flatten_after
        at_cutoff = cutoff is not None and signal.timestamp >= cutoff
        if at_cutoff and self._long and not self._pending_sell:
            qty = self._entry_qty
            self._pending_sell = True
            _log.info(
                'FORCE FLATTEN at window close: qty=%s ts=%s cutoff=%s',
                qty, signal.timestamp, cutoff,
            )
            return [self._build_action(OrderSide.SELL, qty, signal)]
        preds_raw = signal.values.get('_preds')
        if preds_raw is None:
            _log.info('signal missing _preds; keys=%s', list(signal.values))
            return []
        preds = int(preds_raw)
        was_long = self._long
        _log.info(
            'on_signal fired: preds=%s probs=%s was_long=%s',
            preds, signal.values.get('_probs'), was_long,
        )
        if at_cutoff:
            return []
        if preds == 1 and not was_long and not self._pending_buy:
            qty_raw = (
                self._config.capital * self._config.kelly_pct / Decimal('100')
            ) / self._config.estimated_price
            qty = qty_raw.quantize(Decimal('0.00001'), rounding=ROUND_DOWN)
            self._pending_buy = True
            _log.info(
                'ENTER BUY emit: qty=%s was_long=False pending_buy=True', qty,
            )
            return [self._build_action(OrderSide.BUY, qty, signal)]
        if preds == 0 and was_long and not self._pending_sell:
            qty = self._entry_qty
            self._pending_sell = True
            _log.info(
                'ENTER SELL (exit long) emit: qty=%s was_long=True '
                'pending_sell=True', qty,
            )
            return [self._build_action(OrderSide.SELL, qty, signal)]
        return []

    def _build_action(
        self, side: OrderSide, qty: Decimal, signal: Signal | None,
    ) -> Action:
        del signal
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
        is_buy_maker = (
            self._config.maker_preference and side == OrderSide.BUY
        )
        order_type_value = (
            OrderType.LIMIT if is_buy_maker else OrderType.MARKET
        )
        if is_buy_maker:
            execution_params['price'] = str(self._config.estimated_price)
        return Action(
            action_type=ActionType.ENTER,
            direction=side, size=qty,
            execution_mode=ExecutionMode.SINGLE_SHOT,
            order_type=order_type_value,
            execution_params=execution_params,
            deadline=60,
            trade_id=None, command_id=None,
            maker_preference=None,
            reference_price=self._config.estimated_price,
        )

    def on_outcome(
        self, outcome: TradeOutcome, params: StrategyParams, context: StrategyContext,
    ) -> list[Action]:
        del params, context
        from nexus.infrastructure.praxis_connector.trade_outcome_type import (
            TradeOutcomeType,
        )
        is_buy = self._is_buy_command(outcome.command_id)
        if outcome.outcome_type.is_fill:
            if outcome.fill_size is None:
                msg = (
                    f'TradeOutcome {outcome.outcome_id} is_fill but '
                    f'fill_size is None; cannot reconcile strategy '
                    f'position from a fill outcome with no qty.'
                )
                raise ValueError(msg)
            if is_buy:
                self._long = True
                self._entry_qty += outcome.fill_size
                self._pending_buy = False
                _log.info(
                    'on_outcome BUY %s: fill_size=%s entry_qty=%s long=True',
                    outcome.outcome_type.value, outcome.fill_size,
                    self._entry_qty,
                )
                cutoff = self._config.force_flatten_after
                if (
                    cutoff is not None
                    and outcome.timestamp >= cutoff
                    and not self._pending_sell
                ):
                    qty = self._entry_qty
                    self._pending_sell = True
                    _log.info(
                        'FORCE FLATTEN on BUY fill past cutoff: qty=%s '
                        'fill_ts=%s cutoff=%s',
                        qty, outcome.timestamp, cutoff,
                    )
                    return [
                        self._build_action(OrderSide.SELL, qty, signal=None),
                    ]
            else:
                self._reconcile_sell_fill(outcome)
        elif outcome.outcome_type.is_terminal:
            if is_buy:
                self._pending_buy = False
                _log.info(
                    'on_outcome BUY %s (no fill): pending_buy=False',
                    outcome.outcome_type.value,
                )
            else:
                self._pending_sell = False
                if outcome.outcome_type == TradeOutcomeType.EXPIRED:
                    self._must_close_outstanding = self._long
                else:
                    self._must_close_outstanding = False
                _log.info(
                    'on_outcome SELL %s (no fill): pending_sell=False '
                    'must_close_outstanding=%s',
                    outcome.outcome_type.value,
                    self._must_close_outstanding,
                )
        return []

    def _reconcile_sell_fill(self, outcome: TradeOutcome) -> None:
        from nexus.infrastructure.praxis_connector.trade_outcome_type import (
            TradeOutcomeType,
        )
        if outcome.fill_size is None:
            msg = (
                f'_reconcile_sell_fill: outcome {outcome.outcome_id} '
                f'is_fill but fill_size is None'
            )
            raise ValueError(msg)
        self._entry_qty -= outcome.fill_size
        if (
            outcome.outcome_type == TradeOutcomeType.FILLED
            or self._entry_qty <= Decimal('0')
        ):
            self._long = False
            self._entry_qty = Decimal('0')
        self._pending_sell = False
        self._must_close_outstanding = self._long
        _log.info(
            'on_outcome SELL %s: fill_size=%s entry_qty=%s '
            'long=%s must_close_outstanding=%s',
            outcome.outcome_type.value, outcome.fill_size,
            self._entry_qty, self._long,
            self._must_close_outstanding,
        )

    def _is_buy_command(self, command_id: str) -> bool:
        del command_id
        return self._pending_buy or not self._long

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
        self._pending_buy = bool(payload.get('pending_buy', False))
        self._pending_sell = bool(payload.get('pending_sell', False))
        self._must_close_outstanding = bool(
            payload.get('must_close_outstanding', False),
        )

    def on_save(self) -> bytes:
        payload = {
            'long': self._long, 'entry_qty': str(self._entry_qty),
            'pending_buy': self._pending_buy,
            'pending_sell': self._pending_sell,
            'must_close_outstanding': self._must_close_outstanding,
        }
        return json.dumps(payload).encode('utf-8')

    def on_shutdown(
        self, params: StrategyParams, context: StrategyContext,
    ) -> list[Action]:
        del params, context
        return []
