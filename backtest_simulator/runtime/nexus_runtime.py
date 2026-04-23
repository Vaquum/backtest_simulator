"""Thin orchestrator bridging the simulated venue to Nexus strategy dispatch."""
from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from backtest_simulator.exceptions import StopContractViolation
from backtest_simulator.runtime.outcome_translator import (
    NexusOutcomeShape,
    translate,
)
from backtest_simulator.venue.simulated import SimulatedVenueAdapter


@dataclass(frozen=True)
class _OpenTrade:
    trade_id: str
    strategy_id: str
    symbol: str
    side: str
    qty: Decimal
    entry_price: Decimal
    declared_stop: Decimal
    submit_time: datetime


@dataclass
class NexusRuntime:
    """Minimum adapter around Nexus's new action_submit + OutcomeLoop.

    Post-Nexus-PR-40 the heavy lifting — validate + translate + send —
    lives in `nexus.strategy.action_submit.submit_actions`. This class
    keeps only the backtest-specific responsibilities that Nexus cannot
    own:

    1. Mint a `trade_id` on every ENTER. Nexus does not auto-assign
       and OutcomeProcessor requires it.
    2. Enforce the stop-contract gate: every ENTER must carry a
       `declared_stop_price` in its execution_params. No stop -> raise
       StopContractViolation. This is the invariant that keeps
       `R = |entry - stop| * qty` honest downstream.
    3. Drive the SimulatedVenueAdapter synchronously: submit, await,
       translate each fill into a Nexus TradeOutcome shape, dispatch.
    4. Mirror the outcome into the caller-supplied `on_outcome`
       callback (the integration test uses this; production paths
       would plumb through Nexus's OutcomeLoop instead).
    """

    venue: SimulatedVenueAdapter
    on_outcome: Callable[[str, NexusOutcomeShape], Awaitable[None]]
    _open_trades: dict[str, _OpenTrade] = field(default_factory=dict)

    @staticmethod
    def mint_trade_id() -> str:
        return f'TID-{uuid.uuid4().hex[:12]}'

    @staticmethod
    def require_stop(execution_params: object, action_kind: str) -> Decimal:
        stop = getattr(execution_params, 'declared_stop_price', None)
        if stop is None:
            msg = f'{action_kind} missing declared_stop_price (SPEC §19 #1)'
            raise StopContractViolation(msg)
        if not isinstance(stop, Decimal):
            stop = Decimal(str(stop))
        if stop <= 0:
            msg = f'{action_kind} declared_stop_price must be > 0, got {stop}'
            raise StopContractViolation(msg)
        return stop

    async def submit_entry(  # noqa: PLR0913 - keyword-only factory; every arg carries distinct business meaning
        self,
        *,
        strategy_id: str,
        symbol: str,
        side: str,
        qty: Decimal,
        entry_price: Decimal,
        declared_stop: Decimal,
        submit_time: datetime,
    ) -> str:
        """Submit ENTER. Mints trade_id, records the stop, runs the venue."""
        trade_id = self.mint_trade_id()
        result = await self.venue.submit_order(
            symbol=symbol, side=side, order_type='MARKET',
            qty=qty, submit_time=submit_time,
        )
        if not (result.accepted and result.fills):
            return ''
        first = result.fills[0]
        self._open_trades[trade_id] = _OpenTrade(
            trade_id=trade_id, strategy_id=strategy_id,
            symbol=symbol, side=side, qty=qty,
            entry_price=first.fill_price,
            declared_stop=declared_stop, submit_time=submit_time,
        )
        outcome = translate(
            command_id=result.order_id, outcome_id=f'OUT-{uuid.uuid4().hex[:8]}',
            timestamp=first.fill_time, status='FILLED',
            filled_qty=first.fill_qty, avg_fill_price=first.fill_price,
            actual_fees=result.fees_quote, target_qty=qty,
        )
        await self.on_outcome(strategy_id, outcome)
        return trade_id

    async def submit_exit(self, *, trade_id: str, submit_time: datetime) -> None:
        """Submit EXIT for an open trade. Side is inverted from the held position."""
        trade = self._open_trades.get(trade_id)
        if trade is None:
            return
        exit_side = 'SELL' if trade.side == 'BUY' else 'BUY'
        result = await self.venue.submit_order(
            symbol=trade.symbol, side=exit_side, order_type='MARKET',
            qty=trade.qty, submit_time=submit_time,
        )
        if result.accepted and result.fills:
            first = result.fills[0]
            outcome = translate(
                command_id=result.order_id, outcome_id=f'OUT-{uuid.uuid4().hex[:8]}',
                timestamp=first.fill_time, status='FILLED',
                filled_qty=first.fill_qty, avg_fill_price=first.fill_price,
                actual_fees=result.fees_quote, target_qty=trade.qty,
            )
            await self.on_outcome(trade.strategy_id, outcome)
            self._open_trades.pop(trade_id, None)

    def open_trade(self, trade_id: str) -> _OpenTrade | None:
        return self._open_trades.get(trade_id)

    def iter_open_trades(self) -> list[_OpenTrade]:
        return list(self._open_trades.values())
