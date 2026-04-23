"""Backtest action-submit callback — bypass the full validator, land real TradeCommands."""
from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import cast

from nexus.core.domain.enums import OrderSide
from nexus.core.validator.pipeline_models import (
    InstanceState,
    ValidationAction,
    ValidationDecision,
    ValidationRequestContext,
)
from nexus.infrastructure.praxis_connector.praxis_outbound import PraxisOutbound
from nexus.infrastructure.praxis_connector.translate import translate_to_trade_command
from nexus.instance_config import InstanceConfig as NexusInstanceConfig
from nexus.strategy.action import Action, ActionType

_log = logging.getLogger(__name__)


_ACTION_TYPE_TO_VALIDATION_ACTION: dict[ActionType, ValidationAction] = {
    ActionType.ENTER: ValidationAction.ENTER,
    ActionType.EXIT: ValidationAction.EXIT,
    ActionType.MODIFY: ValidationAction.MODIFY,
    ActionType.ABORT: ValidationAction.ABORT,
}


def build_action_submitter(
    *, nexus_config: NexusInstanceConfig,
    state: InstanceState,
    praxis_outbound: PraxisOutbound,
) -> Callable[[list[Action], str], None]:
    """Return a callback for `PredictLoop(action_submit=...)` that sends real TradeCommands.

    The backtest path intentionally bypasses the full `ValidationPipeline`
    (capital, risk, intake, health, platform_limits, price stages). Those
    validators exist to gate orders at live-production boundaries — they
    reject orders when running capital is unavailable, when platform
    rate-limits would be hit, or when a book is too stale. A backtest is
    checking what a strategy WOULD do given a stream of data, not whether
    the production boundaries would let it through; reproducing the
    validators against a simulated venue would reject honest backtest
    hypotheses for reasons that don't apply offline.

    ENTER / EXIT flow through `translate_to_trade_command` with a
    hardcoded `ValidationDecision(allowed=True)` — same translator
    `submit_actions` would call after the validator pass. ABORT flows
    through `praxis_outbound.send_abort` with the original command_id.

    Mirrors `nexus.strategy.action_submit.submit_actions` in what it
    eventually calls on `PraxisOutbound`, so the TradeCommand shape
    landing in `praxis.Trading` is byte-identical to production.
    """
    def _submit(actions: list[Action], strategy_id: str) -> None:
        for action in actions:
            try:
                if action.action_type == ActionType.ABORT:
                    _submit_abort(praxis_outbound, nexus_config, strategy_id, action)
                else:
                    _submit_translated(
                        praxis_outbound, nexus_config, state, strategy_id, action,
                    )
            except Exception:  # noqa: BLE001 - per-action failure is local; don't abort the tick
                _log.exception(
                    'backtest action submission failed',
                    extra={
                        'strategy_id': strategy_id,
                        'action_type': action.action_type.value,
                    },
                )

    return _submit


def _submit_translated(
    praxis_outbound: PraxisOutbound,
    config: NexusInstanceConfig,
    state: InstanceState,
    strategy_id: str,
    action: Action,
) -> None:
    context = _build_context(config=config, state=state, strategy_id=strategy_id, action=action)
    decision = ValidationDecision(allowed=True)
    cmd = translate_to_trade_command(action, context, decision, config, datetime.now(UTC))
    command_id = praxis_outbound.send_command(cmd)
    _log.info(
        'backtest action submitted',
        extra={
            'strategy_id': strategy_id,
            'action_type': action.action_type.value,
            'command_id': command_id,
        },
    )


def _submit_abort(
    praxis_outbound: PraxisOutbound,
    config: NexusInstanceConfig,
    strategy_id: str,
    action: Action,
) -> None:
    if action.command_id is None:
        _log.warning('ABORT action missing command_id; skipping', extra={'strategy_id': strategy_id})
        return
    praxis_outbound.send_abort(
        command_id=action.command_id, account_id=config.account_id,
        reason='backtest_runtime_abort', created_at=datetime.now(UTC),
    )


def _build_context(
    *, config: NexusInstanceConfig, state: InstanceState,
    strategy_id: str, action: Action,
) -> ValidationRequestContext:
    # `translate_to_trade_command` reads: command_id (required non-empty),
    # action, symbol, order_side, order_size, order_notional, trade_id.
    # All other context fields (estimated_fees, strategy_budget, state,
    # config, current_order_notional) are not read by translate — they
    # exist for the validators, which this backtest path bypasses. We
    # still supply real values where trivially available and sentinel
    # zeros / defaults elsewhere, so the dataclass __post_init__
    # validations pass.
    symbol = _extract_symbol(action)
    order_size = action.size or Decimal('0')
    order_notional = (
        order_size * action.reference_price
        if action.reference_price is not None else Decimal('0')
    )
    return ValidationRequestContext(
        strategy_id=strategy_id,
        order_notional=order_notional,
        estimated_fees=Decimal('0'),
        strategy_budget=Decimal('1'),
        state=state,
        config=config,
        action=_ACTION_TYPE_TO_VALIDATION_ACTION[action.action_type],
        symbol=symbol,
        order_side=_resolve_side(action),
        order_size=action.size,
        trade_id=action.trade_id or f'bts-{uuid.uuid4().hex[:12]}',
        command_id=action.command_id or f'bts-cmd-{uuid.uuid4().hex[:12]}',
        current_order_notional=None,
    )


def _extract_symbol(action: Action) -> str:
    params = action.execution_params or {}
    raw = params.get('symbol')
    return cast('str', raw) if isinstance(raw, str) else 'BTCUSDT'


def _resolve_side(action: Action) -> OrderSide | None:
    if action.direction is None:
        return None
    return action.direction
