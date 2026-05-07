"""Backtest action-submit callback — bypass the full validator, land real TradeCommands."""
from __future__ import annotations

import dataclasses
import importlib
import logging
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from types import SimpleNamespace
from typing import cast

from nexus.core.capital_controller.capital_controller import CapitalController
from nexus.core.domain.enums import OrderSide
from nexus.core.domain.instance_state import InstanceState
from nexus.core.domain.order_types import ExecutionMode as _NexusExecutionMode
from nexus.core.domain.order_types import OrderType as _NexusOrderType
from nexus.core.validator import ValidationPipeline
from nexus.core.validator.pipeline_models import (
    ValidationAction,
    ValidationDecision,
    ValidationRequestContext,
)
from nexus.infrastructure.praxis_connector.praxis_outbound import PraxisOutbound
from nexus.infrastructure.praxis_connector.trade_command import TradeCommand
from nexus.infrastructure.praxis_connector.translate import translate_to_trade_command
from nexus.instance_config import InstanceConfig as NexusInstanceConfig
from nexus.strategy.action import Action, ActionType
from praxis.core.domain.enums import ExecutionMode as _PraxisExecutionMode
from praxis.core.domain.enums import MakerPreference as _PraxisMakerPreference
from praxis.core.domain.enums import OrderSide as _PraxisOrderSide
from praxis.core.domain.enums import OrderType as _PraxisOrderType
from praxis.core.domain.enums import STPMode as _PraxisSTPMode
from praxis.core.domain.single_shot_params import (
    SingleShotParams as _PraxisSingleShotParams,
)

_log = logging.getLogger(__name__)

def _install_nexus_enum_shim() -> None:
    module = importlib.import_module('praxis.core.validate_trade_command')
    raw_allowed = getattr(module, '_ALLOWED_ORDER_TYPES')
    if not isinstance(raw_allowed, dict):
        msg = (
            f'praxis.core.validate_trade_command._ALLOWED_ORDER_TYPES must '
            f'be a dict, got {type(raw_allowed).__name__}'
        )
        raise TypeError(msg)
    allowed = cast('dict[Enum, frozenset[Enum]]', raw_allowed)
    praxis_by_name = {em.name: em for em in _PraxisExecutionMode}
    order_name: dict[str, Enum] = {ot.name: ot for ot in _PraxisOrderType}
    nexus_order_name: dict[str, Enum] = {ot.name: ot for ot in _NexusOrderType}
    for nexus_em in _NexusExecutionMode:
        praxis_em = praxis_by_name[nexus_em.name]
        existing = allowed[praxis_em]
        names = {ot.name for ot in existing}
        synonym_set: frozenset[Enum] = frozenset(
            {order_name[n] for n in names if n in order_name}
            | {nexus_order_name[n] for n in names if n in nexus_order_name},
        )
        allowed[praxis_em] = synonym_set
        allowed[nexus_em] = synonym_set

_install_nexus_enum_shim()

def _install_fake_datetime_serializer_shim() -> None:
    module = importlib.import_module('praxis.infrastructure.event_spine')
    original = getattr(module, '_serialize_default')
    if not callable(original):
        msg = (
            f'praxis.infrastructure.event_spine._serialize_default must '
            f'be callable, got {type(original).__name__}'
        )
        raise TypeError(msg)

    def _patched(obj: object) -> object:
        if isinstance(obj, datetime):
            return obj.isoformat()
        return original(obj)

    setattr(module, '_serialize_default', _patched)

_install_fake_datetime_serializer_shim()

_ACTION_TYPE_TO_VALIDATION_ACTION: dict[ActionType, ValidationAction] = {
    ActionType.ENTER: ValidationAction.ENTER,
    ActionType.EXIT: ValidationAction.EXIT,
    ActionType.MODIFY: ValidationAction.MODIFY,
    ActionType.ABORT: ValidationAction.ABORT,
}

_PRAXIS_ENUM_BY_FIELD: dict[str, type[Enum]] = {
    'side': _PraxisOrderSide,
    'order_type': _PraxisOrderType,
    'execution_mode': _PraxisExecutionMode,
    'maker_preference': _PraxisMakerPreference,
    'stp_mode': _PraxisSTPMode,
}

def _convert_enum_field(field_name: str, value: object) -> object:
    praxis_enum = _PRAXIS_ENUM_BY_FIELD.get(field_name)
    if praxis_enum is None or value is None or isinstance(value, praxis_enum):
        return value
    name = getattr(value, 'name', None)
    if name is None:
        return value
    try:
        return praxis_enum[name]
    except KeyError:
        _log.debug(
            'enum convert skipped: %s=%r has no counterpart in %s',
            field_name, value, praxis_enum.__name__,
        )
        return value

def _coerce_optional_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (str, int, float)):
        return Decimal(str(value))
    msg = (
        f'single-shot price field: expected Decimal|str|int|float|None, '
        f'got {type(value).__name__}'
    )
    raise TypeError(msg)

def _extract_single_shot_price_fields(params: object) -> dict[str, Decimal | None]:
    raw: dict[str, object] = {'price': None, 'stop_price': None, 'stop_limit_price': None}
    if isinstance(params, Mapping):
        typed_params = cast('Mapping[object, object]', params)
        for key in raw:
            raw[key] = typed_params.get(key)
    elif params is not None:
        for key in raw:
            if hasattr(params, key):
                raw[key] = getattr(params, key)
    return {key: _coerce_optional_decimal(value) for key, value in raw.items()}

def _wrap_single_shot_params(attrs: dict[str, object]) -> None:
    if attrs.get('execution_mode') is not _PraxisExecutionMode.SINGLE_SHOT:
        return
    if isinstance(attrs.get('execution_params'), _PraxisSingleShotParams):
        return
    extracted = _extract_single_shot_price_fields(attrs.get('execution_params'))
    declared_stop_price = extracted['stop_price']
    from praxis.core.domain.enums import OrderType as _PraxisOT
    stop_required = {
        _PraxisOT.STOP, _PraxisOT.STOP_LIMIT,
        _PraxisOT.TAKE_PROFIT, _PraxisOT.TP_LIMIT, _PraxisOT.OCO,
    }
    order_type_uses_stop = attrs.get('order_type') in stop_required
    stop_price_for_praxis = (
        extracted['stop_price'] if order_type_uses_stop else None
    )
    attrs['execution_params'] = _PraxisSingleShotParams(
        price=extracted['price'],
        stop_price=stop_price_for_praxis,
        stop_limit_price=extracted['stop_limit_price'],
    )
    attrs['declared_stop_price'] = declared_stop_price

def _to_praxis_enums(cmd: TradeCommand) -> TradeCommand:
    attrs: dict[str, object] = {
        field.name: _convert_enum_field(field.name, getattr(cmd, field.name))
        for field in dataclasses.fields(cmd)
    }
    _wrap_single_shot_params(attrs)
    return cast('TradeCommand', SimpleNamespace(**attrs))

@dataclasses.dataclass(frozen=True)
class SubmitterBindings:

    nexus_config: NexusInstanceConfig
    state: InstanceState
    praxis_outbound: PraxisOutbound
    validation_pipeline: ValidationPipeline
    capital_controller: CapitalController
    strategy_budget: Decimal
    touch_provider: Callable[[str], Decimal | None] | None = None
    tick_provider: Callable[[str], Decimal] | None = None

ReservationHook = Callable[[str, ValidationDecision, ValidationRequestContext, Action], None]
SubmitHook = Callable[[str], None]

def build_action_submitter(
    bindings: SubmitterBindings,
    *,
    on_reservation: ReservationHook | None = None,
    on_submit: SubmitHook | None = None,
) -> Callable[[list[Action], str], None]:
    def _submit(actions: list[Action], strategy_id: str) -> None:
        for raw_action in actions:
            action = _maybe_refresh_limit_to_touch(
                raw_action,
                touch_provider=bindings.touch_provider,
                tick_provider=bindings.tick_provider,
            )
            _log.info(
                'action_submit: type=%s direction=%s size=%s',
                action.action_type.value,
                action.direction.name if action.direction else None,
                action.size,
            )
            if action.action_type == ActionType.ABORT:
                _submit_abort(
                    bindings.praxis_outbound, bindings.nexus_config,
                    strategy_id, action,
                )
                continue
            result = _submit_translated(bindings, strategy_id, action)
            if result is None:
                continue
            command_id, decision, context = result
            if on_reservation is not None and decision.reservation is not None:
                on_reservation(command_id, decision, context, action)
            if on_submit is not None:
                on_submit(command_id)

    return _submit

def _maybe_refresh_limit_to_touch(
    action: Action,
    *,
    touch_provider: Callable[[str], Decimal | None] | None,
    tick_provider: Callable[[str], Decimal] | None,
) -> Action:
    if action.order_type is None or action.order_type.name != 'LIMIT':
        return action
    if touch_provider is None or tick_provider is None:
        return action
    if action.direction is None:
        return action
    params = dict(action.execution_params or {})
    symbol_raw = params.get('symbol')
    symbol = symbol_raw if isinstance(symbol_raw, str) else None
    if symbol is None:
        return action
    touch = touch_provider(symbol)
    if touch is None:
        return action
    tick = tick_provider(symbol)
    biased = (
        touch - tick if action.direction.name == 'BUY' else touch + tick
    )
    params['price'] = str(biased)
    return dataclasses.replace(action, execution_params=params)

def _submit_translated(
    bindings: SubmitterBindings,
    strategy_id: str,
    action: Action,
) -> tuple[str, ValidationDecision, ValidationRequestContext] | None:
    context = _build_context(
        config=bindings.nexus_config, state=bindings.state,
        strategy_id=strategy_id, action=action,
        strategy_budget=bindings.strategy_budget,
    )
    if action.direction == OrderSide.SELL:
        decision = ValidationDecision(allowed=True)
        cmd = translate_to_trade_command(
            action, context, decision, bindings.nexus_config, datetime.now(UTC),
        )
        cmd = _to_praxis_enums(cmd)
        command_id = bindings.praxis_outbound.send_command(cmd)
        _log.info(
            'backtest action submitted (SELL close — pipeline bypassed)',
            extra={
                'strategy_id': strategy_id,
                'action_type': action.action_type.value,
                'command_id': command_id,
            },
        )
        return command_id, decision, context
    decision = bindings.validation_pipeline.validate(context)
    if not decision.allowed:
        if decision.reservation is not None:
            release_result = bindings.capital_controller.release_reservation(
                decision.reservation.reservation_id,
            )
            if not release_result.success:
                msg = (
                    f'pipeline-denied reservation release failed: '
                    f'reservation_id={decision.reservation.reservation_id} '
                    f'failed_stage={decision.failed_stage} '
                    f'reason={release_result.reason!r} '
                    f'category={release_result.category}'
                )
                raise RuntimeError(msg)
        _log.warning(
            'validation denied: stage=%s reason_code=%s message=%s command_id=%s',
            decision.failed_stage.value if decision.failed_stage else None,
            decision.reason_code,
            decision.message,
            context.command_id,
        )
        return None
    cmd = translate_to_trade_command(
        action, context, decision, bindings.nexus_config, datetime.now(UTC),
    )
    cmd = _to_praxis_enums(cmd)
    command_id = bindings.praxis_outbound.send_command(cmd)
    _log.info(
        'backtest action submitted',
        extra={
            'strategy_id': strategy_id,
            'action_type': action.action_type.value,
            'command_id': command_id,
            'reservation_id': decision.reservation.reservation_id if decision.reservation else None,
        },
    )
    return command_id, decision, context

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
    strategy_id: str, action: Action, strategy_budget: Decimal,
) -> ValidationRequestContext:
    symbol = _extract_symbol(action)
    order_size = action.size or Decimal('0')
    base_notional = (
        order_size * action.reference_price
        if action.reference_price is not None else Decimal('0')
    )
    order_notional = base_notional * (Decimal('1') + NOTIONAL_RESERVATION_BUFFER)
    estimated_fees = order_notional * _FEE_ESTIMATE_RATE
    return ValidationRequestContext(
        strategy_id=strategy_id,
        order_notional=order_notional,
        estimated_fees=estimated_fees,
        strategy_budget=strategy_budget,
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

_FEE_ESTIMATE_RATE = Decimal('0.002')

NOTIONAL_RESERVATION_BUFFER = Decimal('0.07')

def _extract_symbol(action: Action) -> str:
    params = action.execution_params or {}
    raw = params.get('symbol')
    return raw if isinstance(raw, str) else 'BTCUSDT'

def _resolve_side(action: Action) -> OrderSide | None:
    if action.direction is None:
        return None
    return action.direction
