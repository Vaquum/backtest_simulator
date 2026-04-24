"""Backtest action-submit callback — bypass the full validator, land real TradeCommands."""
from __future__ import annotations

import dataclasses
import importlib
import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from types import SimpleNamespace
from typing import cast

from nexus.core.domain.enums import OrderSide
from nexus.core.domain.order_types import ExecutionMode as _NexusExecutionMode
from nexus.core.domain.order_types import OrderType as _NexusOrderType
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
from praxis.core.domain.enums import ExecutionMode as _PraxisExecutionMode
from praxis.core.domain.enums import MakerPreference as _PraxisMakerPreference
from praxis.core.domain.enums import OrderSide as _PraxisOrderSide
from praxis.core.domain.enums import OrderType as _PraxisOrderType
from praxis.core.domain.enums import STPMode as _PraxisSTPMode
from praxis.core.domain.single_shot_params import (
    SingleShotParams as _PraxisSingleShotParams,
)

_log = logging.getLogger(__name__)


# --- Nexus/Praxis enum bridge -----------------------------------------------
# Nexus and Praxis each define their own `ExecutionMode` / `OrderType`
# enums. Same names + same values, different class objects. Praxis's
# `validate_trade_command._ALLOWED_ORDER_TYPES` dict is keyed on Praxis
# enums and returns None for a Nexus-enum TradeCommand; the validator
# then raises "no allowed order types configured for mode SINGLE_SHOT".
# Extend the dict with Nexus-keyed entries that accept both enum classes
# as synonyms. Runs once at module import.
def _install_nexus_enum_shim() -> None:
    module = importlib.import_module('praxis.core.validate_trade_command')
    allowed: dict = module._ALLOWED_ORDER_TYPES  # noqa: SLF001 - cross-package shim
    praxis_by_name = {em.name: em for em in _PraxisExecutionMode}
    order_name = {ot.name: ot for ot in _PraxisOrderType}
    nexus_order_name = {ot.name: ot for ot in _NexusOrderType}
    for nexus_em in _NexusExecutionMode:
        praxis_em = praxis_by_name[nexus_em.name]
        existing = allowed[praxis_em]
        names = {ot.name for ot in existing}
        synonym_set = frozenset(
            {order_name[n] for n in names if n in order_name}
            | {nexus_order_name[n] for n in names if n in nexus_order_name},
        )
        allowed[praxis_em] = synonym_set
        allowed[nexus_em] = synonym_set


_install_nexus_enum_shim()


# --- FakeDatetime JSON-serializer shim --------------------------------------
# Praxis's `event_spine` serializes via `orjson.dumps(default=_serialize_default)`.
# orjson's native datetime handler rejects freezegun's `FakeDatetime`
# subclass by type identity. Wrap the default handler to isoformat any
# datetime before falling through to the original (which handles Decimal).
def _install_fake_datetime_serializer_shim() -> None:
    module = importlib.import_module('praxis.infrastructure.event_spine')
    original = module._serialize_default  # noqa: SLF001 - cross-package shim

    def _patched(obj: object) -> object:
        if isinstance(obj, datetime):
            return obj.isoformat()
        return original(obj)

    module._serialize_default = _patched  # noqa: SLF001


_install_fake_datetime_serializer_shim()


_ACTION_TYPE_TO_VALIDATION_ACTION: dict[ActionType, ValidationAction] = {
    ActionType.ENTER: ValidationAction.ENTER,
    ActionType.EXIT: ValidationAction.EXIT,
    ActionType.MODIFY: ValidationAction.MODIFY,
    ActionType.ABORT: ValidationAction.ABORT,
}


# --- Nexus -> Praxis enum converters ----------------------------------------
# Nexus and Praxis define separate Enum classes for the same domain
# concepts (ExecutionMode, OrderType, OrderSide, MakerPreference,
# STPMode). Python's `Enum.__eq__` is identity-based, so
# `NexusExecutionMode.SINGLE_SHOT == PraxisExecutionMode.SINGLE_SHOT`
# evaluates False — and Praxis's `_process_command` contains a direct
# `cmd.execution_mode != PraxisExecutionMode.SINGLE_SHOT` check that
# rejects every Nexus-enum command with
# "unsupported execution mode ... mode=SINGLE_SHOT".
#
# `translate_to_trade_command` (from Nexus) builds a Praxis `TradeCommand`
# but leaves the Action's Nexus-typed enums on the dataclass. We normalize
# by name right before `praxis_outbound.send_command` — the rest of the
# Praxis pipeline then sees Praxis-typed enums and identity compares pass.
_PRAXIS_ENUM_BY_FIELD: dict[str, type[Enum]] = {
    'side': _PraxisOrderSide,
    'order_type': _PraxisOrderType,
    'execution_mode': _PraxisExecutionMode,
    'maker_preference': _PraxisMakerPreference,
    'stp_mode': _PraxisSTPMode,
}


def _to_praxis_enums(cmd: object) -> object:
    """Return a `SimpleNamespace` mirroring `cmd`'s fields with Nexus Enum
    values swapped for Praxis Enums, and `execution_params` wrapped in a
    Praxis `SingleShotParams` when the mode is SINGLE_SHOT.

    Using `dataclasses.replace` would re-run the Nexus `TradeCommand.
    __post_init__` validator which insists on Nexus enums — the exact
    opposite of what we want. `SimpleNamespace` gives Praxis's
    `praxis_outbound.send_command` attribute-access parity with the
    original dataclass and skips Nexus's own validation layer, so
    Praxis's internal `TradeCommand` (constructed inside
    `execution_manager.submit_command` from our kwargs) gets Praxis
    enums and identity comparisons like
    `cmd.execution_mode != ExecutionMode.SINGLE_SHOT` pass.

    Best-effort on enum mapping: if a field's name is not present in
    the corresponding Praxis Enum (e.g. Nexus `STPMode.CANCEL_TAKER`
    vs Praxis `STPMode.EXPIRE_TAKER`), the Nexus value is left in
    place rather than raising. If a downstream Praxis check rejects
    it, the rejection surfaces as a clear error on the next run.
    """
    attrs: dict[str, object] = {}
    for field in dataclasses.fields(cmd):
        value = getattr(cmd, field.name)
        praxis_enum = _PRAXIS_ENUM_BY_FIELD.get(field.name)
        if (
            praxis_enum is not None
            and value is not None
            and not isinstance(value, praxis_enum)
        ):
            name = getattr(value, 'name', None)
            if name is not None:
                try:
                    value = praxis_enum[name]
                except KeyError:
                    _log.debug(
                        'enum convert skipped: %s=%r has no counterpart in %s',
                        field.name, value, praxis_enum.__name__,
                    )
        attrs[field.name] = value
    # Praxis's `TradeCommand.__post_init__` requires `execution_params`
    # be a `praxis.core.domain.single_shot_params.SingleShotParams` when
    # mode is SINGLE_SHOT. Nexus `translate_to_trade_command` just
    # passes `action.execution_params` through unchanged — in our
    # strategy template that's a dict (`{'symbol': ..., 'stop_bps': ...}`)
    # meant for validator context, not a params object. Wrap it into a
    # Praxis `SingleShotParams`; MARKET orders have no price/stop fields
    # so the default-empty instance is correct.
    mode = attrs.get('execution_mode')
    if mode is _PraxisExecutionMode.SINGLE_SHOT and not isinstance(
        attrs.get('execution_params'), _PraxisSingleShotParams,
    ):
        params = attrs.get('execution_params')
        extracted: dict[str, object | None] = {
            'price': None, 'stop_price': None, 'stop_limit_price': None,
        }
        if isinstance(params, dict):
            for key in extracted:
                extracted[key] = params.get(key)
        elif params is not None:
            # Nexus SingleShotParams or another params dataclass — copy
            # matching field values by name.
            for key in extracted:
                if hasattr(params, key):
                    extracted[key] = getattr(params, key)
        attrs['execution_params'] = _PraxisSingleShotParams(
            price=extracted['price'],
            stop_price=extracted['stop_price'],
            stop_limit_price=extracted['stop_limit_price'],
        )
    return SimpleNamespace(**attrs)


def build_action_submitter(
    *, nexus_config: NexusInstanceConfig,
    state: InstanceState,
    praxis_outbound: PraxisOutbound,
    on_submit: Callable[[str], None] | None = None,
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

    `on_submit`, when provided, is called once per successful send with
    the returned `command_id`. `BacktestLauncher` uses this to drive the
    synchronous drain: the clock waits for every submitted command_id
    to have been dispatched to the venue adapter before advancing.
    """
    def _submit(actions: list[Action], strategy_id: str) -> None:
        for action in actions:
            if action.action_type == ActionType.ABORT:
                _submit_abort(praxis_outbound, nexus_config, strategy_id, action)
            else:
                command_id = _submit_translated(
                    praxis_outbound, nexus_config, state, strategy_id, action,
                )
                if on_submit is not None:
                    on_submit(command_id)

    return _submit


def _submit_translated(
    praxis_outbound: PraxisOutbound,
    config: NexusInstanceConfig,
    state: InstanceState,
    strategy_id: str,
    action: Action,
) -> str:
    context = _build_context(config=config, state=state, strategy_id=strategy_id, action=action)
    decision = ValidationDecision(allowed=True)
    cmd = translate_to_trade_command(action, context, decision, config, datetime.now(UTC))
    cmd = _to_praxis_enums(cmd)
    command_id = praxis_outbound.send_command(cmd)
    _log.info(
        'backtest action submitted',
        extra={
            'strategy_id': strategy_id,
            'action_type': action.action_type.value,
            'command_id': command_id,
        },
    )
    return command_id


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
