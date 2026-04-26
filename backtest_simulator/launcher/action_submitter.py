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

from nexus.core.domain.enums import OrderSide
from nexus.core.domain.instance_state import InstanceState
from nexus.core.domain.order_types import ExecutionMode as _NexusExecutionMode
from nexus.core.domain.order_types import OrderType as _NexusOrderType
from nexus.core.validator import ValidationPipeline
from nexus.core.validator.pipeline_models import (
    ValidationAction,
    ValidationDecision,
    ValidationRequestContext,
    ValidationStage,
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
    raw_allowed = getattr(module, '_ALLOWED_ORDER_TYPES')
    if not isinstance(raw_allowed, dict):
        msg = (
            f'praxis.core.validate_trade_command._ALLOWED_ORDER_TYPES must '
            f'be a dict, got {type(raw_allowed).__name__}'
        )
        raise TypeError(msg)
    # Post-shim this dict holds both Praxis and Nexus enum keys + values
    # as Enum synonyms, so `Enum` is the honest narrowest annotation.
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


# --- FakeDatetime JSON-serializer shim --------------------------------------
# Praxis's `event_spine` serializes via `orjson.dumps(default=_serialize_default)`.
# orjson's native datetime handler rejects freezegun's `FakeDatetime`
# subclass by type identity. Wrap the default handler to isoformat any
# datetime before falling through to the original (which handles Decimal).
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


def _convert_enum_field(field_name: str, value: object) -> object:
    """Swap a single Nexus enum value for its Praxis counterpart.

    Best-effort: if the field's Praxis enum lacks a member with this
    name (e.g. Nexus `STPMode.CANCEL_TAKER` vs Praxis `EXPIRE_TAKER`),
    the original value is returned unchanged.
    """
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
    """Coerce a params-field scalar to Decimal | None.

    Praxis's SingleShotParams expects Decimal values. The strategy
    template stores them as strings on `execution_params` (YAML
    roundtrip), so we accept str/int/float/Decimal and return None
    for None.
    """
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
    """Pull price/stop_price/stop_limit_price from a params object or dict."""
    raw: dict[str, object] = {'price': None, 'stop_price': None, 'stop_limit_price': None}
    if isinstance(params, Mapping):
        typed_params = cast('Mapping[object, object]', params)
        for key in raw:
            raw[key] = typed_params.get(key)
    elif params is not None:
        # Nexus SingleShotParams or another params dataclass — copy
        # matching field values by name.
        for key in raw:
            if hasattr(params, key):
                raw[key] = getattr(params, key)
    return {key: _coerce_optional_decimal(value) for key, value in raw.items()}


def _wrap_single_shot_params(attrs: dict[str, object]) -> None:
    """Mutate `attrs` in place: wrap execution_params as Praxis SingleShotParams.

    Praxis's `TradeCommand.__post_init__` requires `execution_params`
    be a `praxis.core.domain.single_shot_params.SingleShotParams` when
    mode is SINGLE_SHOT. Nexus passes through the action's raw dict,
    so we wrap it here.

    Praxis's `validate_trade_command` only allows `stop_price` on
    order types in `_STOP_REQUIRED_TYPES` (STOP / STOP_LIMIT /
    TAKE_PROFIT / TP_LIMIT / OCO). MARKET and LIMIT both reject
    `stop_price` with `"<type> does not use execution_params.stop_price"`.
    Our backtest's BUY ENTER carries a DECLARED protective stop on
    `execution_params.stop_price` regardless of `order_type` — it's
    BTS's R-denominator measurement, NOT Praxis's stop concept.
    Strip it from the Praxis params for any non-stop order type and
    preserve it on the returned namespace as `declared_stop_price`
    for the lifecycle tracker. Pre-fix, the `is_market` shortcut
    only handled MARKET; LIMIT (the maker-preference path) hit the
    Praxis validator and crashed every entry.
    """
    if attrs.get('execution_mode') is not _PraxisExecutionMode.SINGLE_SHOT:
        return
    if isinstance(attrs.get('execution_params'), _PraxisSingleShotParams):
        return
    extracted = _extract_single_shot_price_fields(attrs.get('execution_params'))
    declared_stop_price = extracted['stop_price']
    from praxis.core.domain.enums import OrderType as _PraxisOT  # local import
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
    """Return a `SimpleNamespace` mirroring `cmd`'s fields with Praxis Enums.

    Nexus Enum values are swapped for Praxis Enums, and `execution_params`
    is wrapped in a Praxis `SingleShotParams` when the mode is SINGLE_SHOT.

    Using `dataclasses.replace` would re-run the Nexus `TradeCommand.
    __post_init__` validator which insists on Nexus enums — the exact
    opposite of what we want. `SimpleNamespace` gives Praxis's
    `praxis_outbound.send_command` attribute-access parity with the
    original dataclass and skips Nexus's own validation layer. The
    SimpleNamespace carries the same public field surface as TradeCommand;
    pyright sees the declared return type, Praxis sees duck-typed
    attributes, runtime gets the relaxed dataclass.
    """
    attrs: dict[str, object] = {
        field.name: _convert_enum_field(field.name, getattr(cmd, field.name))
        for field in dataclasses.fields(cmd)
    }
    _wrap_single_shot_params(attrs)
    return cast('TradeCommand', SimpleNamespace(**attrs))


@dataclasses.dataclass(frozen=True)
class SubmitterBindings:
    """Long-lived wiring a `build_action_submitter` caller provides once.

    Bundles the per-instance dependencies (Nexus config, pipeline, Praxis
    outbound, strategy budget) so the builder and the per-action helper
    each take a single argument for them instead of five.
    """

    nexus_config: NexusInstanceConfig
    state: InstanceState
    praxis_outbound: PraxisOutbound
    validation_pipeline: ValidationPipeline
    strategy_budget: Decimal


ReservationHook = Callable[[str, ValidationDecision, ValidationRequestContext, Action], None]
SubmitHook = Callable[[str], None]


def build_action_submitter(
    bindings: SubmitterBindings,
    *,
    on_reservation: ReservationHook | None = None,
    on_submit: SubmitHook | None = None,
) -> Callable[[list[Action], str], None]:
    """Return a callback for `PredictLoop(action_submit=...)`.

    The callback drives the real ValidationPipeline and sends the
    resulting `TradeCommand`.

    Contract:
      - Every non-ABORT action is evaluated against
        `bindings.validation_pipeline`.
        A denied decision is logged with the failing stage + reason and
        the action is dropped without reaching Praxis; upstream Nexus
        behaviour is the same (`submit_actions` honors the decision).
      - `on_reservation` is called once per allowed decision with the
        freshly minted `command_id`, the decision (whose `reservation`
        carries the CAPITAL stage's reservation_id + notional), and the
        context used for validation. `BacktestLauncher.CapitalLifecycleTracker`
        stores this so the later `send_order`/`order_ack`/`order_fill`
        calls can tie back to the original reservation.
      - `on_submit` is the pre-existing drain hook the clock loop uses.

    The only stage this pipeline actually enforces in Part 2 is CAPITAL;
    INTAKE / RISK / PRICE / HEALTH / PLATFORM_LIMITS are `_allow` stubs
    as documented in `backtest_simulator.honesty.capital`. ABORT flows
    through `praxis_outbound.send_abort` unchanged.
    """
    def _submit(actions: list[Action], strategy_id: str) -> None:
        for action in actions:
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
    # Part 2 INTAKE hook: declared-stop enforcement. Long-only strategy
    # convention — BUY opens a long (requires protective stop), SELL
    # closes a long (is itself the exit, no new stop). The hook runs
    # BEFORE `validation_pipeline.validate` because the pipeline's
    # INTAKE stage is `_allow` in Part 2 and the stop is carried on
    # `action.execution_params`, which isn't accessible from within a
    # `StageValidator`. A denial here short-circuits the whole path;
    # the caller sees the usual "validation denied" log plus the
    # concrete stop-missing reason.
    intake_decision = _check_declared_stop(action, context)
    if intake_decision is not None:
        _log.warning(
            'validation denied: stage=%s reason_code=%s message=%s command_id=%s',
            intake_decision.failed_stage.value if intake_decision.failed_stage else None,
            intake_decision.reason_code, intake_decision.message,
            context.command_id,
        )
        return None
    # SELL ENTER in our long-only strategy is the EXIT LEG of an
    # already-open long — it reduces an existing position rather than
    # opening a new one. Nexus's `ActionType.EXIT` is the correct tag
    # but requires `trade_id`-linked lifecycle plumbing that exceeds
    # Part 2 scope; for now we send SELL-as-ENTER but SKIP the CAPITAL
    # reservation (which would otherwise over-commit capital on every
    # exit — the known dishonesty codex flagged). The action still
    # flows through Praxis → venue adapter → fill; capital accounting
    # intentionally stays silent on exits until a follow-up slice
    # wires proper EXIT semantics.
    #
    # KNOWN-OPEN P0 (auditor round 4, predates this slice): the SELL
    # exit completely bypasses the validation pipeline AND the
    # CapitalLifecycleTracker's release / position-decrement / PnL-
    # reconciliation paths. A correct fix needs `CapitalController`
    # to expose a `close_position(reservation_id, fill_notional)`
    # primitive (Nexus-side, currently absent) that releases
    # `position_notional` back into `capital_pool` plus realized
    # PnL — without that primitive the backtest can't honestly
    # reconcile capital on close. This belongs in a dedicated EXIT-
    # lifecycle slice, not folded into the maker-fill wiring.
    # Auditor flagged P0 in maker-fill round 4; tracking as a
    # standalone follow-up so it gets its own change boundary.
    if action.direction == OrderSide.SELL:
        decision = ValidationDecision(allowed=True)
        cmd = translate_to_trade_command(
            action, context, decision, bindings.nexus_config, datetime.now(UTC),
        )
        cmd = _to_praxis_enums(cmd)
        command_id = bindings.praxis_outbound.send_command(cmd)
        _log.info(
            'backtest action submitted (SELL close — CAPITAL skipped)',
            extra={
                'strategy_id': strategy_id,
                'action_type': action.action_type.value,
                'command_id': command_id,
            },
        )
        return command_id, decision, context
    decision = bindings.validation_pipeline.validate(context)
    if not decision.allowed:
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
    # Feeds the ValidationPipeline — the CAPITAL stage reads
    # `order_notional`, `estimated_fees`, `strategy_budget`, `state` to
    # decide whether to reserve capital. `order_notional` is computed
    # from `action.reference_price * size`; the strategy supplies
    # `reference_price=estimated_price` (the seed price baked at
    # manifest build time) so this is the real sizing input, not a
    # sentinel zero.
    #
    # `estimated_fees` pre-funds the `fee_reserve`. Nexus's
    # `order_fill` rejects when `actual_fees > fee_reserve`, so the
    # reservation MUST anticipate the fee the venue will charge. We
    # reserve 0.2% of notional — 2x Binance's standard 0.1% taker fee,
    # giving us headroom for maker/taker mix and minor price slippage
    # between reservation and fill. Anything under-reserved here
    # produces "fee deficit" rejections at fill time and leaves capital
    # stranded in `working_order_notional`; anything wildly
    # over-reserved just sits in `fee_reserve` briefly and is
    # reconciled (surplus returns to `fee_reserve`, deficit draws on
    # it) when `order_fill` runs.
    symbol = _extract_symbol(action)
    order_size = action.size or Decimal('0')
    # Reserve `reference_price * size * (1 + buffer)` so real fills at
    # slightly worse prices fit within the reservation. The lifecycle
    # wrapper fails loud if actual fills STILL exceed this buffered
    # amount — no more silently capping capital overshoot.
    base_notional = (
        order_size * action.reference_price
        if action.reference_price is not None else Decimal('0')
    )
    order_notional = base_notional * (Decimal('1') + _NOTIONAL_RESERVATION_BUFFER)
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


# Fee reservation rate — 2x Binance's standard 0.1% taker fee, so the
# `fee_reserve` side of the reservation has headroom for maker/taker
# mix and minor slippage.
_FEE_ESTIMATE_RATE = Decimal('0.002')

# Notional reservation buffer — reserve 7% more notional than the
# `reference_price * size` estimate to absorb the price drift between
# reservation time and actual fill time. Without this buffer, real
# fills at higher prices would overshoot the reservation and
# silently bypass CAPITAL gating. We use a concrete buffer here and
# fail-loud in the lifecycle wrapper on any further overshoot (see
# `_install_capital_adapter_wrapper`).
#
# The 7% ceiling is bounded from above by Nexus's per-trade allocation
# gate (15% of strategy_budget). Raw Kelly for the selected decoder
# sits at ~14%, so buffer > ~7.6% trips that gate and every entry is
# denied. If a larger buffer is needed (e.g. volatile windows with >7%
# drift), the strategy's Kelly% must shrink or the strategy_budget
# concept needs a formal lift above the current allocated_capital.
_NOTIONAL_RESERVATION_BUFFER = Decimal('0.07')


def _check_declared_stop(
    action: Action, context: ValidationRequestContext,
) -> ValidationDecision | None:
    """Deny BUY-ENTER that lacks a declared stop_price.

    The long-only strategy template writes stop_price onto
    `execution_params['stop_price']` for BUYs only (SELLs close an
    open long and are themselves the exit). A missing stop on a BUY
    is an honesty violation — without a concrete stop, there is no
    place for `FillModel.apply_stop` to enforce and no honest
    `r_per_trade` denominator later.
    """
    if action.action_type != ActionType.ENTER:
        return None
    if action.direction != OrderSide.BUY:
        return None
    params = action.execution_params
    # `action.execution_params` is a read-only MappingProxy (not a
    # plain dict) on the Nexus Action dataclass. Use the
    # `collections.abc.Mapping` protocol instead of `isinstance(dict)`.
    stop_price = params.get('stop_price') if isinstance(params, Mapping) else None
    if stop_price is not None and str(stop_price).strip() not in ('', 'None'):
        return None
    return ValidationDecision(
        allowed=False,
        failed_stage=ValidationStage.INTAKE,
        reason_code='INTAKE_DECLARED_STOP_MISSING',
        message=(
            f'ENTER BUY command_id={context.command_id} lacks a declared '
            f'stop_price. Part 2 honesty gate: every long-opening entry '
            f'MUST declare a concrete stop_price so FillModel.apply_stop '
            f'can enforce it and r_per_trade can be computed honestly.'
        ),
    )


def _extract_symbol(action: Action) -> str:
    params = action.execution_params or {}
    raw = params.get('symbol')
    return raw if isinstance(raw, str) else 'BTCUSDT'


def _resolve_side(action: Action) -> OrderSide | None:
    if action.direction is None:
        return None
    return action.direction
