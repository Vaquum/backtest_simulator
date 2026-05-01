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

from backtest_simulator.honesty.atr import AtrSanityGate

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

    `touch_provider` (optional) returns the most recent pre-submit
    trade price for a symbol. When supplied AND the action is a
    LIMIT, the action_submitter rewrites
    `execution_params['price']` to `touch ± tick` (BUY: -tick,
    SELL: +tick) so the order rests strictly inside the touch.
    Pre-fix the same biasing happened inside the venue, which made
    `bts sweep --maker` execute a different price than the one
    the strategy emitted (codex round 4 P2). Doing the rewrite
    here keeps the touch decision in the action / validation
    audit trail; the venue just executes the price it receives.
    `tick_provider` returns the symbol's tick size (used for the
    bias amount). Both default to None, in which case the action
    is forwarded with its original `price`.
    """

    nexus_config: NexusInstanceConfig
    state: InstanceState
    praxis_outbound: PraxisOutbound
    validation_pipeline: ValidationPipeline
    # Slice #28 — pipeline.validate's CAPITAL stage runs before
    # HEALTH/PLATFORM_LIMITS, so a denial from a later stage carries
    # the reservation forward (per nexus pipeline_executor). Without
    # explicit release, available capital leaks. The action_submitter
    # uses this controller to release reservations attached to denied
    # decisions; the launcher constructs both pipeline + controller
    # from the same `build_validation_pipeline` call so they share
    # state. REQUIRED — making it optional would let a caller silently
    # regress to the leak shape (slice-#28 audit, finding 5).
    capital_controller: CapitalController
    strategy_budget: Decimal
    touch_provider: Callable[[str], Decimal | None] | None = None
    tick_provider: Callable[[str], Decimal] | None = None
    # Slice #17 Task 29 — ATR R-denominator gameability gate.
    # When BOTH non-None, ENTER+BUY whose declared stop is closer
    # than `gate.k * ATR(window)` from entry is denied at INTAKE
    # before `validation_pipeline.validate`. Provider closes over
    # the feed; `compute_atr_from_tape` does the math. Either
    # being None disables the gate.
    atr_gate: AtrSanityGate | None = None
    atr_provider: Callable[[str, datetime], Decimal | None] | None = None


ReservationHook = Callable[[str, ValidationDecision, ValidationRequestContext, Action], None]
SubmitHook = Callable[[str], None]
AtrRejectHook = Callable[[ValidationDecision, Action], None]


def build_action_submitter(
    bindings: SubmitterBindings,
    *,
    on_reservation: ReservationHook | None = None,
    on_submit: SubmitHook | None = None,
    on_atr_reject: AtrRejectHook | None = None,
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

    Slice #28: every Nexus pipeline stage runs a real `validate_*_stage`
    call. INTAKE / RISK / PRICE / CAPITAL / HEALTH / PLATFORM_LIMITS are
    all backed by their respective Nexus validators with MMVP-lenient
    defaults; operator-supplied limits + snapshot providers in
    `backtest_simulator.honesty.capital.build_validation_pipeline` dial
    in real denial behavior. CAPITAL runs before HEALTH/PLATFORM_LIMITS
    in stage order, so a late-stage denial that carries the reservation
    forward triggers an explicit `capital_controller.release_reservation`
    in `_submit_translated`. ABORT flows through `praxis_outbound.send_abort`
    unchanged.
    """
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
            result = _submit_translated(
                bindings, strategy_id, action,
                on_atr_reject=on_atr_reject,
            )
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
    """Rewrite a LIMIT action's `execution_params['price']` to touch ± tick.

    Returns the input action unchanged for non-LIMIT actions, when
    no providers are supplied, or when the touch provider returns
    None (no recent trade). Otherwise constructs a new Action via
    `dataclasses.replace` with updated `execution_params`. The
    biased price is `touch - tick` for BUY (post just inside the
    bid) and `touch + tick` for SELL (post just inside the ask).

    The rewrite happens BEFORE validation so the audit trail
    (event_spine, ValidationRequestContext, TradeCommand) all see
    the same price the venue eventually executes — a single
    source of truth for sweep economics. Codex round 4 P2 caught
    the prior shape: the rewrite was venue-side and silently
    diverged from the strategy's emitted price.
    """
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
    *,
    on_atr_reject: AtrRejectHook | None = None,
) -> tuple[str, ValidationDecision, ValidationRequestContext] | None:
    context = _build_context(
        config=bindings.nexus_config, state=bindings.state,
        strategy_id=strategy_id, action=action,
        strategy_budget=bindings.strategy_budget,
    )
    # BTS-side INTAKE pre-hook supplementing the real Nexus INTAKE stage.
    # Slice #28 lit up the pipeline's INTAKE with real
    # `validate_intake_stage(...)` checks; these two hooks layer on top
    # because the data they need (`execution_params['stop_price']`) is
    # not on `ValidationRequestContext`. Closure path: a Nexus PR
    # extending `ValidationRequestContext.execution_params`. Long-only
    # strategy convention — BUY opens a long (requires protective stop),
    # SELL closes a long (is itself the exit, no new stop).
    intake_decision = _check_declared_stop(action, context)
    if intake_decision is None:
        intake_decision = _check_atr_sanity(
            action, context,
            gate=bindings.atr_gate, atr_provider=bindings.atr_provider,
            touch_provider=bindings.touch_provider,
        )
        if intake_decision is not None and on_atr_reject is not None:
            on_atr_reject(intake_decision, action)
    if intake_decision is not None:
        _log.warning(
            'validation denied: stage=%s reason_code=%s message=%s command_id=%s',
            intake_decision.failed_stage.value if intake_decision.failed_stage else None,
            intake_decision.reason_code, intake_decision.message,
            context.command_id,
        )
        return None
    # SELL fast-path: long-only convention treats `SELL+ENTER` as the
    # close of an open long. The fast-path bypasses
    # `validation_pipeline.validate` because (a) the bts long-only
    # strategy template emits SELLs without propagating the BUY's
    # `trade_id`, so `make_reference_integrity_hook` would deny every
    # close with `INTAKE_TRADE_REFERENCE_INVALID`; (b) `CapitalController`
    # has no `close_position` primitive, so a real CAPITAL stage on
    # SELL would over-reserve. Closure path: a Nexus PR adding
    # `close_position` AND a strategy-template change to propagate
    # `trade_id` from BUY to SELL. Both upstream; tracked as
    # follow-up. The bts-side close accounting still lands in the
    # launcher's adapter wrapper via
    # `CapitalLifecycleTracker.record_close_position`, which mirrors
    # `CapitalController.order_fill` exactly: releases
    # `cost_basis + entry_fees` from `position_notional` and
    # decrements `per_strategy_deployed[strategy_id]`. `capital_pool`
    # stays untouched.
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
        # Slice #28: CAPITAL is stage 4 of 6, so HEALTH/PLATFORM_LIMITS
        # denials carry the CAPITAL-stage reservation forward in the
        # denied decision (nexus pipeline_executor preserves it). The
        # action is dropped without reaching Praxis, so no fill / no
        # `order_ack` / no `order_fill` will retire the reservation.
        # Release it explicitly so available capital is not leaked.
        if decision.reservation is not None:
            release_result = bindings.capital_controller.release_reservation(
                decision.reservation.reservation_id,
            )
            if not release_result.success:
                # Fail loud — the reservation was just minted by the same
                # pipeline.validate() call on the same controller; any
                # non-success here is a wiring/state bug (controller
                # mismatch, concurrent release, lost reservation), not an
                # expected race. Mirrors `CapitalLifecycleTracker.record_rejection`'s
                # pattern at honesty/capital.py:461. Letting the bug
                # through with a warning would silently re-introduce
                # the very leak this branch exists to prevent
                # (bit-mis no-defects-conviction P1).
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
NOTIONAL_RESERVATION_BUFFER = Decimal('0.07')


def _check_declared_stop(
    action: Action, context: ValidationRequestContext,
) -> ValidationDecision | None:
    """Deny BUY-ENTER that lacks a declared stop_price.

    BTS-side pre-hook supplementing the real Nexus INTAKE stage.
    Slice #28 lit up `validate_intake_stage` (order-rate, duplicate-
    window, reference-integrity, mode, notional, budget). This hook
    runs in addition because `ValidationRequestContext` does not carry
    `action.execution_params['stop_price']` and the stop check needs
    that field. Closure path: a Nexus PR extending
    `ValidationRequestContext.execution_params` lets this hook
    collapse into the pipeline's INTAKE stage.

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


def _check_atr_sanity(
    action: Action, context: ValidationRequestContext, *,
    gate: AtrSanityGate | None,
    atr_provider: Callable[[str, datetime], Decimal | None] | None,
    touch_provider: Callable[[str], Decimal | None] | None = None,
) -> ValidationDecision | None:
    """Reject ENTER+BUY whose declared stop is closer than `k * ATR(window)`.

    BTS-side pre-hook supplementing the real Nexus INTAKE stage. The
    pipeline's INTAKE runs `validate_intake_stage` with the standard
    Praxis hooks; this ATR-floor hook runs in addition because the
    data it needs (`execution_params['stop_price']` plus the symbol's
    pre-decision ATR window) is not on `ValidationRequestContext`.
    Closure path: a Nexus PR extending `ValidationRequestContext` so
    this floor lands inside `validate_intake_stage` itself.

    Slice #17 Task 29 closes the R-denominator gameability vector
    `_check_declared_stop` only half-blocks: a 1 bp stop reaches
    Praxis untouched and the R̄ in `bts sweep` becomes a knob the
    strategy can dial by tightening stops. Returns `None` when
    the gate is disabled, when not ENTER+BUY, when stop is
    missing (caller already handled), or when ATR allows. Else
    returns a denial with reason_code prefixed `ATR_`.

    BTS-only floor — paper/live measures realized PnL, bts measures
    R-multiples on R̄, and only the latter is gameable by tightening
    stops. Upstreaming this floor to paper/live requires either (a)
    the Nexus context extension above, or (b) moving the floor into
    the strategy template itself so it runs upstream of any
    deployment. Both are out of slice scope.

    Entry-price proxy priority (codex round 1 P1: must match the
    R-denominator's actual entry — a stale seed price drifts vs
    the declared stop and lets gameability leak through):
      1. LIMIT `execution_params['price']` when set
         (`_maybe_refresh_limit_to_touch` may have rewritten this
         to `touch ± tick` for maker posts).
      2. `touch_provider(symbol)` — most recent pre-submit trade.
      3. `action.reference_price` — the baked window-start seed.

    Long-only: only BUY entries are gated; short-side requires
    intent plumbing (same constraint as strict-impact).
    """
    if gate is None or atr_provider is None:
        return None
    # `k=0` is the standalone gate's "disabled" knob. Honor it
    # BEFORE calling `atr_provider` so a disabled gate never
    # rejects on ATR_UNCALIBRATED for an empty pre-decision
    # tape. Symmetric to `AtrSanityGate.evaluate`'s own k=0
    # short-circuit. Codex round 4 P1 caught this.
    if gate.k == Decimal('0'):
        return None
    if action.action_type != ActionType.ENTER or action.direction != OrderSide.BUY:
        return None
    params = action.execution_params
    stop_raw = params.get('stop_price') if isinstance(params, Mapping) else None
    if stop_raw is None or str(stop_raw).strip() in ('', 'None'):
        return None
    # Decimal coercion fails loud on malformed stop_price strings.
    # `_check_declared_stop` already validates non-blank; a string
    # that parses-to-blank but is otherwise non-Decimal is a
    # strategy template bug, not something to silently bypass.
    stop_price = Decimal(str(stop_raw))
    entry_price = _resolve_atr_entry_price(action, touch_provider)
    if entry_price is None:
        return None
    atr = atr_provider(_extract_symbol(action), datetime.now(UTC))
    if atr is None:
        return ValidationDecision(
            allowed=False, failed_stage=ValidationStage.INTAKE,
            reason_code='ATR_UNCALIBRATED',
            message=(
                f'ENTER BUY command_id={context.command_id} ATR uncalibrated '
                f'(empty pre-decision tape over configured window).'
            ),
        )
    decision = gate.evaluate(
        entry_price=entry_price, stop_price=stop_price, atr=atr,
    )
    if decision.allowed:
        return None
    return ValidationDecision(
        allowed=False, failed_stage=ValidationStage.INTAKE,
        reason_code=f'ATR_{(decision.reason or "denied").upper()}',
        message=(
            f'ENTER BUY command_id={context.command_id} stop_distance='
            f'{decision.stop_distance} < min={decision.min_required_distance} '
            f'(atr={atr}, k={gate.k}, window={gate.atr_window_seconds}s).'
        ),
    )


def _resolve_atr_entry_price(
    action: Action,
    touch_provider: Callable[[str], Decimal | None] | None,
) -> Decimal | None:
    """Best-effort live-entry proxy for the ATR gate.

    Codex round 1 P1: the gate's entry-distance comparison must
    track the same value R̄'s denominator will use, not the
    stale window-start seed price. Falls back through (LIMIT
    rewritten price, current touch, seed) — see
    `_check_atr_sanity` docstring.
    """
    params = action.execution_params if isinstance(action.execution_params, Mapping) else None
    if action.order_type is not None and action.order_type.name == 'LIMIT' and params is not None:
        limit_raw = params.get('price')
        if limit_raw is not None and str(limit_raw).strip() not in ('', 'None'):
            # Decimal coercion fails loud on malformed LIMIT price.
            # `execution_params['price']` is set by
            # `_maybe_refresh_limit_to_touch`'s own Decimal-arithmetic
            # write or by the strategy template; either way a
            # non-Decimal string is a template bug.
            return Decimal(str(limit_raw))
    if touch_provider is not None and params is not None:
        symbol = params.get('symbol')
        if isinstance(symbol, str):
            touch = touch_provider(symbol)
            if touch is not None:
                return touch
    return action.reference_price


def _extract_symbol(action: Action) -> str:
    params = action.execution_params or {}
    raw = params.get('symbol')
    return raw if isinstance(raw, str) else 'BTCUSDT'


def _resolve_side(action: Action) -> OrderSide | None:
    if action.direction is None:
        return None
    return action.direction
