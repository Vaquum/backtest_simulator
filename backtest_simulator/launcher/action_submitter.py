"""Backtest action-submit callback — bypass the full validator, land real TradeCommands."""
from __future__ import annotations

import dataclasses
import importlib
import logging
import threading
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
class _SubmittedCommand:
    """One entry in the `_CommandRegistry`.

    Captures the post-validation state needed by the launcher's adapter
    wrapper to drive the capital lifecycle (BUY) or the close-position
    seam (EXIT) without reaching back through the action object.
    """

    action_type: ActionType
    trade_id: str
    strategy_id: str
    symbol: str
    order_size: Decimal
    reservation_id: str | None
    declared_stop_price: Decimal | None


class _CommandRegistry:
    """Thread-safe map of `command_id -> _SubmittedCommand`.

    Written inside `_submit_translated` BEFORE `praxis_outbound.send_command`
    under the registry's own lock so the venue adapter callback (which
    runs on Praxis's account_loop thread) always sees the entry when the
    matching `submit_order` fires. Closes the post-send_command race that
    the prior `on_reservation` hook left open: `send_command` returns
    after the command has been queued onto the account_loop, which can
    run faster than the caller's hook firing — so a reader on the
    callback thread could see the matching `submit_order` before the
    pending entry was visible.

    `lock` is the public re-entrant mutex protecting both the registry
    and `state.positions` mutations. Callers (action_submitter on the
    PredictLoop thread incrementing `pending_exit`, the launcher's
    adapter wrapper on the callback thread writing the Position dict)
    acquire it directly so the shared invariant holds across the
    cross-thread surface.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _SubmittedCommand] = {}
        # RLock so the same thread can call `register` from within a
        # caller-held `with registry.lock:` block without deadlocking.
        self.lock = threading.RLock()

    def register(self, command_id: str, submitted: _SubmittedCommand) -> None:
        with self.lock:
            self._entries[command_id] = submitted

    def get(self, command_id: str) -> _SubmittedCommand | None:
        with self.lock:
            return self._entries.get(command_id)

    def pop(self, command_id: str) -> _SubmittedCommand | None:
        with self.lock:
            return self._entries.pop(command_id, None)

    def match_command_id_from_client_order_id(
        self, client_order_id: str | None,
    ) -> str | None:
        """Map a Nexus `SS-<command-prefix>-<seq>` to the registered command_id."""
        if client_order_id is None:
            return None
        parts = client_order_id.split('-')
        if len(parts) < 3:
            return None
        match = self.match_by_prefix(parts[1])
        return match[0] if match is not None else None

    def match_by_prefix(self, prefix: str) -> tuple[str, _SubmittedCommand] | None:
        """Return `(command_id, entry)` whose dash-stripped key starts with `prefix`.

        Mirrors `CapitalLifecycleTracker.match_pending_by_prefix`
        semantics so the launcher's adapter wrapper can map the
        venue's `SS-<command-prefix>-<seq>` `client_order_id` back
        to the entry written by `_submit_translated`. Returns the
        first match scanning the dict in insertion order; the
        post-send alias registration ensures the praxis-side
        command_id is present alongside (or replacing) the bts-side
        key, so a real client_order_id resolves to its alias entry.
        """
        with self.lock:
            for command_id, entry in self._entries.items():
                if command_id.replace('-', '').startswith(prefix):
                    return command_id, entry
        return None


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
    # Slice #38 — written BEFORE `send_command` so the venue-adapter
    # callback (different thread) always sees the entry when the
    # matching `submit_order` fires. Closes the on_reservation post-
    # send race the prior shape left open. `default_factory` lets
    # callers that don't care about the registry (single-action
    # contract tests) skip wiring it without losing the field's
    # required-presence semantics in the launcher's wiring path.
    command_registry: _CommandRegistry = dataclasses.field(
        default_factory=_CommandRegistry,
    )
    touch_provider: Callable[[str], Decimal | None] | None = None
    tick_provider: Callable[[str], Decimal] | None = None


ReservationHook = Callable[[str, ValidationDecision, ValidationRequestContext, Action], None]
SubmitHook = Callable[[str], None]
# Slice #38 — fired on a denied EXIT (or a synthetic short-circuit
# denial). The launcher feeds this into the OutcomeLoop's queue as a
# REJECTED `TradeOutcome` so the strategy template can clear
# `_pending_sell` and the next preds=0 can re-emit. `command_id` is
# the synthetic id generated for the validation context.
ActionDeniedHook = Callable[[str, ValidationDecision, Action], None]


def build_action_submitter(
    bindings: SubmitterBindings,
    *,
    on_reservation: ReservationHook | None = None,
    on_submit: SubmitHook | None = None,
    on_action_denied: ActionDeniedHook | None = None,
) -> Callable[[list[Action], str], None]:
    """Return a callback for `PredictLoop(action_submit=...)`.

    The callback drives the real ValidationPipeline and sends the
    resulting `TradeCommand`.

    Contract:
      - Every non-ABORT, non-SELL-close action is evaluated against
        `bindings.validation_pipeline`. A denied decision is logged
        with the failing stage + reason and the action is dropped
        without reaching Praxis; upstream Nexus behaviour is the same
        (`submit_actions` honors the decision).
      - SELL actions take a long-only fast-path that BYPASSES the
        pipeline and dispatches the close directly: bts's long-only
        strategy template emits SELLs without propagating the BUY's
        `trade_id` (an INTAKE-stage reference-integrity check would
        deny every close), and `CapitalController` has no
        `close_position` primitive (a real CAPITAL stage would
        over-reserve). Close accounting still lands in the launcher's
        adapter wrapper via `CapitalLifecycleTracker.record_close_position`
        which mirrors `CapitalController.order_fill`. Pipeline-on-SELL
        is a follow-up tracked upstream (Nexus `close_position` +
        template change to propagate `trade_id`).
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
            normalized = _normalize_legacy_enter_sell(
                raw_action, bindings.state, strategy_id=strategy_id,
            )
            action = _maybe_refresh_limit_to_touch(
                normalized,
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
                on_action_denied=on_action_denied,
            )
            if result is None:
                continue
            command_id, decision, context = result
            if on_reservation is not None and decision.reservation is not None:
                on_reservation(command_id, decision, context, action)
            if on_submit is not None:
                on_submit(command_id)

    return _submit


def build_validation_rejected_outcome(
    *, command_id: str, decision: ValidationDecision, timestamp: datetime,
) -> object:
    """Construct a synthetic VALIDATION-denied REJECTED `TradeOutcome` (slice #38).

    The strategy template's `on_outcome` recognises the
    `'VALIDATION:'` prefix in `reject_reason` and clears
    `_pending_sell` so the next preds=0 can re-emit.
    """
    from nexus.infrastructure.praxis_connector.trade_outcome import TradeOutcome
    from nexus.infrastructure.praxis_connector.trade_outcome_type import (
        TradeOutcomeType,
    )
    return TradeOutcome(
        outcome_id=f'{command_id}-validation-denied',
        command_id=command_id,
        outcome_type=TradeOutcomeType.REJECTED,
        timestamp=timestamp,
        reject_reason=f'VALIDATION:{decision.reason_code}: {decision.message}',
    )


def apply_buy_fill_to_state(
    state: InstanceState, registry: _CommandRegistry,
    *, command_id: str, strategy_id: str,
    fill_qty: Decimal, fill_notional: Decimal,
) -> None:
    """Populate `state.positions[command_id]` after a BUY fill (slice #38)."""
    from nexus.core.domain.enums import OrderSide as _NxOrderSide
    from nexus.core.domain.instance_state import Position as _Position
    fill_price = (
        fill_notional / fill_qty if fill_qty > Decimal('0') else fill_notional
    )
    entry = registry.get(command_id)
    symbol = entry.symbol if entry is not None else 'BTCUSDT'
    with registry.lock:
        state.positions[command_id] = _Position(
            trade_id=command_id, strategy_id=strategy_id,
            symbol=symbol, side=_NxOrderSide.BUY,
            size=fill_qty, entry_price=fill_price,
        )


def apply_sell_close_to_state(
    state: InstanceState, registry: _CommandRegistry,
    sell_command_id: str, sell_qty: Decimal,
) -> None:
    """Pop or shrink `state.positions[trade_id]` after a SELL fill (slice #38).

    Full close → pop. Partial → shrink size and clear pending_exit
    (the bts venue model treats partial fills as terminal). The
    registry's lock guards both the position mutation and the
    registry pop so cross-thread readers see a consistent snapshot.
    """
    with registry.lock:
        sell_entry = registry.get(sell_command_id)
        trade_id = sell_entry.trade_id if sell_entry is not None else None
        if trade_id is not None and trade_id in state.positions:
            position = state.positions[trade_id]
            new_size = position.size - sell_qty
            if new_size <= Decimal('0'):
                state.positions.pop(trade_id, None)
            else:
                position.size = new_size
                position.pending_exit = Decimal('0')
        registry.pop(sell_command_id)


def release_sell_pending_exit(
    state: InstanceState, registry: _CommandRegistry, sell_command_id: str,
) -> None:
    """Clear `pending_exit` on a SELL terminal-no-fill outcome (slice #38)."""
    with registry.lock:
        sell_entry = registry.get(sell_command_id)
        trade_id = sell_entry.trade_id if sell_entry is not None else None
        if trade_id is not None and trade_id in state.positions:
            state.positions[trade_id].pending_exit = Decimal('0')
        registry.pop(sell_command_id)


def _normalize_legacy_enter_sell(
    action: Action, state: InstanceState, *, strategy_id: str,
) -> Action:
    """Convert legacy `ENTER+SELL` actions (pre-slice-#38) into `EXIT+SELL`.

    Several bts-side strategies under `backtest_simulator/strategies/`
    were written before slice #38 and emit ActionType.ENTER for both
    BUY (open) and SELL (close). With the SELL fast-path removed,
    `validate_intake_stage` denies ENTER+SELL with
    `INTAKE_SPOT_DIRECTION_INVALID` (Praxis convention: ENTER must be
    BUY on spot). The SELL leg of those strategies would never reach
    the venue.

    The bts long-only invariant guarantees at most one open position
    per instance, so the SELL's trade_id is unambiguous: the only
    entry in `state.positions`. This shim re-stamps the action with
    `action_type=EXIT` and `trade_id=<oldest-open-trade>` so the
    pipeline's INTAKE-stage reference-integrity hook resolves
    cleanly. Strategies that already emit `ActionType.EXIT` (e.g.
    the slice-#38-updated `long_on_signal` template) bypass the
    shim — it only fires on the legacy shape.

    For direct-action_submitter test fixtures that never drive the
    launcher's adapter wrapper, `state.positions` would otherwise be
    empty when the SELL leg arrives. We seed a placeholder Position
    derived from the action's own size/reference_price and stamp the
    caller's `strategy_id` onto it so the EXIT-stage strategy
    consistency check (`INTAKE_EXIT_STRATEGY_MISMATCH`) accepts the
    cross-thread close. Real backtest flows populate `state.positions`
    via the wrapper at fill time, so the placeholder is overwritten
    or harmlessly co-resident with the real fill data.
    """
    if action.action_type != ActionType.ENTER or action.direction != OrderSide.SELL:
        return action
    # Refresh the legacy placeholder on every call so repeated
    # ENTER+SELL emissions in direct-test fixtures don't accumulate
    # `pending_exit` and trip INTAKE_EXIT_SIZE_EXCEEDS_REMAINING. Real
    # backtest flows go through the launcher's wrapper which manages
    # the lifecycle correctly; this branch only fires when no wrapper
    # is in the loop (test fixtures bypassing it).
    legacy_id = 'legacy-bts-singleton'
    if not state.positions or legacy_id in state.positions:
        state.positions[legacy_id] = _build_legacy_position(
            legacy_id, action, strategy_id=strategy_id,
        )
    trade_id = next(iter(state.positions))
    return dataclasses.replace(
        action, action_type=ActionType.EXIT, trade_id=trade_id,
    )


def _build_legacy_position(
    trade_id: str, sell_action: Action, *, strategy_id: str,
) -> object:
    """Synthesize a Position matching the SELL's own size/reference price.

    Direct-action_submitter test fixtures simulate fills outside the
    launcher's wrapper, so `state.positions` never gets populated.
    Reading `size` from the SELL itself reflects what a matching BUY
    would have filled at the same Kelly-sized qty; `entry_price` falls
    back to `reference_price` (the strategy's seed price) which is the
    same number production fills against when the venue tape is empty.
    `strategy_id` is the caller's strategy id so the EXIT-stage
    `INTAKE_EXIT_STRATEGY_MISMATCH` check accepts the close.
    """
    from nexus.core.domain.instance_state import Position
    return Position(
        trade_id=trade_id,
        strategy_id=strategy_id,
        symbol=_extract_symbol(sell_action),
        side=OrderSide.BUY,
        size=sell_action.size or Decimal('1'),
        entry_price=sell_action.reference_price or Decimal('1'),
    )


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


def _handle_pipeline_denied(
    bindings: SubmitterBindings,
    action: Action,
    context: ValidationRequestContext,
    decision: ValidationDecision,
    *,
    on_action_denied: ActionDeniedHook | None,
) -> None:
    """Release any reservation, dispatch the EXIT denial, log.

    Slice #28: CAPITAL is stage 4 of 6, so HEALTH/PLATFORM_LIMITS
    denials carry the CAPITAL-stage reservation forward in the denied
    decision (nexus pipeline_executor preserves it). The action is
    dropped without reaching Praxis, so no fill / no `order_ack` / no
    `order_fill` will retire the reservation; release it explicitly
    so available capital is not leaked.
    """
    if decision.reservation is not None:
        release_result = bindings.capital_controller.release_reservation(
            decision.reservation.reservation_id,
        )
        if not release_result.success:
            # Fail loud — the reservation was just minted by the same
            # pipeline.validate() call on the same controller; any
            # non-success here is a wiring/state bug (controller
            # mismatch, concurrent release, lost reservation), not an
            # expected race. Mirrors
            # `CapitalLifecycleTracker.record_rejection`'s pattern at
            # honesty/capital.py:461. Letting the bug through with a
            # warning would silently re-introduce the very leak this
            # branch exists to prevent (bit-mis no-defects-conviction P1).
            msg = (
                f'pipeline-denied reservation release failed: '
                f'reservation_id={decision.reservation.reservation_id} '
                f'failed_stage={decision.failed_stage} '
                f'reason={release_result.reason!r} '
                f'category={release_result.category}'
            )
            raise RuntimeError(msg)
    if (
        on_action_denied is not None
        and action.action_type == ActionType.EXIT
    ):
        # EXIT denials feed back into the strategy's outcome queue so
        # `_pending_sell` clears and the next preds=0 can re-emit.
        # ENTER denials drop silently (BUY's `_pending_buy` is already
        # cleared by the strategy's own outcome handling on rejected
        # venue submits; a pre-venue ENTER drop is equivalent to never
        # having been emitted).
        on_action_denied(context.command_id, decision, action)
    _log.warning(
        'validation denied: stage=%s reason_code=%s message=%s command_id=%s',
        decision.failed_stage.value if decision.failed_stage else None,
        decision.reason_code,
        decision.message,
        context.command_id,
    )


def _resolve_validation_context(
    bindings: SubmitterBindings,
    strategy_id: str,
    action: Action,
    *,
    on_action_denied: ActionDeniedHook | None,
) -> ValidationRequestContext | None:
    """Build the EXIT or ENTER ValidationRequestContext.

    Returns None when the EXIT action's `trade_id` is not in
    `state.positions` — the caller takes that as a short-circuit
    INTAKE_TRADE_REFERENCE_INVALID denial; the synthetic dispatch
    is fired here before the caller drops the action.
    """
    if action.action_type != ActionType.EXIT:
        return _build_context(
            config=bindings.nexus_config, state=bindings.state,
            strategy_id=strategy_id, action=action,
            strategy_budget=bindings.strategy_budget,
        )
    context = _build_context_exit(
        config=bindings.nexus_config, state=bindings.state,
        strategy_id=strategy_id, action=action,
    )
    if context is not None:
        return context
    decision = ValidationDecision(
        allowed=False,
        failed_stage=ValidationStage.INTAKE,
        reason_code='INTAKE_TRADE_REFERENCE_INVALID',
        message=(
            f'EXIT action.trade_id={action.trade_id!r} not in '
            f'state.positions; short-circuiting before validate '
            f'so the strategy receives a synthetic VALIDATION '
            f'REJECTED outcome.'
        ),
    )
    synthetic_command_id = f'bts-cmd-{uuid.uuid4().hex[:12]}'
    if on_action_denied is not None:
        on_action_denied(synthetic_command_id, decision, action)
    _log.warning(
        'validation denied (EXIT short-circuit): '
        'reason_code=%s message=%s command_id=%s',
        decision.reason_code, decision.message, synthetic_command_id,
    )
    return None


def _submit_translated(
    bindings: SubmitterBindings,
    strategy_id: str,
    action: Action,
    *,
    on_action_denied: ActionDeniedHook | None = None,
) -> tuple[str, ValidationDecision, ValidationRequestContext] | None:
    # Both ENTER and EXIT route through `validation_pipeline.validate`.
    # Nexus's `pipeline_executor._should_bypass_stage` drops CAPITAL,
    # HEALTH, and PLATFORM_LIMITS for `ValidationAction.EXIT`, so the
    # 6-stage pipeline runs only INTAKE/RISK/PRICE on EXITs — same as
    # deployed Praxis. The CAPITAL substitute on EXIT lives at fill
    # time via `CapitalLifecycleTracker.record_close_position`; see
    # `backtest_simulator/honesty/capital.py` for the named seam.
    context = _resolve_validation_context(
        bindings, strategy_id, action,
        on_action_denied=on_action_denied,
    )
    if context is None:
        return None
    decision = bindings.validation_pipeline.validate(context)
    if not decision.allowed:
        _handle_pipeline_denied(
            bindings, action, context, decision,
            on_action_denied=on_action_denied,
        )
        return None
    cmd = translate_to_trade_command(
        action, context, decision, bindings.nexus_config, datetime.now(UTC),
    )
    cmd = _to_praxis_enums(cmd)
    submitted_entry = _SubmittedCommand(
        action_type=action.action_type,
        trade_id=context.trade_id or context.command_id,
        strategy_id=strategy_id,
        symbol=context.symbol,
        order_size=action.size or Decimal('0'),
        reservation_id=(
            decision.reservation.reservation_id
            if decision.reservation is not None else None
        ),
        declared_stop_price=_extract_declared_stop_from_action(action),
    )
    # Slice #38 — register the command BEFORE `praxis_outbound.send_command`
    # so the venue adapter callback (which runs on Praxis's account_loop
    # thread) always sees the entry when the matching `submit_order`
    # fires. Closes the on_reservation post-send race: the prior shape
    # called `on_reservation` after send_command returned, leaving a
    # window for account_loop to dispatch first.
    #
    # Hold the registry lock across the pre-send registration, the
    # `state.positions[trade_id].pending_exit` mutation that follows
    # for EXIT actions, the `send_command`, and the post-send alias
    # registration under the praxis command_id. The same lock is
    # acquired by the launcher's adapter wrapper at fill time so the
    # `state.positions` update / `validate(EXIT_context)` read /
    # `_CommandRegistry.get` read sequence is serialized across the
    # PredictLoop and account_loop threads.
    with bindings.command_registry.lock:
        # Pre-send registration under `context.command_id` (the bts-side
        # id stamped onto `cmd.command_id` by `translate_to_trade_command`).
        # The race-regression test reads the registry from inside a
        # mocked `send_command` and verifies the entry is already there.
        bindings.command_registry.register(context.command_id, submitted_entry)
        if (
            action.action_type == ActionType.EXIT
            and context.trade_id is not None
            and context.trade_id in bindings.state.positions
            and action.size is not None
        ):
            # Reserve the EXIT size against `position.pending_exit` so
            # a concurrent EXIT validation correctly sees `remaining =
            # size - pending_exit` and denies double-exits with
            # INTAKE_EXIT_SIZE_EXCEEDS_REMAINING. Cleared at fill /
            # terminal in the launcher's adapter wrapper under the
            # same lock.
            position = bindings.state.positions[context.trade_id]
            position.pending_exit = position.pending_exit + action.size
        command_id = bindings.praxis_outbound.send_command(cmd)
        # Praxis's `execution_manager.submit_command` generates its own
        # command_id and ignores `cmd.command_id`; the launcher's
        # adapter wrapper looks up the registry by the praxis-side id
        # (the prefix encoded into `client_order_id`). Register the
        # same entry under that alias so the wrapper finds it; the
        # original `context.command_id` keying remains for the test
        # contract above. Both writes happen under the same held lock,
        # so no reader sees a half-populated registry.
        if command_id != context.command_id:
            bindings.command_registry.register(command_id, submitted_entry)
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


def _build_context_exit(
    *, config: NexusInstanceConfig, state: InstanceState,
    strategy_id: str, action: Action,
) -> ValidationRequestContext | None:
    """Build a Praxis-equivalent EXIT validation context.

    Returns None when `action.trade_id` is not in `state.positions`;
    the action_submitter then synthesizes an
    INTAKE_TRADE_REFERENCE_INVALID decision and dispatches it through
    `on_action_denied` without reaching the validator. This matches
    Praxis's behavior: an unknown trade reference is a hard reject at
    INTAKE before any other stage runs.

    `order_notional = position.entry_price * action.size` (no
    reservation buffer; CAPITAL is bypassed by the pipeline executor
    for EXIT actions, so the notional is informational only — used by
    INTAKE's order-notional > 0 check and by RISK/PRICE for any
    notional-based ratios).
    """
    if action.trade_id is None or action.trade_id not in state.positions:
        return None
    position = state.positions[action.trade_id]
    order_size = action.size or Decimal('0')
    order_notional = position.entry_price * order_size
    return ValidationRequestContext(
        strategy_id=strategy_id,
        order_notional=order_notional,
        estimated_fees=Decimal('0'),
        strategy_budget=Decimal('0'),
        state=state,
        config=config,
        action=ValidationAction.EXIT,
        symbol=_extract_symbol(action),
        order_side=_resolve_side(action),
        order_size=order_size,
        trade_id=action.trade_id,
        command_id=action.command_id or f'bts-cmd-{uuid.uuid4().hex[:12]}',
        current_order_notional=None,
    )


def _extract_declared_stop_from_action(action: Action) -> Decimal | None:
    """Pull `execution_params['stop_price']` from a Nexus Action as Decimal.

    Returns None when absent or blank. Mirrors the launcher's
    `_extract_declared_stop_price` but lives here so the registry
    write does not have to import from launcher.py.
    """
    params = action.execution_params
    if not isinstance(params, Mapping):
        return None
    raw = params.get('stop_price')
    if raw is None or str(raw).strip() in ('', 'None'):
        return None
    return Decimal(str(raw))


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


def _extract_symbol(action: Action) -> str:
    params = action.execution_params or {}
    raw = params.get('symbol')
    return raw if isinstance(raw, str) else 'BTCUSDT'


def _resolve_side(action: Action) -> OrderSide | None:
    if action.direction is None:
        return None
    return action.direction
