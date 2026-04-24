"""INTAKE stage hooks — declared-stop enforcement for opening entries.

Part 2 invariant: every ENTER action that OPENS a new directional
position must declare a `stop_price` in its `execution_params`. This
is the Nexus side of the honesty gate: no stop, no entry. The venue
adapter's `FillModel.apply_stop` enforces the stop during the trade
walk; together the two halves guarantee every opening position has a
protective stop that was honestly declared before the fill.

Long-only convention: a BUY opens a new long; a SELL (with
`action=ENTER` in our strategy template) closes an existing long and
IS itself the risk close — no new stop needed. The hook therefore
requires `stop_price` only on BUY entries. A short-biased strategy
would invert: SELL opens, BUY closes.
"""
from __future__ import annotations

from nexus.core.domain.enums import OrderSide
from nexus.core.validator.pipeline_models import (
    ValidationAction,
    ValidationDecision,
    ValidationRequestContext,
    ValidationStage,
)

DECLARED_STOP_MISSING_CODE = 'INTAKE_DECLARED_STOP_MISSING'


def _extract_stop_price(context: ValidationRequestContext) -> str | None:
    """Pull the declared stop_price from the action's `execution_params` dict.

    The strategy template stores it as `execution_params['stop_price']`;
    Nexus's `translate_to_trade_command` passes the dict through to
    `TradeCommand.execution_params`. At validation time we're before
    that translation, but `ValidationRequestContext` has no direct
    reference to the action's execution_params. We reach through the
    context's `state.instance_config` via a side-channel the action
    submitter populates — see action_submitter._build_context for the
    mirror entry.

    Part 2 wires this in through `execution_params` on the action;
    the INTAKE hook reads the SAME dict via the context's side-channel
    attribute `_declared_stop_price` which the action_submitter sets
    when it builds the context.
    """
    return getattr(context, '_declared_stop_price', None)


def require_declared_stop_on_long_entry(
    context: ValidationRequestContext,
) -> ValidationDecision | None:
    """Deny ENTER actions on OrderSide.BUY that lack a declared stop_price.

    Returns `None` (pass-through) for any non-BUY-entry action, or
    if the action already has a stop_price. Returns a denying
    ValidationDecision otherwise.
    """
    if context.action != ValidationAction.ENTER:
        return None
    if context.order_side != OrderSide.BUY:
        return None
    stop_price = _extract_stop_price(context)
    if stop_price is not None and str(stop_price).strip() not in ('', 'None'):
        return None
    return ValidationDecision(
        allowed=False,
        failed_stage=ValidationStage.INTAKE,
        reason_code=DECLARED_STOP_MISSING_CODE,
        message=(
            f'ENTER BUY command_id={context.command_id} lacks a declared '
            f'stop_price. Part 2 honesty gate: every long-opening entry '
            f'MUST declare a concrete stop_price so FillModel.apply_stop '
            f'can enforce it and r_per_trade can be computed honestly.'
        ),
    )
