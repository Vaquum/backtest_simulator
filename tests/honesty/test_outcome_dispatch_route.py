"""bts launcher routes outcomes to the launcher's queue dict, not Trading's.

Pre-fix, `_route_and_translate` (BacktestLauncher's override of
`ExecutionManager._on_trade_outcome`) read `Trading._outcome_queues`
via a `_outcome_queues_for(trading)` helper. That dict is the
Praxis `Trading` per-account registry — populated only when callers
invoke `Trading.register_outcome_queue`, which neither Praxis 0.48.0
nor bts ever does. So `queues.get(account_id)` always returned None
and every translated Nexus outcome was silently dropped before
reaching `OutcomeLoop.PraxisInbound`.

The actual queues that `OutcomeLoop`'s consumer drains live on the
launcher itself: `praxis.Launcher._outcome_queues` (created in
`praxis/launcher.py:1063`, populated in `1122`). The fix points the
override at `self._outcome_queues` so translated outcomes land
where the consumer reads them.

This module exercises the extracted `_make_outcome_router` factory
end-to-end: build a real Praxis `TradeOutcome` (FILLED), feed it
to the router, assert the launcher queue receives a translated
Nexus outcome with `outcome_type=FILLED`.
"""
from __future__ import annotations

import asyncio
import queue
from datetime import UTC, datetime
from decimal import Decimal

from nexus.infrastructure.praxis_connector.trade_outcome import (
    TradeOutcome as NexusTradeOutcome,
)
from nexus.infrastructure.praxis_connector.trade_outcome_type import (
    TradeOutcomeType,
)
from praxis.core.domain.enums import TradeStatus
from praxis.core.domain.trade_outcome import TradeOutcome as PraxisTradeOutcome

from backtest_simulator.launcher.launcher import _make_outcome_router

_ACCOUNT = 'bts-test'


def _filled_praxis_outcome() -> PraxisTradeOutcome:
    """Build a FILLED Praxis outcome with all `__post_init__` invariants satisfied.

    Post-FINAL-MAJOR-07 Praxis adds `cumulative_notional` and
    requires it to be positive when `filled_qty > 0`. Older Praxis
    versions reject the kwarg entirely. We pass it through
    `_kwargs` only when the dataclass accepts it, so the test
    runs cleanly on either schema; the bts translator itself does
    not read this field.
    """
    import dataclasses
    fields = {f.name for f in dataclasses.fields(PraxisTradeOutcome)}
    base: dict[str, object] = {
        'command_id': 'cmd-1',
        'trade_id': 'trade-1',
        'account_id': _ACCOUNT,
        'status': TradeStatus.FILLED,
        'target_qty': Decimal('0.1'),
        'filled_qty': Decimal('0.1'),
        'avg_fill_price': Decimal('70000'),
        'slices_completed': 1,
        'slices_total': 1,
        'reason': None,
        'created_at': datetime(2026, 4, 6, 12, 0, tzinfo=UTC),
    }
    if 'cumulative_notional' in fields:
        base['cumulative_notional'] = Decimal('7000')
    return PraxisTradeOutcome(**base)  # type: ignore[arg-type]


def test_outcome_router_enqueues_filled_outcome_into_launcher_queue() -> None:
    """A FILLED Praxis outcome lands as a FILLED Nexus outcome in the launcher queue."""
    account_queue: queue.Queue[NexusTradeOutcome] = queue.Queue()
    queues: dict[str, queue.Queue[NexusTradeOutcome]] = {_ACCOUNT: account_queue}
    router = _make_outcome_router(queues)
    asyncio.run(router(_filled_praxis_outcome()))
    assert account_queue.qsize() == 1, (
        f'expected 1 outcome enqueued; got {account_queue.qsize()}. '
        f'a regression where the router reads the wrong queue dict '
        f'leaves the queue empty (silent drop)'
    )
    nexus_outcome = account_queue.get_nowait()
    assert nexus_outcome.outcome_type == TradeOutcomeType.FILLED
    assert nexus_outcome.command_id == 'cmd-1'


def test_outcome_router_drops_unknown_account_silently() -> None:
    """An outcome for an unregistered account is dropped (no exception, no enqueue).

    This case only arises during shutdown teardown when the launcher
    has already deregistered the account; the router must not raise
    — it would crash the OutcomeLoop's daemon thread.
    """
    queues: dict[str, queue.Queue[NexusTradeOutcome]] = {}  # no accounts registered
    router = _make_outcome_router(queues)
    asyncio.run(router(_filled_praxis_outcome()))  # should not raise


def test_outcome_router_translates_pending_to_expired() -> None:
    """PENDING is mapped to EXPIRED so bounded-lookahead non-fills clear pending_buy.

    Praxis emits PENDING when the order was accepted but no fill
    landed. In a bounded-lookahead backtest that IS terminal (no
    later outcome will arrive). Mapping to EXPIRED clears the
    strategy's `_pending_buy` gate so the next signal can re-enter.
    """
    account_queue: queue.Queue[NexusTradeOutcome] = queue.Queue()
    queues: dict[str, queue.Queue[NexusTradeOutcome]] = {_ACCOUNT: account_queue}
    router = _make_outcome_router(queues)
    pending = PraxisTradeOutcome(
        command_id='cmd-pending',
        trade_id='trade-pending',
        account_id=_ACCOUNT,
        status=TradeStatus.PENDING,
        target_qty=Decimal('0.1'),
        filled_qty=Decimal('0'),
        avg_fill_price=None,
        slices_completed=0,
        slices_total=1,
        # Nexus's TradeOutcome rejects non-None reject_reason on
        # non-REJECTED outcomes; the bts translator forwards
        # `praxis_outcome.reason` verbatim for non-REJECTED. Keep
        # reason=None so the routed Nexus outcome stays valid.
        reason=None,
        created_at=datetime(2026, 4, 12, 23, 59, tzinfo=UTC),
    )
    asyncio.run(router(pending))
    assert account_queue.qsize() == 1
    nexus_outcome = account_queue.get_nowait()
    assert nexus_outcome.outcome_type == TradeOutcomeType.EXPIRED, (
        'PENDING with no fill in bounded-lookahead must map to EXPIRED '
        '(terminal) so the strategy clears pending_buy. Mapping it to '
        'PENDING (non-terminal) jams the pending-buy gate forever.'
    )
