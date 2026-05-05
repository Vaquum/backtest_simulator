"""Deferred SELL retry on PARTIAL/EXPIRED — exit-only, until flat or hard reject.

Operator invariant: "We have to treat getting out of the position
exactly like we would handle it intraday, no difference. (...) you
keep doing until you have gotten out of the position. (...) If we
get expired we of course re-submit."

This module pins the contract end-to-end:
  - `_must_close_outstanding` flag is set on PARTIAL with residual
    or EXPIRED zero-fill (SELL only) and consumed by the next
    `on_signal` to re-emit the SELL for the held `_entry_qty`.
  - The retry runs at the TOP of `on_signal`, before both the
    cutoff branch and the `_preds is None` early-return.
  - Hard rejections (REJECTED / CANCELED) clear the latch — the
    retry would just hit the same gate.
  - Translator (`launcher.py::_translate_praxis_outcome`) maps
    Praxis EXPIRED with `filled_qty > 0` to Nexus PARTIAL so the
    strategy's existing fill branch reconciles `_entry_qty`.
"""
from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from nexus.core.domain.enums import OrderSide
from nexus.core.domain.operational_mode import OperationalMode
from nexus.infrastructure.praxis_connector.trade_outcome import TradeOutcome
from nexus.infrastructure.praxis_connector.trade_outcome_type import (
    TradeOutcomeType,
)
from nexus.strategy.context import StrategyContext
from nexus.strategy.params import StrategyParams
from nexus.strategy.signal import Signal
from praxis.core.domain.enums import TradeStatus
from praxis.core.domain.trade_outcome import TradeOutcome as PraxisTradeOutcome

from backtest_simulator.launcher.launcher import _translate_praxis_outcome

_T0 = datetime(2026, 4, 20, 0, 0, tzinfo=UTC)
_KLINE_SECONDS = 14400  # 4-hour klines (matches r0011)
_WINDOW_END = datetime(2026, 4, 20, 23, 59, 59, tzinfo=UTC)
_FORCE_FLATTEN_AFTER = _WINDOW_END - timedelta(seconds=_KLINE_SECONDS)


def _load_template(tmp_path: Path) -> object:
    template_path = (
        Path(__file__).resolve().parents[2]
        / 'backtest_simulator' / 'pipeline'
        / '_strategy_templates' / 'long_on_signal.py'
    )
    rendered = template_path.read_text(encoding='utf-8').replace(
        '__BTS_PARAMS__',
        json.dumps({
            'symbol': 'BTCUSDT',
            'capital': '100000',
            'kelly_pct': '10',
            'estimated_price': '70000',
            'stop_bps': '50',
            'force_flatten_after': _FORCE_FLATTEN_AFTER.isoformat(),
        }),
    )
    rendered_path = tmp_path / 'long_on_signal.py'
    rendered_path.write_text(rendered, encoding='utf-8')
    spec = importlib.util.spec_from_file_location(
        f'long_on_signal_expired_retry_{tmp_path.name}', rendered_path,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ctx() -> StrategyContext:
    return StrategyContext(
        positions=(),
        capital_available=Decimal('100000'),
        operational_mode=OperationalMode.ACTIVE,
    )


def _signal(
    ts: datetime,
    preds: int | None = None,
    *,
    drop_preds: bool = False,
) -> Signal:
    """Construct a Nexus Signal with optional `_preds`.

    `drop_preds=True` produces a signal with NO `_preds` key at
    all, which exercises the strategy's `preds_raw is None` early
    return path. `preds=None` (the default) matches the same
    semantic. `drop_preds=False` and `preds=int` puts a real value.
    """
    values: dict[str, object] = {'_probs': 0.5}
    if not drop_preds and preds is not None:
        values['_preds'] = preds
    return Signal(
        predictor_fn_id='expired-retry-fixture',
        timestamp=ts,
        values=values,
    )


def _sell_outcome(
    *,
    outcome_type: TradeOutcomeType,
    fill_size: Decimal | None = None,
    fill_price: Decimal | None = None,
    cmd_id: str = 'cmd-sell',
) -> TradeOutcome:
    """Build a Nexus TradeOutcome for a SELL command (post-translator shape)."""
    is_fill = outcome_type.is_fill
    return TradeOutcome(
        outcome_id=f'oc-{cmd_id}',
        command_id=cmd_id,
        outcome_type=outcome_type,
        timestamp=_T0 + timedelta(seconds=1),
        fill_size=fill_size if is_fill else None,
        fill_price=fill_price if is_fill else None,
        fill_notional=(
            (fill_size * fill_price) if is_fill and fill_size and fill_price
            else None
        ),
        actual_fees=Decimal('0') if is_fill else None,
        reject_reason=None,
        cancel_reason=None,
    )


def _strategy_in_long_state(
    mod: object,
    *,
    entry_qty: Decimal = Decimal('0.10'),
    pending_sell: bool = True,
    must_close: bool = False,
) -> object:
    """Construct a strategy and force it into a `_long=True` state.

    Drives `on_startup` and then sets internal state directly to
    avoid threading a real BUY+fill outcome through every test.
    """
    strategy = mod.Strategy('expired-retry-fixture')
    strategy.on_startup(StrategyParams(raw={}), _ctx())
    strategy._long = True
    strategy._entry_qty = entry_qty
    strategy._pending_buy = False
    strategy._pending_sell = pending_sell
    strategy._must_close_outstanding = must_close
    # Slice #38 — SELL EXITs require trade_id; mirror the value the
    # strategy normally captures from outcome.command_id at BUY fill.
    strategy._open_trade_id = 'fixture-buy-cmd'
    return strategy


# -------------------------------------------------------------------
# Strategy-side tests (long_on_signal.py)
# -------------------------------------------------------------------


def test_partial_sell_residual_flags_must_close(tmp_path: Path) -> None:
    """PARTIAL SELL with residual sets the must_close latch and clears pending_sell."""
    mod = _load_template(tmp_path)
    strategy = _strategy_in_long_state(mod, entry_qty=Decimal('0.10'))
    outcome = _sell_outcome(
        outcome_type=TradeOutcomeType.PARTIAL,
        fill_size=Decimal('0.04'),
        fill_price=Decimal('70000'),
    )
    actions = strategy.on_outcome(outcome, StrategyParams(raw={}), _ctx())
    assert actions == []
    assert strategy._entry_qty == Decimal('0.06')
    assert strategy._long is True
    assert strategy._must_close_outstanding is True
    assert strategy._pending_sell is False


def test_expired_zero_fill_sell_flags_must_close(tmp_path: Path) -> None:
    """EXPIRED SELL with no fill sets the must_close latch."""
    mod = _load_template(tmp_path)
    strategy = _strategy_in_long_state(mod, entry_qty=Decimal('0.10'))
    outcome = _sell_outcome(outcome_type=TradeOutcomeType.EXPIRED)
    actions = strategy.on_outcome(outcome, StrategyParams(raw={}), _ctx())
    assert actions == []
    assert strategy._entry_qty == Decimal('0.10')
    assert strategy._long is True
    assert strategy._must_close_outstanding is True
    assert strategy._pending_sell is False


def test_next_signal_consumes_must_close_overrides_model(
    tmp_path: Path,
) -> None:
    """The deferred-retry latch consumes on next signal, overrides preds=1."""
    mod = _load_template(tmp_path)
    strategy = _strategy_in_long_state(
        mod, entry_qty=Decimal('0.06'),
        pending_sell=False, must_close=True,
    )
    actions = strategy.on_signal(
        _signal(_T0 + timedelta(seconds=300), preds=1),
        StrategyParams(raw={}), _ctx(),
    )
    assert len(actions) == 1
    assert actions[0].direction == OrderSide.SELL
    assert actions[0].size == Decimal('0.06')
    assert strategy._must_close_outstanding is False
    assert strategy._pending_sell is True


def test_must_close_does_not_re_emit_when_pending_sell_in_flight(
    tmp_path: Path,
) -> None:
    """A retry already in flight (`_pending_sell=True`) blocks a second emit."""
    mod = _load_template(tmp_path)
    strategy = _strategy_in_long_state(
        mod, entry_qty=Decimal('0.06'),
        pending_sell=True, must_close=True,
    )
    actions = strategy.on_signal(
        _signal(_T0 + timedelta(seconds=300), preds=1),
        StrategyParams(raw={}), _ctx(),
    )
    assert actions == []
    assert strategy._must_close_outstanding is True
    assert strategy._pending_sell is True


def test_must_close_runs_before_preds_none_guard(tmp_path: Path) -> None:
    """The latch branch sits above the `_preds is None` early return.

    A signal with no `_preds` AND timestamp BEFORE `force_flatten_after`
    only emits a SELL if the must_close branch runs FIRST. An
    implementation that puts the must_close branch BELOW the
    `if preds_raw is None: return []` guard would return [] here
    and fail this test.
    """
    mod = _load_template(tmp_path)
    strategy = _strategy_in_long_state(
        mod, entry_qty=Decimal('0.06'),
        pending_sell=False, must_close=True,
    )
    pre_cutoff_ts = _T0 + timedelta(seconds=600)
    assert pre_cutoff_ts < _FORCE_FLATTEN_AFTER
    actions = strategy.on_signal(
        _signal(pre_cutoff_ts, drop_preds=True),
        StrategyParams(raw={}), _ctx(),
    )
    assert len(actions) == 1
    assert actions[0].direction == OrderSide.SELL
    assert actions[0].size == Decimal('0.06')
    assert strategy._must_close_outstanding is False
    assert strategy._pending_sell is True


def test_filled_sell_clears_must_close(tmp_path: Path) -> None:
    """A clean SELL FILLED clears the latch and flips _long=False."""
    mod = _load_template(tmp_path)
    strategy = _strategy_in_long_state(
        mod, entry_qty=Decimal('0.10'),
        pending_sell=True, must_close=True,
    )
    outcome = _sell_outcome(
        outcome_type=TradeOutcomeType.FILLED,
        fill_size=Decimal('0.10'),
        fill_price=Decimal('70000'),
    )
    actions = strategy.on_outcome(outcome, StrategyParams(raw={}), _ctx())
    assert actions == []
    assert strategy._long is False
    assert strategy._entry_qty == Decimal('0')
    assert strategy._must_close_outstanding is False
    assert strategy._pending_sell is False


def test_expired_buy_does_not_set_must_close(tmp_path: Path) -> None:
    """EXPIRED BUY clears _pending_buy without setting the must_close latch.

    Operator invariant is exit-only — entries on stale intent are
    out of scope. This test pins that scope.
    """
    mod = _load_template(tmp_path)
    strategy = mod.Strategy('expired-retry-fixture')
    strategy.on_startup(StrategyParams(raw={}), _ctx())
    strategy._pending_buy = True
    strategy._must_close_outstanding = False
    outcome = TradeOutcome(
        outcome_id='oc-buy-expired',
        command_id='cmd-buy',
        outcome_type=TradeOutcomeType.EXPIRED,
        timestamp=_T0,
        fill_size=None, fill_price=None, fill_notional=None,
        actual_fees=None, reject_reason=None, cancel_reason=None,
    )
    actions = strategy.on_outcome(outcome, StrategyParams(raw={}), _ctx())
    assert actions == []
    assert strategy._pending_buy is False
    assert strategy._must_close_outstanding is False


def test_rejected_sell_clears_must_close(tmp_path: Path) -> None:
    """REJECTED SELL clears the must_close latch (hard venue rejection)."""
    mod = _load_template(tmp_path)
    strategy = _strategy_in_long_state(
        mod, entry_qty=Decimal('0.10'),
        pending_sell=True, must_close=True,
    )
    outcome = TradeOutcome(
        outcome_id='oc-sell-rejected',
        command_id='cmd-sell',
        outcome_type=TradeOutcomeType.REJECTED,
        timestamp=_T0,
        fill_size=None, fill_price=None, fill_notional=None,
        actual_fees=None,
        reject_reason='lot size below filter',
        cancel_reason=None,
    )
    actions = strategy.on_outcome(outcome, StrategyParams(raw={}), _ctx())
    assert actions == []
    assert strategy._must_close_outstanding is False
    assert strategy._pending_sell is False
    assert strategy._long is True


def test_canceled_sell_clears_must_close(tmp_path: Path) -> None:
    """CANCELED SELL clears the must_close latch (operator-side abort)."""
    mod = _load_template(tmp_path)
    strategy = _strategy_in_long_state(
        mod, entry_qty=Decimal('0.10'),
        pending_sell=True, must_close=True,
    )
    outcome = TradeOutcome(
        outcome_id='oc-sell-canceled',
        command_id='cmd-sell',
        outcome_type=TradeOutcomeType.CANCELED,
        timestamp=_T0,
        fill_size=None, fill_price=None, fill_notional=None,
        actual_fees=None, reject_reason=None,
        cancel_reason='operator canceled',
    )
    actions = strategy.on_outcome(outcome, StrategyParams(raw={}), _ctx())
    assert actions == []
    assert strategy._must_close_outstanding is False
    assert strategy._pending_sell is False
    assert strategy._long is True


def test_must_close_flag_persists_across_save_load(tmp_path: Path) -> None:
    """The latch survives `on_save` + fresh-strategy + `on_load`."""
    mod = _load_template(tmp_path)
    strategy = _strategy_in_long_state(
        mod, entry_qty=Decimal('0.06'), must_close=True,
    )
    saved = strategy.on_save()
    fresh = mod.Strategy('expired-retry-fixture')
    fresh.on_startup(StrategyParams(raw={}), _ctx())
    fresh.on_load(saved)
    assert fresh._must_close_outstanding is True
    assert fresh._long is True
    assert fresh._entry_qty == Decimal('0.06')


# -------------------------------------------------------------------
# Translator tests (launcher.py::_translate_praxis_outcome)
# -------------------------------------------------------------------


def _praxis_outcome(
    *, status: TradeStatus, filled_qty: Decimal,
    avg_fill_price: Decimal | None = None,
    reason: str | None = None,
) -> PraxisTradeOutcome:
    """Build a Praxis TradeOutcome with all `__post_init__` invariants satisfied."""
    import dataclasses
    fields = {f.name for f in dataclasses.fields(PraxisTradeOutcome)}
    base: dict[str, object] = {
        'command_id': 'cmd-1',
        'trade_id': 'trade-1',
        'account_id': 'bts-test',
        'status': status,
        'target_qty': Decimal('0.10'),
        'filled_qty': filled_qty,
        'avg_fill_price': avg_fill_price,
        'slices_completed': 1 if filled_qty > 0 else 0,
        'slices_total': 1,
        'reason': reason,
        'created_at': datetime(2026, 4, 12, 17, 0, tzinfo=UTC),
    }
    if 'cumulative_notional' in fields and filled_qty > 0:
        if avg_fill_price is None:
            msg = 'avg_fill_price required when filled_qty > 0'
            raise ValueError(msg)
        base['cumulative_notional'] = filled_qty * avg_fill_price
    return PraxisTradeOutcome(**base)  # type: ignore[arg-type]


def test_translator_maps_expired_with_partial_fill_to_partial() -> None:
    """Praxis EXPIRED + filled_qty>0 translates to Nexus PARTIAL with fill data."""
    praxis = _praxis_outcome(
        status=TradeStatus.EXPIRED,
        filled_qty=Decimal('0.04'),
        avg_fill_price=Decimal('70000'),
        reason='deadline exceeded',
    )
    nexus_outcome = _translate_praxis_outcome(praxis)
    assert nexus_outcome is not None
    assert nexus_outcome.outcome_type == TradeOutcomeType.PARTIAL
    assert nexus_outcome.fill_size == Decimal('0.04')
    assert nexus_outcome.fill_price == Decimal('70000')
    # Critical: Nexus rejects reject_reason on non-REJECTED outcomes.
    assert nexus_outcome.reject_reason is None


def test_translator_keeps_zero_fill_expired_as_expired() -> None:
    """Praxis EXPIRED + filled_qty=0 stays Nexus EXPIRED with no fill data."""
    praxis = _praxis_outcome(
        status=TradeStatus.EXPIRED,
        filled_qty=Decimal('0'),
        reason='deadline exceeded',
    )
    nexus_outcome = _translate_praxis_outcome(praxis)
    assert nexus_outcome is not None
    assert nexus_outcome.outcome_type == TradeOutcomeType.EXPIRED
    assert nexus_outcome.fill_size is None
    assert nexus_outcome.reject_reason is None


def test_translator_carries_reject_reason_on_real_rejected() -> None:
    """REJECTED outcomes forward the Praxis reason as `reject_reason`."""
    praxis = _praxis_outcome(
        status=TradeStatus.REJECTED,
        filled_qty=Decimal('0'),
        reason='ATR gate denied',
    )
    nexus_outcome = _translate_praxis_outcome(praxis)
    assert nexus_outcome is not None
    assert nexus_outcome.outcome_type == TradeOutcomeType.REJECTED
    assert nexus_outcome.reject_reason == 'ATR gate denied'


def test_translator_drops_reason_on_canceled() -> None:
    """CANCELED outcomes drop the Praxis reason (Nexus rejects reject_reason on CANCELED).

    Pins the latch-clear path: `on_outcome` must actually receive
    a valid Nexus CANCELED to clear `_must_close_outstanding`. If
    the translator raised on this case, the latch would never
    clear and the retry loop would spin until window-end.
    """
    praxis = _praxis_outcome(
        status=TradeStatus.CANCELED,
        filled_qty=Decimal('0'),
        reason='operator canceled',
    )
    nexus_outcome = _translate_praxis_outcome(praxis)
    assert nexus_outcome is not None
    assert nexus_outcome.outcome_type == TradeOutcomeType.CANCELED
    assert nexus_outcome.fill_size is None
    assert nexus_outcome.reject_reason is None
