"""close-by-window-end honesty: every BUY filled inside a window MUST close before window ends.

Three invariants pinned here:

1. Force-flatten at the cutoff bar fires REGARDLESS of `_preds` value.
   Threshold models (Limen's `threshold_logreg_binary`) legitimately
   emit `_preds=None` on uncertain bars; if the cutoff bar is one of
   those, the prior code returned `[]` immediately and force-flatten
   never fired. The strategy must close held inventory whenever
   `signal.timestamp >= cutoff`, irrespective of the predictor's
   abstain output.

2. The cutoff branch in `LongOnSignal.on_signal` is positioned
   STRUCTURALLY BEFORE the `_preds is None` early-return guard. A
   future PR that re-orders these branches breaks the close-by-end
   invariant silently — this static check catches the regression at
   the source level so it cannot ride through.

3. Sweep aborts loudly when ANY window ends with open inventory.
   Stats compiled on the leftover days after the most active days are
   silently dropped are not honest. The sweep refuses to ship a numeric
   verdict on a partial sample.
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
from nexus.strategy.action import ActionType
from nexus.strategy.context import StrategyContext
from nexus.strategy.params import StrategyParams
from nexus.strategy.signal import Signal

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
        f'long_on_signal_close_by_end_{tmp_path.name}', rendered_path,
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


def _signal(ts: datetime, preds: int | None) -> Signal:
    values: dict[str, object] = {'_probs': 0.5}
    if preds is not None:
        values['_preds'] = preds
    return Signal(
        predictor_fn_id='close-by-end-fixture',
        timestamp=ts,
        values=values,
    )


def _drive_into_long_state(mod: object) -> object:
    """Open a position via preds=1 BUY signal then deliver the FILLED outcome."""
    strategy = mod.Strategy('close-by-end-fixture')
    strategy.on_startup(StrategyParams(raw={}), _ctx())
    actions = strategy.on_signal(
        _signal(_T0, preds=1), StrategyParams(raw={}), _ctx(),
    )
    assert len(actions) == 1
    assert actions[0].action_type == ActionType.ENTER
    assert actions[0].direction == OrderSide.BUY
    fill_qty = actions[0].size
    outcome = TradeOutcome(
        outcome_id='oc-buy', command_id='cmd-buy',
        outcome_type=TradeOutcomeType.FILLED,
        timestamp=_T0 + timedelta(seconds=1),
        fill_size=fill_qty,
        fill_price=Decimal('70000'),
        fill_notional=fill_qty * Decimal('70000'),
        actual_fees=Decimal('0'),
        remaining_size=Decimal('0'),
        reject_reason=None, cancel_reason=None,
    )
    strategy.on_outcome(outcome, StrategyParams(raw={}), _ctx())
    assert getattr(strategy, '_long') is True
    return strategy


def test_force_flatten_fires_when_preds_is_none(tmp_path: Path) -> None:
    """The bug we're fixing: at the cutoff bar, _preds=None must NOT short-circuit force-flatten.

    Threshold models routinely emit _preds=None on uncertain bars
    (probability inside hysteresis band, below min_prob_floor, etc.).
    Pre-fix, the strategy returned [] immediately on _preds=None,
    skipping the cutoff branch entirely. A BUY filled earlier in the
    window would carry past window close as orphaned open inventory.
    Post-fix, the cutoff branch fires before the _preds None guard.
    """
    mod = _load_template(tmp_path)
    strategy = _drive_into_long_state(mod)
    cutoff_signal = _signal(_FORCE_FLATTEN_AFTER, preds=None)
    actions = strategy.on_signal(
        cutoff_signal, StrategyParams(raw={}), _ctx(),
    )
    assert len(actions) == 1, (
        f'expected 1 SELL action at cutoff with _preds=None, got '
        f'{len(actions)} actions: {actions!r}'
    )
    assert actions[0].action_type == ActionType.ENTER
    assert actions[0].direction == OrderSide.SELL


def test_force_flatten_branch_precedes_preds_none_guard() -> None:
    """Static contract: the cutoff branch MUST be positioned before the _preds None guard.

    A future PR that re-orders these branches breaks the close-by-end
    invariant silently. Pin the source-level position so the slice's
    capability cannot regress without this test failing first.
    """
    template_path = (
        Path(__file__).resolve().parents[2]
        / 'backtest_simulator' / 'pipeline'
        / '_strategy_templates' / 'long_on_signal.py'
    )
    src = template_path.read_text(encoding='utf-8')
    # zero-bang + Copilot caught this: the prior assertion used
    # `'signal.timestamp >= cutoff'` which also appears verbatim in
    # the file's preceding comment. `src.index()` would hit the
    # comment first; a future PR could move the executable check
    # below the preds-None guard while leaving the comment in
    # place, and the test would still pass silently. Anchor the
    # assertion on the unique executable line instead.
    cutoff_check_pos = src.index(
        'at_cutoff = cutoff is not None and signal.timestamp >= cutoff',
    )
    preds_none_pos = src.index('preds_raw is None')
    assert cutoff_check_pos < preds_none_pos, (
        f'cutoff check must precede _preds None guard in '
        f'on_signal source; got cutoff at offset '
        f'{cutoff_check_pos} and preds_none at offset '
        f'{preds_none_pos}'
    )


def test_sweep_aborts_on_open_inventory_exclusion() -> None:
    """Static contract: sweep MUST raise ParityViolation when n_runs_with_trailing_inventory > 0.

    Stats compiled on the surviving subset after the most active days
    are silently excluded are not honest. The sweep must abort loud
    rather than print a numeric verdict on a partial sample. This
    test pins the abort site so a future PR cannot remove it.
    """
    sweep_path = (
        Path(__file__).resolve().parents[2]
        / 'backtest_simulator' / 'cli' / 'commands' / 'sweep.py'
    )
    src = sweep_path.read_text(encoding='utf-8')
    assert 'n_runs_with_trailing_inventory > 0' in src, (
        'sweep must check n_runs_with_trailing_inventory > 0 and abort'
    )
    abort_block_start = src.index('n_runs_with_trailing_inventory > 0')
    next_section = src[abort_block_start:abort_block_start + 800]
    assert 'ParityViolation' in next_section, (
        'sweep abort site must raise ParityViolation, not log+continue'
    )
    assert 'raise' in next_section, (
        'sweep abort site must contain a raise, not just a print'
    )
