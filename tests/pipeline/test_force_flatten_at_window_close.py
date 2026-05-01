"""long_on_signal force-flatten: at window-close cutoff, exit open inventory and refuse new BUYs."""
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


def _load_template(tmp_path: Path, *, force_flatten_after: datetime | None) -> object:
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
            'force_flatten_after': (
                None if force_flatten_after is None
                else force_flatten_after.isoformat()
            ),
        }),
    )
    rendered_path = tmp_path / 'long_on_signal.py'
    rendered_path.write_text(rendered, encoding='utf-8')
    spec = importlib.util.spec_from_file_location(
        f'long_on_signal_force_flatten_{tmp_path.name}', rendered_path,
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


def _signal(ts: datetime, preds: int) -> Signal:
    return Signal(
        predictor_fn_id='force-flatten-fixture',
        timestamp=ts,
        values={'_preds': preds, '_probs': 0.5},
    )


def _build_long_strategy(mod: object) -> object:
    """Build a strategy and drive it into a long state via signal+outcome."""
    strategy = mod.Strategy('force-flatten-fixture')
    strategy.on_startup(StrategyParams(raw={}), _ctx())
    enter = strategy.on_signal(
        _signal(_T0, preds=1), StrategyParams(raw={}), _ctx(),
    )
    assert len(enter) == 1, f'expected one BUY action, got {enter!r}'
    fill_qty = enter[0].size
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


def test_force_flatten_emits_sell_at_cutoff_signal_when_long(
    tmp_path: Path,
) -> None:
    """A signal at force_flatten_after must emit SELL even with preds=1."""
    mod = _load_template(tmp_path, force_flatten_after=_FORCE_FLATTEN_AFTER)
    strategy = _build_long_strategy(mod)

    actions = strategy.on_signal(
        _signal(_FORCE_FLATTEN_AFTER, preds=1),
        StrategyParams(raw={}), _ctx(),
    )

    assert len(actions) == 1
    assert actions[0].action_type == ActionType.ENTER
    assert actions[0].direction == OrderSide.SELL


def test_force_flatten_emits_sell_after_cutoff_signal_when_long(
    tmp_path: Path,
) -> None:
    """A signal past the cutoff must emit SELL when long."""
    mod = _load_template(tmp_path, force_flatten_after=_FORCE_FLATTEN_AFTER)
    strategy = _build_long_strategy(mod)

    past_cutoff = _FORCE_FLATTEN_AFTER + timedelta(seconds=10)
    actions = strategy.on_signal(
        _signal(past_cutoff, preds=1), StrategyParams(raw={}), _ctx(),
    )

    assert len(actions) == 1
    assert actions[0].direction == OrderSide.SELL


def test_force_flatten_blocks_new_buys_at_cutoff_when_flat(
    tmp_path: Path,
) -> None:
    """preds=1 at the cutoff must NOT open a new position."""
    mod = _load_template(tmp_path, force_flatten_after=_FORCE_FLATTEN_AFTER)
    strategy = mod.Strategy('force-flatten-fixture')
    strategy.on_startup(StrategyParams(raw={}), _ctx())
    assert getattr(strategy, '_long') is False

    actions = strategy.on_signal(
        _signal(_FORCE_FLATTEN_AFTER, preds=1),
        StrategyParams(raw={}), _ctx(),
    )

    assert actions == []


def test_force_flatten_does_not_fire_before_cutoff_when_long(
    tmp_path: Path,
) -> None:
    """A pre-cutoff preds=1 signal must NOT emit SELL when long."""
    mod = _load_template(tmp_path, force_flatten_after=_FORCE_FLATTEN_AFTER)
    strategy = _build_long_strategy(mod)

    pre_cutoff = _FORCE_FLATTEN_AFTER - timedelta(seconds=1)
    actions = strategy.on_signal(
        _signal(pre_cutoff, preds=1), StrategyParams(raw={}), _ctx(),
    )

    assert actions == []


def test_force_flatten_disabled_when_after_is_none(tmp_path: Path) -> None:
    """force_flatten_after=None preserves legacy behaviour (positions can orphan)."""
    mod = _load_template(tmp_path, force_flatten_after=None)
    strategy = _build_long_strategy(mod)

    way_past = _WINDOW_END + timedelta(hours=1)
    actions = strategy.on_signal(
        _signal(way_past, preds=1), StrategyParams(raw={}), _ctx(),
    )

    assert actions == []
