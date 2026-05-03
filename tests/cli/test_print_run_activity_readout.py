"""print_run honest activity readout — trader's five-question scan."""
from __future__ import annotations

from datetime import UTC, datetime

from backtest_simulator.cli._metrics import Trade, pair_trades, print_run


def _trade(side: str, qty: str = '1', price: str = '70000') -> Trade:
    return Trade(
        coid=f'coid-{side.lower()}-{qty}-{price}',
        side=side, qty=qty, price=price,
        fee='0', fee_asset='USDT',
        ts=datetime(2026, 4, 12, 12, 0, tzinfo=UTC).isoformat(),
    )


def test_pair_trades_returns_trailing_unclosed_buy() -> None:
    """A BUY without a matching SELL must surface as trailing, not silently dropped."""
    trades = [_trade('BUY')]
    pairs, trailing = pair_trades(trades)
    assert pairs == []
    assert len(trailing) == 1
    assert trailing[0].side.name == 'BUY'


def test_pair_trades_pairs_buy_sell_round_trip() -> None:
    trades = [_trade('BUY'), _trade('SELL')]
    pairs, trailing = pair_trades(trades)
    assert len(pairs) == 1
    assert trailing == []


def test_print_run_genuinely_flat_day_no_extras(capsys: object) -> None:
    """All-zero counters → no parenthetical."""
    print_run(0, '2026-04-09', [], {})
    captured = capsys.readouterr().out  # type: ignore[attr-defined]
    headline = captured.splitlines()[0]
    assert 'trades 0' in headline
    assert '(' not in headline.split('PF')[0]
    assert '+' not in headline.split('PF')[0]


def test_print_run_canonical_expire_lifecycle(capsys: object) -> None:
    """The canonical 04-12 r0011 case: intent submitted, expired, no fill.

    EventSpine pattern observed: `OrderSubmitIntent → OrderSubmitted →
    OrderExpired → TradeOutcomeProduced`. Counter helper returns
    `intents=1 fills=0 pending=0 rejects=1` (OrderExpired counts as
    a reject — terminal non-fill). The headline must show the
    activity, not just `trades 0`.
    """
    print_run(
        0, '2026-04-12', [], {},
        n_intents=1, n_fills=0, n_pending=0, n_rejects=1,
    )
    captured = capsys.readouterr().out  # type: ignore[attr-defined]
    headline = captured.splitlines()[0]
    assert 'trades 0 (intents 1, rejects 1)' in headline


def test_print_run_pending_at_close(capsys: object) -> None:
    """1 intent still in flight at window close (PENDING outcome)."""
    print_run(
        0, '2026-04-12', [], {},
        n_intents=1, n_fills=0, n_pending=1, n_rejects=0,
    )
    captured = capsys.readouterr().out  # type: ignore[attr-defined]
    headline = captured.splitlines()[0]
    assert 'trades 0 (intents 1, pending 1)' in headline


def test_print_run_open_position_after_buy_fill(capsys: object) -> None:
    """BUY filled, no closing SELL fill → unmatched filled inventory."""
    print_run(
        0, '2026-04-12', [_trade('BUY')], {},
        n_intents=2, n_fills=1, n_pending=1, n_rejects=0,
    )
    captured = capsys.readouterr().out  # type: ignore[attr-defined]
    headline = captured.splitlines()[0]
    assert 'trades 0 (intents 2, fills 1, pending 1, +1 open)' in headline


def test_print_run_clean_round_trip(capsys: object) -> None:
    """Standard 1-pair day: 2 intents, 2 fills, no anomalies."""
    print_run(
        0, '2026-04-06', [_trade('BUY'), _trade('SELL')], {},
        n_intents=2, n_fills=2, n_pending=0, n_rejects=0,
    )
    captured = capsys.readouterr().out  # type: ignore[attr-defined]
    headline = captured.splitlines()[0]
    assert 'trades 1 (intents 2, fills 2)' in headline


def test_print_run_all_blocked_validator(capsys: object) -> None:
    """Both intents validator-rejected → 0 fills, 0 pending, all rejects."""
    print_run(
        0, '2026-04-12', [], {},
        n_intents=2, n_fills=0, n_pending=0, n_rejects=2,
    )
    captured = capsys.readouterr().out  # type: ignore[attr-defined]
    headline = captured.splitlines()[0]
    assert 'trades 0 (intents 2, rejects 2)' in headline


def test_print_run_default_kwargs_preserve_legacy_format(
    capsys: object,
) -> None:
    """Callers that don't thread the new params get a clean legacy headline."""
    print_run(0, '2026-04-09', [], {})
    captured = capsys.readouterr().out  # type: ignore[attr-defined]
    headline = captured.splitlines()[0]
    pre_pf = headline.split('PF')[0]
    assert 'trades 0' in headline
    assert 'intents' not in pre_pf
    assert 'fills' not in pre_pf
    assert 'pending' not in pre_pf
    assert 'rejects' not in pre_pf
