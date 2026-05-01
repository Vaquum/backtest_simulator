"""print_run honest activity readout: trades / +open / orders distinguish flat from fail-to-fill."""
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
    """trades=0 + orders=0 + no fills => no parenthetical activity extras."""
    print_run(0, '2026-04-09', [], {}, n_orders=0)
    captured = capsys.readouterr().out  # type: ignore[attr-defined]
    headline = captured.splitlines()[0]
    assert 'trades 0' in headline
    assert '(' not in headline.split('PF')[0]
    assert '+' not in headline.split('PF')[0]


def test_print_run_orders_fired_no_fill_shows_orders(capsys: object) -> None:
    """trades=0 + orders=1 (e.g. PENDING outcome) MUST surface orders count.

    Reproduces perm 3 2026-04-12 in the canonical r0011 sweep:
    EventSpine had 1 CommandAccepted + 1 OrderSubmitIntent +
    1 OrderSubmitted + 1 TradeOutcomeProduced(PENDING). account.trades
    was empty (no fill landed). Pre-fix this rendered as `trades 0`
    indistinguishable from a flat day.
    """
    print_run(0, '2026-04-12', [], {}, n_orders=1)
    captured = capsys.readouterr().out  # type: ignore[attr-defined]
    headline = captured.splitlines()[0]
    assert 'trades 0 (orders 1)' in headline


def test_print_run_open_position_shows_plus_open(capsys: object) -> None:
    """A BUY filled with no closing SELL fill must show `+1 open`."""
    print_run(0, '2026-04-12', [_trade('BUY')], {}, n_orders=2)
    captured = capsys.readouterr().out  # type: ignore[attr-defined]
    headline = captured.splitlines()[0]
    assert 'trades 0 (+1 open, orders 2)' in headline


def test_print_run_closed_pair_shows_orders_too(capsys: object) -> None:
    """Standard 1-pair day: trades 1 (orders 2) — order count contextualises."""
    print_run(
        0, '2026-04-06', [_trade('BUY'), _trade('SELL')], {}, n_orders=2,
    )
    captured = capsys.readouterr().out  # type: ignore[attr-defined]
    headline = captured.splitlines()[0]
    assert 'trades 1 (orders 2)' in headline


def test_print_run_orders_default_zero_preserves_legacy_format(
    capsys: object,
) -> None:
    """When no caller threads n_orders, output is unchanged from prior behaviour."""
    print_run(0, '2026-04-09', [], {})
    captured = capsys.readouterr().out  # type: ignore[attr-defined]
    headline = captured.splitlines()[0]
    assert 'trades 0' in headline
    assert '(' not in headline.split('PF')[0]
