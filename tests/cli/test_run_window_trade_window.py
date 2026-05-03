"""`bts` venue walks one kline_size of trade tape per submit, not a hardcoded 60s.

Pre-fix, `_run_window.py` constructed `SimulatedVenueAdapter`
with `trade_window_seconds=60`. With BTCUSDT carrying multi-BTC
depth at every price level, a 60-second tape window often did
not contain enough volume on the right side of the book to fill
a 0.3 BTC MARKET order — even though a real exchange would have
crossed the book in microseconds. The result was "phantom
intents": orders the strategy emitted, the venue refused to
fill, and the strategy then dropped (or, post-PR-#55, latched
for retry on the SELL side). Re-running r0011 / r0014 sweeps on
April 2026 windows showed 5-10% of all trade activity invisible
behind this artefact.

The fix: pass `trade_window_seconds=interval_seconds`, where
`interval_seconds == kline_size` from the bundle's manifest. On
r0014 that's 14 400 s = 4 hours of tape per submit — well
beyond any realistic depth threshold on BTCUSDT.

This test pins the contract at the AST level (not a substring
or regex match) so a future PR cannot satisfy the gate by
leaving the comment in place while quietly regressing the
keyword to a hardcoded constant.
"""
from __future__ import annotations

import ast
from pathlib import Path

_RUN_WINDOW_PATH = (
    Path(__file__).resolve().parents[2]
    / 'backtest_simulator' / 'cli' / '_run_window.py'
)


def _find_run_window_in_process(tree: ast.Module) -> ast.FunctionDef:
    """Locate the `run_window_in_process` function definition in the AST."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == 'run_window_in_process'
        ):
            return node
    msg = (
        '`run_window_in_process` not found in '
        f'{_RUN_WINDOW_PATH}; either renamed or deleted'
    )
    raise AssertionError(msg)


def _find_simulated_venue_adapter_call(
    fn: ast.FunctionDef,
) -> ast.Call:
    """Locate the `SimulatedVenueAdapter(...)` call inside the function body."""
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == 'SimulatedVenueAdapter'
        ):
            return node
    msg = (
        '`SimulatedVenueAdapter(...)` call not found in '
        '`run_window_in_process`'
    )
    raise AssertionError(msg)


def test_run_window_passes_interval_seconds_as_trade_window() -> None:
    """`trade_window_seconds` MUST be `interval_seconds`, not a constant.

    AST-based check (not substring): walks `_run_window.py`'s AST,
    finds the `SimulatedVenueAdapter(...)` call inside
    `run_window_in_process`, locates the `trade_window_seconds`
    keyword argument, and asserts the value is the bare name
    `interval_seconds`. A constant integer (e.g. 60), a
    different name, or a comment carrying the string `interval_
    seconds` would all fail this check.
    """
    src = _RUN_WINDOW_PATH.read_text(encoding='utf-8')
    tree = ast.parse(src)
    fn = _find_run_window_in_process(tree)
    adapter_call = _find_simulated_venue_adapter_call(fn)
    trade_window_kwarg = next(
        (
            kw for kw in adapter_call.keywords
            if kw.arg == 'trade_window_seconds'
        ),
        None,
    )
    assert trade_window_kwarg is not None, (
        'SimulatedVenueAdapter(...) is missing the '
        'trade_window_seconds keyword argument; the venue would '
        'fall back to its 3600s default which is not aligned with '
        'the bundle kline_size.'
    )
    value = trade_window_kwarg.value
    assert isinstance(value, ast.Name), (
        f'trade_window_seconds must be `interval_seconds` (a name '
        f'reference to the kline-size local), got '
        f'{ast.dump(value)} of type {type(value).__name__}. '
        f'A constant or expression here re-introduces the '
        f'phantom-intent bug.'
    )
    assert value.id == 'interval_seconds', (
        f'trade_window_seconds must be the local '
        f'`interval_seconds` (= kline_size from the bundle '
        f'manifest); got name `{value.id}`.'
    )


def test_interval_seconds_is_assigned_from_kline_size() -> None:
    """`interval_seconds` is sourced from `read_kline_size_from_experiment_dir`.

    Pins the lineage from bundle manifest → tape walk window:
    the `trade_window_seconds` we pass to the venue (above) is
    only operator-correct if the local it references is the
    bundle's actual kline_size. A future PR could keep the
    name `interval_seconds` but reassign it from a different
    source, silently breaking the contract — this test catches
    that drift.
    """
    src = _RUN_WINDOW_PATH.read_text(encoding='utf-8')
    tree = ast.parse(src)
    fn = _find_run_window_in_process(tree)
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == 'interval_seconds'
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == 'read_kline_size_from_experiment_dir'
        ):
            return
    msg = (
        '`interval_seconds = read_kline_size_from_experiment_dir(...)` '
        'assignment not found in `run_window_in_process`. The '
        'venue trade-window contract requires this lineage.'
    )
    raise AssertionError(msg)
