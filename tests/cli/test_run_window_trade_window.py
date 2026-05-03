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


def test_run_window_trade_window_references_interval_seconds() -> None:
    """`trade_window_seconds` MUST reference the bundle's `interval_seconds`.

    AST-based check (not substring) — walks `_run_window.py`'s AST,
    locates the `SimulatedVenueAdapter(...)` keyword
    `trade_window_seconds`, and asserts the local `interval_seconds`
    appears somewhere in the value expression. Allows future
    implementations that wrap the value (e.g. a `min(...)` clamp,
    a helper, or an explicit `int(...)` cast) as long as the
    bundle-derived kline size is the source of truth — the only
    regression this guards against is re-introducing a hardcoded
    constant or sourcing the window from anywhere other than the
    bundle.
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
    assert not isinstance(value, ast.Constant), (
        f'trade_window_seconds must not be a hardcoded constant '
        f'(was {ast.dump(value)}); use `interval_seconds` from '
        f'the bundle manifest.'
    )
    name_descendants = {
        n.id for n in ast.walk(value) if isinstance(n, ast.Name)
    }
    assert 'interval_seconds' in name_descendants, (
        f'trade_window_seconds expression must reference the local '
        f'`interval_seconds` (= bundle kline_size); '
        f'value AST: {ast.dump(value)} '
        f'(names referenced: {sorted(name_descendants)})'
    )


def test_run_window_passes_window_end_clamp_to_adapter() -> None:
    """`SimulatedVenueAdapter(... window_end_clamp=window_end ...)`.

    The wider tape walk (kline_size instead of 60s) creates a real
    risk: a SELL submitted near the run window's close can pull
    fills from past `window_end` if the venue isn't told where the
    window ends. This is silent lookahead — the strategy thinks
    it closed cleanly when in fact the closing fill came from
    future data.

    The fix lives at the venue layer (a per-submit clamp), but
    the venue can only enforce it if the run-window caller passes
    `window_end` explicitly. This test pins that plumbing.
    """
    src = _RUN_WINDOW_PATH.read_text(encoding='utf-8')
    tree = ast.parse(src)
    fn = _find_run_window_in_process(tree)
    adapter_call = _find_simulated_venue_adapter_call(fn)
    clamp_kwarg = next(
        (
            kw for kw in adapter_call.keywords
            if kw.arg == 'window_end_clamp'
        ),
        None,
    )
    assert clamp_kwarg is not None, (
        'SimulatedVenueAdapter(...) is missing the '
        'window_end_clamp keyword argument; without it, a submit '
        'near window-end can peek at post-window tape (silent '
        'lookahead, the bug Copilot caught on PR #57).'
    )
    name_descendants = {
        n.id for n in ast.walk(clamp_kwarg.value) if isinstance(n, ast.Name)
    }
    assert 'window_end' in name_descendants, (
        f'window_end_clamp expression must reference the local '
        f'`window_end`; got AST {ast.dump(clamp_kwarg.value)} '
        f'(names referenced: {sorted(name_descendants)}).'
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
