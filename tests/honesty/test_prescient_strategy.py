"""Honesty gate: a deliberately-cheating strategy raises LookAheadViolation.

Pins slice #17 Task 6 MVC and SPEC §9.3 (Layer 1 honesty gate).

A `PrescientStrategy` deliberately tries to read `price[t+1]` /
`label[t+1]` through every public API path the simulator exposes to a
strategy:

  1. `HistoricalFeed.get_window(symbol, kline_size, n_rows)` — slices
     backward from `frozen_now()`; cannot reach future bars by
     construction. Pin the invariant explicitly so a regression that
     swaps the slice direction fails here.
  2. `HistoricalFeed.get_trades(symbol, start, end)` — must refuse
     `end > frozen_now()`. The strategy-facing Protocol has *no*
     `venue_lookahead_seconds` kwarg: the venue carve-out is on the
     `get_trades_for_venue` method which is on `VenueFeed` (extends
     `HistoricalFeed`) but not on `HistoricalFeed`. A cheating
     strategy with a `HistoricalFeed`-typed reference cannot reach
     the carve-out; the venue-only name documents the contract and
     a strategy that types its feed as `VenueFeed` to bypass it
     is tested below.
  3. `SignalsTable.lookup(t)` — must refuse `t > frozen_now()`. This
     is the path closed in this slice; before Task 6 the lookup
     silently returned the latest pre-existing row at `<= t`, which
     leaked future predictions to any strategy that constructed its
     own `t` ahead of the simulator clock.
  4. `SignalsTable._frame` — the underlying Polars frame is private
     (underscore-prefixed); strategies must not introspect past the
     `lookup(t)` causal accessor. The introspection test below
     documents this contract and verifies the public name `frame`
     is gone (so a strategy cannot peek without explicitly reaching
     past the underscore).

Every public-API cheating attempt MUST raise `LookAheadViolation`.
If any path silently succeeds, the gate has a hole — and the
operator's backtest-vs-live delta will diverge as soon as a strategy
ships that exercises the hole.

The test runs with `freezegun` set to a fixed wall-clock and asks the
strategy to peek with a `t` strictly after that clock. Each path is
exercised on its own so a future regression in any *one* path fails
in isolation rather than being masked by an earlier failure.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from freezegun import freeze_time

from backtest_simulator.exceptions import LookAheadViolation
from backtest_simulator.feed.parquet_fixture import ParquetFixtureFeed
from backtest_simulator.feed.protocol import HistoricalFeed
from backtest_simulator.sensors.precompute import (
    PredictionsInput,
    SignalsTable,
)

_BTCUSDT = 'BTCUSDT'
_FIXTURE = (
    Path(__file__).resolve().parents[1] / 'fixtures' / 'market'
    / 'btcusdt_1h_fixture.parquet'
)
_FROZEN_NOW = datetime(2020, 4, 1, 0, 0, tzinfo=UTC)
_PEEK_DELTA = timedelta(hours=1)


class PrescientStrategy:
    """Deliberately cheating: tries to peek `t+1` through every public path.

    A clean strategy in this codebase consumes a `Signal` from the
    sensor layer and emits actions; it never reaches behind the curtain
    to query the feed or the SignalsTable for a future `t`. The
    `attempt_*` methods here are the cheats this gate must catch — each
    one corresponds to a public API path that, if unguarded, would
    silently leak `price[t+1]` to the strategy.
    """

    def __init__(self, feed: HistoricalFeed, signals: SignalsTable) -> None:
        self._feed = feed
        self._signals = signals

    def attempt_feed_get_trades_past_now(self) -> pl.DataFrame:
        """Cheat #1: read trades from a window whose `end` is in the future."""
        return self._feed.get_trades(
            _BTCUSDT,
            start=_FROZEN_NOW - timedelta(days=1),
            end=_FROZEN_NOW + _PEEK_DELTA,
        )

    def attempt_signals_table_lookup_past_now(self) -> object:
        """Cheat #2: ask the SignalsTable for a `t` strictly after frozen_now."""
        return self._signals.lookup(_FROZEN_NOW + _PEEK_DELTA)


def _build_signals_table() -> SignalsTable:
    """Synthetic SignalsTable with one row before and one AT the peek.

    The future row's timestamp matches the peek (`_FROZEN_NOW + 1h`).
    Without the new `lookup` gate the future row would be returned
    and the test would silently pass — that's the leak the gate
    closes. With the gate the lookup raises BEFORE the filter runs.
    """
    timestamps = [
        _FROZEN_NOW - timedelta(hours=2),
        _FROZEN_NOW + _PEEK_DELTA,
    ]
    probs = np.array([0.4, 0.7], dtype=np.float64)
    preds = np.array([0, 1], dtype=np.int64)
    predictions = PredictionsInput(
        timestamps=timestamps,
        probs=probs,
        preds=preds,
        label_horizon_bars=1,
        bar_seconds=3600,
    )
    return SignalsTable.from_predictions(
        decoder_id='prescient-fixture',
        split_config=(0, 1, 2),
        predictions=predictions,
    )


def _strategy() -> PrescientStrategy:
    feed = ParquetFixtureFeed(_FIXTURE)
    signals = _build_signals_table()
    return PrescientStrategy(feed=feed, signals=signals)


# ---- One test per public API path. A future regression in *one* path
#      fails in isolation rather than being masked by an earlier path.


def test_prescient_via_feed_get_trades_raises() -> None:
    """`feed.get_trades(end > frozen_now)` must raise LookAheadViolation."""
    strategy = _strategy()
    with freeze_time(_FROZEN_NOW), pytest.raises(LookAheadViolation):
        strategy.attempt_feed_get_trades_past_now()


def test_prescient_via_signals_table_lookup_raises() -> None:
    """`SignalsTable.lookup(t > frozen_now)` must raise LookAheadViolation.

    The SignalsTable fixture explicitly contains a row at
    `_FROZEN_NOW + 2 hours`; without the gate this would be returned
    silently and a cheating strategy would have access to the future
    label. The gate added in `SignalsTable.lookup` raises before the
    frame is filtered.
    """
    strategy = _strategy()
    with freeze_time(_FROZEN_NOW), pytest.raises(LookAheadViolation):
        strategy.attempt_signals_table_lookup_past_now()


def test_prescient_at_frozen_now_passes_signals_table() -> None:
    """Lookup at exactly `frozen_now()` is causal and must NOT raise.

    The gate's contract is `t > frozen_now()`, not `t >= frozen_now()`.
    Querying the table at the *current* bar is the normal strategy
    path; a tighter gate would break every legitimate consumer.
    """
    strategy = _strategy()
    with freeze_time(_FROZEN_NOW):
        # The fixture has a row at `_FROZEN_NOW - 2h` (before now), so
        # lookup at exactly `_FROZEN_NOW` returns that row, not None.
        row = strategy._signals.lookup(_FROZEN_NOW)
    assert row is not None, (
        'lookup at frozen_now() must return the most recent prior row, '
        'not None — the gate is over-tightened (>= instead of >).'
    )
    assert row.timestamp == _FROZEN_NOW - timedelta(hours=2)


def test_prescient_lookup_rejects_naive_datetime() -> None:
    """Naive datetimes are loud-rejected to prevent gate bypass.

    A tz-naive `t` would compare implementation-defined against the
    tz-aware `frozen_now()` and then crash the Polars filter against
    the UTC-typed frame. Both failure modes hide a real lookahead
    leak from the operator's eye — `lookup` must raise before either
    can land.
    """
    strategy = _strategy()
    naive = datetime(2020, 4, 1, 0, 0)  # no tzinfo
    with freeze_time(_FROZEN_NOW), pytest.raises(ValueError, match='tz-aware'):
        strategy._signals.lookup(naive)


def test_prescient_signals_table_has_no_public_frame() -> None:
    """`SignalsTable.frame` must not be reachable as a public attribute.

    Codex round 1: a strategy with a SignalsTable reference could
    bypass `lookup` entirely via `signals.frame.filter(...)`. The
    fix renames the field to `_frame`; this test pins the contract
    so a regression that re-exposes `frame` (or adds any other
    public attribute that returns the underlying Polars frame) is
    caught loudly.
    """
    strategy = _strategy()
    public_attrs = [a for a in dir(strategy._signals) if not a.startswith('_')]
    leaks = [a for a in public_attrs
             if 'frame' in a.lower() or 'rows' in a.lower()]
    assert leaks == [], (
        f'SignalsTable exposes potentially future-leaking public '
        f'attributes: {leaks}. The underlying Polars frame must be '
        f'reachable only via `lookup(t)` (which checks the gate).'
    )


def test_prescient_signals_table_repr_does_not_dump_frame() -> None:
    """`repr(signals)` must not include future timestamps / labels.

    Codex round 3: dataclass default `repr` includes every field, so
    `repr(signals)` would dump the entire `_frame` Polars table
    contents. A strategy that simply logs/prints its decoder reference
    (a normal-looking diagnostic) would then have free access to every
    future row in plain text. The `_frame` field carries `repr=False`
    so its contents do not land in `repr` output. This test pins that
    contract: the rendered repr contains the decoder_id and
    split_config but no future-row contents.
    """
    strategy = _strategy()
    # The fixture has a row at `_FROZEN_NOW + 1h` (the leak target).
    future_iso = (_FROZEN_NOW + _PEEK_DELTA).isoformat()
    rendered = repr(strategy._signals)
    assert 'prescient-fixture' in rendered, (
        f'repr should still include `decoder_id` (a public, non-leaking '
        f'identifier); got: {rendered!r}'
    )
    assert future_iso not in rendered, (
        f'repr(signals) leaked the future row timestamp {future_iso} — '
        f'the `_frame` field must carry `repr=False`. Rendered: {rendered!r}'
    )
    # Also assert the leak doesn't show up as the row's float prob/pred
    # in any obvious encoded form. The fixture's future row has prob=0.7,
    # pred=1; a leaked frame repr would print these.
    assert '0.7' not in rendered, (
        f'repr(signals) leaked the future-row prob 0.7. Rendered: {rendered!r}'
    )


def test_prescient_via_feed_get_trades_has_no_venue_kwarg() -> None:
    """The strategy-facing `HistoricalFeed.get_trades` exposes no carve-out kwarg.

    Codex round 1: a cheating strategy could pass
    `venue_lookahead_seconds=3600` and bypass the strict gate. The
    fix removes the kwarg from the `HistoricalFeed` Protocol and
    the implementation's strategy-facing `get_trades`. The bounded
    venue carve-out lives on `VenueFeed.get_trades_for_venue`
    (a separate Protocol, not extended by `HistoricalFeed`). This
    test pins the contract via the Protocol's signature.
    """
    import inspect
    sig = inspect.signature(HistoricalFeed.get_trades)
    forbidden = {'venue_lookahead_seconds'}
    leaked = [name for name in sig.parameters if name in forbidden]
    assert leaked == [], (
        f'HistoricalFeed.get_trades exposes carve-out kwarg(s) '
        f'{leaked} on the strategy-facing Protocol. A cheating '
        f'strategy can bypass the lookahead gate by passing them.'
    )


_PROTOCOL_MOD = 'backtest_simulator.feed.protocol'
_FEED_MOD = 'backtest_simulator.feed'
_CARVE_OUT_ATTR = 'get_trades_for_venue'
_DYNAMIC_IMPORT_NAMES = frozenset({'__import__', 'import_module'})
_DYNAMIC_EXEC_NAMES = frozenset({'eval', 'exec'})
# Modules strategy code has NO legitimate need to import at all.
# Each is a reach-around lever a cheating strategy could use to bind
# `__import__` / `import_module` / `eval` / `exec` (or attribute-
# walk into the protocol module). Root-matched, so submodules
# (`importlib.util`, `ctypes.pythonapi`) are also banned.
_BANNED_HELPER_MODULES = frozenset({
    'importlib', 'builtins', 'sys', 'inspect',
    'operator', 'ctypes', 'gc', 'pkgutil', 'pydoc',
})
# Banned callable names — matched whether bound by direct name
# (`getattr(_)`), attribute (`logging.sys.modules['builtins'].getattr(_)`),
# or alias (`ga = getattr; ga(_)`). Strategy code uses simple
# named attribute reads; everything in this set is escape-hatch
# territory.
_BANNED_CALLABLES = frozenset({
    'getattr', 'hasattr', 'setattr', 'delattr',
    'globals', 'locals', 'vars', 'dir', 'type',
    'eval', 'exec',
    '__import__', 'import_module',
    'attrgetter', 'methodcaller',  # operator.attrgetter / methodcaller
    'resolve_name', 'locate',  # pkgutil.resolve_name / pydoc.locate
})
# Meta-introspection attribute names — banning these closes the
# `type(feed).__dict__['carve_out']`, `feed.__getattribute__('...')`,
# `func.__globals__['__builtins__'][...]`, `print.__self__.foo`
# style bypasses.
_BANNED_META_ATTRS = frozenset({
    '__dict__', '__class__', '__mro__', '__subclasses__',
    '__getattribute__', '__getattr__', '__builtins__',
    '__globals__', '__self__', '__func__', '__code__',
    '__closure__', '__module__', '__base__', '__bases__',
    '__reduce__', '__reduce_ex__',
})


def _module_path_matches(node_module: str | None, level: int, target: str) -> bool:
    """Return True if `node` resolves (with relative-level normalization) to `target`.

    `from ..feed.protocol import VenueFeed` lands as `module="feed.protocol"`
    with `level=2`, etc. For the bypass scan we don't need full
    package resolution — any `from <relative>.feed.protocol import _`
    or `from <relative>.feed import protocol` is suspicious regardless
    of the leading dots.
    """
    if level == 0:
        return node_module == target
    suffix = target.split('.', 1)[1] if '.' in target else target
    return node_module is not None and (
        node_module == target or node_module.endswith(suffix) or node_module == suffix
    )


def _root_module(name: str | None) -> str | None:
    """Return the leftmost dotted segment (`importlib.util` → `importlib`)."""
    if name is None:
        return None
    return name.split('.', 1)[0]


def _scan_import_from(node: object, rel: Path) -> list[str]:
    """Catch protocol imports + banned helper-module imports (root-matched)."""
    import ast
    if not isinstance(node, ast.ImportFrom):
        return []
    leaks: list[str] = []
    level = node.level or 0
    if _module_path_matches(node.module, level, _PROTOCOL_MOD):
        for a in node.names:
            if a.name in ('VenueFeed', '*'):
                dots = '.' * level
                leaks.append(f'{rel}: from {dots}{node.module or ""} import {a.name}')
    elif _module_path_matches(node.module, level, _FEED_MOD):
        for a in node.names:
            if a.name == 'protocol':
                tail = f' as {a.asname}' if a.asname else ''
                dots = '.' * level
                leaks.append(f'{rel}: from {dots}{node.module or ""} import protocol{tail}')
    if _root_module(node.module) in _BANNED_HELPER_MODULES:
        names = [a.name for a in node.names]
        leaks.append(f'{rel}: from {node.module} import {names!r} (banned helper module)')
    return leaks


def _scan_import(node: object, rel: Path) -> list[str]:
    """Catch `import PROTOCOL` and `import {importlib|builtins|sys|inspect}[.x]`."""
    import ast
    if not isinstance(node, ast.Import):
        return []
    leaks: list[str] = []
    for a in node.names:
        if a.name == _PROTOCOL_MOD:
            leaks.append(f'{rel}: import {a.name}')
        if _root_module(a.name) in _BANNED_HELPER_MODULES:
            leaks.append(f'{rel}: import {a.name} (banned helper module root)')
    return leaks


def _scan_call(node: object, rel: Path) -> list[str]:
    """Ban every banned-callable call shape (Name, Attribute, alias).

    Resolves the function's final binding name and matches against
    `_BANNED_CALLABLES`. Catches: bare `getattr(_)`,
    `obj.getattr(_)`, `logging.sys.modules['builtins'].getattr(_)`,
    `operator.attrgetter('...')(_)`, `pkgutil.resolve_name('...')(_)`.
    """
    import ast
    if not isinstance(node, ast.Call):
        return []
    f = node.func
    name = (f.id if isinstance(f, ast.Name)
            else f.attr if isinstance(f, ast.Attribute) else None)
    if name in _BANNED_CALLABLES:
        return [f'{rel}: banned-callable call `{name}(...)`']
    return []


def _scan_attribute(node: object, rel: Path) -> list[str]:
    """Catch carve-out access AND meta-introspection attrs."""
    import ast
    if not isinstance(node, ast.Attribute):
        return []
    if node.attr == _CARVE_OUT_ATTR:
        return [f'{rel}: attribute access `_.{_CARVE_OUT_ATTR}`']
    if node.attr in _BANNED_META_ATTRS:
        return [f'{rel}: attribute access `_.{node.attr}` (banned meta attr)']
    return []


def _scan_name(node: object, rel: Path) -> list[str]:
    """Catch any read of a banned callable / `__builtins__` global.

    A cheating strategy can alias a banned callable
    (`ga = getattr`) and call the alias instead. This `Name`-Load
    walker rejects any `Load`-context read of a banned name —
    catches the alias-binding shape AND the bare-`__builtins__`
    shape from the round-5 lockdown.
    """
    import ast
    if not isinstance(node, ast.Name):
        return []
    if not isinstance(getattr(node, 'ctx', None), ast.Load):
        return []
    banned = _BANNED_CALLABLES | {'__builtins__'}
    if node.id in banned:
        return [f'{rel}: banned-name read `{node.id}` (alias bypass)']
    return []


def _scan_for_venue_feed_import(path: Path, repo: Path) -> list[str]:
    """Return one entry per VenueFeed-bypass path found in `path`.

    Catches every statically-detectable path a cheating strategy
    could use to reach `VenueFeed` / `get_trades_for_venue` (codex
    post-auditor-4 round 3 expanded the scan to close additional
    surfaces the round-2 enumeration missed):

      1. `from backtest_simulator.feed.protocol import VenueFeed | *`
      2. `from backtest_simulator.feed import protocol [as X]`
      3. `import backtest_simulator.feed.protocol`
      4. ANY dynamic-import call: `__import__(...)`,
         `importlib.import_module(...)` / `il.import_module(...)` /
         `import_module(...)` (regardless of how the callable got
         bound — strategy modules have NO legitimate need for
         dynamic imports, so the entire call shape is banned).
      5. `getattr(_, 'get_trades_for_venue')`.
      6. ANY attribute access `_.get_trades_for_venue` (catches the
         direct-call form `self._feed.get_trades_for_venue(...)`,
         attribute-read form, and any chained reach).
      7. `eval(...)` / `exec(...)` — banned in strategy modules
         (catches the constant-string variant of the bypass).

    Each entry returned is one leak with file + kind + source line.
    """
    import ast
    tree = ast.parse(path.read_text(encoding='utf-8'))
    rel = path.relative_to(repo)
    leaks: list[str] = []
    for node in ast.walk(tree):
        leaks.extend(_scan_import_from(node, rel))
        leaks.extend(_scan_import(node, rel))
        leaks.extend(_scan_call(node, rel))
        leaks.extend(_scan_attribute(node, rel))
        leaks.extend(_scan_name(node, rel))
    return leaks


def test_no_strategy_module_imports_venue_feed() -> None:
    """Strategy modules MUST NOT import `VenueFeed`.

    Codex post-auditor-4 P1: when `_get_trades_for_venue` was
    renamed to `get_trades_for_venue` (no underscore) and stayed
    on the `VenueFeed` Protocol, a cheating strategy could
    `from backtest_simulator.feed.protocol import VenueFeed` then
    `cast(VenueFeed, self._feed).get_trades_for_venue(...,
    venue_lookahead_seconds=3600)` and bypass the strict gate.
    The honesty signal — "this Protocol is adapter-only" — is
    preserved at the import-graph level: this test fails if any
    strategy template / strategy module imports the carve-out
    Protocol.

    Strategies legitimately type their feed as `HistoricalFeed`
    (the strategy-facing surface, no carve-out kwarg). A strategy
    that needs to import `VenueFeed` is — by construction — not a
    strategy. The static check at this level catches the import
    before the cheating call site can be written.
    """
    repo = Path(__file__).resolve().parents[2]
    strategy_dirs = (
        repo / 'backtest_simulator' / 'strategies',
        repo / 'backtest_simulator' / 'pipeline' / '_strategy_templates',
    )
    leaks: list[str] = []
    for d in strategy_dirs:
        for path in d.rglob('*.py'):
            leaks.extend(_scan_for_venue_feed_import(path, repo))
    assert leaks == [], (
        'Strategy modules must not import VenueFeed (the venue-only '
        'carve-out Protocol). The following modules leak the import:\n'
        + '\n'.join(f'  - {leak}' for leak in leaks)
    )


def test_prescient_via_feed_get_window_returns_only_causal_rows() -> None:
    """`feed.get_window` walks backward from now; cannot reach future bars.

    By construction, `get_window` returns the latest `n_rows` with
    `timestamp <= frozen_now()`. There is no `n_rows` argument that
    causes future rows to appear — but pin the invariant explicitly
    so a future regression that swaps the slice direction fails here.
    """
    strategy = _strategy()
    with freeze_time(_FROZEN_NOW):
        window = strategy._feed.get_window(_BTCUSDT, kline_size=3600, n_rows=200)
    assert not window.is_empty(), (
        'fixture window is unexpectedly empty; cannot pin the no-look-ahead '
        'invariant on an empty frame.'
    )
    latest = window['open_time'].max()
    assert latest <= _FROZEN_NOW, (
        f'feed.get_window returned a row with open_time={latest} > '
        f'frozen_now()={_FROZEN_NOW}; the no-look-ahead gate has a hole '
        f'in the slice-direction code path.'
    )
