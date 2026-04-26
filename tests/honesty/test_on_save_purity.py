"""Honesty gate: every strategy's on_save is bytes-pure under save→load→save.

Pins slice #17 Task 9 MVC and SPEC §9.5 — "`on_save` purity sub-rule":

> A strategy's `on_save` must be a *read-only* snapshot of state. If
> it mutates state (incrementing a save counter, stamping "last save
> time", touching any attribute), the bytes of the first and second
> save differ after a load round-trip — which quietly breaks the
> spine-bytes determinism gate for any run that crosses a snapshot
> boundary. Discovered in the phase-3 probe: our demo strategy had
> `_save_calls += 1` inside `on_save` and the round-trip test
> caught it.

For every strategy registered in `backtest_simulator.strategies`:

    blob1 = strategy.on_save()          # snapshot post-mutations
    fresh = SAME_STRATEGY_CTOR(...)
    fresh.on_load(blob1)                 # re-hydrate from blob1
    blob2 = fresh.on_save()              # snapshot the loaded state
    assert blob1 == blob2

Failure raises `AssertionError` (the production gate would raise
`DeterminismViolation`, but the per-strategy round-trip is a unit-
level pin we keep cheap). The test runs each strategy through a
handful of signals first so the saved bytes carry actual state
(`_long`, `_entry_qty`, RNG counters), not a fresh-init snapshot
which would round-trip trivially even for a buggy strategy.
"""
from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from nexus.core.domain.operational_mode import OperationalMode
from nexus.strategy.context import StrategyContext
from nexus.strategy.params import StrategyParams
from nexus.strategy.signal import Signal

from backtest_simulator.strategies import (
    BuyAndHoldStrategy,
    InversePrescientStrategy,
    OverTradingStrategy,
    RandomTimingStrategy,
    ZeroTradeStrategy,
)


def _ctx() -> StrategyContext:
    return StrategyContext(
        positions=(),
        capital_available=Decimal('100000'),
        operational_mode=OperationalMode.ACTIVE,
    )


def _drive_through_signals(
    strategy: object, signals: list[Signal],
) -> None:
    """Call `on_signal` for each fixture signal so state actually mutates."""
    params = StrategyParams(raw={})
    context = _ctx()
    for sig in signals:
        strategy.on_signal(sig, params, context)


def _signal_at(ts: datetime, **values: object) -> Signal:
    return Signal(predictor_fn_id='on-save-fixture', timestamp=ts, values=values)


def _freeze(value: object) -> object:
    """Recursively normalise `value` into something comparable by value.

    Codex round 9 pinned the shallow-snapshot gap: `_full_snapshot`
    used to keep raw references for nested mutables. An `on_save`
    that mutated an existing list/dict/set in place (or mutated a
    nested attr under `_config`) would leave pre and post
    snapshots equal because both pointed at the same now-mutated
    object. This recursion freezes containers and plain-class
    holders into immutable / value-comparable structures so a
    pre-snapshot is preserved across a subsequent in-place
    mutation.

    Normalisations applied (depth-first):
      - `random.Random` → `getstate()` tuple (default `__eq__` is
        identity, getstate() is value).
      - `list` / `tuple` / `set` / `frozenset` → tuple of frozen
        elements (sets sorted by repr to make order deterministic
        for set comparisons that cross hash-randomisation).
      - `dict` → tuple of `(key, frozen(value))` pairs sorted by
        key-repr — same rationale as set sorting.
      - any object with `__dict__` (plain classes, dataclasses,
        nested config holders) → recurse via
        `_freeze(dict(vars(obj)))`.
      - everything else (Decimal, str, int, bool, float, bytes,
        None) → pass through unchanged.
    """
    import random
    if isinstance(value, random.Random):
        return value.getstate()
    if isinstance(value, (str, bytes, bool, int, float, Decimal)) or value is None:
        return value
    if isinstance(value, dict):
        # Freeze BOTH keys and values, preserve INSERTION ORDER. A
        # `repr(k)` shortcut would let a strategy stash state in an
        # identity-hashed key object and mutate the key's internals
        # during on_save without changing repr — codex round 10
        # pinned that bypass. Sorting the pairs would erase
        # insertion order, which is itself observable state in
        # Python 3.7+ — codex round 11 pinned that bypass: an
        # on_save that does `pop`/reinsert (or rebuilds the dict
        # with identical pairs in a new order) would change
        # iteration order without changing the sorted-tuple
        # representation. Preserving insertion order via a plain
        # tuple of (frozen_key, frozen_value) catches both.
        return tuple(
            (_freeze(k), _freeze(v)) for k, v in value.items()
        )
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(repr(_freeze(v)) for v in value))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if hasattr(value, '__dict__'):
        # Plain class / dataclass holder — recurse one level via
        # vars(); _freeze of the inner dict handles its own keys
        # and values.
        return _freeze(dict(vars(value)))
    # Fail-loud on unknown types. The earlier `return repr(value)`
    # fallback was not honest: a mutable object without `__dict__`
    # (e.g. a third-party type with internal state) could mutate in
    # place while keeping the same repr and slip past every check.
    # Codex round 10 pinned this gap. Production strategies must
    # only carry primitives, containers, RNG instances, or plain
    # config classes; anything else surfaces here as a TypeError so
    # the test author extends `_freeze` for the new type explicitly.
    msg = (
        f'_freeze: unsupported attribute type {type(value).__name__!r} '
        f'(value={value!r}). Add an explicit normalisation branch '
        f'rather than rely on a repr() fallback that can hide in-place '
        f'mutation. Honesty gates must fail loud on unknowns.'
    )
    raise TypeError(msg)


def _full_snapshot(strategy: object) -> object:
    """Capture EVERY attribute of `strategy.__dict__` for purity comparison.

    Routes the strategy's `__dict__` through `_freeze`, which (a)
    walks every attribute (catches creation of new attributes — the
    SPEC §9.5 `_save_calls` regression class), (b) recursively
    normalises nested mutables to value-comparable form (catches
    in-place container mutation), AND (c) preserves insertion
    order (catches `pop`+reinsert of an existing key). Codex
    rounds 8 / 9 / 10 / 11 / 12 progressively pinned each layer of
    that contract.

    Returning `_freeze(vars(strategy))` rather than sorting the
    top-level pairs is the round-12 fix: a sort at this level
    would itself erase the strategy's own `__dict__` insertion
    order, which is observable state in Python 3.7+.
    """
    return _freeze(vars(strategy))


_T0 = datetime(2026, 4, 20, 0, 0, tzinfo=UTC)
_BUY_HOLD_SIGNALS = [
    _signal_at(_T0),  # opens long
]
_OVER_TRADING_SIGNALS = [
    _signal_at(_T0, _preds=0),
    _signal_at(_T0, _preds=1),
    _signal_at(_T0, _preds=0),  # ends long-after-flat-after-long
]
_RANDOM_TIMING_SIGNALS = [
    _signal_at(_T0, _preds=i, _probs=0.5) for i in range(7)
]
_INVERSE_PRESCIENT_SIGNALS = [
    _signal_at(_T0, _future_pred=0),  # inverse=1 → BUY
    _signal_at(_T0, _future_pred=0),  # inverse=1 → already long, hold
    _signal_at(_T0, _future_pred=1),  # inverse=0 → SELL close
    _signal_at(_T0, _future_pred=0),  # inverse=1 → BUY again
]
_ZERO_TRADE_SIGNALS = [
    _signal_at(_T0),
    _signal_at(_T0),
]


def _round_trip_bytes(
    strategy_factory: callable,
    signals: list[Signal],
    precondition: callable | None = None,
    snapshot: callable | None = None,
) -> tuple[bytes, bytes]:
    """save → load (into a fresh instance) → save; return (blob1, blob2).

    `precondition`, if supplied, runs against the driven strategy
    BEFORE the first `on_save` is called. Running it after `on_save`
    would let a mutating implementation flip `_long` / `_entry_qty` /
    `_consumed_calls` and make a default-state fixture look
    non-trivial — exactly the regression this gate must catch.

    `snapshot`, if supplied, is a callable `(strategy) -> hashable`
    returning a comparable value covering ALL live state that
    `on_save` MUST NOT mutate (including the RNG via
    `random.Random.getstate()`, not only serialised scalar fields).
    The helper takes the snapshot before `on_save`, re-takes after,
    and asserts equality. Without this check a broken `on_save` that
    serialises a cleared state and ALSO clears the live attributes
    would pass the byte-equality test (both blobs serialise the
    same cleared state) while still being a real bug — `on_save`
    would have left the live strategy in a wrong state for every
    post-snapshot tick. Codex pinned this gap. Per-strategy
    `snapshot` callbacks (instead of a `state_attrs` allow-list)
    make the live-state coverage opaque to the strategy's own
    representation: e.g. RandomTiming includes
    `strategy._rng.getstate()` which the helper cannot inspect by
    attribute-name lookup.
    """
    strategy = strategy_factory()
    _drive_through_signals(strategy, signals)
    if precondition is not None:
        precondition(strategy)
    pre_snap = snapshot(strategy) if snapshot is not None else None
    blob1 = strategy.on_save()
    post_snap = snapshot(strategy) if snapshot is not None else None
    if snapshot is not None and pre_snap != post_snap:
        msg = (
            f'on_save mutated live strategy state across the call. '
            f'pre={pre_snap} post={post_snap}. blob1 byte-equality '
            f'cannot save this regression: a mutating on_save can serialise '
            f'a cleared state, leaving every post-snapshot tick wrong.'
        )
        raise AssertionError(msg)
    fresh = strategy_factory()
    fresh.on_load(blob1)
    # Load-back fidelity: the freshly-loaded instance MUST observe the
    # same snapshot as the pre-save state. A `pure but wrong` on_save
    # that serialises default state (instead of the live mutations)
    # would pass `blob1 == blob2` because both are the cleared blob,
    # but `fresh.on_load(blob1)` would restore the cleared state, NOT
    # the real driven state — which the live system would replay
    # incorrectly after any snapshot. This check catches that.
    if snapshot is not None:
        loaded_snap = snapshot(fresh)
        if loaded_snap != pre_snap:
            msg = (
                f'on_load did not restore the pre-save snapshot. '
                f'pre={pre_snap} loaded={loaded_snap}. on_save serialised a '
                f'state different from the live one, OR on_load failed to '
                f'reconstruct it.'
            )
            raise AssertionError(msg)
    blob2 = fresh.on_save()
    # Bracket the SECOND save with a snapshot too. A `loaded-only`
    # mutating on_save — one that serialises the right bytes for blob2
    # then corrupts `fresh` after building them — would otherwise slip
    # through every existing check: blob1 == blob2 holds, loaded_snap ==
    # pre_snap held *before* this mutation. The corruption only shows
    # up at the next post-snapshot tick on the loaded instance, which
    # is exactly the live regression we cannot afford. Codex round 7
    # pinned this last gap.
    if snapshot is not None:
        post_load_save_snap = snapshot(fresh)
        if post_load_save_snap != loaded_snap:
            msg = (
                f'on_save mutated the freshly-loaded strategy across the '
                f'second save call. loaded={loaded_snap} '
                f'post_second_save={post_load_save_snap}. A loaded-only '
                f'mutating on_save corrupts state for every tick after the '
                f'first cross-snapshot boundary, even when blob bytes match.'
            )
            raise AssertionError(msg)
    return blob1, blob2


def test_on_save_purity_zero_trade() -> None:
    """ZeroTradeStrategy: on_save returns empty bytes; round-trip identity.

    ZeroTrade has no per-trade state machine, but the purity gate
    still applies: a regression that creates `_save_calls` or any
    new attribute inside `on_save` would produce a real cross-snapshot
    bug even on a no-state strategy. Codex round 8 pinned this gap;
    the gate now snapshots full `vars(strategy)`.
    """

    def _build() -> ZeroTradeStrategy:
        return ZeroTradeStrategy('zero-trade-fixture')

    blob1, blob2 = _round_trip_bytes(
        _build, _ZERO_TRADE_SIGNALS, snapshot=_full_snapshot,
    )
    assert blob1 == blob2 == b'', (
        f'ZeroTradeStrategy on_save round-trip differs OR is non-empty; '
        f'a strategy that never trades should never carry state. '
        f'blob1={blob1!r} blob2={blob2!r}'
    )


def test_on_save_purity_buy_and_hold() -> None:
    """BuyAndHoldStrategy: long state round-trips bytes-identical."""

    def _build() -> BuyAndHoldStrategy:
        return BuyAndHoldStrategy(
            'buy-hold-fixture',
            symbol='BTCUSDT',
            capital=Decimal('100000'),
            kelly_pct=Decimal('1'),
            estimated_price=Decimal('70000'),
            stop_bps=Decimal('50'),
        )

    def _precondition(strategy: BuyAndHoldStrategy) -> None:
        # Pre-save state must be non-trivial (entered long with
        # non-zero qty). Runs BEFORE the first `on_save` so a
        # mutating implementation can't make a default-state
        # fixture look real.
        assert strategy._long is True, (
            f'BuyAndHoldStrategy did not enter long across '
            f'{_BUY_HOLD_SIGNALS}; pre-save state is trivial.'
        )
        assert strategy._entry_qty != Decimal('0'), (
            'BuyAndHoldStrategy entered long but `_entry_qty == 0`; '
            'sizing logic regressed.'
        )

    # `_full_snapshot` covers EVERY attribute on `vars(strategy)` —
    # runtime state (`_long`, `_entry_qty`) AND constructor config
    # (`_symbol`, `_capital`, `_kelly_pct`, `_estimated_price`,
    # `_stop_bps`) AND any future attribute a regression introduces
    # (e.g. `_save_calls`). Codex round 8 promoted snapshots from
    # explicit allow-lists to a full `vars()` dump because allow-lists
    # cannot catch attribute *creation* — only mutation of the
    # listed keys.
    blob1, blob2 = _round_trip_bytes(
        _build, _BUY_HOLD_SIGNALS, _precondition,
        snapshot=_full_snapshot,
    )
    assert blob1 == blob2, (
        f'BuyAndHoldStrategy on_save mutates state. blob1={blob1!r} '
        f'blob2={blob2!r}'
    )


def test_on_save_purity_over_trading() -> None:
    """OverTradingStrategy: flip-flop state round-trips bytes-identical."""

    def _build() -> OverTradingStrategy:
        return OverTradingStrategy(
            'over-trading-fixture',
            capital=Decimal('100000'),
            kelly_pct=Decimal('1'),
            estimated_price=Decimal('70000'),
            stop_bps=Decimal('50'),
        )

    def _precondition(strategy: OverTradingStrategy) -> None:
        # The 3-signal pattern (BUY → SELL → BUY) leaves the
        # strategy long. Pre-save check.
        assert strategy._long is True, (
            'OverTradingStrategy did not end long after BUY/SELL/BUY '
            'signal sequence; flip-flop state machine regressed.'
        )
        assert strategy._entry_qty != Decimal('0'), (
            'OverTradingStrategy long but `_entry_qty == 0`.'
        )

    blob1, blob2 = _round_trip_bytes(
        _build, _OVER_TRADING_SIGNALS, _precondition,
        snapshot=_full_snapshot,
    )
    assert blob1 == blob2, (
        f'OverTradingStrategy on_save mutates state. blob1={blob1!r} '
        f'blob2={blob2!r}'
    )


def test_on_save_purity_random_timing() -> None:
    """RandomTimingStrategy: RNG counter + flip state round-trips bytes-identical.

    Critical for this strategy specifically: `_consumed_calls` is the
    counter that lets `on_load` reseed and re-advance the RNG to the
    saved stream position. If `on_save` increments the counter (or
    seeds in a way that includes a non-deterministic value like a
    timestamp), the second save would diverge from the first.
    """

    def _build() -> RandomTimingStrategy:
        return RandomTimingStrategy(
            'random-timing-fixture',
            seed=42,
            capital=Decimal('100000'),
            kelly_pct=Decimal('1'),
            estimated_price=Decimal('70000'),
            stop_bps=Decimal('50'),
        )

    def _precondition(strategy: RandomTimingStrategy) -> None:
        # Every on_signal must advance `_consumed_calls`, whether
        # or not it emits an action. A regression where the counter
        # is bumped only on flips would silently round-trip to a
        # different stream position. Pre-save check.
        assert strategy._consumed_calls == len(_RANDOM_TIMING_SIGNALS), (
            f'RandomTimingStrategy `_consumed_calls`='
            f'{strategy._consumed_calls} != {len(_RANDOM_TIMING_SIGNALS)} '
            f'signals; the RNG counter is not tracking every on_signal call.'
        )

    # `_full_snapshot` normalises `_rng` to its `getstate()` tuple
    # so the RNG's full internal stream position participates in the
    # comparison alongside every other attribute (state + config +
    # any new attribute a regression introduces).
    blob1, blob2 = _round_trip_bytes(
        _build, _RANDOM_TIMING_SIGNALS, _precondition,
        snapshot=_full_snapshot,
    )
    assert blob1 == blob2, (
        f'RandomTimingStrategy on_save mutates state — likely the '
        f'`_consumed_calls` counter or the RNG itself. blob1={blob1!r} '
        f'blob2={blob2!r}'
    )


def test_on_save_purity_inverse_prescient() -> None:
    """InversePrescientStrategy: long state round-trips bytes-identical."""

    def _build() -> InversePrescientStrategy:
        return InversePrescientStrategy(
            'inverse-prescient-fixture',
            capital=Decimal('100000'),
            kelly_pct=Decimal('1'),
            estimated_price=Decimal('70000'),
            stop_bps=Decimal('50'),
        )

    def _precondition(strategy: InversePrescientStrategy) -> None:
        # 4-signal pattern (labels 0,0,1,0 → inverse 1,1,0,1):
        # BUY (1), already-long (1), SELL close (0), BUY again (1).
        # End state is long with non-zero qty. Pre-save check.
        assert strategy._long is True, (
            'InversePrescientStrategy did not end long after the '
            '4-signal fixture sequence; state machine regressed.'
        )
        assert strategy._entry_qty != Decimal('0'), (
            'InversePrescientStrategy long but `_entry_qty == 0`.'
        )

    blob1, blob2 = _round_trip_bytes(
        _build, _INVERSE_PRESCIENT_SIGNALS, _precondition,
        snapshot=_full_snapshot,
    )
    assert blob1 == blob2, (
        f'InversePrescientStrategy on_save mutates state. blob1={blob1!r} '
        f'blob2={blob2!r}'
    )


def test_on_save_purity_random_timing_load_advances_rng_correctly() -> None:
    """RandomTimingStrategy: post-load RNG produces the same next sample.

    `on_load` must restore the RNG to the same stream position the
    saved blob recorded, otherwise after-load behavior diverges from
    a no-snapshot continuation. This is stronger than save→load→save
    bytes-identity: it pins that the loaded RNG actually rolls the
    same next number as the un-snapshotted instance.

    Seed and signal count chosen so the next-roll-after-5-driven and
    the unadvanced-first-roll straddle the 0.5 flip threshold:

      seed=0 rolls = [0.844, 0.758, 0.421, 0.259, 0.511, 0.405, ...]
      after 5 driven signals the 6th roll is 0.405 (< 0.5 → flip)
      a broken `on_load` that does not advance the RNG would re-roll
      the 1st value 0.844 (>= 0.5 → no flip), producing zero actions
      where the correct path produces one — the test's len(actions)
      check would catch the regression.
    """
    seed = 0  # see docstring above
    drive_signals = _RANDOM_TIMING_SIGNALS[:5]  # five on_signal calls

    def _build() -> RandomTimingStrategy:
        return RandomTimingStrategy(
            'rt-rng-fixture',
            seed=seed,
            capital=Decimal('100000'),
            kelly_pct=Decimal('1'),
            estimated_price=Decimal('70000'),
            stop_bps=Decimal('50'),
        )

    # Path A: build, drive 5 signals, drive a 6th, capture qty/long.
    a = _build()
    _drive_through_signals(a, drive_signals)
    actions_a = a.on_signal(_RANDOM_TIMING_SIGNALS[5], StrategyParams(raw={}), _ctx())
    # Path A must produce a flip on the 6th roll (else the seed/count
    # choice no longer differentiates broken-vs-correct on_load).
    assert len(actions_a) == 1, (
        f'seed/count choice broken: 6th roll for seed={seed} after 5 '
        f'driven signals must produce 1 action; got {len(actions_a)}. '
        f'Update _RANDOM_TIMING_SIGNALS / seed / drive_signals slice '
        f'so the comparison straddles the 0.5 flip threshold.'
    )

    # Path B: build, drive 5 signals, save, fresh, load, drive a 6th.
    b = _build()
    _drive_through_signals(b, drive_signals)
    blob = b.on_save()
    b_fresh = _build()
    b_fresh.on_load(blob)
    actions_b = b_fresh.on_signal(
        _RANDOM_TIMING_SIGNALS[5], StrategyParams(raw={}), _ctx(),
    )

    assert len(actions_a) == len(actions_b), (
        f'load-advances-RNG-correctly: post-load action count differs '
        f'from no-snapshot path. a={len(actions_a)} b={len(actions_b)}'
    )
    assert actions_a[0].direction == actions_b[0].direction, (
        f'post-load action direction differs: '
        f'a={actions_a[0].direction.name} b={actions_b[0].direction.name}'
    )
    assert actions_a[0].size == actions_b[0].size, (
        f'post-load action size differs: '
        f'a={actions_a[0].size} b={actions_b[0].size}'
    )


def test_on_save_purity_long_on_signal_template(tmp_path: object) -> None:
    """`long_on_signal` template strategy: same save→load→save bytes contract.

    SPEC §9.5 requires the gate "for every strategy registered in the
    manifest". `long_on_signal.py` is the manifest-rendered strategy
    consumed by every production sweep run, so it must satisfy the
    same contract as the in-tree fixtures. The strategy file is
    rendered by `ManifestBuilder` with a `__BTS_PARAMS__` sentinel
    replaced by a JSON config; this test mirrors Nexus's loader path:
    render the templated source to a temp `.py`, then load via
    `importlib.util.spec_from_file_location` /
    `module_from_spec` / `spec.loader.exec_module(...)` — the same
    sequence Nexus's strategy loader uses, with no `sys.modules`
    registration that could mask loader-only failures.
    """
    import importlib.util
    import json
    from pathlib import Path

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
            'kelly_pct': '1',
            'estimated_price': '70000',
            'stop_bps': '50',
        }),
    )
    rendered_path = Path(str(tmp_path)) / 'long_on_signal.py'
    rendered_path.write_text(rendered, encoding='utf-8')

    spec = importlib.util.spec_from_file_location(
        'long_on_signal_fixture', rendered_path,
    )
    assert spec is not None and spec.loader is not None, (
        f'spec_from_file_location returned None for {rendered_path}; '
        f'cannot mirror Nexus loader path.'
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def _build() -> object:
        strategy = mod.Strategy('long-on-signal-fixture')
        strategy.on_startup(StrategyParams(raw={}), _ctx())
        return strategy

    # Drive a single `_preds=1` signal so an ENTER BUY action is
    # emitted, then synthesize the matching FILLED outcome so the
    # pre-save state carries `_long=True` and a non-zero
    # `_entry_qty`. The previous version flipped `_long` inside
    # `on_signal` directly; with the maker-fill wiring the
    # strategy now reconciles state from `on_outcome` (BUY LIMITs
    # may zero-fill — flipping at emit would lie about
    # inventory). To keep this round-trip pin testing meaningful
    # state (not a trivial flat snapshot), emit + outcome the
    # entry through the live strategy contract.
    from nexus.infrastructure.praxis_connector.trade_outcome import (
        TradeOutcome,
    )
    from nexus.infrastructure.praxis_connector.trade_outcome_type import (
        TradeOutcomeType,
    )
    enter_signal = _signal_at(_T0, _preds=1)
    strategy = _build()
    actions = strategy.on_signal(
        enter_signal, StrategyParams(raw={}), _ctx(),
    )
    assert len(actions) == 1, (
        f'expected exactly one BUY action from the entry signal, got '
        f'{len(actions)}'
    )
    fill_qty = actions[0].size
    outcome = TradeOutcome(
        outcome_id='OUT-FIXTURE-001',
        command_id='CMD-FIXTURE-001',
        outcome_type=TradeOutcomeType.FILLED,
        timestamp=_T0,
        fill_size=fill_qty,
        fill_price=Decimal('70000'),
        fill_notional=fill_qty * Decimal('70000'),
        actual_fees=Decimal('0.7'),
    )
    strategy.on_outcome(outcome, StrategyParams(raw={}), _ctx())
    assert strategy._long is True, (
        '`long_on_signal` did not enter long on `_preds=1` from flat '
        'after FILLED outcome; the fixture pre-save state is trivial '
        'and the round-trip cannot pin save→load→save honesty.'
    )
    assert strategy._entry_qty != Decimal('0'), (
        '`long_on_signal` entered long but `_entry_qty == 0`; sizing '
        'logic regressed.'
    )

    # `_full_snapshot` walks `vars(strategy)` so it captures
    # EVERY attribute, including the runtime `_long`/`_entry_qty`
    # and the `_config` plain class (which `_full_snapshot`
    # recurses into via `vars()`). A regression that introduces
    # `_save_calls`, mutates `_config.capital`, or any other
    # attribute is caught without further enumeration. Inline
    # bracketing (rather than `_round_trip_bytes`) is required
    # because the strategy is loaded via `importlib`, not the
    # in-tree class.
    pre = _full_snapshot(strategy)
    blob1 = strategy.on_save()
    post = _full_snapshot(strategy)
    assert pre == post, (
        f'`long_on_signal` on_save mutated live state. pre={pre} post={post}'
    )
    fresh = _build()
    fresh.on_load(blob1)
    loaded = _full_snapshot(fresh)
    assert loaded == pre, (
        f'`long_on_signal` on_load did not restore the pre-save snapshot. '
        f'pre={pre} loaded={loaded}. on_save serialised a state different '
        f'from the live one, OR on_load failed to reconstruct it.'
    )
    blob2 = fresh.on_save()
    # Bracket the second save: catches a loaded-only mutating on_save
    # that produces correct bytes then corrupts `fresh`. Codex round 7.
    post_load_save = _full_snapshot(fresh)
    assert post_load_save == loaded, (
        f'`long_on_signal` on_save mutated the freshly-loaded strategy '
        f'across the second save. loaded={loaded} '
        f'post_second_save={post_load_save}.'
    )
    assert blob1 == blob2, (
        f'`long_on_signal` template strategy on_save mutates state. '
        f'blob1={blob1!r} blob2={blob2!r}'
    )
