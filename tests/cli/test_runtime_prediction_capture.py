"""Mutation-proof tests for `capture_runtime_prediction`.

Codex (post-auditor-3 P1): real Nexus `Signal.values` is a
`types.MappingProxyType` — a read-only dict view, NOT a `dict`.
The earlier capture predicate's `isinstance(..., dict)` check
silently dropped every real-runtime prediction, so sweep parity
reported "no comparisons made" forever. The fix accepts
`collections.abc.Mapping` (which both `dict` and
`MappingProxyType` satisfy). These tests pin the contract using
the EXACT type Nexus emits.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType

from backtest_simulator.cli._run_window import capture_runtime_prediction


@dataclass(frozen=True)
class _FakeWired:
    """Minimal stub matching Nexus's WiredSensor for our access pattern."""

    sensor_id: str


@dataclass(frozen=True)
class _FakeSignal:
    """Minimal stub matching Nexus's Signal for our access pattern.

    `values` is a `MappingProxyType` — same as real Nexus emits.
    """

    values: MappingProxyType[str, object]
    timestamp: datetime


def test_capture_runtime_prediction_accepts_mapping_proxy() -> None:
    """Nexus's `Signal.values` is a MappingProxyType; capture must work.

    Mutation proof: replacing `isinstance(values_obj, Mapping)`
    with `isinstance(values_obj, dict)` makes this test fail
    because MappingProxyType IS NOT a dict subclass. The earlier
    auditor-3 round had this exact bug — sweep parity reported
    "no comparisons made" in production despite the wrapper
    being installed correctly.
    """
    wired = _FakeWired(sensor_id='perm_3')
    ts = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    # MappingProxyType only — NOT a dict subclass.
    proxy = MappingProxyType({'_preds': 1, '_probs': 0.7})
    signal = _FakeSignal(values=proxy, timestamp=ts)
    sink: list[dict[str, object]] = []

    capture_runtime_prediction(wired=wired, signal=signal, sink=sink)

    assert len(sink) == 1, (
        f'MappingProxyType `signal.values` must be accepted; got '
        f'sink={sink!r}. Reverting `isinstance(..., Mapping)` to '
        f'`isinstance(..., dict)` makes this fire.'
    )
    entry = sink[0]
    assert entry == {
        'sensor_id': 'perm_3',
        'timestamp': ts.isoformat(),
        'pred': 1,
    }


def test_capture_runtime_prediction_accepts_plain_dict() -> None:
    """Plain dict also works (forward-compat / test fixtures)."""
    wired = _FakeWired(sensor_id='perm_5')
    ts = datetime(2026, 4, 20, 12, 30, tzinfo=UTC)
    signal = _FakeSignal(
        values=MappingProxyType({'_preds': 0}),
        timestamp=ts,
    )
    sink: list[dict[str, object]] = []
    capture_runtime_prediction(wired=wired, signal=signal, sink=sink)
    assert len(sink) == 1
    assert sink[0]['pred'] == 0


def test_capture_runtime_prediction_skips_when_preds_missing() -> None:
    """`_preds` key absent → silent no-op (no exception)."""
    wired = _FakeWired(sensor_id='d1')
    ts = datetime(2026, 4, 20, tzinfo=UTC)
    signal = _FakeSignal(
        values=MappingProxyType({'_probs': 0.5}),
        timestamp=ts,
    )
    sink: list[dict[str, object]] = []
    capture_runtime_prediction(wired=wired, signal=signal, sink=sink)
    assert sink == []


def test_capture_runtime_prediction_skips_when_preds_not_int() -> None:
    """`_preds` is a float → silent no-op.

    Nexus's `_extract_values` converts numpy scalars to int when
    they're integer-shaped, but the predicate must defend against
    edge cases (e.g. a float-typed pred from a regression head).
    """
    wired = _FakeWired(sensor_id='d1')
    ts = datetime(2026, 4, 20, tzinfo=UTC)
    signal = _FakeSignal(
        values=MappingProxyType({'_preds': 0.5}),
        timestamp=ts,
    )
    sink: list[dict[str, object]] = []
    capture_runtime_prediction(wired=wired, signal=signal, sink=sink)
    assert sink == []


def test_capture_runtime_prediction_skips_when_signal_lacks_values() -> None:
    """Signal without `.values` attribute → silent no-op."""
    @dataclass(frozen=True)
    class _MinimalSignal:
        timestamp: datetime
    wired = _FakeWired(sensor_id='d1')
    signal = _MinimalSignal(timestamp=datetime(2026, 4, 20, tzinfo=UTC))
    sink: list[dict[str, object]] = []
    capture_runtime_prediction(wired=wired, signal=signal, sink=sink)
    assert sink == []


def test_capture_runtime_prediction_skips_when_timestamp_missing() -> None:
    """Signal with `_preds` but no `timestamp` → silent no-op.

    Without a timestamp, the parity check has nothing to look up
    against. Silent no-op is correct.
    """
    @dataclass(frozen=True)
    class _NoTimestamp:
        values: MappingProxyType[str, object]
    wired = _FakeWired(sensor_id='d1')
    signal = _NoTimestamp(values=MappingProxyType({'_preds': 1}))
    sink: list[dict[str, object]] = []
    capture_runtime_prediction(wired=wired, signal=signal, sink=sink)
    assert sink == []
