"""Slice 0 (#64) tests for `backtest_simulator.launcher.replay_clock`.

Every MVC predicate in the slice spec is encoded here as a `test_*`,
plus the behavioural pins that prove the runtime contract.

Suites in this file map to the MVC sections A-H in the issue body:

  - `TestModuleStructure`        → A
  - `TestSurfacesRemoved`        → B
  - `TestLauncherUsesReplayClock`→ C
  - `TestTimerLoopFailLoud`      → D (E is also covered here)
  - `TestComputeKlineBoundaries` → F
  - `TestBookkeeping`            → G
  - `TestDriveWindow`            → H (behavioural)
  - `TestModuleImportClean`      → H (clean-subprocess import)
"""
from __future__ import annotations

import ast
import json
import pathlib
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

from backtest_simulator.launcher.replay_clock import (
    ReplayClock,
    compute_kline_boundaries,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_LAUNCHER_PY = _REPO_ROOT / 'backtest_simulator' / 'launcher' / 'launcher.py'
_CLOCK_PY = _REPO_ROOT / 'backtest_simulator' / 'launcher' / 'clock.py'
_REPLAY_CLOCK_PY = _REPO_ROOT / 'backtest_simulator' / 'launcher' / 'replay_clock.py'
_PYPROJECT = _REPO_ROOT / 'pyproject.toml'
_CHANGELOG = _REPO_ROOT / 'CHANGELOG.md'
_MODULE_BUDGETS = _REPO_ROOT / '.github' / 'module_budgets.json'


def _parse(p: pathlib.Path) -> ast.Module:
    return ast.parse(p.read_text())


# ---------- Section A: module structure ----------


class TestModuleStructure:

    def test_module_defines_compute_kline_boundaries_and_replay_clock(self) -> None:
        tree = _parse(_REPLAY_CLOCK_PY)
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.ClassDef):
                names.add(node.name)
        assert 'compute_kline_boundaries' in names
        assert 'ReplayClock' in names

    def test_compute_kline_boundaries_keyword_only_parameters(self) -> None:
        tree = _parse(_REPLAY_CLOCK_PY)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == 'compute_kline_boundaries'
        )
        positional = [a.arg for a in fn.args.args]
        kwonly = [a.arg for a in fn.args.kwonlyargs]
        assert positional == []
        assert kwonly == ['window_start', 'window_end', 'interval_seconds']

    def test_drive_window_keyword_only_parameters_in_canonical_order(self) -> None:
        tree = _parse(_REPLAY_CLOCK_PY)
        cls = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef) and n.name == 'ReplayClock'
        )
        method = next(
            n for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == 'drive_window'
        )
        kwonly = [a.arg for a in method.args.kwonlyargs]
        assert kwonly == [
            'window_start',
            'window_end',
            'wired_sensors',
            'predict_loop',
            'outcome_loop',
            'drain_pending_submits',
            'freezer',
        ]


# ---------- Section B: old surfaces removed ----------


class TestSurfacesRemoved:

    def test_clock_no_longer_defines_frozen_aware_timer_run(self) -> None:
        tree = _parse(_CLOCK_PY)
        defined = any(
            isinstance(n, ast.FunctionDef) and n.name == '_frozen_aware_timer_run'
            for n in ast.walk(tree)
        )
        assert not defined

    def test_clock_no_longer_patches_threading_timer(self) -> None:
        tree = _parse(_CLOCK_PY)
        patches: list[ast.AST] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == 'setattr'
                and any(
                    isinstance(a, ast.Attribute) and a.attr == 'Timer'
                    for a in node.args
                )
            ):
                patches.append(node)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and target.attr == 'run'
                        and isinstance(target.value, ast.Attribute)
                        and target.value.attr == 'Timer'
                    ):
                        patches.append(node)
                        break
        assert patches == []

    def test_launcher_no_longer_defines_advance_clock_until(self) -> None:
        tree = _parse(_LAUNCHER_PY)
        defined = any(
            isinstance(n, ast.FunctionDef) and n.name == '_advance_clock_until'
            for n in ast.walk(tree)
        )
        assert not defined


# ---------- Section C: launcher uses ReplayClock ----------


class TestLauncherUsesReplayClock:

    def test_launcher_imports_replay_clock(self) -> None:
        tree = _parse(_LAUNCHER_PY)
        imports = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom)
            and n.module == 'backtest_simulator.launcher.replay_clock'
        ]
        assert len(imports) >= 1

    def test_run_window_calls_drive_window_with_required_kwargs(self) -> None:
        tree = _parse(_LAUNCHER_PY)
        cls = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef) and n.name == 'BacktestLauncher'
        )
        run_window = next(
            n for n in ast.walk(cls)
            if isinstance(n, ast.FunctionDef) and n.name == 'run_window'
        )
        drive_calls = [
            n for n in ast.walk(run_window)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == 'drive_window'
        ]
        assert len(drive_calls) >= 1
        kwargs = {k.arg for k in drive_calls[0].keywords}
        required = {
            'window_start',
            'window_end',
            'wired_sensors',
            'predict_loop',
            'outcome_loop',
            'drain_pending_submits',
            'freezer',
        }
        assert kwargs >= required, (
            f'drive_window kwargs missing: {required - kwargs}'
        )

    def test_launcher_does_not_start_predict_or_outcome_loop(self) -> None:
        tree = _parse(_LAUNCHER_PY)
        starts = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == 'start'
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id in {'predict_loop', 'outcome_loop'}
        ]
        assert starts == []

    def test_launcher_constructs_praxis_inbound_with_zero_poll_timeout(self) -> None:
        tree = _parse(_LAUNCHER_PY)
        kw_values: list[ast.AST] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == 'PraxisInbound'
            ):
                for k in node.keywords:
                    if k.arg == 'poll_timeout':
                        kw_values.append(k.value)
        assert len(kw_values) == 1, (
            f'expected exactly one PraxisInbound(... poll_timeout=...) call, '
            f'found {len(kw_values)}'
        )
        only = kw_values[0]
        assert isinstance(only, ast.Constant)
        assert only.value == 0.0


# ---------- Section D: TimerLoop fail-loud ----------


class TestTimerLoopFailLoud:

    def test_launcher_does_not_construct_timer_loop(self) -> None:
        tree = _parse(_LAUNCHER_PY)
        calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == 'TimerLoop'
        ]
        assert calls == []

    def test_launcher_raises_runtime_error_on_non_empty_timer_specs(self) -> None:
        tree = _parse(_LAUNCHER_PY)
        # AST walk descends into f-string `JoinedStr` fragments so the
        # check tolerates `raise RuntimeError(f'... timer_specs ...')`
        # without requiring a flat `Constant` arg.
        raises = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Raise)
            and isinstance(n.exc, ast.Call)
            and isinstance(n.exc.func, ast.Name)
            and n.exc.func.id == 'RuntimeError'
            and any(
                isinstance(c, ast.Constant)
                and isinstance(c.value, str)
                and 'timer_specs' in c.value
                for c in ast.walk(n.exc)
            )
        ]
        assert len(raises) >= 1


# ---------- Section F: compute_kline_boundaries behaviour ----------


class TestComputeKlineBoundaries:

    def test_one_day_4h_returns_five_boundaries(self) -> None:
        boundaries = compute_kline_boundaries(
            window_start=datetime(2026, 4, 4, 0, 0, tzinfo=UTC),
            window_end=datetime(2026, 4, 4, 23, 59, tzinfo=UTC),
            interval_seconds=14400,
        )
        assert [t.isoformat() for t in boundaries] == [
            '2026-04-04T04:00:00+00:00',
            '2026-04-04T08:00:00+00:00',
            '2026-04-04T12:00:00+00:00',
            '2026-04-04T16:00:00+00:00',
            '2026-04-04T20:00:00+00:00',
        ]

    def test_half_open_interval_at_start(self) -> None:
        # `window_start` lies exactly on an epoch-aligned 4h boundary
        # (00:00 UTC). The first returned tick must be at +interval, not
        # at window_start itself.
        boundaries = compute_kline_boundaries(
            window_start=datetime(2026, 4, 4, 0, 0, tzinfo=UTC),
            window_end=datetime(2026, 4, 4, 8, 0, tzinfo=UTC),
            interval_seconds=14400,
        )
        assert boundaries[0] == datetime(2026, 4, 4, 4, 0, tzinfo=UTC)

    def test_inclusive_at_end(self) -> None:
        # `window_end` exactly on a boundary IS included.
        boundaries = compute_kline_boundaries(
            window_start=datetime(2026, 4, 4, 0, 0, tzinfo=UTC),
            window_end=datetime(2026, 4, 4, 8, 0, tzinfo=UTC),
            interval_seconds=14400,
        )
        assert boundaries[-1] == datetime(2026, 4, 4, 8, 0, tzinfo=UTC)

    def test_negative_interval_raises(self) -> None:
        with pytest.raises(ValueError, match='interval_seconds must be positive'):
            compute_kline_boundaries(
                window_start=datetime(2026, 4, 4, 0, 0, tzinfo=UTC),
                window_end=datetime(2026, 4, 4, 8, 0, tzinfo=UTC),
                interval_seconds=0,
            )

    def test_naive_window_start_raises(self) -> None:
        with pytest.raises(ValueError, match='timezone-aware'):
            compute_kline_boundaries(
                window_start=datetime(2026, 4, 4, 0, 0),
                window_end=datetime(2026, 4, 4, 8, 0, tzinfo=UTC),
                interval_seconds=14400,
            )

    def test_window_end_before_start_raises(self) -> None:
        with pytest.raises(ValueError, match='must be >='):
            compute_kline_boundaries(
                window_start=datetime(2026, 4, 4, 8, 0, tzinfo=UTC),
                window_end=datetime(2026, 4, 4, 0, 0, tzinfo=UTC),
                interval_seconds=14400,
            )


# ---------- Section G: bookkeeping ----------


class TestBookkeeping:

    def test_module_budgets_includes_replay_clock(self) -> None:
        budgets = json.loads(_MODULE_BUDGETS.read_text())
        assert isinstance(
            budgets.get('backtest_simulator/launcher/replay_clock.py'), int,
        )

    def test_changelog_first_version_header_is_2_5_0(self) -> None:
        for line in _CHANGELOG.read_text().splitlines():
            if line.strip():
                assert re.match(r'#\s+v2\.5\.0\b', line) is not None
                return
        msg = 'CHANGELOG.md is empty; expected `# v2.5.0` header'
        raise AssertionError(msg)

    def test_pyproject_version_is_2_5_0(self) -> None:
        data = tomllib.loads(_PYPROJECT.read_text())
        assert data['project']['version'] == '2.5.0'


# ---------- Section H: behavioural ----------


@dataclass
class _MockWired:
    """Minimal stand-in for `nexus.startup.sequencer.WiredSensor`."""
    sensor_id: str
    interval_seconds: int


class _MockPredictLoop:
    """Pre-started Nexus PredictLoop stand-in.

    Records every `tick_once(wired)` invocation for parity assertions.
    """

    def __init__(self) -> None:
        self.running: bool = False
        self.calls: list[tuple[str, datetime]] = []

    def tick_once(self, wired: object) -> None:
        from datetime import datetime as _dt
        self.calls.append((
            getattr(wired, 'sensor_id', repr(wired)),
            _dt.now(UTC),
        ))


class _MockOutcomeLoop:
    """Pre-started Nexus OutcomeLoop stand-in.

    Each `tick_once()` consumes one queued outcome and (optionally) emits
    a follow-on action by appending to a tracked list. Returns False when
    the queue is empty.
    """

    def __init__(self, outcomes: list[str] | None = None) -> None:
        self.running: bool = False
        self._outcomes: list[str] = list(outcomes or [])
        self.consumed: list[str] = []
        self.tick_count: int = 0

    def queue(self, item: str) -> None:
        self._outcomes.append(item)

    def tick_once(self) -> bool:
        self.tick_count += 1
        if not self._outcomes:
            return False
        self.consumed.append(self._outcomes.pop(0))
        return True


class _MockFreezer:
    """Records every move_to(...) call for monotonic-order assertions."""

    def __init__(self) -> None:
        self.moves: list[datetime] = []

    def move_to(self, target_datetime: datetime) -> None:
        self.moves.append(target_datetime)


def _drain() -> Callable[[], None]:
    drained = {'count': 0}

    def _do_drain() -> None:
        drained['count'] += 1

    setattr(_do_drain, 'count_ref', drained)
    return _do_drain


class TestDriveWindow:

    def _build_args(
        self,
        *,
        wired_sensors: Sequence[_MockWired] | None = None,
        outcomes_per_tick: list[str] | None = None,
    ) -> dict[str, Any]:
        if wired_sensors is None:
            wired_sensors = [_MockWired(sensor_id='exp:1', interval_seconds=14400)]
        return {
            'window_start': datetime(2026, 4, 4, 0, 0, tzinfo=UTC),
            'window_end': datetime(2026, 4, 4, 23, 59, tzinfo=UTC),
            'wired_sensors': wired_sensors,
            'predict_loop': _MockPredictLoop(),
            'outcome_loop': _MockOutcomeLoop(outcomes=outcomes_per_tick),
            'drain_pending_submits': _drain(),
            'freezer': _MockFreezer(),
        }

    def test_calls_tick_once_per_boundary_per_sensor(self) -> None:
        wired_sensors = [
            _MockWired(sensor_id='s1', interval_seconds=14400),
            _MockWired(sensor_id='s2', interval_seconds=14400),
        ]
        args = self._build_args(wired_sensors=wired_sensors)
        ReplayClock().drive_window(**args)
        # 5 boundaries on a 4h kline-day, 2 sensors → 10 tick_once calls.
        assert len(args['predict_loop'].calls) == 10
        # Order: boundary0(s1), boundary0(s2), boundary1(s1), ...
        sensor_order = [sid for sid, _ in args['predict_loop'].calls]
        assert sensor_order == ['s1', 's2'] * 5

    def test_advances_freezer_to_each_boundary_in_order(self) -> None:
        args = self._build_args()
        ReplayClock().drive_window(**args)
        moves = args['freezer'].moves
        # 5 kline boundaries + 1 final move_to(window_end).
        assert moves == [
            datetime(2026, 4, 4, 4, 0, tzinfo=UTC),
            datetime(2026, 4, 4, 8, 0, tzinfo=UTC),
            datetime(2026, 4, 4, 12, 0, tzinfo=UTC),
            datetime(2026, 4, 4, 16, 0, tzinfo=UTC),
            datetime(2026, 4, 4, 20, 0, tzinfo=UTC),
            datetime(2026, 4, 4, 23, 59, tzinfo=UTC),
        ]
        # Strict monotonic ascending.
        assert moves == sorted(moves)

    def test_drains_submits_and_outcomes_to_quiescence_repeat_until_fixed_point(self) -> None:
        # Pre-queue 3 outcomes that survive the first sensor tick on the
        # first boundary; the drain loop must consume all 3 across
        # successive passes within the same boundary's drain phase.
        args = self._build_args(outcomes_per_tick=['o1', 'o2', 'o3'])
        ReplayClock().drive_window(**args)
        # All 3 outcomes consumed.
        assert args['outcome_loop'].consumed == ['o1', 'o2', 'o3']
        # Drain ran at least once per boundary + final.
        drain_count = getattr(args['drain_pending_submits'], 'count_ref')['count']
        assert drain_count >= 6  # 5 boundaries + 1 final, each at least once

    def test_drain_waits_for_delayed_routed_outcome_before_advancing(self) -> None:
        # Simulate a router that enqueues an outcome only on the SECOND
        # call to drain_pending_submits (i.e. after the first
        # submit-drain returns, the asyncio router finally enqueues).
        # ReplayClock's repeat-until-fixed-point loop must call
        # drain_pending_submits AGAIN after consuming the late outcome.
        outcome_loop = _MockOutcomeLoop(outcomes=[])
        drain_calls: list[int] = []

        def delayed_router_drain() -> None:
            drain_calls.append(len(drain_calls))
            # On the second call (drain pass within the same boundary),
            # enqueue an outcome that the next outcome_loop.tick_once()
            # will consume.
            if len(drain_calls) == 2:
                outcome_loop.queue('delayed-outcome')

        args = {
            'window_start': datetime(2026, 4, 4, 0, 0, tzinfo=UTC),
            'window_end': datetime(2026, 4, 4, 4, 0, tzinfo=UTC),
            'wired_sensors': [_MockWired(sensor_id='s1', interval_seconds=14400)],
            'predict_loop': _MockPredictLoop(),
            'outcome_loop': outcome_loop,
            'drain_pending_submits': delayed_router_drain,
            'freezer': _MockFreezer(),
        }
        ReplayClock().drive_window(**args)
        # The delayed outcome must have been consumed (drive_window did
        # NOT advance past the boundary while the router still had the
        # outcome pending).
        assert outcome_loop.consumed == ['delayed-outcome']

    def test_fails_loud_on_heterogeneous_cadence(self) -> None:
        wired_sensors = [
            _MockWired(sensor_id='s1', interval_seconds=14400),
            _MockWired(sensor_id='s2', interval_seconds=3600),
        ]
        args = self._build_args(wired_sensors=wired_sensors)
        with pytest.raises(ValueError, match='heterogeneous'):
            ReplayClock().drive_window(**args)

    def test_fails_loud_when_predict_loop_running(self) -> None:
        args = self._build_args()
        args['predict_loop'].running = True
        with pytest.raises(RuntimeError, match=r'predict_loop\.running is True'):
            ReplayClock().drive_window(**args)

    def test_fails_loud_when_outcome_loop_running(self) -> None:
        args = self._build_args()
        args['outcome_loop'].running = True
        with pytest.raises(RuntimeError, match=r'outcome_loop\.running is True'):
            ReplayClock().drive_window(**args)

    def test_fails_loud_on_empty_wired_sensors(self) -> None:
        args = self._build_args(wired_sensors=[])
        with pytest.raises(ValueError, match='non-empty sequence'):
            ReplayClock().drive_window(**args)

    def test_is_deterministic_across_runs(self) -> None:
        # Two independent calls with identical inputs produce identical
        # recorded `tick_once` invocation sequences. The mocks are
        # state-free across runs (each call builds fresh).
        args_a = self._build_args()
        args_b = self._build_args()
        ReplayClock().drive_window(**args_a)
        ReplayClock().drive_window(**args_b)
        assert (
            [sid for sid, _ in args_a['predict_loop'].calls]
            == [sid for sid, _ in args_b['predict_loop'].calls]
        )
        assert args_a['freezer'].moves == args_b['freezer'].moves


# ---------- clean-subprocess import (Section H) ----------


class TestModuleImportClean:

    def test_import_replay_clock_from_clean_interpreter(self) -> None:
        # Importing replay_clock cold (in a fresh subprocess, no other
        # nexus modules pre-loaded) must not raise. The Nexus circular
        # import (nexus.startup.shutdown_sequencer ↔ nexus.strategy.predict_loop)
        # is sensitive to import order in the launcher path; replay_clock
        # itself only imports stdlib + freezegun + dataclasses, so it
        # must work standalone.
        result = subprocess.run(
            [
                sys.executable,
                '-c',
                'import backtest_simulator.launcher.replay_clock',
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, (
            f'clean import failed; stderr={result.stderr}'
        )
