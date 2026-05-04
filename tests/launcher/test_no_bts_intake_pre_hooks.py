"""bts must NOT add INTAKE pre-hooks on top of Nexus's pipeline.

The action_submitter previously layered two BTS-side intake checks
(`_check_declared_stop`, `_check_atr_sanity`) before
`validation_pipeline.validate`. Both produced PENDING phantoms in
the EventSpine: the strategy emitted an `OrderSubmitIntent`, the
hook denied the action silently, and `TradeOutcomeProduced` settled
to status=PENDING with no `OrderSubmitted` / `OrderRejected` /
`OrderExpired` ever flowing through. The result: bts diverged from
paper/live execution and silently dropped strategy decisions.

Five Principles, third bullet: "not building a parallel universe
around praxis/nexus -> extending them." Strategy invariants live
in the Nexus strategy file; bts replays whatever the strategy
decides. This test pins both the absence of the two hooks and the
absence of their support module.
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

_ACTION_SUBMITTER_PATH = (
    Path(__file__).resolve().parents[2]
    / 'backtest_simulator' / 'launcher' / 'action_submitter.py'
)
_LAUNCHER_PATH = (
    Path(__file__).resolve().parents[2]
    / 'backtest_simulator' / 'launcher' / 'launcher.py'
)
_ATR_PATH = (
    Path(__file__).resolve().parents[2]
    / 'backtest_simulator' / 'honesty' / 'atr.py'
)


def test_action_submitter_has_no_intake_pre_hook_functions() -> None:
    """No `_check_declared_stop` or `_check_atr_sanity` defined or referenced."""
    src = _ACTION_SUBMITTER_PATH.read_text(encoding='utf-8')
    tree = ast.parse(src)
    forbidden = {'_check_declared_stop', '_check_atr_sanity', '_resolve_atr_entry_price'}
    defined = {
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    leaked = forbidden & defined
    assert not leaked, (
        f'action_submitter.py defines bts-side INTAKE pre-hooks '
        f'{sorted(leaked)}; these silently drop strategy actions '
        f'and produce PENDING phantoms. Nexus pipeline INTAKE is '
        f'the only authority — the strategy file enforces strategy '
        f'invariants, not bts.'
    )
    referenced = {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }
    leaked_refs = forbidden & referenced
    assert not leaked_refs, (
        f'action_submitter.py references {sorted(leaked_refs)}; '
        f'remove the call sites along with the function bodies.'
    )


def test_atr_module_is_gone() -> None:
    """`backtest_simulator.honesty.atr` must not exist."""
    assert not _ATR_PATH.exists(), (
        f'{_ATR_PATH} still exists; the ATR sanity gate was a '
        f'parallel-universe gate. The strategy file enforces R-floor; '
        f'bts does not second-guess.'
    )
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module('backtest_simulator.honesty.atr')


def test_launcher_has_no_atr_counter_state() -> None:
    """`BacktestLauncher` carries no `_n_atr_rejected` / `_record_atr_rejection`."""
    src = _LAUNCHER_PATH.read_text(encoding='utf-8')
    forbidden_substrings = (
        '_n_atr_rejected',
        '_n_atr_uncalibrated',
        '_record_atr_rejection',
        'AtrSanityGate',
        'atr_gate',
        'atr_provider',
    )
    leaked = [s for s in forbidden_substrings if s in src]
    assert not leaked, (
        f'launcher.py still references {leaked}; the ATR plumbing '
        f'must be removed in lockstep with the action_submitter '
        f'pre-hooks.'
    )


def test_submitter_bindings_has_no_atr_fields() -> None:
    """`SubmitterBindings` has no `atr_gate` / `atr_provider`."""
    from backtest_simulator.launcher.action_submitter import SubmitterBindings
    field_names = {f.name for f in SubmitterBindings.__dataclass_fields__.values()}
    leaked = field_names & {'atr_gate', 'atr_provider'}
    assert not leaked, (
        f'SubmitterBindings carries {sorted(leaked)}; the ATR gate '
        f'wiring must be removed.'
    )


def test_build_action_submitter_has_no_on_atr_reject_param() -> None:
    """`build_action_submitter` accepts no `on_atr_reject` keyword."""
    import inspect

    from backtest_simulator.launcher.action_submitter import (
        build_action_submitter,
    )
    sig = inspect.signature(build_action_submitter)
    assert 'on_atr_reject' not in sig.parameters, (
        'build_action_submitter still accepts on_atr_reject; the '
        'ATR rejection hook must be removed.'
    )
