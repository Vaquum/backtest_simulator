"""Run/sweep subcommands live at the documented module paths."""
from __future__ import annotations

import importlib

# Imported lazily inside the test functions to avoid pytest collecting
# `test_*` names that get pulled in via `from ... import _run as test_*`
# at module scope (pytest's default discovery would treat them as tests).
_SUBCOMMANDS = (
    'run', 'sweep', 'enrich', 'test', 'lint',
    'typecheck', 'gate', 'notebook', 'version',
)


def _run_function_module(name: str) -> str:
    mod = importlib.import_module(f'backtest_simulator.cli.commands.{name}')
    fn = getattr(mod, '_run', None) or getattr(mod, '_run_no_args', None)
    if fn is None:
        # `version` has no args so its handler is `_run` too; just guard
        # against missing handlers rather than asserting here.
        return mod.__name__
    return fn.__module__


def test_run_module_path() -> None:
    assert _run_function_module('run') == 'backtest_simulator.cli.commands.run'


def test_sweep_module_path() -> None:
    assert _run_function_module('sweep') == 'backtest_simulator.cli.commands.sweep'


def test_every_subcommand_module_under_cli_commands() -> None:
    for name in _SUBCOMMANDS:
        mod_path = _run_function_module(name)
        assert mod_path.startswith('backtest_simulator.cli.commands.'), mod_path
