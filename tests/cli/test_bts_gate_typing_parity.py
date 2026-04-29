"""`bts gate typing` must produce the same pass/fail state as CI.

Slice for issue #31. Two prior divergences:

1. CI plants PEP 561 `py.typed` markers in nexus / praxis / limen /
   clickhouse_connect via the `pr_checks_typing.yml` step "Plant
   py.typed markers in sibling libraries" before pyright runs.
   Without those markers locally, pyright emits ~80
   `reportMissingTypeStubs` + cascading reportUnknown* errors.

2. CI runs pyright against system Python (where everything is
   `pip install --system`-ed); pyright auto-detects that interpreter.
   Locally, pyright invoked via `python -m pyright` does NOT pick the
   `.venv/` interpreter automatically and reports ~145
   reportMissingImports for polars / numpy / nexus / praxis.

Both gaps are closed by the `_build_command('typing', ...)` helper in
`backtest_simulator/cli/commands/gate.py`. This test locks the
contract by inspecting the generated command string for both
mitigations.
"""
from __future__ import annotations

from pathlib import Path

from backtest_simulator.cli.commands import gate as gate_cmd

REPO_ROOT = Path(__file__).resolve().parents[2]
TYPING_WORKFLOW = REPO_ROOT / '.github/workflows/pr_checks_typing.yml'


def _typing_command_source() -> str:
    """Return the literal `-c` snippet `bts gate typing` runs.

    The build_command return shape is `[sys.executable, '-c', snippet]`;
    we sanity-check the wrapper before returning the snippet.
    """
    cmd = gate_cmd._build_command('typing', REPO_ROOT)
    assert len(cmd) >= 3 and cmd[1] == '-c', cmd
    return str(cmd[2])


def test_command_plants_pytyped_markers_for_each_sibling() -> None:
    """The local typing gate plants py.typed in nexus/praxis/limen/clickhouse_connect."""
    snippet = _typing_command_source()
    for sibling in ('nexus', 'praxis', 'limen', 'clickhouse_connect'):
        assert sibling in snippet, (
            f'`bts gate typing` does not plant py.typed for {sibling!r}. '
            f'CI does — see pr_checks_typing.yml step "Plant py.typed '
            f'markers in sibling libraries". Without the marker, '
            f'pyright reports reportMissingTypeStubs locally for code '
            f'CI sees cleanly.'
        )
    assert "'py.typed'" in snippet, (
        '`bts gate typing` must plant a literal `py.typed` file at '
        'each sibling package root (PEP 561). Look for the '
        "`open(..., 'a').close()` line in `_build_command('typing')`."
    )


def test_command_passes_pythonpath_to_pyright() -> None:
    """The local invocation forwards `sys.executable` as `--pythonpath` to pyright."""
    snippet = _typing_command_source()
    assert '"--pythonpath", sys.executable' in snippet, (
        '`bts gate typing` must pass `--pythonpath sys.executable` to '
        'pyright so the venv\'s site-packages is discovered. Without '
        'this flag, pyright auto-detects a different interpreter and '
        'reports ~145 reportMissingImports for polars / numpy / etc. '
        'CI does not need the flag because its pyright sees system '
        'Python directly.'
    )


def test_command_uses_protected_base_budget_not_bootstrap() -> None:
    """The local gate runs against the same base-budget oracle CI uses."""
    snippet = _typing_command_source()
    # CI fetches origin/main:.github/typing_budget.json as the base oracle.
    # Locally we mirror that, with a HEAD: fallback for the bootstrap
    # commit. Bootstrap mode (which would disable the ratchet) must NOT
    # be in the snippet — passing it would let a local gate succeed
    # while CI rejected after raising the budget.
    assert 'origin/main:.github/typing_budget.json' in snippet, (
        'local typing gate must read the protected base-ref budget '
        '(origin/main:.github/typing_budget.json) the same way CI does'
    )
    assert '--bootstrap' not in snippet, (
        '`--bootstrap` disables the typing-budget ratchet. Letting the '
        'local gate run in bootstrap while CI runs in compare mode is '
        'the exact divergence this slice closes — reject the flag.'
    )


def test_command_calls_typing_gate_with_pyright_json() -> None:
    """The local invocation pipes pyright JSON into tools/typing_gate.py."""
    snippet = _typing_command_source()
    assert 'tools/typing_gate.py' in snippet
    assert '"--pyright-json", "pyright_output.json"' in snippet
    assert '"--base-budget", "base_budget.json"' in snippet


def test_ci_workflow_plants_pytyped_markers_too() -> None:
    """CI's pr_checks_typing.yml step plants py.typed for the same siblings.

    The local gate's parity claim only holds if CI is doing the same
    thing. Pin the workflow shape so that a future PR moving CI off
    the py.typed-planting step also breaks this test.
    """
    workflow = TYPING_WORKFLOW.read_text(encoding='utf-8')
    assert 'Plant py.typed markers in sibling libraries' in workflow
    for sibling in ('nexus', 'praxis', 'limen', 'clickhouse_connect'):
        assert sibling in workflow, (
            f'pr_checks_typing.yml no longer plants py.typed for '
            f'{sibling!r}. Either update CI in lockstep, or remove the '
            f'sibling from `bts gate typing`\'s planting list.'
        )
