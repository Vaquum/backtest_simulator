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

Both gaps are closed by `backtest_simulator.cli._typing_runner`,
which `bts gate typing` shells out to. The runner mirrors
`pr_checks_typing.yml`'s steps: plant markers, run pyright with
`--pythonpath sys.executable`, then resolve the base-budget oracle
with the same bootstrap-or-fail-loud conditional CI uses (NOT a
silent fallback to `HEAD:` — codex post-auditor-2 P1 caught the
prior shape).

This test reads the runner's source by symbol so a refactor that
keeps the contract still passes.
"""
from __future__ import annotations

import inspect
from pathlib import Path

from backtest_simulator.cli import _typing_runner
from backtest_simulator.cli.commands import gate as gate_cmd

REPO_ROOT = Path(__file__).resolve().parents[2]
TYPING_WORKFLOW = REPO_ROOT / '.github/workflows/pr_checks_typing.yml'


def _runner_source() -> str:
    """Return the runner module's source — the readable parity surface."""
    return inspect.getsource(_typing_runner)


def test_gate_command_invokes_typing_runner() -> None:
    """`bts gate typing` shells out to `backtest_simulator.cli._typing_runner`."""
    cmd = gate_cmd._build_command('typing', REPO_ROOT)
    assert cmd[1:] == ['-m', 'backtest_simulator.cli._typing_runner'], (
        f'gate.py must shell out to the typing runner module. Got cmd[1:]={cmd[1:]!r}.'
    )


def test_command_plants_pytyped_markers_for_each_sibling() -> None:
    """The local typing gate plants py.typed in nexus/praxis/limen/clickhouse_connect."""
    src = _runner_source()
    for sibling in ('nexus', 'praxis', 'limen', 'clickhouse_connect'):
        assert sibling in src, (
            f'`bts gate typing` does not plant py.typed for {sibling!r}. '
            f'CI does — see pr_checks_typing.yml step "Plant py.typed '
            f'markers in sibling libraries". Without the marker, '
            f'pyright reports reportMissingTypeStubs locally for code '
            f'CI sees cleanly.'
        )
    assert "'py.typed'" in src, (
        '`bts gate typing` must plant a literal `py.typed` file at '
        'each sibling package root (PEP 561). Look for the marker '
        "string in `backtest_simulator.cli._typing_runner`."
    )


def test_pytyped_planting_is_idempotent_skip_when_present() -> None:
    """Skip planting when the marker exists — codex post-auditor-2 P2.

    `open(..., 'a').close()` is content-idempotent but not
    behavior-equivalent to CI: it touches the file even when present,
    requiring write permission. CI's shape is
    `if not os.path.exists(marker): with open(marker, 'w'): pass`.
    The runner mirrors that.
    """
    src = _runner_source()
    assert 'if not os.path.exists(marker):' in src, (
        'py.typed planting must skip-if-present rather than '
        'unconditionally `open(..., "a").close()`. The latter touches '
        'a read-only filesystem unnecessarily. Codex post-auditor-2 '
        'P2 flagged this — match CI\'s `if not os.path.exists` shape.'
    )


def test_command_passes_pythonpath_to_pyright() -> None:
    """The local invocation forwards `sys.executable` as `--pythonpath` to pyright."""
    src = _runner_source()
    assert "'--pythonpath', sys.executable" in src, (
        '`bts gate typing` must pass `--pythonpath sys.executable` to '
        'pyright so the venv\'s site-packages is discovered. Without '
        'this flag, pyright auto-detects a different interpreter and '
        'reports ~145 reportMissingImports for polars / numpy / etc. '
        'CI does not need the flag because its pyright sees system '
        'Python directly.'
    )


def test_base_budget_resolution_is_fail_loud_not_silent_fallback() -> None:
    """Codex post-auditor-2 P1: missing origin/main must NOT silently fall back to HEAD.

    CI's `pr_checks_typing.yml` resolves the base ref by:
      1. Trying `origin/<base>:.github/typing_budget.json`
      2. Bootstrapping ONLY if HEAD adds the file
      3. Failing loud otherwise (`::error::base ref has no .github/typing_budget.json...`)

    The prior local shape silently fell back to `HEAD:.github/typing_budget.json`
    on origin/main miss — letting a PR raise the head budget locally,
    pass `bts gate typing`, then fail CI. The runner now mirrors CI:
    bootstrap or fail loud, never silent.
    """
    src = _runner_source()
    # Must fetch from origin/main first.
    assert "'origin/main:.github/typing_budget.json'" in src, (
        'local runner must consult origin/main first as the base oracle'
    )
    # Bootstrap mode is conditionally enabled when HEAD adds the file.
    assert "'--bootstrap'" in src, (
        'local runner must support bootstrap mode (HEAD introduces budget)'
    )
    assert "'.github/typing_budget.json' not in diff.stdout.split()" in src, (
        'local runner must check that HEAD adds the file before bootstrapping; '
        'a silent unconditional fallback was the divergence post-auditor-2 caught'
    )
    # Fail-loud path on missing-and-not-introduced.
    assert 'sys.exit(2)' in src, (
        'local runner must fail loud (sys.exit(2)) when origin/main is missing '
        'AND HEAD does not introduce the budget — same shape as CI'
    )
    # The legacy silent-fallback to HEAD must NOT exist.
    assert "'HEAD:.github/typing_budget.json'" not in src, (
        'silent fallback to HEAD ref was the codex-flagged divergence; '
        'remove any `git show HEAD:.github/typing_budget.json` invocation '
        'from the runner — the only HEAD-related path is checking whether '
        'HEAD introduces the budget via `git diff --name-only origin/main...HEAD`.'
    )


def test_runner_calls_typing_gate_with_pyright_json() -> None:
    """The runner pipes pyright JSON into tools/typing_gate.py."""
    src = _runner_source()
    assert 'tools/typing_gate.py' in src
    assert "'--pyright-json', 'pyright_output.json'" in src


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
