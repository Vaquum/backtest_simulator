"""`bts gate typing` mirrors CI's `pr_checks_typing.yml`."""
from __future__ import annotations

from pathlib import Path

from backtest_simulator.cli.commands import gate as gate_cmd

REPO_ROOT = Path(__file__).resolve().parents[2]
TYPING_WORKFLOW = REPO_ROOT / '.github/workflows/pr_checks_typing.yml'
LOCAL_GATE = REPO_ROOT / 'tools/local_typing_gate.py'


def _runner_source() -> str:
    """Return the local typing-gate script's source — the readable parity surface."""
    return LOCAL_GATE.read_text(encoding='utf-8')


def test_gate_command_invokes_local_typing_gate() -> None:
    """`bts gate typing` shells out to `tools/local_typing_gate.py`."""
    cmd = gate_cmd._build_command('typing', REPO_ROOT)
    assert cmd[1:] == ['tools/local_typing_gate.py'], (
        f'gate.py must shell out to the local typing-gate script. Got cmd[1:]={cmd[1:]!r}.'
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
        "string in `tools/local_typing_gate.py`."
    )


def test_pytyped_planting_skips_when_marker_already_present() -> None:
    """Planting is skip-if-present, not unconditional touch.

    CI's shape is `if not os.path.exists(marker): with open(marker, 'w'): pass`.
    `open(..., 'a').close()` would touch a read-only filesystem
    unnecessarily and is not byte-equivalent to CI.
    """
    src = _runner_source()
    assert 'if not os.path.exists(marker):' in src, (
        'py.typed planting must skip-if-present rather than '
        'unconditionally `open(..., "a").close()` — match CI\'s '
        '`if not os.path.exists` shape.'
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


def test_command_uses_protected_base_budget_not_bootstrap() -> None:
    """The runner reads the protected base-ref budget; bootstraps only when HEAD adds the file.

    CI resolves the base oracle by trying `origin/<base>:.github/typing_budget.json`
    first; bootstrapping only if HEAD introduces the file; failing loud
    otherwise. A silent fallback would let a PR raise the head budget
    locally, pass `bts gate typing`, then fail CI.
    """
    src = _runner_source()
    assert "'origin/main:.github/typing_budget.json'" in src
    assert "'--bootstrap'" in src
    assert "'.github/typing_budget.json' not in diff.stdout.split()" in src
    assert 'sys.exit(2)' in src
    assert "'HEAD:.github/typing_budget.json'" not in src


def test_runner_fetches_origin_main_before_reading_base_budget() -> None:
    """CI runs `git fetch origin <base> --depth=1` before reading the base budget.

    Without the fetch, a local clone one (or many) commits behind
    `origin/main` would read a stale budget — a budget raised on `main`
    since the last fetch would be invisible locally and `bts gate typing`
    could pass while CI failed (or vice versa).
    """
    src = _runner_source()
    assert "'fetch', 'origin', 'main', '--depth=1'" in src, (
        '`tools/local_typing_gate.py` must fetch origin/main before '
        "reading `origin/main:.github/typing_budget.json`. Without the "
        'fetch, the local base ref is stale and the gate diverges from CI.'
    )


def test_runner_fails_loud_on_empty_pyright_output() -> None:
    """CI fails loud when pyright produces no output; the local runner mirrors that.

    A pyright crash, missing module, or unrecognized flag yields empty
    JSON. CI's `pr_checks_typing.yml:184-189` checks the file exists
    and is non-empty, prints `::error::pyright produced no output`,
    and exits 2. Without the same check locally, an empty JSON gets
    handed to `tools/typing_gate.py` which surfaces a downstream
    JSON-decode error rather than the actual pyright stderr.
    """
    src = _runner_source()
    assert "out_path.stat().st_size == 0" in src, (
        'runner must check pyright_output.json size and fail loud on '
        'empty output, mirroring CI'
    )
    assert 'pyr.stderr' in src, (
        "runner must surface pyright's stderr when output is empty"
    )


def test_command_calls_typing_gate_with_pyright_json() -> None:
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
