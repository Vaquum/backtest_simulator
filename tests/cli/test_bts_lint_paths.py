"""`bts lint` default paths must match CI's `pr_checks_lint` ruff invocation.

Slice for issue #30: `scripts/` was retired in PR #21 (merged into
`tools/`); the stale `'scripts'` entry in `_DEFAULT_PATHS` caused
`bts lint` to fail with E902 "No such file or directory: scripts" while
`bts gate lint` and CI both passed. The CLI-first contract requires
byte-equivalence between the local acceptance run and CI's.
"""
from __future__ import annotations

from pathlib import Path

from backtest_simulator.cli.commands import lint as lint_cmd

REPO_ROOT = Path(__file__).resolve().parents[2]
LINT_WORKFLOW = REPO_ROOT / '.github/workflows/pr_checks_lint.yml'


def test_default_paths_contains_no_scripts_entry() -> None:
    """Regression: `scripts/` was retired; `_DEFAULT_PATHS` must not list it."""
    assert 'scripts' not in lint_cmd._DEFAULT_PATHS, (
        '`scripts/` was retired in PR #21 (merged into `tools/`). '
        'Leaving it in `_DEFAULT_PATHS` makes `bts lint` E902 with '
        '"No such file or directory: scripts" while CI passes — the '
        'CLI-first contract divergence this slice closes.'
    )


def test_default_paths_match_ci_workflow_set() -> None:
    """`_DEFAULT_PATHS` is the same set CI's pr_checks_lint runs ruff on."""
    workflow = LINT_WORKFLOW.read_text(encoding='utf-8')
    # CI invocation:
    #   .venv-lint/bin/python -m ruff check backtest_simulator tools tests
    assert '-m ruff check backtest_simulator tools tests' in workflow, (
        'pr_checks_lint workflow no longer matches the path set this '
        'test was written against; update both in lockstep.'
    )
    # The CLI's default must equal the CI path set, in the same order
    # (ruff respects argv ordering for include filters).
    assert lint_cmd._DEFAULT_PATHS == ('backtest_simulator', 'tools', 'tests'), (
        f'`bts lint` default paths {lint_cmd._DEFAULT_PATHS!r} diverge '
        f'from the CI workflow set (backtest_simulator, tools, tests). '
        f'Either update _DEFAULT_PATHS or update CI in lockstep.'
    )


def test_default_paths_all_exist_on_disk() -> None:
    """Each default path resolves to a real directory under the repo root."""
    for rel in lint_cmd._DEFAULT_PATHS:
        path = REPO_ROOT / rel
        assert path.is_dir(), (
            f'`bts lint` default path {rel!r} does not resolve to a '
            f'directory at {path}. ruff E902 is the symptom; remove the '
            f'stale entry or restore the missing directory.'
        )


def test_lint_help_text_does_not_mention_scripts() -> None:
    """The --paths help text must not advertise the retired `scripts` path."""
    import argparse
    p = argparse.ArgumentParser()
    sp = p.add_subparsers()

    def _add(name: str, help_text: str) -> argparse.ArgumentParser:
        return sp.add_parser(name, help=help_text)

    lint_cmd.register(_add)
    # Find the lint subparser and pull its --paths help.
    for action in sp._name_parser_map.values():
        for sub_action in action._actions:
            if sub_action.option_strings == ['--paths']:
                assert 'scripts' not in (sub_action.help or ''), (
                    f'`bts lint --paths` help text still advertises '
                    f'`scripts`: {sub_action.help!r}'
                )
                return
    msg = '`--paths` argument not found on the lint subparser'
    raise AssertionError(msg)
