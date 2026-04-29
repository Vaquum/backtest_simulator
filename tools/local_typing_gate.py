"""`bts gate typing` runner — mirror `pr_checks_typing.yml`."""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
from typing import Final

_SIBLINGS: Final[tuple[str, ...]] = ('nexus', 'praxis', 'limen', 'clickhouse_connect')


def _plant_pytyped_markers() -> None:
    """Plant PEP 561 `py.typed` in each sibling package (idempotent)."""
    import importlib
    for name in _SIBLINGS:
        module = importlib.import_module(name)
        module_file = module.__file__
        if module_file is None:
            msg = f'sibling module {name!r} has no __file__; cannot locate package root'
            raise RuntimeError(msg)
        marker = os.path.join(os.path.dirname(module_file), 'py.typed')
        if not os.path.exists(marker):
            # Open with 'w' (write) ONLY when missing — match CI's
            # `if not os.path.exists(marker): with open(marker, 'w'): pass`
            # shape exactly. `'a'` would touch the file even when
            # present, requiring write permission unnecessarily.
            with open(marker, 'w'):
                pass


def _run_pyright() -> None:
    """Capture pyright JSON; fail loud on missing/empty output."""
    pyr = subprocess.run(
        [
            sys.executable, '-m', 'pyright',
            '--pythonpath', sys.executable,
            '--outputjson', 'backtest_simulator',
        ],
        capture_output=True, text=True, check=False,
    )
    with open('pyright_output.json', 'w') as f:
        f.write(pyr.stdout)
    # Mirror pr_checks_typing.yml:184-189 — pyright crash / unrecognized
    # flag yields empty JSON; surface stderr and exit 2 so the operator
    # sees the actual cause, not a downstream JSON-decode error.
    out_path = pathlib.Path('pyright_output.json')
    if not out_path.exists() or out_path.stat().st_size == 0:
        if pyr.stderr:
            sys.stderr.write(pyr.stderr)
            if not pyr.stderr.endswith('\n'):
                sys.stderr.write('\n')
        sys.stderr.write('local_typing_gate: pyright produced no output\n')
        sys.exit(2)


def _resolve_base_budget() -> list[str]:
    """Return the typing_gate.py args for the base-budget oracle.

    Fetch the current protected base before reading it (CI runs
    `git fetch origin <base_ref> --depth=1` first; without it the
    local clone reads whatever stale origin/main it last fetched).
    Then prefer `origin/main:.github/typing_budget.json`; if that
    fails AND HEAD introduces the file, return `--bootstrap`;
    otherwise fail loud.
    """
    fetch = subprocess.run(
        ['git', 'fetch', 'origin', 'main', '--depth=1'],
        capture_output=True, text=True, check=False,
    )
    if fetch.returncode != 0:
        sys.stderr.write(
            'local_typing_gate: git fetch origin main failed; cannot '
            'verify the base-ref budget. Stderr:\n' + fetch.stderr,
        )
        sys.exit(2)
    base = subprocess.run(
        ['git', 'show', 'origin/main:.github/typing_budget.json'],
        capture_output=True, text=True, check=False,
    )
    if base.returncode == 0:
        with open('base_budget.json', 'w') as f:
            f.write(base.stdout)
        return ['--base-budget', 'base_budget.json']
    diff = subprocess.run(
        ['git', 'diff', '--name-only', 'origin/main...HEAD'],
        capture_output=True, text=True, check=False,
    )
    if diff.returncode != 0 or '.github/typing_budget.json' not in diff.stdout.split():
        sys.stderr.write(
            'local_typing_gate: origin/main has no .github/typing_budget.json '
            'and HEAD does not introduce it.\n',
        )
        sys.exit(2)
    return ['--bootstrap']


def main() -> int:
    _plant_pytyped_markers()
    _run_pyright()
    base_args = _resolve_base_budget()
    return subprocess.run(
        [
            sys.executable, 'tools/typing_gate.py',
            '--pyright-json', 'pyright_output.json',
            *base_args,
        ],
        check=False,
    ).returncode


if __name__ == '__main__':
    sys.exit(main())
