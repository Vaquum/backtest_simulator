"""`bts gate typing` runner — mirror `pr_checks_typing.yml` byte-for-byte."""

# Step-by-step parity (each helper below names the workflow step it
# mirrors):
#
# 1. `_plant_pytyped_markers` — PEP 561 `py.typed` in nexus / praxis /
#    limen / clickhouse_connect. Idempotent skip-if-present (NOT
#    `open(..., 'a')`, which would touch a read-only filesystem).
# 2. `_run_pyright` — `pyright --pythonpath sys.executable
#    --outputjson backtest_simulator`. The `--pythonpath` flag is the
#    fix for pyright's failure to auto-detect the venv when invoked
#    via `python -m pyright`; CI does not need it because system
#    Python is auto-detected there.
# 3. `_resolve_base_budget` — fetch `origin/main:.github/typing_budget.json`;
#    bootstrap only if HEAD introduces the file; fail loud otherwise.
#    Same conditional CI uses; never silent fallback to HEAD.
# 4. `main` — pipe pyright JSON + resolved base into
#    `tools/typing_gate.py` and exit on its return code.
#
# This file exists ONLY for `bts gate typing` parity with CI; nothing
# else imports it.
from __future__ import annotations

import os
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
    """Capture pyright JSON to `pyright_output.json`."""
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


def _resolve_base_budget() -> list[str]:
    """Return the typing_gate.py args for the base-budget oracle.

    Mirrors `pr_checks_typing.yml`'s "Resolve base ref" step: prefer
    `origin/main:.github/typing_budget.json`; if that fails AND HEAD
    introduces the file, return `--bootstrap`; otherwise fail loud.
    """
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
        msg = (
            'typing_runner: origin/main has no .github/typing_budget.json '
            'and HEAD does not introduce it. Either the protected base ref '
            'lost the budget (restore it before merging) or your branch is '
            'missing origin/main; run `git fetch origin main`. Same fail-loud '
            'shape as `pr_checks_typing.yml`.'
        )
        sys.stderr.write(msg + '\n')
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
