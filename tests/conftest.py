"""Pytest conftest: gracefully skip integration-dependent test modules.

The lint gate's coverage step runs `pytest tests/` inside a lean
`.venv-lint` that only installs `ruff`, `vulture`, `coverage`,
`pytest`. Every test that transitively imports nexus / praxis / limen
/ polars (i.e. most of this suite) would error on collection under
that environment.

We make collection resilient by dropping the dep-heavy paths from
`collect_ignore` when the required sibling libraries are missing.
The honesty gate (`pr_checks_honesty.yml`) installs `.[integration]`
so those paths are collected and executed there; the lint gate skips
them and measures coverage only on whatever pure-Python tests CAN
run without the deps — the coverage floor (50/45) was set with that
reality in mind.

An honesty check (`test_all_three_siblings_present_together`) at
module-load guards against the silent-skip failure mode: if ANY of
the three libs is present we assert ALL are, so a partially-installed
environment can't silently skip a subset.
"""
from __future__ import annotations

import importlib.util


def _module_present(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


_HAS_NEXUS = _module_present('nexus')
_HAS_PRAXIS = _module_present('praxis')
_HAS_LIMEN = _module_present('limen')
_HAS_POLARS = _module_present('polars')

_ANY_SIBLING_PRESENT = _HAS_NEXUS or _HAS_PRAXIS or _HAS_LIMEN
_ALL_SIBLINGS_PRESENT = _HAS_NEXUS and _HAS_PRAXIS and _HAS_LIMEN

if _ANY_SIBLING_PRESENT and not _ALL_SIBLINGS_PRESENT:
    msg = (
        f'Partial sibling-library install detected: '
        f'nexus={_HAS_NEXUS} praxis={_HAS_PRAXIS} limen={_HAS_LIMEN}. '
        f'All three must be installed together (install `.[integration]`) '
        f'or none — a partial install would silently skip coverage of '
        f'integration paths while still appearing to run.'
    )
    raise RuntimeError(msg)


# Paths that transitively import nexus / praxis / limen / polars at
# module scope and therefore cannot be collected without those libs
# installed. The lint gate's lean venv lacks them; the honesty gate
# has them.
_INTEGRATION_DEPENDENT_GLOBS: tuple[str, ...] = (
    'honesty/*',
    'launcher/*',
    'pipeline/*',
    'integration/*',
)

collect_ignore_glob: list[str] = (
    [] if _ALL_SIBLINGS_PRESENT and _HAS_POLARS
    else list(_INTEGRATION_DEPENDENT_GLOBS)
)
