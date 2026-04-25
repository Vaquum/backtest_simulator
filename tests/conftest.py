"""Pytest conftest: gracefully skip integration-dependent test modules.

Both the honesty gate (`pr_checks_honesty.yml`) and the lint gate
(`pr_checks_lint.yml`) install `.[integration]`, so in CI the full
suite runs in both places and `collect_ignore_glob` resolves to an
empty list. The skip logic stays for two local-dev cases where the
sibling libraries (nexus / praxis / limen / polars) may legitimately
be absent:

  1. A contributor running `pytest tests/tools/` in a clean venv to
     iterate on bloat-gate scripts, without installing the integration
     extras.
  2. Fork CI where the `SIBLING_INSTALL_TOKEN` secret is not
     provisioned and the `.[integration]` install silently fails.

In both cases the dep-heavy paths (honesty/*, launcher/*, pipeline/*,
integration/*) would error on collection, so we drop them from
collection and let the pure-Python tests still run.

An honesty check (`_ANY_SIBLING_PRESENT and not _ALL_SIBLINGS_PRESENT`)
at module-load guards against the silent-skip failure mode: if ANY of
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
    # `tests/venue/test_simulated.py` imports praxis at module scope
    # for the OrderSide / OrderStatus / OrderType enums, so it can't
    # be collected without the integration libs. `tests/venue/test_filters.py`
    # only needs `backtest_simulator.venue.filters` (pure-Python),
    # but the directory pattern is dropped wholesale to keep the skip
    # rule simple. (Slim install sees only `tests/test_lazy_imports.py`
    # and `tests/tools/`.)
    'venue/*',
)

collect_ignore_glob: list[str] = (
    [] if _ALL_SIBLINGS_PRESENT and _HAS_POLARS
    else list(_INTEGRATION_DEPENDENT_GLOBS)
)
