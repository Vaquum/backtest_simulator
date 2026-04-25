"""Pin the package's lazy-import behaviour for slim installs.

`backtest_simulator/__init__.py` keeps the pure-Python re-exports
eager (exceptions, ClickHouseFeed, FeeSchedule, BinanceSpotFilters,
FillModelConfig — none require Praxis/Nexus/Limen) and defers the
integration re-exports (BacktestLauncher, SimulatedVenueAdapter,
install_cache, BacktestMarketDataPoller) behind PEP 562 `__getattr__`.
On a slim install where the integration extra wasn't installed,
`import backtest_simulator` succeeds and accessing one of the lazy
names raises a clear `ImportError` with install guidance.

These tests run on the integration install (so the lazy names ARE
loaded). To exercise the slim-install branch we temporarily inject
a fake `_integration_error` into the module and verify the fallback
behaviour, then restore it.
"""
from __future__ import annotations

import pytest

import backtest_simulator as bs


def test_pure_python_reexports_always_work() -> None:
    # These never depend on the integration extra. Importing them
    # directly off the package must work whether or not Praxis/Nexus
    # are present.
    assert bs.HonestyViolation.__name__ == 'HonestyViolation'
    assert bs.LookAheadViolation.__name__ == 'LookAheadViolation'
    assert bs.BinanceSpotFilters.__name__ == 'BinanceSpotFilters'
    assert bs.FeeSchedule.__name__ == 'FeeSchedule'
    assert bs.ClickHouseConfig.__name__ == 'ClickHouseConfig'


@pytest.mark.skipif(
    bs._integration_error is not None,
    reason='requires the [integration] extra to be installed',
)
def test_integration_names_load_eagerly_with_extras() -> None:
    # On the integration install (the test runner's environment) the
    # eager try-block succeeded, _integration_error is None, and the
    # integration names resolve directly off the module. On a slim
    # install this test is skipped — the slim path is exercised by
    # `test_getattr_raises_loud_when_integration_missing` below
    # (which works in either install by injecting a fake error).
    assert bs.BacktestLauncher.__name__ == 'BacktestLauncher'
    assert bs.SimulatedVenueAdapter.__name__ == 'SimulatedVenueAdapter'
    assert bs.BacktestMarketDataPoller.__name__ == 'BacktestMarketDataPoller'


def test_getattr_raises_loud_when_integration_missing() -> None:
    # Simulate a slim install by injecting an ImportError into the
    # module's `_integration_error` slot. `__getattr__` must raise an
    # ImportError with install guidance, NOT silently fall through
    # to AttributeError or return None.
    saved = bs._integration_error
    bs._integration_error = ImportError('simulated: limen not installed')
    try:
        with pytest.raises(ImportError, match=r'requires the \[integration\] extra'):
            bs.__getattr__('BacktestLauncher')
        with pytest.raises(ImportError, match=r'requires the \[integration\] extra'):
            bs.__getattr__('SimulatedVenueAdapter')
        with pytest.raises(ImportError, match=r'requires the \[integration\] extra'):
            bs.__getattr__('install_cache')
    finally:
        bs._integration_error = saved


def test_getattr_unknown_name_raises_attribute_error() -> None:
    # Names outside the lazy registry must raise AttributeError, not
    # ImportError, so dir()/hasattr() on the package behave normally.
    with pytest.raises(AttributeError, match='no attribute'):
        bs.__getattr__('SomeNonExistentSymbol')


def test_partial_integration_cleanup_drops_succeeded_names() -> None:
    """If an integration import fails, no earlier-imported name leaks to globals.

    Pre-cleanup pattern: `_limen_cache` succeeds, `launcher` fails.
    `install_cache` stays in module globals from the first import,
    `bs.install_cache` returns the live function, but `_integration_error`
    is set — the surface is inconsistent (some lazy names work, others
    raise). Post-fix the except branch drops every name in `_LAZY_NAMES`
    so all four resolve uniformly through `__getattr__`.

    This test simulates the partial state by re-running the cleanup
    block manually after seeding globals with stub names. It must
    save and restore EVERY name in `_LAZY_NAMES`, otherwise on
    integration installs subsequent tests in this session would see
    `AttributeError` on `bs.BacktestLauncher` etc.
    """
    bs_globals = vars(bs)
    # Snapshot every lazy-name binding (including absent ones).
    saved = {n: bs_globals.get(n, _ABSENT) for n in bs._LAZY_NAMES}
    saved_error = bs._integration_error
    sentinel = object()
    try:
        bs_globals['install_cache'] = sentinel
        bs._integration_error = ImportError('simulated partial')
        # Re-run the cleanup snippet the except branch uses.
        for _n in bs._LAZY_NAMES:
            bs_globals.pop(_n, None)
        # Sentinel must be gone.
        assert 'install_cache' not in bs_globals, (
            'expected partial-integration cleanup to drop install_cache '
            'from globals; the sentinel survived.'
        )
        # And __getattr__ must now raise the install-guidance error.
        with pytest.raises(ImportError, match=r'requires the \[integration\] extra'):
            bs.__getattr__('install_cache')
    finally:
        bs._integration_error = saved_error
        # Restore every snapshot — set the ones that were present,
        # leave the ones that were absent absent.
        for name, value in saved.items():
            if value is _ABSENT:
                bs_globals.pop(name, None)
            else:
                bs_globals[name] = value


_ABSENT = object()
