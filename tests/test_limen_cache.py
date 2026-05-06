from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def test_package_import_does_not_patch_limen_historical_data() -> None:
    pytest.importorskip('limen')
    code = textwrap.dedent(
        """
        import inspect
        from limen.data import HistoricalData

        before = inspect.getsourcefile(HistoricalData.get_spot_klines)
        import backtest_simulator  # noqa: F401
        after = inspect.getsourcefile(HistoricalData.get_spot_klines)

        assert before == after, (before, after)
        assert not getattr(HistoricalData, '_bts_cache_installed', False)
        """,
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
