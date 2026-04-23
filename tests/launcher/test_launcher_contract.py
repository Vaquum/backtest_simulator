"""Contract: BacktestLauncher is a real praxis.launcher.Launcher subclass."""
from __future__ import annotations

import pytest
from praxis.launcher import InstanceConfig, Launcher

from backtest_simulator.launcher import BacktestLauncher, BacktestMarketDataPoller


def test_backtest_launcher_is_real_praxis_launcher_subclass() -> None:
    # The shim mandate: don't reimplement Praxis orchestration. Extend it.
    assert issubclass(BacktestLauncher, Launcher)


def test_poller_lifecycle_without_network() -> None:
    # Poller with empty kline_intervals must start/stop cleanly without
    # touching the network. Stays clean under CI without ClickHouse.
    poller = BacktestMarketDataPoller(kline_intervals={})
    assert not poller.running
    poller.start()
    assert poller.running
    poller.stop()
    assert not poller.running


def test_poller_implements_market_data_poller_shape() -> None:
    # praxis.Launcher calls these five on its poller; all must exist.
    poller = BacktestMarketDataPoller(kline_intervals={})
    for name in ('start', 'stop', 'running', 'get_market_data',
                 'add_kline_size', 'remove_kline_size'):
        assert hasattr(poller, name), f'BacktestMarketDataPoller missing {name!r}'


def test_poller_get_market_data_returns_empty_for_unknown_size() -> None:
    poller = BacktestMarketDataPoller(kline_intervals={})
    result = poller.get_market_data(3600)
    assert result.is_empty()


def test_instance_config_construction_is_stable() -> None:
    # Shim contract: we pass real InstanceConfig values to Launcher.
    # Construct one with plausible paths; no disk read required.
    from pathlib import Path
    cfg = InstanceConfig(
        account_id='bts-acct-0',
        manifest_path=Path('/tmp/manifest.yaml'),
        strategies_base_path=Path('/tmp/strategies'),
        state_dir=Path('/tmp/state'),
    )
    assert cfg.account_id == 'bts-acct-0'


def test_launcher_rejects_both_event_spine_and_db_path() -> None:
    # Inherited invariant from praxis.Launcher: exactly one of the two.
    from pathlib import Path

    from praxis.infrastructure.event_spine import EventSpine
    from praxis.trading_config import TradingConfig

    from backtest_simulator.venue.fees import FeeSchedule
    from backtest_simulator.venue.filters import BinanceSpotFilters
    from backtest_simulator.venue.simulated import SimulatedVenueAdapter

    class _StubFeed:
        def get_trades(self, *_a: object, **_k: object) -> object:
            import polars as pl
            return pl.DataFrame()
        def get_window(self, *_a: object, **_k: object) -> object:
            import polars as pl
            return pl.DataFrame()

    adapter = SimulatedVenueAdapter(
        feed=_StubFeed(),  # type: ignore[arg-type]  # Protocol duck-typed stub
        filters=BinanceSpotFilters.binance_spot('BTCUSDT'),
        fees=FeeSchedule(),
    )
    tc = TradingConfig(
        epoch_id=1, venue_rest_url='http://sim', venue_ws_url='ws://sim',
        account_credentials={'x': ('k', 's')}, shutdown_timeout=1.0,
    )
    with pytest.raises(ValueError, match='exactly one'):
        BacktestLauncher(
            trading_config=tc, instances=[], venue_adapter=adapter,
            event_spine=object.__new__(EventSpine),  # type: ignore[arg-type]
            db_path=Path('/tmp/x.sqlite'),
        )
