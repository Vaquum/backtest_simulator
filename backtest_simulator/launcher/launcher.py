"""BacktestLauncher — real praxis.launcher.Launcher subclass, historical seams."""
from __future__ import annotations

import logging
from pathlib import Path
from types import FrameType

from limen import HistoricalData
from praxis.infrastructure.event_spine import EventSpine
from praxis.infrastructure.venue_adapter import VenueAdapter
from praxis.launcher import InstanceConfig, Launcher
from praxis.trading_config import TradingConfig

from backtest_simulator.launcher.poller import BacktestMarketDataPoller

_log = logging.getLogger(__name__)


class BacktestLauncher(Launcher):
    """Praxis `Launcher` subclass that swaps live seams for historical ones.

    Overrides (the only three boundary seams this shim is allowed to swap):
      1. `_start_poller` — `BacktestMarketDataPoller` reads klines from
         Limen HistoricalData instead of Binance REST.
      2. `_signal_handler` — no-op; backtests terminate on window end
         rather than SIGINT/SIGTERM.
      3. Venue seam — passed via `venue_adapter=` in Launcher's existing
         constructor; our `SimulatedVenueAdapter` plugs in unchanged.

    NOT overridden: `_start_trading`, `_start_nexus_instances`,
    `_run_nexus_instance`, StartupSequencer call, PredictLoop/TimerLoop
    construction, PraxisOutbound wiring. Those are all Nexus/Praxis
    internals that MUST run identically to production for the
    apples-to-apples guarantee to hold.

    Time advancement under `freeze_time` is a follow-up concern — in its
    current form the launcher runs in real time, so a backtest over a
    1h window takes 1h of wall clock. HonestyStatus flags this as
    BACKTEST_CLOCK=REAL_TIME until the ticker integration lands.
    """

    def __init__(
        self,
        trading_config: TradingConfig,
        instances: list[InstanceConfig],
        venue_adapter: VenueAdapter,
        *,
        event_spine: EventSpine | None = None,
        db_path: Path | None = None,
        historical_data: HistoricalData | None = None,
    ) -> None:
        super().__init__(
            trading_config=trading_config,
            instances=instances,
            event_spine=event_spine,
            db_path=db_path,
            venue_adapter=venue_adapter,
            healthz_port=None,
        )
        self._historical_data = historical_data or HistoricalData()

    def _start_poller(self) -> None:
        kline_intervals = self._collect_kline_intervals()
        self._poller = BacktestMarketDataPoller(
            kline_intervals=kline_intervals or {},
            historical_data=self._historical_data,
        )
        self._poller.start()
        _log.info(
            'backtest poller started',
            extra={'kline_sizes': sorted(kline_intervals or {})},
        )

    def _signal_handler(self, _signum: int, _frame: FrameType | None) -> None:
        # Backtests aren't daemons; termination is driven by the outer
        # harness calling `request_stop()` when the window ends, not by
        # SIGINT/SIGTERM landing mid-run.
        _log.info('backtest launcher ignoring external signal')

    def request_stop(self) -> None:
        """Ask `launch()` to return. Outer harness uses this at window end."""
        self._stop_event.set()
