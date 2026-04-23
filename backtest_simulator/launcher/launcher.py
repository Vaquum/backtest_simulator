"""BacktestLauncher — real praxis.launcher.Launcher subclass, historical seams."""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType

from limen import HistoricalData
from praxis.infrastructure.event_spine import EventSpine
from praxis.infrastructure.venue_adapter import VenueAdapter
from praxis.launcher import InstanceConfig, Launcher
from praxis.trading_config import TradingConfig

from backtest_simulator.launcher.clock import accelerated_clock
from backtest_simulator.launcher.poller import BacktestMarketDataPoller

_log = logging.getLogger(__name__)

_BOOT_TIMEOUT_SECONDS = 60
_POLL_INTERVAL_SECONDS = 0.05
_REAL_TIME_CAP_SECONDS = 300
_SHUTDOWN_TIMEOUT_SECONDS = 30


class BacktestLauncher(Launcher):
    """Praxis `Launcher` subclass that swaps live seams for historical ones.

    Three overrides — the boundary seams this shim is allowed to swap:
      1. `_start_poller` — `BacktestMarketDataPoller` reads klines from
         Limen HistoricalData instead of Binance REST.
      2. `_signal_handler` — no-op; backtests terminate on window end
         rather than SIGINT/SIGTERM.
      3. Venue seam — passed via `venue_adapter=` in Launcher's existing
         constructor; our `SimulatedVenueAdapter` plugs in unchanged.

    NOT overridden: `_start_trading`, `_start_nexus_instances`,
    `_run_nexus_instance`, StartupSequencer call, PredictLoop/TimerLoop
    construction, PraxisOutbound wiring. Nexus / Praxis internals that
    MUST run identically to production for apples-to-apples to hold.

    Time advancement: `run_window(start, end)` enters an
    `accelerated_clock` block where `asyncio.sleep(N)` ticks the frozen
    clock by N seconds instead of sleeping real wall time. PredictLoop
    / TimerLoop drive historical time forward via their native interval
    sleeps; a heartbeat from the main thread nudges time along when
    the loop is otherwise idle.
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

    def run_window(self, start: datetime, end: datetime) -> None:
        """Run the backtest from `start` to `end` with an accelerated clock.

        Everything — boot, sensor wiring, PredictLoop ticking — runs
        under the accelerated_clock block. Limen's Trainer emits a
        SystemTimeWarning during its HF-dataset fetch (the frozen clock
        is years before the TLS cert's notBefore), but urllib3 proceeds
        regardless and the fetch succeeds. Starting PredictLoop outside
        accelerated_clock and transitioning in later doesn't work:
        PredictLoop's initial `asyncio.sleep(interval)` is scheduled
        before the monkey-patch engages, so the first tick waits real
        seconds instead of ticking the frozen clock forward. Keeping
        the whole flow inside the block catches every sleep from the
        start.
        """
        if end <= start:
            msg = f'run_window: end {end} must be after start {start}'
            raise ValueError(msg)

        launch_thread = threading.Thread(
            target=self.launch, daemon=True, name='backtest-launch',
        )
        with accelerated_clock(start) as freezer:
            launch_thread.start()
            self._wait_until_trading_ready()
            self._advance_clock_until(end, freezer)
            self.request_stop()
        launch_thread.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
        if launch_thread.is_alive():
            _log.warning('backtest launch thread did not terminate within shutdown timeout')

    def _wait_until_trading_ready(self) -> None:
        # freezegun patches every public `time.*` clock (time.monotonic,
        # time.perf_counter, even freezegun's own cached real_monotonic
        # reference is affected because `_time` is patched at the C
        # module level). `os.times()[4]` is the elapsed wall-clock field
        # from the POSIX `times()` syscall — freezegun does not touch
        # it, so it's the real-time deadline source that survives an
        # active freeze_time block.
        start = os.times()[4]
        while (os.times()[4] - start) < _BOOT_TIMEOUT_SECONDS:
            if self._trading is not None and self._trading.started:
                return
            time.sleep(_POLL_INTERVAL_SECONDS)
        msg = f'Trading did not start within {_BOOT_TIMEOUT_SECONDS}s of real wall time'
        raise RuntimeError(msg)

    @staticmethod
    def _advance_clock_until(end: datetime, freezer: object) -> None:
        # Progress is driven entirely by the loop thread's monkey-patched
        # `asyncio.sleep` calls — each one ticks the frozen clock by its
        # delay. No heartbeat: heartbeats race ahead of Nexus's startup
        # (Trainer's HTTPS fetch is real-time I/O) and burn through the
        # backtest window before PredictLoop ever ticks. Real-clock
        # bound (`_max_real_seconds`) protects against a pathological
        # run where nothing ever sleeps.
        del freezer
        real_start = os.times()[4]
        max_real_seconds = _REAL_TIME_CAP_SECONDS
        while datetime.now(UTC) < end:
            if os.times()[4] - real_start > max_real_seconds:
                _log.warning(
                    'backtest window exceeded %ds of real wall time without '
                    'reaching end=%s; forcing stop at frozen %s',
                    max_real_seconds, end, datetime.now(UTC),
                )
                return
            time.sleep(_POLL_INTERVAL_SECONDS)
