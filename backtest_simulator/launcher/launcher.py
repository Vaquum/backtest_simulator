"""BacktestLauncher — real praxis.launcher.Launcher subclass, historical seams."""
from __future__ import annotations

import importlib
import json
import logging
import os
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import FrameType

from limen import HistoricalData
from nexus.infrastructure.manifest import load_manifest
from praxis.infrastructure.event_spine import EventSpine
from praxis.infrastructure.venue_adapter import VenueAdapter
from praxis.launcher import InstanceConfig, Launcher
from praxis.trading_config import TradingConfig

from backtest_simulator.launcher.clock import accelerated_clock
from backtest_simulator.launcher.poller import BacktestMarketDataPoller

_log = logging.getLogger(__name__)

_BOOT_TIMEOUT_SECONDS = 60
_NEXUS_RUN_TIMEOUT_SECONDS = 120
_POLL_INTERVAL_SECONDS = 0.05
_REAL_TIME_CAP_SECONDS = 600
_SHUTDOWN_TIMEOUT_SECONDS = 30
_NEXUS_RUNNING_MESSAGE = 'nexus instance running'
_CLOCK_TICK_SECONDS = timedelta(seconds=60)
_CLOCK_TICK_REAL_PAUSE_SECONDS = 0.01


class _NexusRunningHandler(logging.Handler):
    """Signal an event once every Nexus instance has logged 'nexus instance running'."""

    def __init__(self, event: threading.Event, expected: int) -> None:
        super().__init__()
        self._event = event
        self._expected = expected
        self._seen = 0
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        if _NEXUS_RUNNING_MESSAGE in record.getMessage():
            with self._lock:
                self._seen += 1
                if self._seen >= self._expected:
                    self._event.set()


class BacktestLauncher(Launcher):
    """Praxis `Launcher` subclass that swaps live seams for historical ones.

    Three overrides — the boundary seams this shim is allowed to swap:
      1. `_start_poller` — `BacktestMarketDataPoller` reads klines from
         Limen HistoricalData instead of Binance REST.
      2. `_signal_handler` — no-op; backtests terminate on window end
         rather than SIGINT/SIGTERM.
      3. Venue seam — passed via `venue_adapter=` in Launcher's existing
         constructor; our `SimulatedVenueAdapter` plugs in unchanged.

    Nothing else is reimplemented. `_start_trading`, `_start_nexus_instances`,
    `_run_nexus_instance`, StartupSequencer, PredictLoop, TimerLoop, and
    PraxisOutbound all run from upstream Nexus / Praxis unchanged — the
    apples-to-apples guarantee requires that production plumbing is the
    same plumbing the backtest drives.
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
        kline_intervals = self._resolve_kline_intervals_from_manifests()
        self._poller = BacktestMarketDataPoller(
            kline_intervals=kline_intervals,
            historical_data=self._historical_data,
        )
        self._poller.start()
        _log.info(
            'backtest poller started',
            extra={'kline_sizes': sorted(kline_intervals)},
        )

    def _resolve_kline_intervals_from_manifests(self) -> dict[int, int]:
        # Praxis's inherited `_collect_kline_intervals` reads
        # `sensor._limen_manifest.data_source_config.params['kline_size']`,
        # but that attribute is never attached to the frozen `SensorSpec`
        # dataclass returned from `load_manifest` — it's an artifact of
        # an earlier wiring step that no longer happens in this release.
        # Result: the inherited extractor returns `{}` and the poller
        # fetches no klines, so the strategy's `on_signal` gets called
        # but `signal.values` is empty (no probability, no close).
        #
        # We read the kline_size straight from the experiment_dir's
        # `metadata.json` → sfd_module → `sfd.manifest().data_source_config`
        # instead. That matches what Limen's `Trainer` does when it
        # reconstructs the sensor, so the poller's data source and
        # Trainer's data source stay aligned.
        intervals: dict[int, int] = {}
        for inst in self._instances:
            try:
                manifest = load_manifest(inst.manifest_path)
            except Exception:  # noqa: BLE001 - per-instance failure is isolated
                _log.exception(
                    'failed to load manifest for kline extraction',
                    extra={'account_id': inst.account_id},
                )
                continue
            for spec in manifest.strategies:
                for sensor_spec in spec.sensors:
                    kline_size = self._kline_size_from_experiment_dir(
                        sensor_spec.experiment_dir,
                    )
                    if kline_size is None:
                        continue
                    current = intervals.get(kline_size)
                    if current is None or sensor_spec.interval_seconds < current:
                        intervals[kline_size] = sensor_spec.interval_seconds
        return intervals

    @staticmethod
    def _kline_size_from_experiment_dir(experiment_dir: Path) -> int | None:
        metadata_path = experiment_dir / 'metadata.json'
        if not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            sfd_module_name = metadata.get('sfd_module')
            if not sfd_module_name:
                return None
            sfd = importlib.import_module(sfd_module_name)
            limen_manifest = sfd.manifest()
            config = getattr(limen_manifest, 'data_source_config', None)
            if config is None:
                return None
            kline_size = config.params.get('kline_size')
            return int(kline_size) if kline_size is not None else None
        except Exception:  # noqa: BLE001 - missing experiment is non-fatal
            _log.exception(
                'failed to extract kline_size from experiment_dir',
                extra={'experiment_dir': str(experiment_dir)},
            )
            return None

    def _signal_handler(self, _signum: int, _frame: FrameType | None) -> None:
        # Backtests aren't daemons; termination is driven by the outer
        # harness calling `request_stop()` when the window ends, not by
        # SIGINT/SIGTERM landing mid-run.
        _log.info('backtest launcher ignoring external signal')

    def request_stop(self) -> None:
        """Ask `launch()` to return. Outer harness uses this at window end."""
        self._stop_event.set()

    def run_window(self, start: datetime, end: datetime) -> None:
        """Run the backtest from `start` to `end`; boot then accelerate.

        Boot (Trading + Nexus StartupSequencer + Trainer + strategy
        on_startup) runs under REAL wall time. Nexus's Trainer re-streams
        the full BTCUSDT-klines dataset from HuggingFace on every boot
        (Limen has no on-disk cache) and that fetch takes ~20 real
        seconds plus sklearn refit. Running that under `accelerated_clock`
        lets concurrent asyncio.sleep sites tick the frozen clock
        forward during the real-time HTTPS wait and burn through the
        backtest window before PredictLoop ticks at all.

        The flow instead:
          1. Launch thread starts under real time — Trainer fetches at
             full speed, SSL validates against a 2026 real clock, and
             every Nexus instance logs 'nexus instance running' when its
             StartupSequencer completes and PredictLoop.start() has
             been called.
          2. A log handler keyed on that message releases a barrier
             once all instances are up.
          3. We enter `accelerated_clock(start)` at that point. The
             very next `asyncio.sleep(kline_size)` from PredictLoop /
             TimerLoop hits the monkey-patch and advances frozen time
             from `start` forward in kline-sized chunks.
          4. Main thread blocks until `datetime.now(UTC) >= end`, then
             calls `request_stop()`. A real-wall-clock cap protects
             against runs that never produce a sleep.
        """
        if end <= start:
            msg = f'run_window: end {end} must be after start {start}'
            raise ValueError(msg)

        running_event = threading.Event()
        handler = _NexusRunningHandler(
            running_event, expected=max(len(self._instances), 1),
        )
        nexus_logger = logging.getLogger('praxis.launcher')
        nexus_logger.addHandler(handler)

        launch_thread = threading.Thread(
            target=self.launch, daemon=True, name='backtest-launch',
        )
        try:
            launch_thread.start()
            self._wait_until_trading_ready()
            self._wait_until_all_nexus_running(running_event)
        finally:
            nexus_logger.removeHandler(handler)

        with accelerated_clock(start) as freezer:
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
    def _wait_until_all_nexus_running(event: threading.Event) -> None:
        if not event.wait(timeout=_NEXUS_RUN_TIMEOUT_SECONDS):
            msg = (
                f'Nexus instances did not all reach "running" within '
                f'{_NEXUS_RUN_TIMEOUT_SECONDS}s of real wall time'
            )
            raise RuntimeError(msg)

    @staticmethod
    def _advance_clock_until(end: datetime, freezer: object) -> None:
        # PredictLoop schedules its ticks via `threading.Timer`, whose
        # wait loop uses real monotonic time regardless of freezegun.
        # `accelerated_clock` patches `threading.Timer.run` to poll the
        # frozen clock instead, so the timer fires when enough frozen
        # time has elapsed — but that requires someone to actually
        # advance the frozen clock. Here we do: tick by
        # `_CLOCK_TICK_SECONDS` each iteration, pause briefly in real
        # time to let the Timer thread + strategy callbacks + venue
        # adapter + reconciliation process the tick, repeat until the
        # frozen window end.
        real_start = os.times()[4]
        while datetime.now(UTC) < end:
            if os.times()[4] - real_start > _REAL_TIME_CAP_SECONDS:
                _log.warning(
                    'backtest window exceeded %ds of real wall time without '
                    'reaching end=%s; forcing stop at frozen %s',
                    _REAL_TIME_CAP_SECONDS, end, datetime.now(UTC),
                )
                return
            freezer.tick(_CLOCK_TICK_SECONDS)  # type: ignore[attr-defined]
            time.sleep(_CLOCK_TICK_REAL_PAUSE_SECONDS)
