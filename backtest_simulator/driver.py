"""SimulationDriver — the DES loop that replaces PredictLoop + TimerLoop."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Final

from backtest_simulator.runtime.nexus_runtime import NexusRuntime
from backtest_simulator.runtime.outcome_translator import NexusOutcomeShape
from backtest_simulator.sensors.precompute import SignalsTable


@dataclass(frozen=True)
class DriverStats:
    """Summary counters emitted at run end."""

    bars_processed: int
    entries: int
    exits: int
    rejects: int


_PROB_IN: Final[float] = 0.55
_PROB_OUT: Final[float] = 0.45


@dataclass
class SimulationDriver:
    """Walk one bar at a time. Bypasses PredictLoop; calls venue via NexusRuntime.

    Strategy contract for M1: long on `prob > 0.55`, flat on `prob < 0.45`,
    hold otherwise. Every ENTER carries a `declared_stop = entry * (1 - stop_frac)`
    for BUY (`(1 + stop_frac)` for SELL). The stop is honored strictly by
    the venue's fill model. The resulting R = |entry - stop| * qty is the
    publishable denominator for `reporting/metrics.r_per_trade`.

    Outcomes are collected as tuples `(strategy_id, outcome)` and exposed via
    `pop_outcomes()` so the reporting layer can derive R/PF/etc.
    """

    bars: list[dict[str, object]]  # iterable of rows with 'open_time', 'close', 'high', 'low'
    signals: SignalsTable
    runtime: NexusRuntime
    strategy_id: str = 'strategy-0'
    symbol: str = 'BTCUSDT'
    base_qty: Decimal = Decimal('0.01')
    stop_frac: Decimal = Decimal('0.01')
    side_bias: str = 'BUY'
    _open_trade_id: str | None = None
    _stats: dict[str, int] = field(default_factory=lambda: {'entries': 0, 'exits': 0, 'rejects': 0})
    _outcomes: list[tuple[str, NexusOutcomeShape]] = field(default_factory=list)

    def capture_outcome(self, strategy_id: str, outcome: NexusOutcomeShape) -> None:
        self._outcomes.append((strategy_id, outcome))

    def pop_outcomes(self) -> list[tuple[str, NexusOutcomeShape]]:
        out = list(self._outcomes)
        self._outcomes.clear()
        return out

    async def run(self) -> DriverStats:
        bars_processed = 0
        for bar in self.bars:
            bars_processed += 1
            ts = _as_dt(bar['open_time'])
            close = Decimal(str(bar['close']))
            signal = self.signals.lookup(ts)
            if signal is None:
                continue
            if self._open_trade_id is None and signal.prob > _PROB_IN:
                await self._enter(ts=ts, close=close)
            elif self._open_trade_id is not None and signal.prob < _PROB_OUT:
                await self._exit(ts=ts)
        if self._open_trade_id is not None:
            final_ts = _as_dt(self.bars[-1]['open_time']) + timedelta(hours=1)
            await self._exit(ts=final_ts)
        return DriverStats(
            bars_processed=bars_processed,
            entries=self._stats['entries'], exits=self._stats['exits'],
            rejects=self._stats['rejects'],
        )

    async def _enter(self, *, ts: datetime, close: Decimal) -> None:
        stop_multiplier = (Decimal('1') - self.stop_frac) if self.side_bias == 'BUY' else (Decimal('1') + self.stop_frac)
        declared_stop = close * stop_multiplier
        trade_id = await self.runtime.submit_entry(
            strategy_id=self.strategy_id, symbol=self.symbol,
            side=self.side_bias, qty=self.base_qty,
            entry_price=close, declared_stop=declared_stop,
            submit_time=ts,
        )
        if trade_id:
            self._open_trade_id = trade_id
            self._stats['entries'] += 1
        else:
            self._stats['rejects'] += 1

    async def _exit(self, *, ts: datetime) -> None:
        assert self._open_trade_id is not None
        await self.runtime.submit_exit(trade_id=self._open_trade_id, submit_time=ts)
        self._open_trade_id = None
        self._stats['exits'] += 1


def _as_dt(value: object) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    msg = f'expected datetime in bar["open_time"], got {type(value).__name__}'
    raise TypeError(msg)


StrategyCallback = Callable[[SimulationDriver, datetime], Awaitable[None]]
