"""Sweep-level statistical-honesty stats for `bts sweep` summary."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from backtest_simulator.cli._metrics import STARTING_CAPITAL, Trade, pair_metrics, pair_trades
from backtest_simulator.honesty.deflated_sharpe import DeflatedSharpeResult
from backtest_simulator.honesty.spa import SpaResult


@dataclass(frozen=True)
class SweepStats:
    dsr: DeflatedSharpeResult | None
    spa: SpaResult | None
    best_decoder: str | None
    best_sharpe: float | None
    n_decoders: int
    n_observations: int

def daily_return_for_run(trades: list[Trade], declared_stops: dict[str, Decimal]) -> float | None:
    pairs, trailing = pair_trades(trades)
    if trailing:
        return None
    net = Decimal('0')
    for pair in pairs:
        declared = declared_stops.get(pair[0].client_order_id)
        pair_net, _, _ = pair_metrics(pair, declared)
        net += pair_net
    return float(net / STARTING_CAPITAL)

def make_seed_price_from_parquet(tape_path: Path) -> Callable[[datetime], Decimal]:
    import polars as _pl
    frame = _pl.read_parquet(str(tape_path)).set_sorted('time')

    def _seed(ts: datetime) -> Decimal:
        rows = frame.filter(_pl.col('time') >= ts).head(1)
        if rows.is_empty():
            msg = f'--trades-tape {tape_path} has no tick at or after {ts.isoformat()}; tape must cover every replay day open + close.'
            raise RuntimeError(msg)
        return Decimal(str(rows['price'][0]))
    return _seed

def announce_operator_trades_tape(tape_path: Path, t_start: float) -> str:
    import time as _time
    if not tape_path.is_file():
        msg = f'bts sweep: --trades-tape file not found: {tape_path}'
        raise FileNotFoundError(msg)
    elapsed = _time.perf_counter() - t_start
    print(f'[{elapsed:7.2f}s] trades tape ready: {tape_path.name} (operator-supplied)', flush=True)
    return str(tape_path)

@dataclass(frozen=True)
class CpcvPboResult:
    pbo: float
    n_paths: int
    n_decoders: int
    purge_seconds: int
    embargo_seconds: int
    n_paths_skipped: int
