"""Single-window backtest run + subprocess child entry point."""

# `run_window_in_process` builds the manifest, wires the launcher, runs
# the window, and returns picklable trade tuples. `run_window_in_subprocess`
# calls this module as `python -m backtest_simulator.cli._run_window` so
# every window gets a fresh interpreter (necessary on macOS Python 3.12,
# where `multiprocessing.fork` deadlocks inside libdispatch after asyncio
# state has been touched).
#
# The `__main__` entry point reads a JSON payload from stdin, runs the
# window in-process, and writes the JSON result to stdout. Subprocess
# isolation makes between-run state bleed (closed-loop RuntimeError on
# the next submit) impossible — the process exits and a new one starts.
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from backtest_simulator.cli._pipeline import SYMBOL, WORK_DIR, seed_price_at


def _fresh_work_dir(suffix: str) -> Path:
    d = WORK_DIR / suffix
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def run_window_in_process(
    perm_id: int,
    kelly_pct: Decimal,
    window_start: datetime,
    window_end: datetime,
    experiment_dir: Path,
) -> dict[str, object]:
    """In-process single-window run — picklable result for cross-process use."""
    from praxis.launcher import InstanceConfig
    from praxis.trading_config import TradingConfig

    from backtest_simulator.feed.clickhouse import ClickHouseConfig, ClickHouseFeed
    from backtest_simulator.launcher import BacktestLauncher
    from backtest_simulator.pipeline.manifest_builder import (
        AccountSpec,
        ManifestBuilder,
        SensorBinding,
        StrategyParamsSpec,
    )
    from backtest_simulator.venue.fees import FeeSchedule
    from backtest_simulator.venue.filters import BinanceSpotFilters
    from backtest_simulator.venue.simulated import SimulatedVenueAdapter

    seed_price = seed_price_at(window_start)
    suffix = f'{perm_id}_{window_start.date().isoformat()}'
    work = _fresh_work_dir(suffix)
    manifest_dir = work / 'manifest'
    state_dir = work / 'state'
    state_dir.mkdir()

    built = ManifestBuilder(output_dir=manifest_dir).build(
        account=AccountSpec(
            account_id='bts-sweep',
            allocated_capital=Decimal('100000'),
            capital_pool=Decimal('100000'),
        ),
        strategy_id='long_on_signal',
        sensor=SensorBinding(
            experiment_dir=experiment_dir,
            permutation_ids=(perm_id,),
            interval_seconds=3600,
        ),
        strategy_params=StrategyParamsSpec(
            symbol=SYMBOL,
            capital=Decimal('100000'),
            kelly_pct=kelly_pct,
            estimated_price=seed_price,
            stop_bps=Decimal('50'),
        ),
    )
    feed = ClickHouseFeed(config=ClickHouseConfig.from_env(), symbol=SYMBOL)
    adapter = SimulatedVenueAdapter(
        feed=feed,
        filters=BinanceSpotFilters.binance_spot(SYMBOL),
        fees=FeeSchedule(),
        trade_window_seconds=60,
    )
    tc = TradingConfig(
        epoch_id=1, venue_rest_url='http://sim', venue_ws_url='ws://sim',
        account_credentials={'bts-sweep': ('k', 's')}, shutdown_timeout=5.0,
    )
    launcher = BacktestLauncher(
        trading_config=tc,
        instances=[InstanceConfig(
            account_id='bts-sweep',
            manifest_path=built.manifest_path,
            strategies_base_path=built.strategies_base_path,
            state_dir=state_dir,
        )],
        venue_adapter=adapter,
        db_path=work / 'event_spine.sqlite',
    )
    launcher.run_window(window_start, window_end)
    account = adapter.history('bts-sweep')
    trade_tuples = [
        (
            t.client_order_id, t.side.name, str(t.qty), str(t.price),
            str(t.fee), t.fee_asset, t.timestamp.isoformat(),
        )
        for t in account.trades
    ]
    declared_stops = {k: str(v) for k, v in launcher._declared_stops.items()}
    return {
        'trades': trade_tuples,
        'declared_stops': declared_stops,
        'orders': len(account.orders),
    }


def run_window_in_subprocess(
    perm_id: int,
    kelly_pct: Decimal,
    window_start: datetime,
    window_end: datetime,
    experiment_dir: Path,
) -> dict[str, object]:
    """Run one window in a fresh Python interpreter for state isolation."""
    payload = json.dumps({
        'perm_id': perm_id,
        'kelly_pct': str(kelly_pct),
        'window_start': window_start.isoformat(),
        'window_end': window_end.isoformat(),
        'experiment_dir': str(experiment_dir),
    })
    proc = subprocess.run(
        [sys.executable, '-m', 'backtest_simulator.cli._run_window'],
        input=payload,
        capture_output=True, text=True, check=False, timeout=120,
    )
    if proc.returncode != 0:
        msg = f'child exit={proc.returncode}; stderr tail: {proc.stderr[-400:]}'
        raise RuntimeError(msg)
    last: str | None = None
    for line in proc.stdout.splitlines():
        s = line.strip()
        if s.startswith('{') and s.endswith('}'):
            last = s
    if last is None:
        msg = f'child produced no JSON result; stdout tail: {proc.stdout[-400:]}'
        raise RuntimeError(msg)
    return json.loads(last)


def _child_main() -> int:
    """Child-process entry: read JSON payload from stdin, run, write JSON result."""
    payload = json.loads(sys.stdin.read())
    result = run_window_in_process(
        int(payload['perm_id']),
        Decimal(payload['kelly_pct']),
        datetime.fromisoformat(payload['window_start']),
        datetime.fromisoformat(payload['window_end']),
        Path(payload['experiment_dir']),
    )
    print(json.dumps(result), flush=True)
    return 0


if __name__ == '__main__':  # pragma: no cover - subprocess child entry
    sys.exit(_child_main())
