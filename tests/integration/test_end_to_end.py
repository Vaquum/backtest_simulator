"""End-to-end M1 integration: one manifest, one window, all critical gates green."""
from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from freezegun import freeze_time

from backtest_simulator.driver import SimulationDriver
from backtest_simulator.feed.parquet_fixture import ParquetFixtureFeed
from backtest_simulator.reporting.ledger import assert_fee_nonnegative
from backtest_simulator.reporting.metrics import TradeRecord, summarize
from backtest_simulator.runtime.nexus_runtime import NexusRuntime
from backtest_simulator.sensors.precompute import SignalsTable
from backtest_simulator.venue.fees import FeeSchedule
from backtest_simulator.venue.filters import SymbolFilters
from backtest_simulator.venue.simulated import SimulatedVenueAdapter

FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures' / 'market' / 'btcusdt_1h_fixture.parquet'


def _synthetic_signals(bars: pl.DataFrame) -> SignalsTable:
    """Build a synthetic signal stream: 1 for first N, 0 for next N, repeating.

    This is not a real predictor. It exercises the DES loop through
    multiple enter/exit cycles to prove the driver + venue + runtime
    stack actually drives trades end-to-end.
    """
    n = bars.height
    probs = np.concatenate([
        np.full(10, 0.80), np.full(10, 0.30),
        np.full(10, 0.85), np.full(10, 0.25),
    ] * (1 + n // 40))[:n]
    preds = (probs > 0.5).astype(int)
    stamps = bars['open_time'].to_list()
    return SignalsTable.from_predictions(
        decoder_id='synthetic-0', split_config=(8, 1, 2),
        timestamps=stamps, probs=probs, preds=preds,
        label_horizon_bars=1, bar_seconds=3600,
    )


def _bars_as_list(bars: pl.DataFrame) -> list[dict[str, object]]:
    return [dict(row) for row in bars.iter_rows(named=True)]


def _synthetic_trade_feed(bars: pl.DataFrame) -> ParquetFixtureFeed:
    """Wrap the bars parquet as a ParquetFixtureFeed with synthetic trades.

    One trade per bar at close_price with qty=1. The fixture feed's
    get_trades honors the no-look-ahead invariant, so the driver
    never sees trades past frozen_now().
    """
    import tempfile
    tmp = Path(tempfile.mkdtemp()) / 'trades.parquet'
    trades = pl.DataFrame({
        'time': bars['open_time'].to_list(),
        'price': bars['close'].to_list(),
        'qty': [1.0] * bars.height,
    })
    trades.write_parquet(tmp)
    return ParquetFixtureFeed(FIXTURE, trades_path=tmp)


@pytest.mark.asyncio
async def test_end_to_end_one_manifest_one_window() -> None:
    bars = pl.read_parquet(FIXTURE).sort('open_time')
    first_ts = bars['open_time'][0]
    frozen_at = first_ts + timedelta(hours=bars.height)

    feed = _synthetic_trade_feed(bars)
    venue = SimulatedVenueAdapter(
        feed=feed, fees=FeeSchedule(), filters=SymbolFilters.binance_spot('BTCUSDT'),
        trade_window_seconds=3600 * 3,
    )
    outcomes: list[tuple[str, object]] = []
    runtime = NexusRuntime(
        venue=venue, on_outcome=lambda sid, o: _append(outcomes, sid, o),
    )
    signals = _synthetic_signals(bars)
    driver = SimulationDriver(
        bars=_bars_as_list(bars), signals=signals, runtime=runtime,
    )

    with freeze_time(frozen_at):
        stats = await driver.run()

    # Exercised end-to-end: at least some entries/exits, all outcomes emitted.
    assert stats.bars_processed == bars.height
    assert stats.entries >= 1
    assert stats.exits >= 1

    # Fee accounting is non-negative across every outcome.
    fill_outcomes = [o for _, o in outcomes if hasattr(o, 'actual_fees')]
    assert_fee_nonnegative(fill_outcomes)


async def _append(dest: list[tuple[str, object]], sid: str, outcome: object) -> None:
    dest.append((sid, outcome))


def test_metrics_summarize_produces_headline_numbers() -> None:
    """Proves metrics.summarize aggregates trades into R/PF/sum_pnl_net shape."""
    trades = [
        TradeRecord(
            trade_id=f'T{i}', side='BUY',
            entry_price=Decimal('7000'), exit_price=Decimal('7000') + Decimal(str(delta)),
            declared_stop=Decimal('6930'),  # 1% below
            qty=Decimal('0.1'),
            entry_fees=Decimal('0.7'), exit_fees=Decimal('0.7'),
        )
        for i, delta in enumerate([50, -30, 80, -20, 40, -10, 70])
    ]
    summary = summarize(trades)
    assert summary.n_trades == len(trades)
    assert summary.profit_factor > 0
    assert summary.total_fees > 0
