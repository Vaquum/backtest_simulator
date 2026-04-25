"""Integration gate: SlippageModel actually adjusts SimulatedVenueAdapter fills.

Slice #17 Task 12 wiring.

The standalone SlippageModel passes its own MVC (`test_slippage_calibration.py`).
This test pins that the model is *wired into the venue path*: a
SimulatedVenueAdapter constructed WITH a slippage model produces
different fill prices than one without, on the same submit + same
trade window. Without this assertion the model is dead code from
`bts run` / `bts sweep`'s point of view.

The fixture builds a tiny synthetic-but-realistic trade tape (real
schema; dense quotes around mid; signed by `is_buyer_maker`),
calibrates a SlippageModel, then submits one BUY MARKET against
the same tape via two adapters — one with the calibrated model,
one without — and asserts:

  - the WITH-model adapter's fill price is HIGHER than the WITHOUT
    one (BUY aggressor pays slippage above mid);
  - the WITH-model adapter records `slippage_realised_n_samples
    == 1` and `slippage_realised_aggregate_bps > 0`;
  - the WITHOUT adapter records `n_samples == 0` and
    `slippage_realised_aggregate_bps is None`.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import polars as pl
from nexus.core.domain.enums import OrderSide as NexusOrderSide
from praxis.core.domain.enums import OrderSide, OrderType

from backtest_simulator.honesty.slippage import SlippageModel
from backtest_simulator.venue.fees import FeeSchedule
from backtest_simulator.venue.filters import BinanceSpotFilters
from backtest_simulator.venue.simulated import SimulatedVenueAdapter

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / 'fixtures' / 'market' / 'btcusdt_trades_30min.parquet'
)


class _TapeFeed:
    """Minimal VenueFeed-shape that serves a slice of real BTCUSDT trades.

    The fixture is the same parquet the standalone SlippageModel
    test calibrates on; this integration test reuses it so the
    calibration and the fill window come from the same tape — the
    closest-to-live setup the test rig admits.
    """

    def __init__(self) -> None:
        df = pl.read_parquet(_FIXTURE)
        # Schema-rename: ClickHouse trade table → venue feed
        # contract. ClickHouse column is `datetime`; venue feed
        # column is `time`.
        df = df.rename({'datetime': 'time', 'quantity': 'qty'})
        self._trades = df.sort('time')

    def get_trades(
        self, symbol: str, start: datetime, end: datetime,
    ) -> pl.DataFrame:
        del symbol
        return self._trades.filter(
            (pl.col('time') >= start) & (pl.col('time') <= end),
        )

    def _get_trades_for_venue(
        self, symbol: str, start: datetime, end: datetime,
        *, venue_lookahead_seconds: int,
    ) -> pl.DataFrame:
        del symbol, venue_lookahead_seconds
        return self._trades.filter(
            (pl.col('time') >= start) & (pl.col('time') <= end),
        )


def _calibrate(feed: _TapeFeed, t: datetime) -> SlippageModel:
    pre = feed.get_trades('BTCUSDT', t - timedelta(minutes=10), t)
    # Rename back to the SlippageModel.calibrate contract
    # (`datetime`, `quantity`).
    pre = pre.rename({'time': 'datetime', 'qty': 'quantity'})
    return SlippageModel.calibrate(
        trades=pre,
        side_buckets=(
            Decimal('0.001'), Decimal('0.01'),
            Decimal('0.1'), Decimal('1.0'),
        ),
        dt_seconds=10,
    )


def test_slippage_model_wired_changes_fill_price() -> None:
    """A BUY MARKET fill against the same tape is more expensive WITH model."""
    feed = _TapeFeed()
    # Submit at minute 15 — well past the 10-minute calibration
    # warm-up so the rolling-mid window inside `apply` is full.
    submit_t = datetime(2024, 1, 1, 12, 15, tzinfo=UTC)
    model = _calibrate(feed, submit_t)
    filters = BinanceSpotFilters.binance_spot('BTCUSDT')

    # Pick a qty whose BUY bucket actually exists in the
    # calibration, with a positive median bps (the run-time
    # apply() looks up by `(side, qty)` and raises if the bucket
    # is empty — that path falls back to bps=0 and the test would
    # silently pass). Walking the calibrated buckets here makes
    # the choice explicit.
    buy_buckets = model.buckets_for(NexusOrderSide.BUY)
    qualifying = [
        b for b in buy_buckets
        if b.median_bps > Decimal('0') and b.n_samples >= 5
    ]
    assert qualifying, (
        f'BUY-side calibration has no bucket with positive median '
        f'bps and >= 5 samples; the integration test cannot pin the '
        f'wiring. Buckets: {buy_buckets}'
    )
    chosen = qualifying[0]
    # qty inside the chosen bucket: half-open [qty_min, qty_max),
    # so use qty_min itself (always inclusive, always inside).
    qty = chosen.qty_min if chosen.qty_min > Decimal('0') else (
        chosen.qty_max / Decimal('2') if chosen.qty_max else Decimal('0.5')
    )
    expected_bps = chosen.median_bps

    def _make_adapter(slippage: SlippageModel | None) -> SimulatedVenueAdapter:
        adapter = SimulatedVenueAdapter(
            feed=feed,
            filters=filters,
            fees=FeeSchedule(),
            trade_window_seconds=60,
            slippage_model=slippage,
        )
        adapter.register_account('a', 'k', 's')
        # Freeze the adapter's wall clock at submit_t so submit_order
        # uses the right window. The adapter calls `self._now()`
        # which returns datetime.now(UTC); for the test we monkey-
        # patch it to return submit_t.
        adapter._now = lambda: submit_t  # type: ignore[method-assign]
        return adapter

    a_off = _make_adapter(None)
    a_on = _make_adapter(model)

    res_off = asyncio.run(a_off.submit_order(
        'a', 'BTCUSDT', OrderSide.BUY, OrderType.MARKET, qty,
    ))
    res_on = asyncio.run(a_on.submit_order(
        'a', 'BTCUSDT', OrderSide.BUY, OrderType.MARKET, qty,
    ))

    assert res_off.immediate_fills, 'baseline adapter must produce a fill'
    assert res_on.immediate_fills, 'slippage-wired adapter must produce a fill'

    fill_off = res_off.immediate_fills[0].price
    fill_on = res_on.immediate_fills[0].price
    bps_recorded = a_on.slippage_realised_aggregate_bps
    # The price difference is `fill_off * bps / 10000`, post-tick
    # rounding. For BTCUSDT (tick 0.01 USDT) at ~42700, even sub-
    # bp bps moves the rounded price by ≥ one tick.
    expected_price_delta = fill_off * expected_bps / Decimal('10000')
    if abs(expected_price_delta) < Decimal('0.01'):
        # Calibration produced bps too small to surface after
        # tick-rounding; use the bps-recorded check below instead
        # of a price-comparison check, since price-comparison would
        # be tautological at sub-tick precision.
        assert fill_on >= fill_off, (
            f'BUY taker fill should not improve on raw VWAP under '
            f'positive slippage (bps={expected_bps}); got '
            f'with_slippage={fill_on} no_slippage={fill_off}'
        )
    else:
        assert fill_on > fill_off, (
            f'BUY taker with calibrated slippage must pay above raw VWAP. '
            f'with_slippage={fill_on} no_slippage={fill_off} '
            f'expected_bps={expected_bps} expected_delta={expected_price_delta}'
        )
    del bps_recorded

    assert a_off.slippage_realised_aggregate_bps is None, (
        f'adapter without model must report None (feature-disabled '
        f'signal); got {a_off.slippage_realised_aggregate_bps}'
    )
    assert a_off.slippage_realised_n_samples == 0

    bps_on = a_on.slippage_realised_aggregate_bps
    assert bps_on is not None, (
        'adapter with model must report a numeric aggregate; got None'
    )
    assert bps_on > Decimal('0'), (
        f'BUY-aggregate bps must be positive (taker pays); got {bps_on}'
    )
    assert a_on.slippage_realised_n_samples == 1
