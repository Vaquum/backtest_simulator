"""Pin SimulatedVenueAdapter.submit_order on the zero-fill terminal status.

Pre-fix: a validated order that walked to zero fills (no liquidity in
the window) was reported as `OrderStatus.REJECTED`, conflating venue-
rejection with no-fill-in-window. This made `query_open_orders()`
permanently terminal for any future resting LIMIT and lied to lifecycle
consumers about why the order didn't fill. Post-fix: the validation-
rejection path still returns REJECTED (validation actually said no),
but the no-liquidity walk outcome returns EXPIRED (window closed
without execution).
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import cast

import polars as pl
from praxis.core.domain.enums import OrderSide, OrderStatus, OrderType

from backtest_simulator.feed.protocol import VenueFeed
from backtest_simulator.venue.fees import FeeSchedule
from backtest_simulator.venue.filters import BinanceSpotFilters
from backtest_simulator.venue.simulated import SimulatedVenueAdapter


class _EmptyFeed:
    """VenueFeed that returns no trades for any window — exercises the no-fill path."""

    def _empty_frame(self) -> pl.DataFrame:
        # `walk_trades` filters on a `time` column comparing against a
        # tz-aware UTC literal — the empty frame must declare the same
        # tz-aware Datetime dtype so the comparison's schema validates.
        return pl.DataFrame(schema={
            'time': pl.Datetime(time_zone='UTC'),
            'price': pl.Float64,
            'qty': pl.Float64,
            'trade_id': pl.Int64,
        })

    def get_trades(self, symbol: str, start: datetime, end: datetime) -> pl.DataFrame:
        del symbol, start, end
        return self._empty_frame()

    def _get_trades_for_venue(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        venue_lookahead_seconds: int,
    ) -> pl.DataFrame:
        del symbol, start, end, venue_lookahead_seconds
        return self._empty_frame()


def _adapter() -> SimulatedVenueAdapter:
    return SimulatedVenueAdapter(
        feed=cast(VenueFeed, _EmptyFeed()),
        filters=BinanceSpotFilters.binance_spot('BTCUSDT'),
        fees=FeeSchedule(),
        trade_window_seconds=60,
    )


def test_zero_fill_returns_expired_not_rejected() -> None:
    # Validated MARKET BUY that finds no liquidity in the empty feed
    # must terminate as EXPIRED. REJECTED is reserved for orders that
    # failed the filter pre-walk.
    adapter = _adapter()
    adapter.register_account('acct-1', 'k', 's')
    result = asyncio.run(adapter.submit_order(
        'acct-1', 'BTCUSDT', OrderSide.BUY,
        OrderType.MARKET,
        Decimal('0.001'),
    ))
    assert result.status == OrderStatus.EXPIRED, (
        f'expected EXPIRED for no-fill-in-window outcome, got {result.status}. '
        f'Pre-fix this returned REJECTED, conflating no-fill with venue-rejection.'
    )
    assert result.immediate_fills == ()


def test_zero_fill_status_routing() -> None:
    """`_I.zero_fill_status` picks OPEN for the GTC family, EXPIRED elsewhere.

    Live Binance keeps any GTC non-MARKET order on the book until it
    triggers / crosses / is cancelled. The simulator reports OPEN so
    `query_open_orders` surfaces still-live orders correctly (M1 only
    emits MARKET, so this isn't exercised end-to-end, but the status
    mapping must be correct for backtest ≡ paper ≡ live status parity).
    MARKET (no resting concept) and IOC / FOK (terminate on no
    immediate execution) map to EXPIRED.
    """
    from backtest_simulator.venue._adapter_internals import zero_fill_status

    # GTC family stays OPEN.
    assert zero_fill_status('LIMIT', 'GTC') == OrderStatus.OPEN
    assert zero_fill_status('LIMIT', 'gtc') == OrderStatus.OPEN  # case-insensitive
    assert zero_fill_status('STOP_LOSS', 'GTC') == OrderStatus.OPEN
    assert zero_fill_status('STOP_LOSS_LIMIT', 'GTC') == OrderStatus.OPEN
    assert zero_fill_status('TAKE_PROFIT', 'GTC') == OrderStatus.OPEN
    # IOC / FOK terminate on no immediate execution.
    assert zero_fill_status('LIMIT', 'IOC') == OrderStatus.EXPIRED
    assert zero_fill_status('LIMIT', 'FOK') == OrderStatus.EXPIRED
    assert zero_fill_status('STOP_LOSS_LIMIT', 'IOC') == OrderStatus.EXPIRED
    # MARKET never rests — always EXPIRED on no fill, even with GTC.
    assert zero_fill_status('MARKET', 'GTC') == OrderStatus.EXPIRED
    assert zero_fill_status('MARKET', 'IOC') == OrderStatus.EXPIRED


def test_validation_rejection_still_rejects() -> None:
    # Below min_qty → fails the filter pre-walk → REJECTED. This path
    # is unchanged by the zero-fill semantic correction.
    adapter = _adapter()
    adapter.register_account('acct-1', 'k', 's')
    result = asyncio.run(adapter.submit_order(
        'acct-1', 'BTCUSDT', OrderSide.BUY,
        OrderType.MARKET,
        Decimal('0.000001'),  # below min_qty 0.00001
    ))
    assert result.status == OrderStatus.REJECTED
