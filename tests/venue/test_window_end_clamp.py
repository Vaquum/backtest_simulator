"""SimulatedVenueAdapter clamps the tape walk to `window_end_clamp`.

Pre-fix: a SELL submitted near the run window's close with a large
`trade_window_seconds` (e.g. `kline_size = 14 400 s` post-PR-#56)
would consume up to `kline_size` of trade tape past `window_end`,
silently peeking at future data. Praxis's per-call causality
guard (`assert_trades_causal`) only ensures `end <=
frozen_now() + venue_lookahead_seconds`, which is satisfied as
long as `venue_lookahead_seconds >= trade_window_seconds` —
that's the bug Copilot caught on PR #57.

Fix: `SimulatedVenueAdapter` accepts a new optional
`window_end_clamp: datetime | None` constructor kwarg. When set,
`submit_order` computes the per-submit walk endpoint as
`min(submit_time + trade_window_seconds, window_end_clamp)` and
passes the corresponding `venue_lookahead_seconds` to the feed
so the trade fetch is honestly bounded by the run window.

These tests pin both branches: the unclamped legacy behaviour
(`window_end_clamp=None`) and the clamped behaviour (clamp set
to a time earlier than `submit_time + trade_window_seconds`).
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import cast

import polars as pl
from praxis.core.domain.enums import OrderSide, OrderType

from backtest_simulator.feed.protocol import VenueFeed
from backtest_simulator.venue.fees import FeeSchedule
from backtest_simulator.venue.filters import BinanceSpotFilters
from backtest_simulator.venue.simulated import SimulatedVenueAdapter


class _RecordingFeed:
    """VenueFeed spy: records every `(start, end, lookahead)` triple it sees."""

    def __init__(self) -> None:
        self.calls: list[tuple[datetime, datetime, int]] = []

    def _empty_frame(self) -> pl.DataFrame:
        return pl.DataFrame(schema={
            'time': pl.Datetime(time_zone='UTC'),
            'price': pl.Float64,
            'qty': pl.Float64,
            'trade_id': pl.Int64,
        })

    def get_trades(
        self, symbol: str, start: datetime, end: datetime,
    ) -> pl.DataFrame:
        del symbol, start, end
        return self._empty_frame()

    def get_trades_for_venue(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        venue_lookahead_seconds: int,
    ) -> pl.DataFrame:
        del symbol
        self.calls.append((start, end, venue_lookahead_seconds))
        return self._empty_frame()


_TRADE_WINDOW = 14_400  # 4h, matches r0014 kline_size


def _submit_market_buy(
    adapter: SimulatedVenueAdapter, account_id: str = 'acct-1',
) -> object:
    adapter.register_account(account_id, 'k', 's')
    return asyncio.run(adapter.submit_order(
        account_id, 'BTCUSDT', OrderSide.BUY,
        OrderType.MARKET,
        Decimal('0.001'),
    ))


def test_no_clamp_default_uses_full_trade_window() -> None:
    """`window_end_clamp=None` (default): walk endpoint = submit_time + trade_window_seconds."""
    feed = _RecordingFeed()
    adapter = SimulatedVenueAdapter(
        feed=cast(VenueFeed, feed),
        filters=BinanceSpotFilters.binance_spot('BTCUSDT'),
        fees=FeeSchedule(),
        trade_window_seconds=_TRADE_WINDOW,
        # window_end_clamp deliberately unset — legacy behaviour.
    )
    _submit_market_buy(adapter)
    assert len(feed.calls) == 1, (
        f'expected one get_trades_for_venue call, got '
        f'{len(feed.calls)}'
    )
    start, end, lookahead = feed.calls[0]
    # Walk endpoint = start + trade_window_seconds when clamp is None.
    # `start` here is the slippage/maker fetch_start (= submit_time
    # adjusted for calibration history). The endpoint we care about
    # is `end`; assert lookahead == trade_window_seconds.
    del start, end
    assert lookahead == _TRADE_WINDOW, (
        f'unclamped path: venue_lookahead_seconds must equal '
        f'trade_window_seconds={_TRADE_WINDOW}, got {lookahead}'
    )


def test_clamp_at_or_after_unbounded_endpoint_is_no_op() -> None:
    """`window_end_clamp` later than `submit_time + trade_window` leaves the walk unbounded."""
    feed = _RecordingFeed()
    far_future = datetime(2030, 1, 1, tzinfo=UTC)
    adapter = SimulatedVenueAdapter(
        feed=cast(VenueFeed, feed),
        filters=BinanceSpotFilters.binance_spot('BTCUSDT'),
        fees=FeeSchedule(),
        trade_window_seconds=_TRADE_WINDOW,
        window_end_clamp=far_future,
    )
    _submit_market_buy(adapter)
    _, _, lookahead = feed.calls[0]
    assert lookahead == _TRADE_WINDOW, (
        f'clamp far in the future must be a no-op; expected '
        f'lookahead={_TRADE_WINDOW}, got {lookahead}'
    )


def test_clamp_before_unbounded_endpoint_shrinks_lookahead() -> None:
    """A clamp 600s after submit_time shrinks the walk to 600s of tape.

    Models the operator-relevant case: a SELL submitted near
    `window_end - 600s` with a 4h `trade_window_seconds`. Without
    the clamp we would peek ~3.83h past the window. With the clamp
    we get a 600-second walk of in-window tape only.
    """
    feed = _RecordingFeed()
    # Pin the simulated wall clock so submit_time is deterministic.
    # `register_account` + `submit_order` use `datetime.now(UTC)`;
    # use freezegun if available, else just compute the expected
    # bound from the actual now and assert >= a sensible floor.
    # For a deterministic check, set the clamp via a delta from
    # the soon-to-be-recorded submit_time and assert the lookahead
    # in the recorded call.
    target_lookahead = 600  # seconds
    # Pre-compute a clamp that's `target_lookahead` past now-ish.
    # The actual submit_time will be a few microseconds later, so
    # the recorded lookahead will be slightly less than 600. Tolerate
    # +/- 5 seconds.
    clamp = datetime.now(UTC) + timedelta(seconds=target_lookahead)
    adapter = SimulatedVenueAdapter(
        feed=cast(VenueFeed, feed),
        filters=BinanceSpotFilters.binance_spot('BTCUSDT'),
        fees=FeeSchedule(),
        trade_window_seconds=_TRADE_WINDOW,
        window_end_clamp=clamp,
    )
    _submit_market_buy(adapter)
    _, end, lookahead = feed.calls[0]
    assert lookahead < _TRADE_WINDOW, (
        f'clamp before unbounded endpoint must shrink the '
        f'lookahead; expected < {_TRADE_WINDOW}, got {lookahead}'
    )
    assert abs(lookahead - target_lookahead) < 5, (
        f'clamped lookahead should be ~{target_lookahead}s; got '
        f'{lookahead}'
    )
    # The walk endpoint must equal the clamp.
    assert end == clamp, (
        f'walk endpoint should equal window_end_clamp ({clamp}); '
        f'got {end}'
    )
