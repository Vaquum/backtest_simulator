"""Integration gate: SimulatedVenueAdapter measures realised slippage.

Slice #17 Task 12 wiring (audit-corrected).

The earlier wiring multiplied `f.fill_price` by `(1 + bps/10000)` on
top of `walk_trades`'s already-tape-priced fill — double-counting
the spread effect (audit P1 #1). This wiring uses `walk_trades`'s
realistic taker prices unchanged and *measures* per-fill slippage
as `(fill_price - rolling_mid) / rolling_mid * 10000` over the
same `dt_seconds` lookback the calibration used.

The contract:
  - `slippage_realised_aggregate_bps` is the signed mean of
    `(fill_price - mid) / mid * 10000` across recorded taker fills:
    positive when the average fill landed above mid, negative when
    below. NOT a cost metric — a SELL above mid produces positive
    bps but is improvement, not paid spread. Side interpretation
    belongs to `cost_bps`.
  - `slippage_realised_cost_bps` is the side-normalized mean —
    operator-visible cost. Positive = run paid spread on
    average; negative = run captured price improvement on
    average. The earlier `mean(|bps|)` design was rejected by
    the auditor for counting favorable fills as cost; the
    `cost_bps` correction normalizes by side (BUY: cost=+bps,
    SELL: cost=-bps) so the sign is honest.
  - `slippage_realised_buy_bps` and `slippage_realised_sell_bps`
    isolate per-side means.
  - `slippage_realised_n_excluded` separates measured-zero from
    no-mid-available (audit P1 #2 in measurement form).
  - The four bps aggregates (signed, cost, buy, sell) return None
    when no model attached (feature-disabled signal); the two
    counters (n_samples, n_excluded) stay integer-typed and read
    0 when no model attached.
  - `fill_price` is byte-identical with vs without model attached
    — the model only measures, never adjusts (audit P1 #1 fix).
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import polars as pl
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
    """Minimal VenueFeed-shape that serves a slice of real BTCUSDT trades."""

    def __init__(self) -> None:
        df = pl.read_parquet(_FIXTURE)
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
    pre = pre.rename({'time': 'datetime', 'qty': 'quantity'})
    return SlippageModel.calibrate(
        trades=pre,
        side_buckets=(
            Decimal('0.001'), Decimal('0.01'),
            Decimal('0.1'), Decimal('1.0'),
        ),
        dt_seconds=10,
    )


def _make_adapter(
    feed: _TapeFeed,
    slippage: SlippageModel | None,
    submit_t: datetime,
) -> SimulatedVenueAdapter:
    adapter = SimulatedVenueAdapter(
        feed=feed,
        filters=BinanceSpotFilters.binance_spot('BTCUSDT'),
        fees=FeeSchedule(),
        trade_window_seconds=60,
        slippage_model=slippage,
    )
    adapter.register_account('a', 'k', 's')
    adapter._now = lambda: submit_t  # type: ignore[method-assign]
    return adapter


def test_slippage_measurement_does_not_change_fill_price() -> None:
    """Audit P1 #1: walk_trades's tape price stands; slippage layer measures only.

    Submitting the same BUY MARKET order against the same trade
    window via two adapters — one with a calibrated SlippageModel,
    one without — must produce byte-identical fills. The earlier
    wiring multiplied `fill_price * (1 + bps/10000)` on top of the
    already-tape-priced fill; that double-counted spread and
    over-charged the strategy. Honest fill price comes from the
    tape walk; slippage exists at this layer to *measure* the
    realised deviation from mid for operator reporting, not to
    pile additional cost on the simulator.
    """
    feed = _TapeFeed()
    submit_t = datetime(2024, 1, 1, 12, 15, tzinfo=UTC)
    model = _calibrate(feed, submit_t)
    a_off = _make_adapter(feed, None, submit_t)
    a_on = _make_adapter(feed, model, submit_t)

    qty = Decimal('0.5')
    res_off = asyncio.run(a_off.submit_order(
        'a', 'BTCUSDT', OrderSide.BUY, OrderType.MARKET, qty,
    ))
    res_on = asyncio.run(a_on.submit_order(
        'a', 'BTCUSDT', OrderSide.BUY, OrderType.MARKET, qty,
    ))

    assert res_off.immediate_fills, 'baseline adapter must produce a fill'
    assert res_on.immediate_fills, 'measurement-wired adapter must produce a fill'
    fill_off = res_off.immediate_fills[0].price
    fill_on = res_on.immediate_fills[0].price
    assert fill_on == fill_off, (
        f'fill_price must be identical with vs without slippage model — '
        f'measurement layer must not adjust the tape price. '
        f'got with={fill_on} without={fill_off}. The audit P1 #1 fix '
        f'has regressed.'
    )


def test_slippage_records_realised_bps_when_model_attached() -> None:
    """A taker fill produces one measurement; aggregates surface it."""
    feed = _TapeFeed()
    submit_t = datetime(2024, 1, 1, 12, 15, tzinfo=UTC)
    model = _calibrate(feed, submit_t)
    adapter = _make_adapter(feed, model, submit_t)

    asyncio.run(adapter.submit_order(
        'a', 'BTCUSDT', OrderSide.BUY, OrderType.MARKET, Decimal('0.05'),
    ))

    assert adapter.slippage_realised_n_samples == 1, (
        f'one taker fill should produce one slippage measurement; got '
        f'{adapter.slippage_realised_n_samples}'
    )
    signed = adapter.slippage_realised_aggregate_bps
    cost = adapter.slippage_realised_cost_bps
    buy_only = adapter.slippage_realised_buy_bps
    sell_only = adapter.slippage_realised_sell_bps
    assert signed is not None
    assert cost is not None
    assert buy_only is not None
    assert sell_only is not None  # zero-sample side returns 0, not None
    # Single BUY fill: per-BUY mean equals the overall signed mean,
    # and `cost = +bps` (BUY normalization). The cost and signed
    # match in sign on a one-sided population — the difference shows
    # up on round trips, exercised in the dedicated test below.
    assert buy_only == signed
    assert cost == signed, (
        f'on a BUY-only population, cost (=+bps) equals signed mean; '
        f'got cost={cost} signed={signed}'
    )
    # No SELL fills: per-SELL mean is zero (the property's "no
    # samples" branch).
    assert sell_only == Decimal('0')


def test_slippage_aggregates_are_none_without_model() -> None:
    """Audit P1 #1: feature-disabled returns None across all aggregates."""
    feed = _TapeFeed()
    submit_t = datetime(2024, 1, 1, 12, 15, tzinfo=UTC)
    adapter = _make_adapter(feed, None, submit_t)
    asyncio.run(adapter.submit_order(
        'a', 'BTCUSDT', OrderSide.BUY, OrderType.MARKET, Decimal('0.05'),
    ))
    assert adapter.slippage_realised_aggregate_bps is None
    assert adapter.slippage_realised_cost_bps is None
    assert adapter.slippage_realised_buy_bps is None
    assert adapter.slippage_realised_sell_bps is None
    assert adapter.slippage_realised_n_samples == 0
    assert adapter.slippage_realised_n_excluded == 0


def test_slippage_cost_is_side_normalized_paid_spread_round_trip() -> None:
    """Audit P1 #3: signed/cost/per-side aggregates report what they claim.

    Construction: a BUY leg pays +5 bps (above mid → cost +5),
    and a SELL leg pays -5 bps (below mid → cost +5 because for
    SELL aggressors `cost = -bps`). The signed mean cancels to 0;
    the side-normalized cost is +5 (both legs paid spread).

    A pre-fix implementation that reported only the signed mean
    would tell the operator "0 bps slippage" on this round trip
    — exactly the audit's P1 #3 failure mode. `cost_bps` is the
    side-normalized correction.
    """
    feed = _TapeFeed()
    submit_t = datetime(2024, 1, 1, 12, 15, tzinfo=UTC)
    model = _calibrate(feed, submit_t)
    adapter = _make_adapter(feed, model, submit_t)

    # Inject the +5 BUY / -5 SELL distribution directly. Real-tape
    # walking is symmetric for BUY/SELL on `walk_trades` (both
    # sides consume the same head-of-queue trades), so a tape-
    # driven test cannot demonstrate the cost-vs-signed split.
    from nexus.core.domain.enums import OrderSide as NexusOrderSide
    adapter._slippage_realised_bps = [Decimal('5'), Decimal('-5')]
    adapter._slippage_realised_sides = [
        NexusOrderSide.BUY, NexusOrderSide.SELL,
    ]

    signed = adapter.slippage_realised_aggregate_bps
    cost = adapter.slippage_realised_cost_bps
    buy_bps = adapter.slippage_realised_buy_bps
    sell_bps = adapter.slippage_realised_sell_bps

    assert signed == Decimal('0'), (
        f'signed mean of +5 / -5 must cancel to 0; got {signed}'
    )
    assert cost == Decimal('5'), (
        f'side-normalized cost: BUY +5 + SELL -(-5) = +5+5; mean = +5; '
        f'got {cost}. A signed-only aggregator would report 0 here.'
    )
    assert buy_bps == Decimal('5'), (
        f'BUY-only mean must equal +5; got {buy_bps}'
    )
    assert sell_bps == Decimal('-5'), (
        f'SELL-only mean must equal -5; got {sell_bps}'
    )


def test_slippage_calibration_loop_predicts_and_reports_gap() -> None:
    """Audit gap D: calibration is answerable, not just observable.

    The model's `apply()` produces a PREDICTED bps; the adapter
    measures the REALISED bps; the difference is the calibration
    error signal. A run with realised cost 5 bps and predicted
    cost 3 bps has a gap of +2 bps — the operator sees the
    calibration is too optimistic by 2 bps.

    Synthetic injection of paired predictions/realisations covers
    the aggregator math without depending on tape symmetry.
    Construction:
      - Fill 0: realised=+5, predicted=+3, side=BUY → cost gap = +5-+3 = +2.
      - Fill 1: realised=-2, predicted=-1, side=SELL → cost gap = (-(-2)) - (-(-1)) = 2 - 1 = +1.
    Mean gap = +1.5 (calibration too optimistic).
    """
    from nexus.core.domain.enums import OrderSide as NexusOrderSide
    feed = _TapeFeed()
    submit_t = datetime(2024, 1, 1, 12, 15, tzinfo=UTC)
    model = _calibrate(feed, submit_t)
    adapter = _make_adapter(feed, model, submit_t)

    adapter._slippage_realised_bps = [Decimal('5'), Decimal('-2')]
    adapter._slippage_realised_sides = [
        NexusOrderSide.BUY, NexusOrderSide.SELL,
    ]
    adapter._slippage_predicted_bps = [Decimal('3'), Decimal('-1')]

    gap = adapter.slippage_predict_vs_realised_gap_bps
    assert gap == Decimal('1.5'), (
        f'predict-vs-realised gap = mean(realised_cost - predicted_cost) '
        f'= mean(+2, +1) = +1.5; got {gap}. The calibration loop is '
        f'broken — operator cannot tell if calibration matches reality.'
    )
    predicted_cost = adapter.slippage_predicted_cost_bps
    assert predicted_cost == Decimal('2'), (
        f'predicted cost: BUY +3 + SELL -(-1) = +3+1=+4; mean = +2; '
        f'got {predicted_cost}'
    )
    realised_cost = adapter.slippage_realised_cost_bps
    assert realised_cost == Decimal('3.5'), (
        f'realised cost: BUY +5 + SELL -(-2) = +5+2=+7; mean = +3.5; '
        f'got {realised_cost}'
    )
    assert realised_cost - predicted_cost == gap, (
        f'gap should equal realised_cost - predicted_cost; got '
        f'gap={gap} but realised-predicted={realised_cost - predicted_cost}'
    )


def test_slippage_uncalibrated_predict_separated_from_excluded_via_live_path() -> None:
    """A real fill against an uncalibrated bucket raises in apply() and increments n_uncalibrated_predict.

    The earlier version of this test mutated the counter
    directly. Auditor flagged that as not a real integration
    test — it pinned the property's read shape but not the
    actual venue → SlippageModel.apply() interaction. This
    version drives the live path:

      - Build a SlippageModel with ONE bucket on BUY side
        (qty_min=0.5, qty_max=None) and an empty SELL side.
      - Submit a BUY MARKET order with qty=0.05 — below the
        bucket's qty_min, so SlippageModel.apply() raises
        ValueError when the adapter calls it for the predicted
        bps. The realised measurement still succeeds (mid is
        available from the synthetic pre-submit prints), so
        n_samples increments AND n_uncalibrated_predict
        increments — the two counters track distinct failure
        modes.
    """
    from nexus.core.domain.enums import OrderSide as NexusOrderSide

    from backtest_simulator.honesty.slippage import (
        SlippageBucket,
        SlippageModel,
    )

    submit_t = datetime(2024, 1, 1, 12, 0, 30, tzinfo=UTC)

    class _SyntheticFeed:
        def __init__(self) -> None:
            rows = [
                # Pre-submit: gives realised measurement a mid.
                (submit_t - timedelta(seconds=5), 100.0, 0.5, 0, 1),
                (submit_t - timedelta(seconds=3), 100.05, 0.5, 0, 2),
                (submit_t - timedelta(seconds=1), 100.10, 0.5, 0, 3),
                # Post-submit: a fill at the order's qty.
                (submit_t + timedelta(seconds=1), 100.20, 1.0, 0, 4),
            ]
            self.df = pl.DataFrame(
                rows,
                schema={
                    'time': pl.Datetime('us', 'UTC'),
                    'price': pl.Float64,
                    'qty': pl.Float64,
                    'is_buyer_maker': pl.UInt8,
                    'trade_id': pl.UInt64,
                },
                orient='row',
            )

        def get_trades(
            self, symbol: str, start: datetime, end: datetime,
        ) -> pl.DataFrame:
            del symbol
            return self.df.filter(
                (pl.col('time') >= start) & (pl.col('time') <= end),
            )

        def _get_trades_for_venue(
            self, symbol: str, start: datetime, end: datetime,
            *, venue_lookahead_seconds: int,
        ) -> pl.DataFrame:
            del symbol, venue_lookahead_seconds
            return self.df.filter(
                (pl.col('time') >= start) & (pl.col('time') <= end),
            )

    # Hand-built model: ONE BUY bucket starting at qty_min=0.5.
    # An order with qty=0.05 is BELOW that, so apply() raises
    # ValueError on the predicted lookup. The realised mid is
    # still computable from the pre-submit prints.
    model = SlippageModel(
        _buckets_by_side={
            NexusOrderSide.BUY: (
                SlippageBucket(
                    side=NexusOrderSide.BUY,
                    qty_min=Decimal('0.5'),
                    qty_max=None,
                    median_bps=Decimal('1.0'),
                    n_samples=100,
                ),
            ),
            NexusOrderSide.SELL: (),
        },
        _dt_seconds=10,
    )
    feed = _SyntheticFeed()
    adapter = SimulatedVenueAdapter(
        feed=feed,
        filters=BinanceSpotFilters.binance_spot('BTCUSDT'),
        fees=FeeSchedule(),
        trade_window_seconds=60,
        slippage_model=model,
    )
    adapter.register_account('a', 'k', 's')
    adapter._now = lambda: submit_t  # type: ignore[method-assign]

    res = asyncio.run(adapter.submit_order(
        'a', 'BTCUSDT', OrderSide.BUY, OrderType.MARKET, Decimal('0.05'),
    ))
    assert res.immediate_fills, 'synthetic tape must produce a fill'

    # Realised side succeeded — preceding mid was available.
    assert adapter.slippage_realised_n_samples == 1, (
        f'realised measurement should succeed (pre-submit mid '
        f'available); got n_samples='
        f'{adapter.slippage_realised_n_samples}'
    )
    assert adapter.slippage_realised_n_excluded == 0, (
        f'realised measurement should NOT be in n_excluded; got '
        f'n_excluded={adapter.slippage_realised_n_excluded}'
    )

    # Predicted side failed: qty=0.05 is below the only BUY
    # bucket's qty_min=0.5. The counter increments distinct from
    # n_excluded.
    assert adapter.slippage_n_uncalibrated_predict == 1, (
        f'predicted apply() must raise on qty=0.05 vs bucket '
        f'qty_min=0.5; got n_uncalibrated_predict='
        f'{adapter.slippage_n_uncalibrated_predict}'
    )

    # n_predicted_samples is the gap denominator. With one
    # uncalibrated fill, no gap can be computed.
    assert adapter.slippage_n_predicted_samples == 0, (
        f'no successful predictions = no gap denominator; got '
        f'n_predicted_samples={adapter.slippage_n_predicted_samples}'
    )

    # Predicted aggregate and gap return Decimal(0) here (model
    # attached, but no successful predictions). The OUTPUT layer
    # is responsible for distinguishing "0 because matched" from
    # "0 because no signal" via n_predicted_samples — see
    # `_metrics.print_run` and the sweep summary.
    assert adapter.slippage_predicted_cost_bps == Decimal('0')
    assert adapter.slippage_predict_vs_realised_gap_bps == Decimal('0')


def test_slippage_cost_is_negative_when_run_captures_price_improvement() -> None:
    """Audit P1 #2: favorable fills must show as NEGATIVE cost, not positive.

    The earlier `adverse_bps = mean(|bps|)` was wrong: a BUY filling
    BELOW mid is FAVORABLE (price improvement), not cost. The
    side-normalized cost: BUY cost = +bps, SELL cost = -bps. So
    a BUY at -3 bps has cost = -3 (improvement); a SELL at +2 bps
    has cost = -2 (improvement). Mean cost = -2.5 (improvement).

    Pre-fix `adverse = mean(|-3|, |2|) = 2.5`, the wrong sign — it
    would tell the operator "you paid 2.5 bps" when the truth was
    "you saved 2.5 bps".
    """
    feed = _TapeFeed()
    submit_t = datetime(2024, 1, 1, 12, 15, tzinfo=UTC)
    model = _calibrate(feed, submit_t)
    adapter = _make_adapter(feed, model, submit_t)

    from nexus.core.domain.enums import OrderSide as NexusOrderSide
    # BUY at -3 bps (favorable: filled below mid).
    # SELL at +2 bps (favorable: received above mid).
    adapter._slippage_realised_bps = [Decimal('-3'), Decimal('2')]
    adapter._slippage_realised_sides = [
        NexusOrderSide.BUY, NexusOrderSide.SELL,
    ]
    cost = adapter.slippage_realised_cost_bps
    assert cost is not None
    expected_cost = Decimal('-2.5')  # ((-3) + (-2)) / 2
    assert cost == expected_cost, (
        f'two favorable fills (BUY -3 bps, SELL +2 bps) must mean '
        f'NEGATIVE cost (price improvement). Expected '
        f'{expected_cost}, got {cost}. The earlier mean(|bps|) '
        f'would report +2.5 — exactly the audit P1 #2 failure mode.'
    )
    assert cost < Decimal('0'), (
        f'cost on a favorable run must be negative; got {cost}'
    )


def test_slippage_first_post_submit_fill_is_measured_not_excluded() -> None:
    """Codex round-2 pin: pre-submit tape reaches the rolling-mid window.

    Strictly pins the fix using a SYNTHETIC tape. The real fixture
    is too dense to discriminate (multiple post-submit prints can
    seed the rolling mid even under the broken fetch). Synthetic
    layout:
      - Pre-submit prints at t-5s, t-3s, t-1s (inside dt_seconds=10).
      - Exactly ONE post-submit print at t+1s with enough qty.
      - Fix-on:  rolling-mid for t+1s fill sees pre-submit prints
                 → n_samples=1, n_excluded=0.
      - Fix-off: pre-submit prints excluded from the fetched frame
                 → n_excluded=1, n_samples=0.
    """

    class _SyntheticFeed:
        def __init__(self, submit_t: datetime) -> None:
            rows = [
                (submit_t - timedelta(seconds=5), 100.0, 0.5, 0, 1),
                (submit_t - timedelta(seconds=3), 100.05, 0.5, 0, 2),
                (submit_t - timedelta(seconds=1), 100.10, 0.5, 0, 3),
                (submit_t + timedelta(seconds=1), 100.20, 1.0, 0, 4),
            ]
            self.df = pl.DataFrame(
                rows,
                schema={
                    'time': pl.Datetime('us', 'UTC'),
                    'price': pl.Float64,
                    'qty': pl.Float64,
                    'is_buyer_maker': pl.UInt8,
                    'trade_id': pl.UInt64,
                },
                orient='row',
            )

        def get_trades(
            self, symbol: str, start: datetime, end: datetime,
        ) -> pl.DataFrame:
            del symbol
            return self.df.filter(
                (pl.col('time') >= start) & (pl.col('time') <= end),
            )

        def _get_trades_for_venue(
            self, symbol: str, start: datetime, end: datetime,
            *, venue_lookahead_seconds: int,
        ) -> pl.DataFrame:
            del symbol, venue_lookahead_seconds
            return self.df.filter(
                (pl.col('time') >= start) & (pl.col('time') <= end),
            )

    submit_t = datetime(2024, 1, 1, 12, 0, 30, tzinfo=UTC)
    feed = _SyntheticFeed(submit_t)
    cal_trades = feed.get_trades(
        'BTCUSDT',
        submit_t - timedelta(minutes=1),
        submit_t + timedelta(minutes=1),
    ).rename({'time': 'datetime', 'qty': 'quantity'})
    model = SlippageModel.calibrate(
        trades=cal_trades,
        side_buckets=(Decimal('0.001'), Decimal('0.01'), Decimal('1.0')),
        dt_seconds=10,
    )
    adapter = SimulatedVenueAdapter(
        feed=feed,
        filters=BinanceSpotFilters.binance_spot('BTCUSDT'),
        fees=FeeSchedule(),
        trade_window_seconds=60,
        slippage_model=model,
    )
    adapter.register_account('a', 'k', 's')
    adapter._now = lambda: submit_t  # type: ignore[method-assign]
    res = asyncio.run(adapter.submit_order(
        'a', 'BTCUSDT', OrderSide.BUY, OrderType.MARKET, Decimal('0.5'),
    ))
    assert res.immediate_fills, 'synthetic tape must produce a fill'
    assert adapter.slippage_realised_n_samples == 1, (
        f'exactly one post-submit fill must produce one measurement; '
        f'got n_samples={adapter.slippage_realised_n_samples} '
        f'n_excluded={adapter.slippage_realised_n_excluded}. The '
        f'pre-submit prefix is not reaching _record_slippage — the '
        f'fetch-start widening regressed.'
    )
    assert adapter.slippage_realised_n_excluded == 0, (
        f'no fill should land in n_excluded — three pre-submit prints '
        f'are inside dt_seconds. Got '
        f'n_excluded={adapter.slippage_realised_n_excluded}'
    )


def test_slippage_n_excluded_separates_no_mid_from_measured_zero() -> None:
    """Audit P1 #2 + codex r3 pin: no-preceding-mid → n_excluded, not n_samples.

    Strict synthetic-tape pin. Layout: ZERO pre-submit prints,
    ONE post-submit print at t+1s. The fill at t+1s has no tape
    inside the rolling-mid window `[t+1s - dt_seconds, t+1s)` —
    the adapter must increment n_excluded by 1 and leave
    n_samples at 0. The earlier real-fixture version of this
    test was vacuous (the fixture's earliest trade is 12:00, the
    submit at 11:30 produces no fills at all, and the test
    returned without proving anything). Codex r3 caught it.
    """

    class _SyntheticFeed:
        def __init__(self, submit_t: datetime) -> None:
            rows = [
                # No pre-submit prints — the rolling-mid window
                # for the post-submit fill must be empty.
                # ONE post-submit print, large enough to fill.
                (submit_t + timedelta(seconds=1), 100.0, 1.0, 0, 1),
            ]
            self.df = pl.DataFrame(
                rows,
                schema={
                    'time': pl.Datetime('us', 'UTC'),
                    'price': pl.Float64,
                    'qty': pl.Float64,
                    'is_buyer_maker': pl.UInt8,
                    'trade_id': pl.UInt64,
                },
                orient='row',
            )

        def get_trades(
            self, symbol: str, start: datetime, end: datetime,
        ) -> pl.DataFrame:
            del symbol
            return self.df.filter(
                (pl.col('time') >= start) & (pl.col('time') <= end),
            )

        def _get_trades_for_venue(
            self, symbol: str, start: datetime, end: datetime,
            *, venue_lookahead_seconds: int,
        ) -> pl.DataFrame:
            del symbol, venue_lookahead_seconds
            return self.df.filter(
                (pl.col('time') >= start) & (pl.col('time') <= end),
            )

    submit_t = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    feed = _SyntheticFeed(submit_t)
    # Pre-built model with the same dt_seconds the test relies on;
    # bypassing SlippageModel.calibrate keeps the test independent
    # of the calibration's own coverage requirements.
    model = SlippageModel(_dt_seconds=10)
    adapter = SimulatedVenueAdapter(
        feed=feed,
        filters=BinanceSpotFilters.binance_spot('BTCUSDT'),
        fees=FeeSchedule(),
        trade_window_seconds=60,
        slippage_model=model,
    )
    adapter.register_account('a', 'k', 's')
    adapter._now = lambda: submit_t  # type: ignore[method-assign]

    res = asyncio.run(adapter.submit_order(
        'a', 'BTCUSDT', OrderSide.BUY, OrderType.MARKET, Decimal('0.5'),
    ))
    assert res.immediate_fills, (
        'synthetic tape must produce a fill so the n_excluded path '
        'is actually exercised'
    )
    assert adapter.slippage_realised_n_samples == 0, (
        f'a fill with no preceding mid must NOT increment n_samples; '
        f'got n_samples={adapter.slippage_realised_n_samples}'
    )
    assert adapter.slippage_realised_n_excluded == 1, (
        f'a fill with no preceding mid must increment n_excluded; '
        f'got n_excluded={adapter.slippage_realised_n_excluded}'
    )
    # And the aggregates: with samples=0 and a model attached, the
    # measurement-active aggregator returns Decimal(0) rather than
    # None. None is reserved for "no model attached" only. Pin all
    # four bps aggregate properties (signed, cost, buy, sell);
    # n_samples / n_excluded are integer counters, not in this
    # set. Without the per-side checks the contract would let
    # buy_bps/sell_bps regress to None on the zero-measurement-
    # with-model state. Codex r4 caught this gap.
    assert adapter.slippage_realised_aggregate_bps == Decimal('0')
    assert adapter.slippage_realised_cost_bps == Decimal('0')
    assert adapter.slippage_realised_buy_bps == Decimal('0'), (
        f'BUY-side aggregate must return Decimal(0) (model attached, '
        f'no measurements yet) — None is reserved for "no model '
        f'attached". Got {adapter.slippage_realised_buy_bps}'
    )
    assert adapter.slippage_realised_sell_bps == Decimal('0'), (
        f'SELL-side aggregate must return Decimal(0) on the zero-'
        f'measurement-with-model state; got '
        f'{adapter.slippage_realised_sell_bps}'
    )
