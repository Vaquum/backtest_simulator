"""SimulatedVenueAdapter — real Praxis VenueAdapter Protocol against historical trades."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal

import polars as pl
from nexus.core.domain.enums import OrderSide as NexusOrderSide
from praxis.core.domain.enums import OrderSide, OrderStatus, OrderType
from praxis.core.domain.health_snapshot import HealthSnapshot
from praxis.infrastructure.venue_adapter import (
    BalanceEntry,
    CancelResult,
    ExecutionReport,
    NotFoundError,
    OrderBookLevel,
    OrderBookSnapshot,
    SubmitResult,
    VenueOrder,
    VenueTrade,
)
from praxis.infrastructure.venue_adapter import (
    SymbolFilters as PraxisSymbolFilters,
)

from backtest_simulator.feed.protocol import VenueFeed
from backtest_simulator.honesty.maker_fill import MakerFillModel
from backtest_simulator.honesty.slippage import SlippageModel
from backtest_simulator.venue import _adapter_internals as _I
from backtest_simulator.venue.fees import FeeSchedule
from backtest_simulator.venue.fills import walk_trades
from backtest_simulator.venue.filters import BinanceSpotFilters
from backtest_simulator.venue.types import FillModelConfig, FillResult, PendingOrder

_BPS = Decimal('10000')


class SimulatedVenueAdapter:
    """VenueAdapter Protocol implementation backed by historical trades."""

    def __init__(
        self,
        feed: VenueFeed,
        filters: BinanceSpotFilters,
        fees: FeeSchedule,
        fill_config: FillModelConfig | None = None,
        trade_window_seconds: int = 3600,
        slippage_model: SlippageModel | None = None,
        maker_fill_model: MakerFillModel | None = None,
        market_impact_bucket_minutes: int | None = None,
        market_impact_threshold_fraction: Decimal = Decimal('0.1'),
        strict_impact_policy: bool = False,
    ) -> None:
        self._feed = feed
        self._filters = filters
        self._fees = fees
        self._fill_config = fill_config or FillModelConfig()
        self._trade_window_seconds = trade_window_seconds
        # `slippage_model` is calibrated externally (`SlippageModel.calibrate`
        # over a pre-window trade slice) and supplied by the operator
        # — usually `cli/_run_window.run_window_in_process`. The
        # adapter does NOT adjust `fill_price` based on this model:
        # `walk_trades` already prices fills against the actual
        # historical trade prints, which IS the live taker reality.
        # The model contributes its `dt_seconds` (rolling-mid window)
        # to a per-fill *measurement* of realised bps — where the
        # fill landed relative to mid (fill-price deviation, sign-
        # neutral; cost / improvement interpretation belongs to the
        # `slippage_realised_cost_bps` aggregate). For operator-
        # visible reporting
        # (signed mean, side-normalized cost, per-side aggregates).
        # When None, the measurement layer is off and aggregate
        # properties return None as the "feature disabled" signal.
        self._slippage_model = slippage_model
        self._slippage_realised_bps: list[Decimal] = []
        self._slippage_realised_sides: list[NexusOrderSide] = []
        # Calibration-loop telemetry: for each measured taker fill,
        # the model's `apply()` is queried for the PREDICTED bps the
        # calibration says this (side, qty) should pay. The
        # `predict_vs_realised_gap` aggregate exposes whether the
        # calibration matches reality — if the gap is large, the
        # calibration window is wrong (too short, wrong volatility
        # regime, qty buckets miscalibrated). When `apply()` raises
        # ValueError (qty outside any calibrated bucket) the
        # corresponding entry is None and `n_uncalibrated_predict`
        # increments — distinct from `n_excluded` (no preceding mid).
        self._slippage_predicted_bps: list[Decimal | None] = []
        self._slippage_n_excluded: int = 0
        self._slippage_n_uncalibrated_predict: int = 0
        # MakerFillModel routing for LIMIT orders. None = legacy
        # O(1) "first crossing tick = full fill" path; set =
        # realistic queue-position + partial-fill engine. Adapter
        # also tracks LIMIT-order telemetry so `bts sweep` can
        # surface n_submitted / n_filled_full / n_partial / n_zero
        # / n_marketable_taker counts.
        self._maker_fill_model = maker_fill_model
        self._n_limit_submitted: int = 0
        self._n_limit_filled_full: int = 0
        self._n_limit_filled_partial: int = 0
        self._n_limit_filled_zero: int = 0
        self._n_limit_marketable_taker: int = 0
        self._maker_fill_efficiencies: list[Decimal] = []
        # MarketImpactModel — STRICT-CAUSAL per-submit estimate.
        # Each submit fetches a fresh `[submit_time -
        # bucket_minutes, submit_time)` slice of pre-submit
        # tape and delegates the qty-to-bps math to
        # `MarketImpactModel.evaluate_rolling`. Measurement
        # runs on every submit (BUY + SELL); the strict-policy
        # rejection scopes to BUY only (entry leg for the
        # long-only template — audit Finding 2 on fe00024).
        # `bucket_minutes is None` disables the feature.
        # `threshold_fraction` controls the size-vs-volume flag
        # threshold (default 10%). `strict_impact_policy=True`
        # makes the venue REJECT flagged BUY orders before
        # `walk_trades` runs — the auditor's "pre-fill gate"
        # semantic. Default `False` preserves the prior
        # measurement-only shape. Codex round 2 caught the
        # original wiring's lookahead (calibration spanned the
        # run window) and absent gate; the audit on fe00024
        # caught the gate's over-broad scope (would have
        # rejected SELL exits too).
        self._market_impact_bucket_minutes = market_impact_bucket_minutes
        self._market_impact_threshold_fraction = (
            market_impact_threshold_fraction
        )
        self._strict_impact_policy = strict_impact_policy
        self._market_impact_bps_samples: list[Decimal] = []
        self._market_impact_n_flagged: int = 0
        self._market_impact_n_uncalibrated: int = 0
        self._market_impact_n_rejected: int = 0
        self._accounts: dict[str, _I.Account] = {}
        self._symbol_filters: dict[str, BinanceSpotFilters] = {filters.symbol: filters}
        self._next_order_seq = 1
        self._next_trade_seq = 1
        self._history: dict[str, _I.Account] = {}

    def touch_for_symbol(self, symbol: str) -> Decimal | None:
        """Return the most recent pre-now trade price, or None if empty.

        The action_submitter's LIMIT-touch refresh hook reads this
        before validation so the strategy's `execution_params['price']`
        is set to the touch ± tick — keeping the touch decision in
        the action audit trail rather than venue-side. Codex round 4
        P2 caught the prior shape (rewrite hidden inside `submit_order`).
        """
        from datetime import timedelta
        now = self._now()
        trades = self._feed._get_trades_for_venue(
            symbol, now - timedelta(minutes=1), now,
            venue_lookahead_seconds=0,
        )
        if trades.is_empty():
            return None
        return Decimal(str(trades.tail(1)['price'].item()))

    def tick_for_symbol(self, symbol: str) -> Decimal:
        """Return the symbol's tick size, or raise if not registered."""
        filters = self._symbol_filters.get(symbol)
        if filters is None:
            msg = (
                f'tick_for_symbol: symbol {symbol!r} not registered; '
                f'call load_filters([{symbol!r}]) first.'
            )
            raise KeyError(msg)
        return filters.tick_size

    def register_account(self, account_id: str, api_key: str, api_secret: str) -> None:
        # If the same account_id was registered + unregistered + re-registered,
        # recover the prior Account so fill/order history survives the cycle.
        # Praxis's shutdown path unregisters the account during normal
        # teardown; post-run inspection needs the trades to still exist.
        account = self._history.pop(account_id, None) or _I.Account(
            account_id=account_id, api_key=api_key, api_secret=api_secret,
        )
        self._accounts[account_id] = account

    def unregister_account(self, account_id: str) -> None:
        if account_id not in self._accounts:
            msg = f'account_id {account_id!r} not registered'
            raise KeyError(msg)
        self._history[account_id] = self._accounts.pop(account_id)

    def _record_slippage(
        self,
        order: PendingOrder,
        fills: list[FillResult],
        trades_window: pl.DataFrame,
    ) -> None:
        """Record realised slippage bps per taker fill — measure, do NOT adjust.

        The previous wiring multiplied `f.fill_price` by `(1 + bps/10000)`
        on top of `walk_trades`'s already-tape-priced fill, which
        double-counts the spread/drift effect (the audit's P1 #1).
        `walk_trades` returns realistic taker prices because it walks
        actual historical trade prints — that IS the price the strategy
        pays in live. Slippage here is observability only: for each
        taker fill, record the deviation from the rolling mid over
        `slippage_model.dt_seconds` preceding `fill_time`. Maker fills
        (`is_maker=True`) are SKIPPED in this method — they pin to the
        declared limit price and the cost convention there is a future
        feature; only taker fills contribute to the measurement
        aggregates today.

        The signed bps is stored alongside the side so the aggregator
        can report per-side and side-normalized cost means (cost =
        +bps for BUY, -bps for SELL). A plain signed mean would let a
        round trip cancel to zero even though the strategy paid spread
        on both legs (the audit's P1 #3); the cost view is the
        operator-visible cost / improvement metric.

        Fills whose preceding `dt_seconds` window has zero trades
        (start-of-tape, halt) are recorded under `n_excluded` rather
        than counted as zero — silent zeros would let an empty mid
        masquerade as "no slippage paid" (the audit's P1 #2 in its
        measurement form).

        When `slippage_model` is None this method is a no-op and the
        aggregate properties return None.
        """
        if self._slippage_model is None:
            return
        if trades_window.is_empty():
            self._slippage_n_excluded += len(
                [f for f in fills if not f.is_maker],
            )
            return
        from datetime import timedelta
        dt = timedelta(seconds=self._slippage_model.dt_seconds)
        for f in fills:
            if f.is_maker:
                # Maker fills land at limit — measuring against mid
                # is a future-tense feature (price improvement
                # reporting); for now skip to keep the signal scoped
                # to taker fills where the calibration semantics
                # match. Track separately if it's needed.
                continue
            window_start = f.fill_time - dt
            preceding = trades_window.filter(
                (pl.col('time') >= window_start)
                & (pl.col('time') < f.fill_time),
            )
            if preceding.is_empty():
                self._slippage_n_excluded += 1
                continue
            mid_value = preceding['price'].median()
            if mid_value is None or mid_value <= 0:
                self._slippage_n_excluded += 1
                continue
            mid = Decimal(str(mid_value))
            bps = (f.fill_price - mid) / mid * _BPS
            side = (
                NexusOrderSide.BUY if order.side == 'BUY'
                else NexusOrderSide.SELL
            )
            self._slippage_realised_bps.append(bps)
            self._slippage_realised_sides.append(side)
            # Calibration-loop step: ask the model what it would
            # predict for this (side, qty, mid, t). Mismatch
            # against the realised bps is the calibration error
            # signal. ValueError = qty bucket uncalibrated; record
            # None and bump the counter so the operator can tell
            # "calibration off-bucket" apart from "bucket says 0".
            try:
                predicted_bps = self._slippage_model.apply(
                    side=side, qty=f.fill_qty, mid=mid, t=f.fill_time,
                )
            except ValueError:
                self._slippage_predicted_bps.append(None)
                self._slippage_n_uncalibrated_predict += 1
            else:
                self._slippage_predicted_bps.append(predicted_bps)

    def _aggregate_bps_when_active(
        self,
        sample_filter: object = None,
    ) -> Decimal | None:
        """Mean of recorded bps under `sample_filter`; None when slippage is off."""
        if self._slippage_model is None:
            return None
        if sample_filter is None:
            samples = list(self._slippage_realised_bps)
        else:
            samples = [
                bps for bps, side in zip(
                    self._slippage_realised_bps,
                    self._slippage_realised_sides,
                    strict=True,
                )
                if sample_filter(bps, side)
            ]
        if not samples:
            return Decimal('0')
        return sum(samples, Decimal('0')) / Decimal(len(samples))

    @property
    def slippage_realised_aggregate_bps(self) -> Decimal | None:
        """Signed mean of `(fill_price - mid) / mid * 10000` across recorded taker fills.

        Positive when the average recorded fill landed above mid;
        negative when below. NOT a cost metric on its own — a SELL
        filling above mid produces positive bps but is price
        improvement, not paid spread. The side interpretation
        belongs to `slippage_realised_cost_bps` (cost = +bps for
        BUY, -bps for SELL). Pair the two: use signed for "where
        did fills land relative to mid?", cost for "what did the
        run pay?". None when no slippage model attached.
        """
        return self._aggregate_bps_when_active()

    @property
    def slippage_realised_cost_bps(self) -> Decimal | None:
        """Side-normalized realised slippage cost in bps.

        For each taker fill, the bps cost relative to mid is:
          - BUY aggressor:  cost =  bps   (paid above mid → positive cost)
          - SELL aggressor: cost = -bps   (received below mid → positive cost)

        Mean of these costs across recorded taker fills (maker
        fills are skipped — see `_record_slippage`). Sign convention:
          - Positive: the run paid spread on average.
          - Negative: the run captured price improvement on average.
          - Zero: no measurements yet (model attached) OR sample-
            mean cancels exactly.

        This replaces the earlier `adverse_bps = mean(|bps|)`, which
        was wrong: it counted favorable fills (BUY below mid, SELL
        above mid) as cost. The audit caught the math error;
        `cost_bps` is the side-normalized correction. None when no
        slippage model attached.
        """
        if self._slippage_model is None:
            return None
        if not self._slippage_realised_bps:
            return Decimal('0')
        cost_samples = [
            bps if side == NexusOrderSide.BUY else -bps
            for bps, side in zip(
                self._slippage_realised_bps,
                self._slippage_realised_sides,
                strict=True,
            )
        ]
        return sum(cost_samples, Decimal('0')) / Decimal(len(cost_samples))

    @property
    def slippage_realised_buy_bps(self) -> Decimal | None:
        """Mean realised bps over BUY-aggressor fills (positive = paid above mid)."""
        return self._aggregate_bps_when_active(
            lambda _bps, side: side == NexusOrderSide.BUY,
        )

    @property
    def slippage_realised_sell_bps(self) -> Decimal | None:
        """Mean realised bps over SELL-aggressor fills (negative = received below mid)."""
        return self._aggregate_bps_when_active(
            lambda _bps, side: side == NexusOrderSide.SELL,
        )

    @property
    def slippage_predicted_cost_bps(self) -> Decimal | None:
        """Side-normalized PREDICTED cost from the calibration's `apply()`.

        Parallel to `slippage_realised_cost_bps` but driven by what
        the model predicts rather than what was measured. Only
        fills where `apply()` succeeded contribute (uncalibrated
        buckets are tracked via `slippage_n_uncalibrated_predict`).
        Pair with `realised` to read the calibration loop:
        gap = realised - predicted (see `slippage_predict_vs_realised_gap_bps`).
        """
        if self._slippage_model is None:
            return None
        paired = [
            (pred, side)
            for pred, side in zip(
                self._slippage_predicted_bps,
                self._slippage_realised_sides,
                strict=True,
            )
            if pred is not None
        ]
        if not paired:
            return Decimal('0')
        cost_samples = [
            pred if side == NexusOrderSide.BUY else -pred
            for pred, side in paired
        ]
        return sum(cost_samples, Decimal('0')) / Decimal(len(cost_samples))

    @property
    def slippage_predict_vs_realised_gap_bps(self) -> Decimal | None:
        """Mean (realised_cost - predicted_cost) per fill where both available.

        The calibration loop's primary signal. Zero means the
        calibration matches reality on average; large positive
        means the run paid more than the calibration predicted
        (calibration is too optimistic); large negative means the
        run paid less than predicted (calibration too
        pessimistic). Either direction is a recalibration trigger.

        Only fills with both a realised measurement AND a
        successful `apply()` contribute. Fills excluded for either
        reason are tracked separately (`n_excluded`,
        `n_uncalibrated_predict`). None when no slippage model
        attached.
        """
        if self._slippage_model is None:
            return None
        gaps: list[Decimal] = []
        for realised, pred, side in zip(
            self._slippage_realised_bps,
            self._slippage_predicted_bps,
            self._slippage_realised_sides,
            strict=True,
        ):
            if pred is None:
                continue
            realised_cost = (
                realised if side == NexusOrderSide.BUY else -realised
            )
            predicted_cost = pred if side == NexusOrderSide.BUY else -pred
            gaps.append(realised_cost - predicted_cost)
        if not gaps:
            return Decimal('0')
        return sum(gaps, Decimal('0')) / Decimal(len(gaps))

    @property
    def slippage_n_uncalibrated_predict(self) -> int:
        """Fills where the calibration's `apply()` raised (qty out of bucket).

        Distinct from `n_excluded` (which counts realised-side
        failures: no preceding mid). A fill can be uncalibrated
        on the predict side AND still measured on the realised
        side — those entries contribute to realised aggregates
        but NOT to predicted/gap aggregates.
        """
        return self._slippage_n_uncalibrated_predict

    def _record_market_impact_pre_fill(
        self,
        order: PendingOrder,
        symbol: str,
        submit_time: datetime,
    ) -> bool:
        """STRICT-CAUSAL per-submit market-impact gate.

        Fetches `[submit_time - bucket_minutes, submit_time)` of
        tape — strictly pre-submit — and delegates the qty-to-bps
        math to `MarketImpactModel.evaluate_rolling`. The venue
        owns:

          1. The tape fetch.
          2. The strict-causal `time < submit_time` filter (the
             feed's `_get_trades_for_venue` may return an
             inclusive range depending on backend; this filter
             enforces the half-open contract).
          3. The column rename to the model's convention
             (`time`/`qty` → `datetime`/`quantity`).
          4. The decision-to-telemetry mapping (record bps,
             increment `n_flagged` / `n_rejected`, return True
             when `strict_impact_policy` rejects).

        The model owns the linear-interpolation math
        (`total_volume`, `price_range_bps`, `impact_bps`,
        `flag`). A `None` return from the model is the
        "uncalibrated" signal — empty slice, zero-volume, or
        non-positive first price. Distinct from a zero-impact
        decision (which only arises against a well-formed
        bucket); never recorded as a sample.

        Strict-policy gate scoping: the gate REJECTS only the
        ENTRY leg, where "entry" is `order.side == 'BUY'` for
        the long-only `long_on_signal` template that ships in
        this slice. SELL orders represent EXIT (closing the
        long position) and are NEVER rejected — rejecting an
        oversized exit would leave the strategy holding risk
        with no way out, and would diverge from paper/live
        semantics where the operator wants the exit to land.
        Measurement (impact_bps + flag aggregates) runs on
        BOTH sides — the operator still sees flagged SELL
        exits in `n_flagged`, just not in `n_rejected`. So
        `n_flagged - n_rejected` includes (a) flagged BUYs
        when `strict_impact_policy=False` and (b) flagged SELLs
        regardless of policy.

        Audit Finding 2 on commit fe00024 caught the prior
        shape: the gate rejected ANY flagged order. A flagged
        SELL exit was rejected, leaving the strategy long the
        position. Short-side strategies (BUY = exit, SELL =
        entry) are out of scope for this slice — when they are
        added, the entry-side identity must be plumbed through
        explicitly (action intent on the order or in the
        strategy template configuration); using `side` as a
        proxy for `entry` only holds for long-only.

        When the order is flagged AND the side is BUY AND
        `strict_impact_policy=True`, returns True so
        `submit_order` routes to REJECTED before `walk_trades`
        runs. Returns False otherwise.

        Why a rolling slice rather than the model's
        wall-clock-bucket `evaluate(t)`? The standalone
        `calibrate` truncates trades to wall-clock minute
        boundaries; a non-boundary submit (e.g. `12:31:15`)
        with a 1-minute window would have its `[submit - 1m,
        submit)` slice split across the 12:30 and 12:31
        buckets, and `evaluate` would match only the partial
        bucket containing `submit_time - 1µs`. The pre-fill
        estimate needs the FULL trailing minute as one bucket
        — exactly what `evaluate_rolling` provides.

        No-op when `market_impact_bucket_minutes is None`.
        """
        if self._market_impact_bucket_minutes is None:
            return False
        from datetime import timedelta

        from backtest_simulator.honesty.market_impact import (
            MarketImpactModel,
        )
        bucket = self._market_impact_bucket_minutes
        raw = self._feed._get_trades_for_venue(
            symbol, submit_time - timedelta(minutes=bucket),
            submit_time,
            venue_lookahead_seconds=0,
        )
        pre = raw.filter(pl.col('time') < submit_time).rename({
            'time': 'datetime', 'qty': 'quantity',
        })
        decision = MarketImpactModel.evaluate_rolling(
            qty=order.qty,
            trades_pre_submit=pre,
            threshold_fraction=self._market_impact_threshold_fraction,
        )
        if decision is None:
            self._market_impact_n_uncalibrated += 1
            return False
        self._market_impact_bps_samples.append(decision.impact_bps)
        if decision.flag:
            self._market_impact_n_flagged += 1
            if self._strict_impact_policy and order.side == 'BUY':
                self._market_impact_n_rejected += 1
                return True
        return False

    @property
    def market_impact_realised_bps(self) -> Decimal | None:
        """Mean estimated impact bps across recorded order submits.

        None when the impact feature is off
        (`bucket_minutes is None`). Returns Decimal('0') when
        on but no calibrated bucket matched any submit
        (`n_uncalibrated > 0` exposes this case to the operator
        separately so a `0.00bp` aggregate can't masquerade as
        "no impact" when the calibration is missing).
        """
        if self._market_impact_bucket_minutes is None:
            return None
        if not self._market_impact_bps_samples:
            return Decimal('0')
        return sum(
            self._market_impact_bps_samples, Decimal('0'),
        ) / Decimal(len(self._market_impact_bps_samples))

    @property
    def market_impact_n_samples(self) -> int:
        """Count of order submits with a matching calibrated bucket."""
        return len(self._market_impact_bps_samples)

    @property
    def market_impact_n_flagged(self) -> int:
        """Count of order submits flagged as too large vs concurrent volume.

        Counts EVERY flagged submit, regardless of side or
        policy. The strict-policy gate (`n_rejected`) is a
        subset: only flagged BUY orders under
        `strict_impact_policy=True` are rejected. Flagged
        SELL exits and flagged BUYs under default
        observability policy are recorded here but pass through
        to `walk_trades`. Net:
        `n_flagged - n_rejected = orders flagged but not
        rejected` — includes (a) flagged BUYs under default
        policy, and (b) flagged SELL exits regardless of
        policy (the strict-policy gate does not reject exits
        — see `_record_market_impact_pre_fill` for why).
        """
        return self._market_impact_n_flagged

    @property
    def market_impact_n_uncalibrated(self) -> int:
        """Count of order submits whose pre-submit slice was empty / pathological.

        Distinct from `n_samples`: an uncalibrated submit means
        the per-submit `MarketImpactModel.calibrate` saw no
        trades in `[submit_time - bucket_minutes, submit_time)`
        OR every trade in that slice had a non-positive price.
        The impact is unknown, not zero. The sweep aggregator
        WARNs when this rises so the operator widens
        `bucket_minutes` or runs against a denser-volume window.
        """
        return self._market_impact_n_uncalibrated

    @property
    def market_impact_n_rejected(self) -> int:
        """Orders rejected by the strict-policy pre-fill gate.

        Always 0 when `strict_impact_policy=False` (default
        observability mode — flagged orders are recorded but
        execute). Non-zero when the operator opts into the
        gate via `--strict-impact` AND a flagged ENTRY order
        (BUY for the long-only template) is submitted. SELL
        orders represent EXITs in the long-only template and
        are NEVER rejected — see
        `_record_market_impact_pre_fill` (audit Finding 2 on
        commit fe00024). Each rejection translates into an
        `OrderStatus.REJECTED` SubmitResult so the downstream
        lifecycle (capital reservation release, strategy
        state) treats it the same as a venue filter
        rejection.
        """
        return self._market_impact_n_rejected

    def _record_limit_outcome(
        self,
        order: PendingOrder,
        fills: list[FillResult],
    ) -> None:
        """Track LIMIT-order outcomes for `bts sweep` telemetry.

        Counts are kept regardless of whether `maker_fill_model` is
        attached — the operator wants to see how many LIMIT orders
        ran through the sweep, how many filled fully vs partially
        vs not at all, and how many were marketable (limit price
        already crossed at submit, fell through to taker). MARKET
        and STOP_* orders are no-ops here. The maker_fill engine
        produces multiple FillResults for a single LIMIT order
        (one per crossing aggressor); they are aggregated to the
        order level.
        """
        if order.order_type != 'LIMIT':
            return
        self._n_limit_submitted += 1
        if not fills:
            self._n_limit_filled_zero += 1
            self._maker_fill_efficiencies.append(Decimal('0'))
            return
        all_taker = all(not f.is_maker for f in fills)
        if all_taker:
            self._n_limit_marketable_taker += 1
            # Marketable LIMITs don't exercise the maker engine —
            # don't pollute fill efficiency with a "100%" entry
            # the operator would mistake for queue-position
            # success. Track separately above.
            return
        total_filled = sum((f.fill_qty for f in fills), Decimal('0'))
        if total_filled >= order.qty:
            self._n_limit_filled_full += 1
        elif total_filled > Decimal('0'):
            self._n_limit_filled_partial += 1
        else:
            self._n_limit_filled_zero += 1
        if order.qty > Decimal('0'):
            self._maker_fill_efficiencies.append(
                total_filled / order.qty,
            )

    @property
    def n_limit_orders_submitted(self) -> int:
        return self._n_limit_submitted

    @property
    def n_limit_filled_full(self) -> int:
        return self._n_limit_filled_full

    @property
    def n_limit_filled_partial(self) -> int:
        return self._n_limit_filled_partial

    @property
    def n_limit_filled_zero(self) -> int:
        return self._n_limit_filled_zero

    @property
    def n_limit_marketable_taker(self) -> int:
        """Marketable LIMITs that fell through to taker (limit crossed at submit)."""
        return self._n_limit_marketable_taker

    @property
    def maker_fill_efficiency_p50(self) -> Decimal | None:
        """Median (filled_qty / order_qty) across passive LIMIT orders.

        Excludes marketable LIMITs (which are taker, not a maker
        engine outcome). None when no passive LIMITs were seen.
        Operator reads this as "of the LIMITs that went on the
        book, what fraction of qty actually filled before the
        window expired?"
        """
        if not self._maker_fill_efficiencies:
            return None
        ordered = sorted(self._maker_fill_efficiencies)
        n = len(ordered)
        # Median: lower-of-pair on even count for determinism.
        if n % 2 == 1:
            return ordered[n // 2]
        return (ordered[n // 2 - 1] + ordered[n // 2]) / Decimal('2')

    @property
    def maker_fill_efficiency_mean(self) -> Decimal | None:
        """Arithmetic mean of `(filled_qty / order_qty)` across passive LIMITs.

        Pair with `maker_fill_efficiency_p50` for the operator
        view: median is robust to skewed runs where one big
        partial pulls the average down; mean is the "true average
        fraction filled" the sweep aggregator wants. The sweep
        summary should weight this by the number of passive
        LIMITs in the run (NOT total LIMITs — n_marketable_taker
        is already excluded from `_maker_fill_efficiencies`) so
        the cross-run aggregate is a real mean across all passive
        orders, not a mean-of-means with mixed denominators
        (codex round 4 P2 caught the mis-weighting).
        """
        if not self._maker_fill_efficiencies:
            return None
        return sum(
            self._maker_fill_efficiencies, Decimal('0'),
        ) / Decimal(len(self._maker_fill_efficiencies))

    @property
    def n_passive_limits(self) -> int:
        """Count of passive LIMIT orders (excludes marketable takers).

        Equal to `len(self._maker_fill_efficiencies)` — matches
        the denominator used by `maker_fill_efficiency_p50` /
        `maker_fill_efficiency_mean`. Sweep summary uses this as
        the per-run weight so cross-run aggregation stays
        denominator-consistent.
        """
        return len(self._maker_fill_efficiencies)

    @property
    def slippage_n_predicted_samples(self) -> int:
        """Fills where the model's `apply()` succeeded — the gap denominator.

        The predict-vs-realised gap aggregate is averaged ONLY
        over these fills. `n_samples` counts realised
        measurements; `n_predicted_samples` counts the subset
        where prediction also succeeded. Operators reading the
        gap need this denominator separately so a low predicted
        count over many realised fills surfaces as "calibration
        coverage is thin even though we measured a lot."
        """
        return sum(
            1 for pred in self._slippage_predicted_bps if pred is not None
        )

    @property
    def slippage_realised_n_samples(self) -> int:
        return len(self._slippage_realised_bps)

    @property
    def slippage_realised_n_excluded(self) -> int:
        """Taker fills excluded because the preceding mid window was empty.

        Honest separation between "measured zero" and "could not measure":
        a sparse-tape window at run start may produce excluded fills
        without any signal — the operator sees this count and knows
        to widen the calibration / pre-window slice. The standalone
        `SlippageModel.apply` raises on uncalibrated buckets; this
        adapter does not call apply on the load-bearing path (we
        measure directly), so that loud-vs-silent gap collapses to
        the n_excluded counter here.
        """
        return self._slippage_n_excluded

    def history(self, account_id: str) -> _I.Account:
        """Return the Account (orders + trades) whether currently registered or not."""
        if account_id in self._accounts:
            return self._accounts[account_id]
        if account_id in self._history:
            return self._history[account_id]
        msg = f'account_id {account_id!r} never registered'
        raise KeyError(msg)

    async def submit_order(
        self,
        account_id: str,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        qty: Decimal,
        *,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        stop_limit_price: Decimal | None = None,
        client_order_id: str | None = None,
        time_in_force: str | None = None,
    ) -> SubmitResult:
        del stop_limit_price  # OCO/stop-limit path not implemented in the simulated fill engine
        account = self._require_account(account_id)
        venue_order_id = self._mint_order_id()
        coid = client_order_id or f'BTS-{venue_order_id}'
        # `OrderType.LIMIT_IOC` collapses to `'LIMIT'` in `TYPE_MAP`,
        # so without this nudge a caller that passes the IOC enum but
        # leaves `time_in_force=None` would land on PendingOrder with
        # `time_in_force='GTC'` and the zero-fill branch would mis-
        # report the order as OPEN (resting) instead of EXPIRED. Force
        # `IOC` whenever the enum carries it. (Other TIF mappings —
        # FOK, GTX, GTC explicit — must be passed by the caller.)
        effective_tif = time_in_force or (
            'IOC' if order_type == OrderType.LIMIT_IOC else 'GTC'
        )
        # Resolve the per-symbol filter record. `_symbol_filters` is the
        # authoritative source (populated via `load_filters()` at boot
        # and seeded from the adapter's init filter). Falling back to
        # `self._filters` only when the symbol hasn't been registered
        # would mask the misroute silently; raise instead.
        symbol_filters = self._symbol_filters.get(symbol)
        if symbol_filters is None:
            msg = (
                f'submit_order: symbol {symbol!r} has no registered filters; '
                f'call load_filters([{symbol!r}]) before submitting'
            )
            raise ValueError(msg)
        # The venue carve-out: peek up to `trade_window_seconds` past
        # `frozen_now()` to simulate a realistic submit→fill window.
        # The strategy-facing `get_trades` does not accept a kwarg for
        # this — `_get_trades_for_venue` is the only path with the
        # bounded peek. See `feed/protocol.py` for the rationale.
        #
        # When slippage measurement is active, extend the START of
        # the fetch by `dt_seconds` so the rolling-mid window for a
        # fill at the very first post-submit tick has pre-submit
        # tape to work with. Codex pinned this gap: the rolling
        # mid for a fill at t = submit_time + ε needs trades from
        # [t - dt_seconds, t), most of which sit before submit_time.
        # `walk_trades` itself filters internally with
        # `pl.col('time') >= submit_ts`, so the pre-submit prefix
        # never reaches the fill computation — it's measurement-only.
        # The lookahead carve-out is on `end`, not `start`, so
        # widening the start is unrestricted.
        submit_time = self._now()
        from datetime import timedelta as _td
        fetch_start = submit_time
        if self._slippage_model is not None:
            fetch_start = min(
                fetch_start,
                submit_time - _td(seconds=self._slippage_model.dt_seconds),
            )
        # Maker queue calibration must seed from a fresh per-submit
        # lookback. Codex P1 caught the prior behaviour: the
        # MakerFillModel was calibrated once at window-start with
        # `[window_start - 30m, window_start)`, but orders submit
        # hours later — `MakerFillModel.evaluate()` then derives a
        # `[submit_time - 30m, submit_time)` slice that's empty
        # (the stored tape ends at window_start), so queue=0 on
        # every late-window submit and `bts sweep --maker` over-
        # reports maker fill efficiency. Widen this submit's fetch
        # to span the lookback so the venue can hand `_walk_limit`
        # a fresh pre-submit slice; the model's stored tape is now
        # only a fallback (test paths that don't pre-fetch).
        if self._maker_fill_model is not None:
            fetch_start = min(
                fetch_start,
                submit_time - _td(
                    minutes=self._maker_fill_model.lookback_minutes,
                ),
            )
        trades = self._feed._get_trades_for_venue(
            symbol, fetch_start,
            submit_time + _I.window_seconds(self._trade_window_seconds),
            venue_lookahead_seconds=self._trade_window_seconds,
        )
        # The maker-LIMIT touch-refresh USED to live here (rewrite
        # `price` to last_trade ± tick when a maker model was
        # attached). Codex round 4 P2 pinned that as wrong-locus:
        # the venue would silently execute at a different price
        # than the strategy/Praxis command requested. The decision
        # now lives in `action_submitter._maybe_refresh_limit_to_touch`
        # — the action's `execution_params['price']` is rewritten
        # BEFORE validation so the entire audit trail (validation
        # context, TradeCommand, event_spine) sees the touch price
        # the venue eventually executes. The venue here just
        # honours the price it receives.
        order = PendingOrder(
            order_id=venue_order_id, side=side.name, order_type=_I.TYPE_MAP[order_type],
            qty=qty, limit_price=price, stop_price=stop_price,
            time_in_force=effective_tif, submit_time=submit_time, symbol=symbol,
        )
        if _I.reject_reason(order, symbol_filters, price) is not None:
            _I.record_rejection(account, order, coid, side, order_type, price)
            return SubmitResult(
                venue_order_id=venue_order_id, status=OrderStatus.REJECTED, immediate_fills=(),
            )
        # Pre-fill market-impact gate. Strict-causal per-submit
        # calibration over `[submit_time - bucket_minutes,
        # submit_time)` — no future tape. Records the predicted
        # impact bps + flag in the running aggregates. When
        # `strict_impact_policy=True` AND the model flags the
        # order as too large, returns True and we route to
        # REJECTED here, before walk_trades. Default
        # observability mode (False) records but never blocks.
        # Codex round 2 P1 caught the prior shape: post-fill
        # measurement only, no gate.
        if self._record_market_impact_pre_fill(order, symbol, submit_time):
            _I.record_rejection(account, order, coid, side, order_type, price)
            return SubmitResult(
                venue_order_id=venue_order_id,
                status=OrderStatus.REJECTED,
                immediate_fills=(),
            )
        # Slice the pre-submit prefix from the same widened fetch.
        # `walk_trades` itself filters `trades` by `time >= submit_time`
        # for the post-submit window; the pre-submit slice is needed
        # only for the maker engine's queue-position calibration. We
        # pass it explicitly so the model gets a fresh per-submit
        # lookback rather than the stale window-start tape.
        if self._maker_fill_model is not None:
            trades_pre_submit = trades.filter(pl.col('time') < submit_time)
        else:
            trades_pre_submit = None
        fills = walk_trades(
            order, trades, self._fill_config, symbol_filters,
            maker_model=self._maker_fill_model,
            trades_pre_submit=trades_pre_submit,
        )
        # Measure realised slippage against rolling mid; do NOT
        # adjust `fills` — the audit's P1 was that adjusting on top
        # of tape-priced fills double-counts spread.
        self._record_slippage(order, fills, trades)
        # Market-impact recording fired BEFORE walk_trades above;
        # it's strict-causal pre-submit and may have routed to
        # REJECTED already if the strict-policy gate triggered.
        # LIMIT order telemetry: surface fill efficiency on the
        # load-bearing `bts sweep` path. Counts are zeroed for
        # MARKET / STOP_* orders.
        self._record_limit_outcome(order, fills)
        immediate = _I.record_fills(
            account, self._fees,
            _I.OrderIdentity(
                venue_order_id=venue_order_id, client_order_id=coid,
                symbol=symbol, side=side,
            ),
            fills, self._mint_trade_id,
        )
        filled_qty = sum((f.qty for f in immediate), Decimal('0'))
        # Validation rejection (filter failure) returned earlier with
        # status=REJECTED via the early branch above. By the time we
        # reach this line the order passed validation. A zero-fill
        # outcome's terminal status depends on `(order_type, TIF)`:
        #   - MARKET (any TIF): there's no resting concept — no
        #     liquidity in the window means the order failed to
        #     execute; map to EXPIRED.
        #   - LIMIT with GTC: live Binance keeps the order on the
        #     book until it crosses or is cancelled. Mark OPEN so
        #     `query_open_orders` surfaces it.
        #   - LIMIT with IOC / FOK / GTX: window closed without
        #     execution → EXPIRED.
        # Pre-fix this branch returned REJECTED uniformly, conflating
        # venue-rejection with no-fill-in-window.
        status = (
            OrderStatus.FILLED if filled_qty >= qty
            else OrderStatus.PARTIALLY_FILLED if filled_qty > 0
            else _I.zero_fill_status(order.order_type, order.time_in_force)
        )
        account.orders[venue_order_id] = VenueOrder(
            venue_order_id=venue_order_id, client_order_id=coid,
            status=status, symbol=symbol, side=side, order_type=order_type,
            qty=qty, filled_qty=filled_qty, price=price,
        )
        return SubmitResult(
            venue_order_id=venue_order_id, status=status, immediate_fills=immediate,
        )

    async def cancel_order(
        self, account_id: str, symbol: str,
        *, venue_order_id: str | None = None, client_order_id: str | None = None,
    ) -> CancelResult:
        account = self._require_account(account_id)
        vo = _I.resolve_order(account, venue_order_id, client_order_id)
        terminal = {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED}
        if vo.status in terminal:
            return CancelResult(venue_order_id=vo.venue_order_id, status=vo.status)
        account.orders[vo.venue_order_id] = VenueOrder(
            venue_order_id=vo.venue_order_id, client_order_id=vo.client_order_id,
            status=OrderStatus.CANCELED, symbol=vo.symbol, side=vo.side,
            order_type=vo.order_type, qty=vo.qty, filled_qty=vo.filled_qty, price=vo.price,
        )
        return CancelResult(venue_order_id=vo.venue_order_id, status=OrderStatus.CANCELED)

    async def cancel_order_list(
        self, account_id: str, symbol: str,
        *, venue_order_id: str | None = None, client_order_id: str | None = None,
    ) -> CancelResult:
        return await self.cancel_order(
            account_id, symbol, venue_order_id=venue_order_id, client_order_id=client_order_id,
        )

    async def query_order(
        self, account_id: str, symbol: str,
        *, venue_order_id: str | None = None, client_order_id: str | None = None,
    ) -> VenueOrder:
        return _I.resolve_order(
            self._require_account(account_id), venue_order_id, client_order_id,
        )

    async def query_open_orders(self, account_id: str, symbol: str) -> list[VenueOrder]:
        account = self._require_account(account_id)
        live = {OrderStatus.SUBMITTING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED}
        return [o for o in account.orders.values() if o.symbol == symbol and o.status in live]

    async def query_balance(
        self, account_id: str, assets: frozenset[str],
    ) -> list[BalanceEntry]:
        account = self._require_account(account_id)
        return [
            BalanceEntry(
                asset=a, free=account.balances.get(a, Decimal('0')),
                locked=account.locked.get(a, Decimal('0')),
            )
            for a in sorted(assets)
        ]

    async def query_trades(
        self, account_id: str, symbol: str,
        *, start_time: datetime | None = None,
    ) -> list[VenueTrade]:
        account = self._require_account(account_id)
        return [
            t for t in account.trades
            if t.symbol == symbol and (start_time is None or t.timestamp >= start_time)
        ]

    async def get_exchange_info(self, symbol: str) -> PraxisSymbolFilters:
        f = self._symbol_filters.get(symbol)
        if f is None:
            msg = f'exchange_info: symbol {symbol!r} not loaded; call load_filters first'
            raise NotFoundError(msg)
        return PraxisSymbolFilters(
            symbol=f.symbol, tick_size=f.tick_size, lot_step=f.step_size,
            lot_min=f.min_qty, lot_max=f.max_qty, min_notional=f.min_notional,
        )

    async def query_order_book(self, symbol: str, *, limit: int = 20) -> OrderBookSnapshot:
        # One-level book sourced from most recent trade. Real depth-20 needs a
        # live book; this passes the Protocol so no hot-path consumer of the
        # book exists in the backtest. HonestyStatus flags as ESTIMATED.
        del limit
        now = self._now()
        trades = self._feed.get_trades(symbol, now - _I.window_seconds(60), now)
        if trades.is_empty():
            return OrderBookSnapshot(bids=(), asks=(), last_update_id=0)
        last = trades.tail(1).row(0, named=True)
        px, qty = Decimal(str(last['price'])), Decimal(str(last['qty']))
        return OrderBookSnapshot(
            bids=(OrderBookLevel(price=px, qty=qty),),
            asks=(OrderBookLevel(price=px, qty=qty),),
            last_update_id=int(last.get('trade_id', 0)),
        )

    async def get_server_time(self) -> int:
        return int(self._now().timestamp() * 1000)

    def get_health_snapshot(self, account_id: str) -> HealthSnapshot:
        # Simulated venue has no network, retries, or drift. Zeros are honest.
        return HealthSnapshot()

    async def load_filters(self, symbols: Sequence[str]) -> None:
        for sym in symbols:
            if sym not in self._symbol_filters:
                self._symbol_filters[sym] = BinanceSpotFilters.binance_spot(sym)

    def parse_execution_report(self, data: Mapping[str, object]) -> ExecutionReport:
        del data
        msg = (
            'SimulatedVenueAdapter.parse_execution_report: WebSocket path is '
            'unused in backtest; fills return inline via SubmitResult.'
        )
        raise NotImplementedError(msg)

    def _require_account(self, account_id: str) -> _I.Account:
        account = self._accounts.get(account_id)
        if account is None:
            msg = f'account_id {account_id!r} not registered'
            raise KeyError(msg)
        return account

    def _mint_order_id(self) -> str:
        oid = f'SIM-O-{self._next_order_seq:08d}'
        self._next_order_seq += 1
        return oid

    def _mint_trade_id(self) -> str:
        tid = f'SIM-T-{self._next_trade_seq:08d}'
        self._next_trade_seq += 1
        return tid

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)
