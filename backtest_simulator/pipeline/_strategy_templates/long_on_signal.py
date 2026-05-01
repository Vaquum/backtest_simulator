"""Preds-based binary-regime, long-only template — `__BTS_PARAMS__` substituted by ManifestBuilder."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from decimal import ROUND_DOWN, Decimal

from nexus.core.domain.enums import OrderSide
from nexus.core.domain.order_types import ExecutionMode, OrderType
from nexus.infrastructure.praxis_connector.trade_outcome import TradeOutcome
from nexus.strategy.action import Action, ActionType
from nexus.strategy.base import Strategy as _StrategyBase
from nexus.strategy.context import StrategyContext
from nexus.strategy.params import StrategyParams
from nexus.strategy.signal import Signal

_log = logging.getLogger(__name__)

# Baked-in config. ManifestBuilder substitutes this literal JSON string
# when copying the template; before substitution the strategy would
# still parse cleanly but would exit early at on_startup.
_BAKED_CONFIG: dict[str, object] = json.loads('__BTS_PARAMS__')


class _Config:
    # Plain class (not @dataclass): Nexus's dynamic strategy loader runs
    # `exec_module` without registering the module in `sys.modules`, so
    # `@dataclass` fails at class-definition time when it tries
    # `sys.modules.get(cls.__module__).__dict__`.

    def __init__(
        self, symbol: str, capital: Decimal, kelly_pct: Decimal,
        estimated_price: Decimal, stop_bps: Decimal,
        force_flatten_after: datetime | None,
        maker_preference: bool = False,
    ) -> None:
        self.symbol = symbol
        self.capital = capital
        self.kelly_pct = kelly_pct
        self.estimated_price = estimated_price
        self.stop_bps = stop_bps
        # Force-flatten cutoff. When the next signal's timestamp is
        # at or after this instant, the strategy emits SELL on any
        # held inventory regardless of `_preds`, AND blocks new BUYs.
        # bts sets it to `window_end - kline_size` so the strategy's
        # last in-window signal closes any open position before the
        # subprocess shuts down. None disables force-flatten (legacy
        # behaviour: positions can survive past window close).
        self.force_flatten_after = force_flatten_after
        # When True the strategy emits LIMIT orders at the
        # estimated price (passive maker post). When False
        # (default) it emits MARKET orders. The runtime venue
        # adapter routes LIMITs through the MakerFillModel —
        # queue position, partial fills, aggressor-size bound
        # — so `bts sweep --maker` produces realistic LIMIT-fill
        # telemetry distinct from MARKET-only sweeps.
        self.maker_preference = maker_preference


class Strategy(_StrategyBase):
    """Binary regime on `_preds`, long-only, Kelly-sized from baked config.

    State machine:
      preds=1 AND flat  -> ENTER BUY  (size = capital * kelly_pct/100 / est_price)
      preds=0 AND long  -> ENTER SELL (size = self._entry_qty recorded at ENTER)
      preds=1 AND long  -> no-op
      preds=0 AND flat  -> no-op

    Kelly% comes from `backtest_mean_kelly_pct` of the selected decoder
    (baked at manifest-build time). `estimated_price` is the ClickHouse
    seed price at window start — the real fill price comes from the venue
    adapter's next-trade walk, so the qty here is a sizing hint, not a
    price promise.

    Class name is `Strategy` because Nexus's loader looks up that exact
    module attribute; the base is aliased `_StrategyBase` to avoid collision.
    """

    def on_startup(
        self, params: StrategyParams, context: StrategyContext,
    ) -> list[Action]:
        # Nexus currently constructs `StrategyParams(raw={})` and does not
        # pass per-strategy YAML params through; `_BAKED_CONFIG` above is
        # the mechanism ManifestBuilder uses to carry per-instance config
        # to each strategy file.
        del params, context
        force_flatten_raw = _BAKED_CONFIG.get('force_flatten_after')
        force_flatten_after = (
            None if force_flatten_raw is None
            else datetime.fromisoformat(str(force_flatten_raw))
        )
        self._config = _Config(
            symbol=str(_BAKED_CONFIG['symbol']),
            capital=Decimal(str(_BAKED_CONFIG['capital'])),
            kelly_pct=Decimal(str(_BAKED_CONFIG['kelly_pct'])),
            estimated_price=Decimal(str(_BAKED_CONFIG['estimated_price'])),
            stop_bps=Decimal(str(_BAKED_CONFIG['stop_bps'])),
            force_flatten_after=force_flatten_after,
            maker_preference=bool(
                _BAKED_CONFIG.get('maker_preference', False),
            ),
        )
        # Fresh-start defaults; `on_load` overwrites if persisted state exists.
        self._long: bool = False
        self._entry_qty: Decimal = Decimal('0')
        # `_pending_buy` blocks duplicate BUY emissions when a prior
        # LIMIT BUY is still resting (passive maker). Without this
        # the strategy would re-emit BUY on every preds=1 signal
        # while the previous LIMIT had not filled — accumulating
        # multiple OPEN orders and a corrupted CAPITAL ledger. Set
        # on emit, cleared in `on_outcome` regardless of terminal
        # outcome (FILLED/PARTIAL flip `_long`; REJECTED/EXPIRED/
        # CANCELED clear pending without flipping `_long`).
        self._pending_buy: bool = False
        # `_pending_sell` mirrors `_pending_buy` for the exit leg:
        # `_long` only flips False once the SELL outcome lands, so
        # without this gate a second `_preds=0` signal arriving
        # before the OutcomeLoop dispatches the prior SELL would
        # emit a duplicate exit (double-selling the same
        # `_entry_qty`, corrupting the position ledger). Codex
        # round 2 P2 caught this. Set on SELL emit, cleared in
        # `on_outcome` for SELL outcomes (terminal or fill).
        self._pending_sell: bool = False
        _log.info(
            'LongOnSignal startup: symbol=%s capital=%s kelly_pct=%s est_price=%s',
            self._config.symbol, self._config.capital,
            self._config.kelly_pct, self._config.estimated_price,
        )
        return []

    def on_signal(
        self, signal: Signal, params: StrategyParams, context: StrategyContext,
    ) -> list[Action]:
        del params, context
        preds_raw = signal.values.get('_preds')
        if preds_raw is None:
            _log.info('signal missing _preds; keys=%s', list(signal.values))
            return []
        preds = int(preds_raw)
        # `was_long` captures the PRIOR state so the log line is unambiguous —
        # readers shouldn't have to infer whether `long=...` is pre- or
        # post-transition. Transitions log their own ENTER BUY / ENTER SELL
        # line below.
        was_long = self._long
        _log.info(
            'on_signal fired: preds=%s probs=%s was_long=%s',
            preds, signal.values.get('_probs'), was_long,
        )
        # Force-flatten at window close: if this signal arrives at or
        # after the configured cutoff (window_end - kline_size), close
        # any open position immediately and refuse new entries. The
        # strategy's last in-window signal becomes a guaranteed
        # exposure-zero point so per-day stats reflect realized PnL
        # without orphaned positions.
        cutoff = self._config.force_flatten_after
        if cutoff is not None and signal.timestamp >= cutoff:
            if was_long and not self._pending_sell:
                qty = self._entry_qty
                self._pending_sell = True
                _log.info(
                    'FORCE FLATTEN at window close: qty=%s ts=%s cutoff=%s',
                    qty, signal.timestamp, cutoff,
                )
                return [self._build_action(OrderSide.SELL, qty, signal)]
            return []
        if preds == 1 and not was_long and not self._pending_buy:
            qty_raw = (
                self._config.capital * self._config.kelly_pct / Decimal('100')
            ) / self._config.estimated_price
            # Round DOWN to Binance's BTCUSDT step_size of 0.00001 so
            # the venue adapter's lot-size filter accepts the qty and
            # the full amount fills (no PARTIALLY_FILLED from a qty
            # that doesn't land on a step boundary). A fractional
            # residue beyond 5 decimals silently turns cmd.qty >
            # filled_qty into a partial status even when the walk
            # consumed everything that was step-legal.
            qty = qty_raw.quantize(Decimal('0.00001'), rounding=ROUND_DOWN)
            # `_long` and `_entry_qty` flip in `on_outcome` once the
            # venue confirms the fill. MARKET BUYs fill in-line so
            # the outcome arrives synchronously and `_long` is True
            # before the next `on_signal`. LIMIT BUYs (maker
            # preference) may rest unfilled — `_pending_buy` keeps
            # the strategy from re-issuing while the prior order is
            # still in the lookahead window.
            self._pending_buy = True
            _log.info(
                'ENTER BUY emit: qty=%s was_long=False pending_buy=True', qty,
            )
            return [self._build_action(OrderSide.BUY, qty, signal)]
        if preds == 0 and was_long and not self._pending_sell:
            qty = self._entry_qty
            # `_long` and `_entry_qty` flip in `on_outcome` when the
            # SELL outcome lands. Clearing `_entry_qty` at emit time
            # would let a SELL that zero/partial-fills (rare for
            # MARKET, but the venue's 60s lookahead can run dry)
            # leave the strategy thinking it's flat with size zero
            # — the next preds=0 would emit a zero-size SELL and
            # the venue's LOT_SIZE filter would reject. Holding the
            # qty until the SELL fill confirms is the honest path.
            self._pending_sell = True
            _log.info(
                'ENTER SELL (exit long) emit: qty=%s was_long=True '
                'pending_sell=True', qty,
            )
            return [self._build_action(OrderSide.SELL, qty, signal)]
        return []

    def _build_action(self, side: OrderSide, qty: Decimal, signal: Signal) -> Action:
        del signal
        # Declared-stop enforcement (Part 2): every OPEN-position ENTER
        # must carry a concrete `stop_price`. For long-only we only
        # open on BUY; SELL here is the EXIT leg of an already-open
        # position and doesn't need a new stop (the SELL itself is the
        # risk close). The stop sits `stop_bps` basis points BELOW the
        # reference price for a BUY — e.g. with `stop_bps=50` and
        # `estimated_price=70000`, `stop_price=69650`. The venue
        # fill engine (`venue/fills.py::_walk_market`) halts the entry
        # walk the moment a tick breaches the stop and returns the
        # already-accumulated partial fill; the residual is NOT booked
        # at the declared stop. A separate STOP_* close fills at the
        # breach tick's actual tape price (gap slippage), not at the
        # declared stop, via `_walk_stop`. The declared stop is the
        # measurement unit for R, not a promise about where fills land.
        stop_price: Decimal | None = None
        if side == OrderSide.BUY:
            bps = self._config.stop_bps
            stop_price = self._config.estimated_price * (
                Decimal('1') - bps / Decimal('10000')
            )
        execution_params: dict[str, object] = {
            'symbol': self._config.symbol,
            'stop_bps': str(self._config.stop_bps),
        }
        if stop_price is not None:
            execution_params['stop_price'] = str(stop_price)
        # When `maker_preference` is set, BUY entries become passive
        # LIMITs (the rebate-capture leg). SELL exits stay MARKET so
        # the strategy's `_long` / `_entry_qty` state — flipped at
        # signal time — actually matches the executed position. A
        # SELL LIMIT that zero-fills would leave the strategy
        # marking flat against an unfilled exit, then double-sell
        # on the next regime change. Asymmetric routing matches
        # live market-maker conventions: passive entries (paid for
        # liquidity) and aggressive exits (paying for execution
        # certainty). The venue refreshes the BUY limit price to
        # `last_trade - 1 tick` so it's strictly passive at submit
        # — see `SimulatedVenueAdapter.submit_order`.
        is_buy_maker = (
            self._config.maker_preference and side == OrderSide.BUY
        )
        order_type_value = (
            OrderType.LIMIT if is_buy_maker else OrderType.MARKET
        )
        if is_buy_maker:
            execution_params['price'] = str(self._config.estimated_price)
        return Action(
            action_type=ActionType.ENTER,
            direction=side, size=qty,
            execution_mode=ExecutionMode.SINGLE_SHOT,
            order_type=order_type_value,
            execution_params=execution_params,
            # `deadline` is a DURATION in seconds (not an epoch timestamp).
            # Praxis computes the concrete deadline as
            # `cmd.created_at + timedelta(seconds=timeout)`, where
            # `timeout = action.deadline` via Nexus's praxis_outbound.
            # 60s is a reasonable fill window for a backtest MARKET order.
            deadline=60,
            trade_id=None, command_id=None,
            maker_preference=None,
            # `reference_price` is what the action-submitter multiplies
            # by `size` to produce the order's notional for the CAPITAL
            # validator. The backtest uses `estimated_price` baked at
            # manifest-build time (ClickHouse seed price at window
            # start) rather than trying to read live book here — the
            # strategy is notified only of the prediction, not the
            # current tick. Real fills come from the venue adapter's
            # historical trade walk regardless; this price is a sizing
            # hint only.
            reference_price=self._config.estimated_price,
        )

    def on_outcome(
        self, outcome: TradeOutcome, params: StrategyParams, context: StrategyContext,
    ) -> list[Action]:
        """Reconcile `_long` / `_entry_qty` from the venue's actual fill.

        Strategy state used to flip at signal-emit time, which was
        safe under MARKET-everywhere (fills always succeed). Under
        `maker_preference=True` BUY entries become passive LIMITs
        that may zero-fill; flipping `_long` on emit would let a
        later preds=0 signal emit a SELL for inventory the venue
        never gave us. Reconcile here so the next signal sees the
        truth.

        Outcome handling:
          - PARTIAL/FILLED + BUY: `_long=True`, `_entry_qty +=
            outcome.fill_size`. Partials accumulate on the same
            command_id until terminal.
          - REJECTED/EXPIRED/CANCELED + BUY: clear `_pending_buy`;
            don't touch `_long` (no inventory was acquired).
          - PARTIAL/FILLED + SELL: `_long=False` (exit confirmed).
          - Any non-fill SELL: leave `_long` alone — exit failed,
            we still hold inventory. The next preds=0 signal will
            re-emit.
        """
        del params, context
        from nexus.infrastructure.praxis_connector.trade_outcome_type import (
            TradeOutcomeType,
        )
        is_buy = self._is_buy_command(outcome.command_id)
        if outcome.outcome_type.is_fill:
            if outcome.fill_size is None:
                msg = (
                    f'TradeOutcome {outcome.outcome_id} is_fill but '
                    f'fill_size is None; cannot reconcile strategy '
                    f'position from a fill outcome with no qty.'
                )
                raise ValueError(msg)
            if is_buy:
                self._long = True
                self._entry_qty += outcome.fill_size
                # Codex P1 caught: in this backtest the maker
                # engine evaluates the entire post-submit lookahead
                # in one shot, so a PARTIAL outcome is TERMINAL
                # — no later FILLED arrives for the same command.
                # Leaving `_pending_buy=True` past PARTIAL would
                # block every subsequent entry AND let the SELL
                # outcome get misclassified as a BUY by
                # `_is_buy_command`. Clear pending on any fill.
                self._pending_buy = False
                _log.info(
                    'on_outcome BUY %s: fill_size=%s entry_qty=%s long=True',
                    outcome.outcome_type.value, outcome.fill_size,
                    self._entry_qty,
                )
            else:
                # Reduce inventory by the fill_size; only flip
                # `_long` to False when the SELL has FULLY closed
                # the position. A PARTIAL SELL (zero/partial-fill
                # on a MARKET exit when the 60s tape window is
                # thin) leaves residual long inventory, which the
                # next preds=0 should still try to close.
                self._entry_qty -= outcome.fill_size
                if (
                    outcome.outcome_type == TradeOutcomeType.FILLED
                    or self._entry_qty <= Decimal('0')
                ):
                    self._long = False
                    self._entry_qty = Decimal('0')
                # SELL outcome arrived — re-open the pending-sell
                # gate so a subsequent preds=0 (e.g. after a
                # bounce-and-bust signal flip) can re-issue.
                self._pending_sell = False
                _log.info(
                    'on_outcome SELL %s: fill_size=%s entry_qty=%s long=%s',
                    outcome.outcome_type.value, outcome.fill_size,
                    self._entry_qty, self._long,
                )
        elif outcome.outcome_type.is_terminal:
            if is_buy:
                # REJECTED / EXPIRED / CANCELED on a BUY: no fill,
                # free the pending slot. The strategy stays flat.
                self._pending_buy = False
                _log.info(
                    'on_outcome BUY %s (no fill): pending_buy=False',
                    outcome.outcome_type.value,
                )
            else:
                # Terminal-no-fill SELL outcome: free the
                # pending-sell gate so the next preds=0 can
                # re-attempt the exit. `_long` stays True (the
                # position was never closed).
                self._pending_sell = False
                _log.info(
                    'on_outcome SELL %s (no fill): pending_sell=False',
                    outcome.outcome_type.value,
                )
        return []

    def _is_buy_command(self, command_id: str) -> bool:
        """Best-effort side inference for the BUY/SELL state machine.

        BUY emits stamp `pending_buy=True`. The outcome's
        `command_id` doesn't carry a side directly, but at most
        one BUY can be in flight (`_pending_buy` gates it) and
        SELLs only emit when `_long=True`. So: if `_pending_buy`
        is True OR `_long` is False, the outcome must be a BUY's.
        Otherwise SELL. Inferring side from state is structurally
        correct for this single-position strategy and avoids
        threading the side through TradeOutcome (which Nexus's
        contract doesn't carry).
        """
        del command_id
        return self._pending_buy or not self._long

    def on_timer(
        self, timer_id: str, params: StrategyParams, context: StrategyContext,
    ) -> list[Action]:
        del timer_id, params, context
        return []

    def on_load(self, data: bytes) -> None:
        if not data:
            return
        payload = json.loads(data.decode('utf-8'))
        self._long = bool(payload['long'])
        self._entry_qty = Decimal(str(payload['entry_qty']))
        # `pending_buy` / `pending_sell` default False on load;
        # in-flight LIMITs/MARKETs don't survive a strategy reload
        # across runs in this slice.
        self._pending_buy = bool(payload.get('pending_buy', False))
        self._pending_sell = bool(payload.get('pending_sell', False))

    def on_save(self) -> bytes:
        payload = {
            'long': self._long, 'entry_qty': str(self._entry_qty),
            'pending_buy': self._pending_buy,
            'pending_sell': self._pending_sell,
        }
        return json.dumps(payload).encode('utf-8')

    def on_shutdown(
        self, params: StrategyParams, context: StrategyContext,
    ) -> list[Action]:
        del params, context
        return []
