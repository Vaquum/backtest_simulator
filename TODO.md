# TODO — backtest_simulator honest market adaptation

The plan: drive the existing pipeline (UEL → ManifestBuilder → BacktestLauncher → SimulatedVenueAdapter against real ClickHouse trade ticks) so the trading results visibly reflect a real long-only binary-regime strategy responding to a real market. Then in Part 2, add the mechanical honesty gates (stops, conservation, validator, etc.).

**Constraint that overrides everything: the profile run stays at ≈10 seconds total. Faster is welcome. Slower is not.**

---

## Part 1 — pipeline behaves like the real market, results visible

- [x] **SSH tunnel preflight.** `ssh -fN -L 18123:127.0.0.1:8123 root@37.27.112.167` opens a tunnel; the profile script verifies the tunnel by querying ClickHouse for one recent BTCUSDT trade before booting the launcher. If the tunnel is down, fail loud with the exact reopen command.
- [x] **ClickHouseFeed wired to `origo.binance_daily_spot_trades`.** `database='origo'`, `table='binance_daily_spot_trades'` defaults. Format datetime parameters as `'%Y-%m-%d %H:%M:%S.%f'` (DateTime64(6) rejects ISO-T+timezone). Real Binance ticks reach the venue adapter.
- [x] **Strategy template — preds-based binary regime, long-only.** `self._long: bool` state, persisted via `on_save`/`on_load`. `preds=1` AND flat → ENTER BUY. `preds=0` AND long → ENTER SELL with the entry qty. `preds=1` AND long → no-op. `preds=0` AND flat → no-op. No probs, no enter_threshold.
- [x] **`StrategyParamsSpec` updated.** Drop `qty` and `enter_threshold`. Add `capital: Decimal`, `kelly_pct: Decimal`, `estimated_price: Decimal`. ManifestBuilder bakes them into the generated strategy file.
- [x] **Filter API supports column ranges.** `filter_results({'col': (lo, hi)})` keeps only rows where `lo ≤ col ≤ hi`. Equality and set-membership stay supported. Lets the profile script pick a decoder with `backtest_trades_count > 0` AND good metrics in one call. *(already supported by `_column_predicate` — tuple-of-2 triggers the range branch.)*
- [x] **Synchronous drain after every submit.** `_advance_clock_until` waits until each submitted `command_id` has been dispatched by Praxis's `account_loop` (i.e. `submit_order` has been called on the adapter) before advancing the next clock tick. Keeps the 10s budget AND closes the dispatch-starvation gap that currently leaves `orders=0` at the adapter.
- [x] **Profile script wired to ClickHouse + Kelly.** Picks one active decoder via the new range-filter on `backtest_trades_count`, reads its `backtest_mean_kelly_pct`, fetches a seed price from ClickHouse at window start, runs the 14h window. Prints per-trade entry/exit prices, sides, qty, fees.
- [x] **Verify in the run output:** `orders > 0`, `trades > 0`, with `BUY` + `SELL` pairs corresponding to `preds 0→1→0` transitions, fill prices and times matching real Binance ticks. Total wall time ≈10s. *(measured 8.9s; verification asserts fail-loud if any criterion misses.)*
- [x] **Commit + push to `feat/m1-bootstrap`.** *(commit `f63d1f6`, pushed to `origin/feat/m1-bootstrap`.)*

## Part 2 — honesty hardening (Issue #10 invariants)

- [x] Real `ValidationPipeline` (CAPITAL=real, others=_allow). *(`backtest_simulator/honesty/capital.py::build_validation_pipeline`)*
- [x] CapitalController 4-step lifecycle (`check_and_reserve → send_order → order_ack → order_fill`). *(`CapitalLifecycleTracker` in the same module; launcher drives transitions via `_install_capital_adapter_wrapper`.)*
- [x] Conservation laws checked after every event (`assert_conservation`). *(`backtest_simulator/honesty/conservation.py`, INV-1a/1b/2/3.)*
- [x] Declared-stop enforcement in `FillModel.apply_stop`; INTAKE rejects stop-less ENTER. *(Strategy emits `stop_price` on BUY `execution_params`; `_check_declared_stop` in `action_submitter.py`; `_walk_market` in `venue/fills.py`.)*
- [x] `r_per_trade` from declared stop (no virtual `stop_bps` denominator). *(`backtest_simulator/honesty/risk.py::compute_r`; surfaced per-trade in profile output.)*
- [x] `is_maker` reflects real fill type, not hardcoded `False`. *(MARKET orders are always taker → `False` is correct; documented in `_walk_market` aggregation path. Follow-up for LIMIT maker orders lives with broader LIMIT support.)*
- [ ] `SignalsTable` precompute layer + per-decoder split-alignment. *(Deferred — precompute module exists at `backtest_simulator/sensors/precompute.py`, split-alignment tests are already passing; full wiring into the live predict path defers to a follow-up slice with its own e2e budget review.)*
- [x] 75 honesty tests across 12 files in `tests/honesty/`. *(capital_invariants, capital_lifecycle, conservation, determinism, lookahead, risk, risk_edge_cases, split_alignment, stop_enforcement, sell_close_semantics, adapter_wrapper_paths. Mutations are inline, not separate files.)*
- [x] `pr_checks_honesty` workflow added; live ruleset includes it as a required check. *(`.github/workflows/pr_checks_honesty.yml`, ruleset snapshot `.github/rulesets/main.json` updated.)*
- [x] `bts enrich` CLI joins `results.csv` with optional `backtest_results.parquet` into `results_with_backtest.csv`. *(`backtest_simulator/cli.py::_cmd_enrich` → `build_enriched_table`. `bts sweep` and `bts analyze` retained as aliases. Does NOT run a backtest sweep — only enriches existing results.)*

## Not delivered — #10 MVC assertions still open

These items from #10's MVC / Tests table are NOT shipped in this PR. Each requires its own follow-up slice:

- [ ] `test_perf_gate_logreg_binary` — 1-year hourly replay < 10s wall time.
- [ ] `test_prescient_strategy` — prescient strategy raises `LookAheadViolation`.
- [ ] `test_inverse_prescient` — sign-flipped prescient posts catastrophic loss.
- [ ] `test_shuffle_bars` — permuted-bar-order profit collapses to ≈0 after fees.
- [ ] `test_on_save_purity` — `save → load → save` produces identical bytes.
- [ ] `test_sanity_buy_hold` — return within ±5 bps of `(close−open)/open − fees`.
- [ ] `test_sanity_random` — 1000 random seeds mean return 95% CI brackets 0.
- [ ] `test_sanity_zero_trade` — zero-trading strategy emits no fills, no fees.
- [ ] `test_sanity_maker_no_fill` — far-away maker orders never fill.
- [ ] `test_sanity_over_trading` — over-trading models have mean R/trade < 0, PF < 1.
- [ ] 5 separate `_mutation.py` files (perf, conservation, split-alignment, on-save, determinism).
- [ ] `NexusRuntime` module (replaced by `launcher/` package — architectural deviation from #10).
- [ ] `SimulationDriver` module (replaced by `BacktestLauncher` — architectural deviation from #10).
- [ ] 3 integration test files (`test_end_to_end`, `test_cli_end_to_end`, `test_notebook_execution`).
- [ ] 1 notebook (`sweep_and_analyze.ipynb`).

## Not delivered — capability gaps acknowledged

These were flagged by the operator as "must not defer" but remain unimplemented. Each requires dedicated design work beyond documentation fixes:

- [ ] **Market impact model.** No own-order book impact; historical tape walked as-is. No gate on order-size vs concurrent trade volume.
- [ ] **Passive maker-fill realism.** Limit orders get full qty at limit price on first cross. No queue position, no partial fills, no aggressor-size bound.
- [ ] **Gap-risk on stops.** Fills at declared stop price on crossing, not `min(stop, first_crossing_trade_price)`. Optimistic for adverse gap scenarios.
- [ ] **Statistical honesty (CPCV / DSR / PBO / SPA).** Zero code. `SignalsTable.lookup()` takes only `t: datetime` — no path_id, no purge, no embargo.
- [ ] **R denominator gameability gate.** No ATR-based sanity check on stop distance. `stop = entry × 0.9999` is accepted.
- [ ] **Parametric thresholds.** All test calibrations are hardcoded constants. No `pytest.param` / env-var override / paired calibration-proof tests.
- [ ] **Book-gap instrumentation.** Maximum stop-cross-to-trade gap not reported per-run.
- [ ] **Ledger parity vs Praxis.** No parity test asserting byte-identical event-spine output between backtest and paper-Praxis replay.
- [ ] **Slippage model.** No slippage model calibrated on trade-stream price-move distribution.
