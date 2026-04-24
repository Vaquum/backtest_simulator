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
- [x] 18 honesty gate tests + 5 mutation tests (per Issue #10 *Tests* table). *(52 tests pass in `tests/honesty/`: pre-existing conservation / determinism / lookahead / split-alignment / stop-enforcement + new capital-invariants + capital-lifecycle + risk, with 5 mutation tests.)*
- [x] `pr_checks_honesty` workflow added; live ruleset includes it as a required check. *(`.github/workflows/pr_checks_honesty.yml`, ruleset snapshot `.github/rulesets/main.json` updated.)*
- [x] `bts sweep` CLI produces `<experiment_dir>/results_with_backtest.csv` with the full enriched column set. *(`backtest_simulator/cli.py::_cmd_sweep` → `build_enriched_table`; tested by producing the CSV against the profile's UEL output.)*
