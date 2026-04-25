# TODO — backtest_simulator honest market adaptation

The plan: drive the existing pipeline (UEL → ManifestBuilder → BacktestLauncher → SimulatedVenueAdapter against real ClickHouse trade ticks) so the trading results visibly reflect a real long-only binary-regime strategy responding to a real market. Then in Part 2, add the mechanical honesty gates (stops, conservation, validator, etc.). Part 3 (M2) takes the remaining honesty / capability / test gaps to closure under a single slice — issue #17.

**Constraint that overrides everything: the profile run stays at ≈10 seconds total. Faster is welcome. Slower is not.**

---

## Part 1 — pipeline behaves like the real market, results visible

- [x] **SSH tunnel preflight.** The profile script opens an SSH tunnel to the internal ClickHouse host and verifies the tunnel by querying ClickHouse for one recent BTCUSDT trade before booting the launcher. If the tunnel is down, the script fails loud with the exact reopen command. Connection details live in operator-local environment/config, not in the repo.
- [x] **ClickHouseFeed wired to `origo.binance_daily_spot_trades`.** `database='origo'`, `table='binance_daily_spot_trades'` defaults. Format datetime parameters as `'%Y-%m-%d %H:%M:%S.%f'` (DateTime64(6) rejects ISO-T+timezone). Real Binance ticks reach the venue adapter.
- [x] **Strategy template — preds-based binary regime, long-only.** `self._long: bool` state, persisted via `on_save`/`on_load`. `preds=1` AND flat → ENTER BUY. `preds=0` AND long → ENTER SELL with the entry qty. `preds=1` AND long → no-op. `preds=0` AND flat → no-op. No probs, no enter_threshold.
- [x] **`StrategyParamsSpec` updated.** Drop `qty` and `enter_threshold`. Add `capital: Decimal`, `kelly_pct: Decimal`, `estimated_price: Decimal`. ManifestBuilder bakes them into the generated strategy file.
- [x] **Filter API supports column ranges.** `filter_results({'col': (lo, hi)})` keeps only rows where `lo ≤ col ≤ hi`. Equality and set-membership stay supported. *(already supported by `_column_predicate` — tuple-of-2 triggers the range branch.)*
- [x] **Synchronous drain after every submit.** `_advance_clock_until` waits until each submitted `command_id` has been dispatched by Praxis's `account_loop` before advancing the next clock tick.
- [x] **Profile script wired to ClickHouse + Kelly.** Picks one active decoder via the new range-filter on `backtest_trades_count`, reads its `backtest_mean_kelly_pct`, fetches a seed price, runs the 14h window. Prints per-trade entry/exit prices, sides, qty, fees.
- [x] **Verify in the run output:** `orders > 0`, `trades > 0`, with `BUY` + `SELL` pairs corresponding to `preds 0→1→0` transitions, fill prices and times matching real Binance ticks. Total wall time ≈10s.
- [x] **Commit + push to `feat/m1-bootstrap`.** *(commit `f63d1f6`.)*

## Part 2 — honesty hardening (Issue #10 invariants)

- [x] Real `ValidationPipeline` (CAPITAL=real, others=_allow). *(`backtest_simulator/honesty/capital.py::build_validation_pipeline`)*
- [x] CapitalController 4-step lifecycle (`check_and_reserve → send_order → order_ack → order_fill`).
- [x] Conservation laws checked after every event (`assert_conservation`). *(INV-1a/1b/2/3.)*
- [x] Declared-stop enforcement; INTAKE rejects stop-less ENTER.
- [x] `r_per_trade` from declared stop (no virtual `stop_bps` denominator).
- [x] `is_maker` reflects real fill type. *(MARKET orders are always taker → `False` is correct; documented.)*
- [ ] **`SignalsTable` precompute layer + per-decoder split-alignment.** Deferred from Part 2 to Part 3 (M2) Task 16, where `SignalsTable.lookup` gains `path_id` / `purge_seconds` / `embargo_seconds` and is wired into the live predict path.
- [x] 75 honesty tests across 12 files in `tests/honesty/`.
- [x] `pr_checks_honesty` workflow added; live ruleset includes it as a required check.
- [x] `bts enrich` CLI joins `results.csv` with optional `backtest_results.parquet`. *(Now part of the master `bts` CLI; see Part 3 Task 1.)*

## Part 3 — M2 (slice issue #17) — package the `bts` master CLI and close every open honesty / capability / test gap

> **Operational protocol (operator-mandated)**
>
> 1. **CLI is the master tool.** All debugging through `bts`. Test runs through `bts test`. Lint through `bts lint`. Typing through `bts typecheck`. Gates through `bts gate <name>`. Verbosity flags `-v`, `-vv`, `-vvv` accepted on every subcommand. **No tool other than `bts`, `codex`, `git`, and `gh` is used over the lifetime of this slice; any direct `pytest` / `ruff` / `pyright` / `python -c` invocation outside `bts` is a workflow violation.**
> 2. **One slice (#17), one PR, 25 commits.** Each task is one commit. Each commit is reviewed by codex (model 5.5, reasoning xhigh) until codex approves. Each approved commit is pushed to the slice's PR before the next task starts. Exactly one PR for the whole slice.
> 3. **Codex first, zero-bang last.** Codex reviews every commit. After all 25 commits land, codex reviews the PR end-to-end and the implementer iterates until codex approves the PR. Only then is zero-bang re-requested. Implementer addresses zero-bang's comments and re-requests until zero-bang approves.
> 4. **Operator review is the final gate.** Operator pinged ONLY when the GitHub merge button is truly green (10 gates green AND `reviewDecision == APPROVED` AND every review thread resolved). Not before.
> 5. **No empty scaffolding.** Every task lands fully-functional code. No placeholders, no stubs. Each task's exit-state must be verifiable via the CLI without additional setup.
> 6. **TODO.md is the live mirror.** Every checkbox below reflects current truth at the latest commit on this branch. The CLI move (Task 1) updates this file and adds the 25-task list; subsequent commits tick their box.
> 7. **Closest-possible-market-simulation standard.** Every modeling decision (fills, stops, slippage, maker/taker, validator sequence, R metric) matches live as tightly as the available data allows. Where data cannot resolve a detail, the model falls back to the conservative (worse-for-strategy) choice. Backtest ≡ paper ≡ live byte-identical event spine is the M3 unlock.

### Tasks (1–25)

- [ ] **Task 1.** Move `/tmp/bts_sweep.py` into the package as `bts` master CLI: subcommands `run`, `sweep`, `enrich`, `test`, `lint`, `typecheck`, `gate`, `notebook`, `version`; verbosity `-v`/`-vv`/`-vvv`; `docs/cli.md`; `tests/cli/`; pyproject 1.5.2 → 1.6.0; CHANGELOG; this TODO.md update.
- [ ] **Task 2.** `test_sanity_zero_trade` — no-op strategy emits no fills, no fees.
- [ ] **Task 3.** `test_sanity_buy_hold` — return within ±5 bps of `(close−open)/open − fees`.
- [ ] **Task 4.** `test_sanity_over_trading` — over-trading models post mean R/trade < 0, PF < 1.
- [ ] **Task 5.** `test_sanity_random` — 1000 random seeds, mean return 95% CI brackets 0.
- [ ] **Task 6.** `test_prescient_strategy` — prescient strategy raises `LookAheadViolation`.
- [ ] **Task 7.** `test_inverse_prescient` — sign-flipped prescient posts catastrophic loss.
- [ ] **Task 8.** `test_shuffle_bars` — permuted-bar-order profit collapses to ≈0 after fees.
- [ ] **Task 9.** `test_on_save_purity` — `save → load → save` produces identical bytes.
- [ ] **Task 10.** R-denominator gameability gate — ATR-based stop-distance check; `stop = entry × 0.9999` is rejected.
- [ ] **Task 11.** Book-gap instrumentation — max stop-cross-to-trade gap reported per-run; surfaced via `bts run --output-format json`.
- [ ] **Task 12.** Slippage model — calibrated empirical slippage from the same symbol/period's trade tape; applied to fills with parametric tolerance.
- [ ] **Task 13.** Market impact model — per-$-traded penalty derived from tape volume distribution; flag for orders exceeding `threshold_fraction` of concurrent trade volume.
- [ ] **Task 14.** Passive maker-fill realism — queue position bound from trade stream; partial fills; aggressor-size bound; LIMIT-OPEN handling in launcher.
- [ ] **Task 15.** `test_sanity_maker_no_fill` — far-away maker orders never fill (depends on Task 14).
- [ ] **Task 16.** `SignalsTable.lookup(t, *, path_id=None, purge_seconds=0, embargo_seconds=0)` — wired into the live predict path; CPCV index + per-decoder split alignment.
- [ ] **Task 17.** Statistical honesty — full CPCV (purge + embargo) + Deflated Sharpe + Probability of Backtest Overfitting + Superior Predictive Ability test.
- [ ] **Task 18.** Ledger parity vs Praxis — byte-identical event-spine assertion between backtest and paper-Praxis replay.
- [ ] **Task 19.** `test_perf_gate_logreg_binary` — 1-year hourly replay < 10 s wall time on a frozen parquet fixture.
- [ ] **Task 20.** Parametric thresholds — `pytest.param` + env-var override across all calibrated honesty tests; paired calibration-proof tests.
- [ ] **Task 21.** 5 separate `_mutation.py` files (perf, conservation, split-alignment, on-save, determinism) — each test fires loud when the mutation it is meant to catch is applied.
- [ ] **Task 22.** `NexusRuntime` module — facade composing `ValidationPipeline` + `CapitalController` + `ActionSubmitter` + `CapitalState`; `submit(action)` returns `SubmissionOutcome`; `assert_conservation(context)` honors INV-1/2/3.
- [ ] **Task 23.** `SimulationDriver` module — drives `BacktestLauncher` end-to-end against `NexusRuntime` + `SimulatedVenueAdapter` + `HistoricalFeed`; `run(window_start, window_end)` returns `RunReport`.
- [ ] **Task 24.** 3 integration test files: `test_end_to_end`, `test_cli_end_to_end`, `test_notebook_execution`.
- [ ] **Task 25.** `notebooks/sweep_and_analyze.ipynb` — operator-facing demo notebook; `jupyter nbconvert --to script | python -` runs end-to-end.

### P0 architectural honesty gaps surfaced mid-slice (added 2026-04-25)

- [ ] **Task 26.** Outcome-driven strategy state. `StrategyContext.positions` and `capital_available` are populated from real `CapitalState` + `Account` (currently empty / zero in `launcher.py:687`). Strategy templates flip `_long` / `_entry_qty` only inside `on_outcome` after a confirmed fill (currently flipped at action-emit time in over_trading / buy_and_hold). Partial fills surface back so the strategy emits a corrective close for the *actual* filled qty. Without this fix a rejection or partial-fill leaves the strategy believing it owns a position it does not, then later emits a SELL for the wrong quantity.
- [ ] **Task 27.** SELL closes flow through CAPITAL stage. `action_submitter.py:367` short-circuits SELL with `ValidationDecision(allowed=True)` and logs "CAPITAL skipped" — the close never lands in `validation_pipeline.validate`. Fix routes SELL through `validate` so reservation release, PnL update, and position decrement land in the canonical lifecycle. Allow remains True for a close, but the *machinery* runs.
- [ ] **Task 28.** Scope `threading.Timer.run` patch to `accelerated_clock`. `launcher/clock.py:117` does `setattr(threading.Timer, 'run', _frozen_aware_timer_run)` at module import — any process that imports `backtest_simulator` gets timer semantics globally rewritten, including paper/live code that imports the package for *any* reason. Move the patch inside the `accelerated_clock` context manager (save the original and restore on exit) so the rewrite is scoped to the frozen-clock backtest only.

### Final phase (after all 28 tasks land)

- Codex full-PR review until codex approves the PR.
- Re-request `zero-bang`; address comments; re-request until zero-bang approves.
- Operator pinged once merge button is truly green.
