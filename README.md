<div align="center">
  <br />
  <a href="https://github.com/Vaquum"><img src="https://github.com/Vaquum/Home/raw/main/assets/Logo.png" alt="Vaquum" width="150" /></a>
  <br />
</div>
<br />
<div align="center"><strong>Vaquum backtest_simulator runs honest, gate-enforced backtests for unmodified Nexus strategies against unmodified Praxis.</strong></div>

<div align="center">
  <a href="#backtest_simulator">backtest_simulator</a> •
  <a href="#what-bts-is-not">What bts Is Not</a> •
  <a href="#capabilities">Capabilities</a> •
  <a href="#first-sweep">First Sweep</a> •
  <a href="#learn-more">Learn More</a>
</div>
<br />

<hr />

<a id="backtest_simulator"></a>

# backtest_simulator — The Market Simulator

*The closest-to-live backtester for Nexus strategies. Every modeling decision (fills, stops, slippage, maker/taker, validator sequence, R metric) matches live as tightly as the available data allows.*

`bts sweep` is the main pathway. The package exists to produce honest distributions of three trading-outcome metrics (R per trade, profit factor, net return) across thousands of strategy-and-hyperparameter samples, so that model selection and risk-adjusted ranking become mechanical instead of anecdotal. The backtest-to-live delta is the project's primary metric; if a modeling shortcut waters down the simulator, it waters down paper and live the same way, and that is rejected.

## What bts Is Not

bts is not:

- a trading platform or order-routing layer (Praxis owns execution; bts wraps it)
- a research / parameter-search engine (Limen owns model search; bts consumes its decoders)
- a generic multi-asset simulator (single-symbol BTC/USDT today; the contract is depth, not breadth)
- a fork of Nexus or Praxis (they are imported unmodified — strategy code that runs in bts is byte-identical to what runs in paper and live)

In the wider Vaquum architecture, Origo sits upstream as the data layer; Limen produces decoders and SignalsTables; Nexus runs the strategy loop; Praxis is the trading layer; Veritas oversees. bts is the seam where Nexus + Praxis run against historical tape under the same contracts they see in production.

## Capabilities

- One-symbol historical backtest with strict no-look-ahead and per-fill measurement against the trade tape
- Real Praxis `VenueAdapter` Protocol implementation backed by historical trades (`SimulatedVenueAdapter`)
- Real Nexus runtime — `BacktestLauncher` is a Praxis `Launcher` subclass; the wired `action_submit` path is the same six-stage `ValidationPipeline` the live system uses, with each stage running the real `validate_*_stage` from `nexus.core.validator` (MMVP-lenient defaults — operator-supplied `nexus_config` / limits / snapshot providers dial in real denial behavior, same dials Praxis paper-trade exposes)
- Slippage, market-impact, and maker-fill models calibrated on the same tape, measured per fill (not adjusted on `fill_price` — the audit rejected the double-counting design)
- ATR R-denominator gameability gate: stop-distance must clear `k * ATR(window)` from entry or the order is rejected pre-fill
- Book-gap instrumentation: `(trigger_time - prev_sub_stop_time)` recorded per STOP/TP fill, surfaced on `bts run --output-format json`
- CPCV + Deflated Sharpe + Probability of Backtest Overfitting + Superior Predictive Ability — built on per-decoder daily returns, not bar counts
- Ledger parity: byte-identical event-spine assertion between backtest and paper-Praxis replay (`bts run --check-parity-vs PATH`)
- 10-gate CI surface: scope (slice), lint + bloat budgets, typing (pyright strict, error-count ratchet), honesty (no-swallowed-violations, fail-loud), CodeQL, ruleset, version, conventional commits

## First Sweep

The fastest first success is a small sweep against a UEL-trained decoder pool.

1. Install the package:

```bash
pip install backtest_simulator
```

2. Set up the environment (Praxis / Nexus / Limen install transitively, BTC/USDT trade tape via ClickHouse):

```bash
cp .env.example .env
# Fill CLICKHOUSE_PASSWORD from the tdw-control-plane host
```

3. Run a small sweep:

```bash
bts sweep --exp-code path/to/exp_code.py \
          --n-decoders 5 --n-permutations 30 \
          --replay-period-start 2026-04-01 --replay-period-end 2026-04-20 \
          --trading-hours-start 08:00 --trading-hours-end 16:00
```

The headline output is one line per `(decoder, day)` pair plus a sweep summary that reports DSR, SPA, CPCV-PBO. Every per-fill measurement (slippage cost, market-impact bps, maker-fill efficiency) is calibrated on the same trade tape the strategy sees.

4. Inspect a single window in detail:

```bash
bts run --exp-code path/to/exp_code.py \
        --window-start 2026-04-20T08:00:00+00:00 \
        --window-end   2026-04-20T16:00:00+00:00 \
        --decoder-id 0 --output-format json | jq
```

The JSON report carries `trades`, `declared_stops`, the slippage / impact / book-gap fields, and the `event_spine_jsonl` path for ledger-parity comparison against paper.

## Learn More

- Start with [docs/CLI.md](docs/CLI.md) — every subcommand, every flag, every printed line, and the market-relevance commitment each one carries
- Read [CHANGELOG.md](CHANGELOG.md) — every shipped slice with the contract it added or sharpened
- The honesty surface lives in `backtest_simulator/honesty/` — slippage, market impact, maker-fill, ATR gate, CPCV, deflated Sharpe, PBO, SPA, ledger parity, conservation laws, capital lifecycle
- The CI gate surface lives in `tools/` (gate scripts) and `.github/workflows/pr_checks_*.yml` (10 gates, all required)
- The Nexus / Praxis seam lives in `backtest_simulator/launcher/` (BacktestLauncher) and `backtest_simulator/venue/simulated.py` (SimulatedVenueAdapter)

## Contributing

The simplest way to start contributing is by [joining an open discussion](https://github.com/Vaquum/backtest_simulator/issues?q=is%3Aissue%20state%3Aopen%20label%3Aquestion%2Fdiscussion), contributing to [the docs](https://github.com/Vaquum/backtest_simulator/tree/main/docs), or by [picking up an open issue](https://github.com/Vaquum/backtest_simulator/issues?q=is%3Aissue%20state%3Aopen%20label%3Abug%20OR%20label%3Aenhancement%20OR%20label%3A%22good%20first%20issue%22%20OR%20label%3A%22help%20wanted%22%20OR%20label%3APriority%20OR%20label%3Aprocess).

Before contributing, read [CLAUDE.md](CLAUDE.md) — the ten laws every PR is enforced against (slice scope, conventional commits, typing discipline, fail-loud, version bump, lint clean, honesty test pass, CodeQL clean, ruleset snapshot, branch protection).

## Vulnerabilities

Report vulnerabilities privately through [GitHub Security Advisories](https://github.com/Vaquum/backtest_simulator/security/advisories/new).

## Citations

If you use backtest_simulator for published work, please cite:

Vaquum backtest_simulator [Computer software]. (2026). Retrieved from https://github.com/Vaquum/backtest_simulator.

## License

[MIT License](LICENSE).
