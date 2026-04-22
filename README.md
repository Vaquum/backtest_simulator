# backtest_simulator

Honest, gate-enforced backtester for unmodified Nexus strategies against unmodified Praxis.

The simulator runs strategies through the same paper-trading engine that drives live execution, against historical market data on hourly bars. Honesty gates fail fast, hard, and loud on look-ahead leakage, conservation violations, non-determinism, parity drift between live and replay, and worker-budget overruns. Statistical methodology — Combinatorial Purged Cross-Validation, Deflated Sharpe, Probability of Backtest Overfitting — follows López de Prado.

## Status

Bootstrap. The CI scaffolding (gate scripts, contract tests, ruleset, budget ratchets) is in place; the `backtest_simulator/` package is an empty placeholder waiting for the runtime layer.

## Layout

- `backtest_simulator/` — installable package (currently empty placeholder).
- `tools/` — CI gate scripts (Conventional Commits, fail-loud, ruleset drift, slice contract, pyright/typing budget, version+changelog).
- `tests/tools/` — contract tests for the gates.
- `tests/fixtures/` — fixtures for gate unit tests.
- `.github/` — workflows, rulesets, issue/PR templates, budget files.

## Governance

Every PR merging into `main` must pass the eight required status checks listed in `.github/rulesets/main.json`:
`PR Checks CodeQL (python)`, `pr_checks_cc`, `pr_checks_lint`, `pr_checks_ruleset`, `pr_checks_slice`, `pr_checks_fail_loud`, `pr_checks_typing`, `pr_checks_version`.
The ruleset itself is enforced server-side; the snapshot in this repo is a copy that `pr_checks_ruleset` compares against the live ruleset on every PR to detect drift.
