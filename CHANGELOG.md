# v1.4.0

- Hard-mechanical bloat gates (closes #11). Every subsequent PR is subject to a set of CI-enforced conditions that fail loud and hard on bloat — no warning mode, no disable flag, no environment-variable override, no conditional skip.
- Added 7 gate scripts under `scripts/`: `check_module_budgets.py` (per-module line ceilings), `check_module_docstrings.py` (one-line module docstring required), `check_file_size_balance.py` (largest file ≤ 2.5× median), `check_test_code_ratio.py` (ratio in [0.6, 2.0]), `check_coverage_floor.py` (line ≥ 95%, branch ≥ 90%), `check_budget_ratchet.py` (budget raises require `[budget-raise: <path>: <reason>]` marker in PR body).
- Added `.github/module_budgets.json` with 37 entries (30 prospective M1 modules under `backtest_simulator/**/*.py` — 2,840-line ceiling — plus 7 self-budgets for the gate scripts themselves).
- Extended `pyproject.toml` `[tool.ruff.lint].select` with 14 new rules: `C901` (complexity ≤ 10), `PLR0912/13/15` (branches/args/statements caps), `T201` (no `print`), `FIX001–004` (no TODO/FIXME/XXX/HACK), `ERA001` (no commented code), `D200/D205/D415` (docstring hygiene), `PIE790`. `tools/` and `scripts/` exempted via `per-file-ignores` from every rule that would force modifying them (they legitimately print banners by design).
- Extended `pr_checks_lint.yml` to run every new gate + vulture + `ruff check backtest_simulator tools tests scripts`. No `|| true`, no `continue-on-error`, no soft-fail anywhere.
- Added `tests/tools/test_bloat_gates_contract.py` (12 tests) and `tests/tools/test_bloat_gate_mutation.py` (12 mutation tests) — every gate has a paired mutation test that proves it fires on the specific violation class it is supposed to catch.

# v1.3.4

- Post-bootstrap hardening sweep (slice #7, closes parent meta-issue #6).
- Replaced every remaining `tdw_control_plane`, `tdw-control-plane`, and `quickstart_etl_tests` reference across `pyproject.toml`, gate scripts, budget files, contract tests, fixtures, and the slice issue template.
- Rewrote `pyproject.toml` `[project].description`, dropped tdw-only dependencies (`dagster*`, `clickhouse*`, `boto3`, `huggingface_hub`, `lz4`, `pyarrow`, `polars`, `pandas`, `matplotlib`, `requests`), removed the `[tool.dagster]` section, and retargeted ruff/pyright/setuptools excludes from `quickstart_etl_tests` to `tests` and `demo`.
- Re-baselined `.github/typing_budget.json` and `.github/fail_loud_budget.json` against the empty `backtest_simulator/` package — every count is now 0/0 instead of carrying tdw's totals.
- Replaced `.github/rulesets/main.json` with a snapshot of the live repo ruleset (id 15401332): `required_approving_review_count: 1`, `allowed_merge_methods: ["merge"]`. The snapshot is now stricter than the previous tdw-imported version.
- Regenerated all four `tests/fixtures/github/ruleset_live_*.json` from the live ruleset response so they are authentic snapshots of this repo's ruleset rather than edited tdw snapshots.
- Removed the `if: vars.RULESET_ID != ''` guard from `pr_checks_ruleset.yml`. The live-ruleset compare step now runs unconditionally; `vars.RULESET_ID` is set on the repo to `15401332`.
- Added `README.md` so `pip install` / wheel builds succeed.

# v1.3.3

- Initial CI scaffolding: imported `tools/` (6 gate scripts) and `tests/tools/` (4 contract tests) from tdw-control-plane.
- Added empty `backtest_simulator/` package with `__init__.py` placeholder.
- Renamed every `tdw_control_plane` reference across imported configs and gate scripts to `backtest_simulator` so the gates target this repo's package.
- Renamed `pyproject.toml` `[project].name` from `quickstart_etl` to `backtest_simulator`.
