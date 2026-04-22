# v1.3.4

- Post-bootstrap hardening sweep (closes #6).
- Replaced every remaining `tdw_control_plane`, `tdw-control-plane`, and `quickstart_etl_tests` reference across `pyproject.toml`, gate scripts, budget files, contract tests, fixtures, and the slice issue template.
- Rewrote `pyproject.toml` `[project].description`, dropped tdw-only dependencies (`dagster*`, `clickhouse*`, `boto3`, `huggingface_hub`, `lz4`, `pyarrow`, `polars`, `pandas`, `matplotlib`, `requests`), removed the `[tool.dagster]` section, and retargeted ruff/pyright excludes from `quickstart_etl_tests` to `tests` and `demo`.
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
