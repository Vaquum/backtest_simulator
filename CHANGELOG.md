# v1.3.3

- Initial CI scaffolding: imported `tools/` (6 gate scripts) and `tests/tools/` (4 contract tests) from tdw-control-plane.
- Added empty `backtest_simulator/` package with `__init__.py` placeholder.
- Renamed every `tdw_control_plane` reference across imported configs and gate scripts to `backtest_simulator` so the gates target this repo's package.
- Renamed `pyproject.toml` `[project].name` from `quickstart_etl` to `backtest_simulator`.
