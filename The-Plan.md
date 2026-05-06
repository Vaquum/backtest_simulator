# The Plan

## BLUF

This package becomes `bts sweep` and nothing else.

The success condition is mechanical:

- one canonical `bts sweep` coverage suite
- one positive golden `bts sweep` run
- package Python LOC `<= 3000`
- test Python LOC `<= 1500`
- at most `3` package files changed per PR
- canonical inputs SHA-256 locked
- canonical outputs normalized and byte-compared
- every kept file proven by canonical `bts sweep`
- every executable package line and branch hit by canonical `bts sweep`
- CI, gate tooling, and CI-contract tests preserved

## Hard Limits

Set `x = 3000`; tests get `x / 2`.

- package Python LOC: `<= 3000`
- test Python LOC: `<= 1500`
- raw tracked Python lines count; blank/comment lines count
- the counter must not be changed without failing CI

Commands:

- package: `git ls-files 'backtest_simulator/**/*.py' 'backtest_simulator/*.py' | sort -u | xargs wc -l`
- tests: `git ls-files 'tests/**/*.py' 'tests/*.py' | sort -u | xargs wc -l`

Current baseline:

- package: `16894`
- tests: `17202`

## Package Touch Limit

Every PR may change at most `3` tracked files under:

`backtest_simulator/`

This includes:

- add
- modify
- delete
- rename

No bypass. If work needs more than `3` package files, the architecture or scope is too wide; split the slice.

Dead-file deletion also follows this limit.

## CI Preservation

Preserve everything that makes CI true.

Protected unless replaced by a stricter equivalent in the same PR:

- `.github/workflows/`
- `.github/rulesets/`
- `.github/*budget*.json`
- `tools/`
- `tests/tools/`
- tests that assert CI, ruleset, lint, typing, budget, or gate contracts

The cleanup target is the product package, not the CI control plane.

CI tests do not count as removable product tests.

## Golden Test

Create the positive golden test:

`tests/golden/test_bts_sweep_canonical.py`

It runs the real CLI:

- subprocess command is `bts sweep`
- no direct `sweep._run`
- no bypass of decoder picking
- no bypass of window execution
- no bypass of feed semantics
- no bypass of parity assertions

Canonical run:

- `2` decoders
- `3` replay days
- canonical real-data bundle
- all surviving `bts sweep` args explicitly supplied
- fixed `--session-id`
- isolated `HOME`, `HF_HOME`, `XDG_CACHE_HOME`
- local kline parquet
- local trade tape
- sockets blocked

The test runs the command twice in one job. Normalized outputs must be byte-equal across both runs and byte-equal to expected artifacts.

Compare:

- normalized stdout
- normalized stderr
- normalized `sweep_per_window.csv`
- normalized `sweep_per_tick.csv`
- normalized event-spine summary
- exit code

Normalize with one script:

`tools/normalize_sweep_outputs.py`

Normalize:

- absolute paths
- durations
- wall-clock timestamps
- session index `started_at` / `ended_at`
- temporary directory names

## Coverage Suite

Package coverage is not a whitelist.

The oracle is live `coverage json` from real `bts sweep` subprocesses.

The suite may contain:

- one positive golden sweep
- minimal negative sweep invocations needed to hit fail-loud branches

The suite may not contain:

- direct function calls into package internals
- mocks replacing sweep mechanics
- coverage-only helper paths
- non-sweep package entrypoints

Every surviving error branch must be hit by a canonical negative `bts sweep` invocation. If that is impossible or not worth doing, delete the branch.

## Canonical Bundle Lock

Add canonical fixture assets under:

`tests/fixtures/canonical/`

Required files:

- `bundle.zip`
- `klines.parquet`
- `trades.parquet`
- `checksums.sha256`
- `expected/stdout.txt`
- `expected/stderr.txt`
- `expected/sweep_per_window.csv`
- `expected/sweep_per_tick.csv`
- `expected/event_spine_summary.json`

Rules:

- fixture data is real data, not synthetic data
- bundle zip entries are sorted
- bundle zip mtimes are fixed
- checksums are computed in Python, not shell `sha256sum`
- expected output changes must appear in the same PR as checksum/reference updates

## Sweep Reference

Create:

`tests/fixtures/canonical/sweep_reference.json`

Produce it only after `bts run` migration is complete and the golden test is green.

Generate with subprocess coverage:

- `coverage run --branch -m pytest tests/golden/test_bts_sweep_canonical.py`
- child-process coverage enabled
- `coverage combine`
- `coverage json`

Coverage config is locked:

- `source = backtest_simulator`
- `branch = True`
- subprocess coverage enabled
- no `omit`
- no `exclude_lines`
- no plugins
- no dynamic context

Record:

- imported `backtest_simulator` files
- exact executable line set per file
- exact executed line set per file
- exact branch set per file
- exact executed branch set per file
- normalized output artifact checksums

CI verifies:

- imported package file set equals the reference
- no package file outside the reference is imported
- every reference file is imported
- executable package lines equal executed package lines
- executable package branches equal executed package branches
- no package line is coverage-excluded
- no `# pragma: no cover` exists under `backtest_simulator/`
- output checksums match

Files absent from the reference are dead and must be deleted.

## Package Coverage Law

The package may contain only code executed by canonical `bts sweep`.

Mechanical gate:

- run canonical `bts sweep` under coverage
- use `git ls-files 'backtest_simulator/**/*.py' 'backtest_simulator/*.py'` as the file universe
- include every tracked `backtest_simulator/**/*.py`
- combine parent and child-process coverage
- parse `coverage json`
- fail if any package file has `missing_lines`
- fail if any package file has `missing_branches`
- fail if any package file has `excluded_lines`
- fail if any tracked package file is absent from coverage
- fail if coverage config has `omit`, `exclude_lines`, plugins, or dynamic context
- fail if coverage JSON lacks branch data
- fail if coverage schema changes without an explicit gate update

No package-only helper, alternate command path, dormant branch, compatibility shim, or unused error path survives this gate.

## Slice Plan

Each cleanup PR closes one OPEN slice-labelled issue and obeys `CLAUDE.md`.

1. Golden test, real fixtures, normalizer.
2. `bts run` inspection and migration decision.
3. CLI prune to `sweep` only.
4. Sweep reference and CI gates.
5. Dead-file deletion.
6. Dead-line deletion.
7. Package-root/doc/pyproject tidy.

Each slice updates version, changelog, budgets, and ruleset when touched.

## Deletion Order

1. Add the golden test and canonical inputs.
2. Inspect `bts run` while existing tests still exist.
3. Move real `bts run` capability into `bts sweep`; delete the rest.
4. Delete every CLI command except `sweep`.
5. Replace the required test gate with the golden test.
6. Delete old non-CI product tests.
7. Preserve CI-contract tests and their fixtures.
8. Delete test-only strategies and non-CI fixtures.
9. Generate `sweep_reference.json`.
10. Delete every package file not in the reference.
11. Ratchet typing, fail-loud, module, and LOC budgets down.
12. Update docs, pyproject, CI, ruleset.
13. Repeat at line level: delete unexecuted lines unless they are required fail-loud error paths.

## `bts run` Rule

`bts run` is deleted.

Before deletion, inspect it and write a keep/delete list into the migration slice issue.

Only these can move into `bts sweep`:

- explicit decoder selection
- explicit replay window
- ledger parity, if made sweep-native

Delete run-only JSON output unless the canonical sweep uses it.

## CI Gates

Add required checks:

- `pr_checks_golden_sweep.yml`
- `pr_checks_loc_budget.yml`
- `pr_checks_package_touch_budget.yml`
- `pr_checks_canonical_bundle.yml`
- `pr_checks_sweep_reference.yml`
- `pr_checks_package_coverage.yml`
- `pr_checks_cli_surface.yml`

The checks are gates only after `.github/rulesets/main.json` requires them.

Required scripts:

- `tools/check_coverage_config.py`
- `tools/check_loc_budget.py`
- `tools/check_package_touch_budget.py`
- `tools/check_canonical_bundle.py`
- `tools/check_sweep_reference.py`
- `tools/check_package_coverage.py`
- `tools/check_cli_surface.py`

The gates must:

- run without network
- fail if more than `3` files under `backtest_simulator/` changed
- fail on any outbound socket during golden sweep
- fail if any CLI subcommand except `sweep` exists
- fail if any surviving `bts sweep` option is absent from the canonical command
- fail if any package line is not hit by canonical `bts sweep`
- fail if coverage has any whitelist or exclusion mechanism
- fail if outputs drift
- fail if budgets rise
- print exact observed counts or checksum mismatches

## Existing Gates

Cleanup must not leave old gates lying.

- preserve CI workflows, tools, budgets, ruleset snapshots, and CI-contract tests
- replace `CLAUDE.md` law 7 target with the golden test
- update `.github/rulesets/main.json`
- ratchet `.github/typing_budget.json` down
- ratchet `.github/fail_loud_budget.json` down
- ratchet `.github/module_budgets.json` down
- keep version and changelog current
- retire obsolete gates only in the same PR that updates ruleset requirements
- retire nothing CI-related without stricter same-PR replacement

## Non-Negotiables

- No parallel backtest universe around Praxis/Nexus.
- No synthetic data.
- No mocks that replace sweep mechanics.
- No live ClickHouse, HuggingFace, DNS, HTTP, or socket dependency in golden test.
- No unmeasured fixture drift.
- No PR changing more than `3` package files.
- No package or test growth past budget.
- No CI file treated as a gate until ruleset-required.
- No package code outside canonical `bts sweep` coverage.
- No coverage whitelist, omit, exclusion, or pragma.
- No deletion of CI tooling or CI-contract tests without stricter same-PR replacement.
- No deleted behavior unless canonical `bts sweep` proves it is unused or moved.

## Done

The cleanup is done when:

- only `bts sweep` remains as a CLI command
- one golden test remains
- canonical sweep runs twice deterministically
- CI enforces LOC ceilings
- CI enforces package touch ceiling
- CI enforces canonical input checksums
- CI enforces canonical normalized outputs
- CI enforces sweep reference
- CI enforces 100% package line and branch coverage from canonical `bts sweep`
- CI enforces CLI surface
- package LOC is `<= 3000`
- test LOC is `<= 1500`
- obsolete gates are retired or rewired
- CI workflows, gate tools, budgets, rulesets, and CI-contract tests are preserved
- the PR series is merged
