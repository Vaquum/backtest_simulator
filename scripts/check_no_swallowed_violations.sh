#!/usr/bin/env bash
# Hand-written grep gate. Forbids any `except` clause that catches a
# HonestyViolation (or subclass) inside backtest_simulator/ or tests/.
# Runs as a CI step in pr_checks_lint alongside the other bloat gates.
set -euo pipefail

pattern='except[[:space:]]*(\(|)[[:space:]]*(HonestyViolation|LookAheadViolation|ConservationViolation|DeterminismViolation|ParityViolation|SanityViolation|PerformanceViolation|StopContractViolation)'

hits=$(grep -rnE "$pattern" backtest_simulator/ tests/ 2>/dev/null || true)
if [ -n "$hits" ]; then
  printf 'NO SWALLOWED VIOLATIONS GATE -- FAIL\n\n%s\n\nMerge blocked.\n' "$hits" >&2
  exit 1
fi
echo 'NO SWALLOWED VIOLATIONS GATE -- PASS'
