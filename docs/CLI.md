# `bts` — backtest_simulator master CLI

The `bts` command is the **only** operator-facing surface of the package. All debugging, gate invocation, sweep / single-window runs, lint / typing, and notebook execution flow through `bts <subcommand>`. The CLI is intentionally the only entry point: nothing else in the project is meant to be invoked directly.

This is a design commitment. `bts` is the contract; everything else is implementation. The same applies to honesty: every CLI surface below states the **market-relevance commitment** it preserves — what the equivalent paper-trade or live invocation would do. If the simulator's behaviour drifts from that commitment, the slice that introduces the drift is rejected.

## Verbosity

Every subcommand that runs work accepts cumulative verbosity flags. The mapping is centralised in `backtest_simulator/cli/_verbosity.py` so every subcommand interprets the flag identically.

| flag | stdlib level | structlog level | tqdm | behaviour |
| --- | --- | --- | --- | --- |
| (none) | ERROR | ERROR | silenced | CLI's own per-run summary lines on stdout. Praxis / Nexus / Limen logs are quiet. |
| `-v` | INFO | INFO | silenced | Pipeline-orchestration INFO + sibling-library INFO logs surface. |
| `-vv` | DEBUG | DEBUG | silenced | DEBUG-level "why did this fill?" diagnostics. |
| `-vvv` | NOTSET | NOTSET | unmasked | Everything: tqdm bars, structlog details, asyncio traces. |

**Market-relevance commitment.** The structlog filter tracks the stdlib level (zero-bang fix, post-v2.0.4): at `-v` the operator sees the same INFO output Praxis would produce in paper / live. Hardcoding ERROR for verbosity<3 (the previous bug) hid logs the operator needed for debugging fill-vs-paper deltas.

---

## `bts run`

`bts run` runs **one backtest window for one decoder** and prints either a human-readable one-line summary plus per-trade detail (`--output-format text`, default) or a structured JSON report (`--output-format json`).

### Inputs

```
bts run --exp-code PATH
        --window-start ISO8601 --window-end ISO8601
        [--decoder-id INT | --n-decoders INT | --input-from-file PATH]
        [--experiment-dir PATH] [--output-format {text,json}]
        [--seed INT]
        [--maker] [--strict-impact]
        [--atr-k FLOAT] [--atr-window-seconds INT]
        [--check-parity-vs PATH] [--parity-tolerance {strict,clock_normalized}]
        [-v|-vv|-vvv]
```

| flag | type | meaning |
| --- | --- | --- |
| `--exp-code` | path (required) | UEL-compliant Python file with `params()` + `manifest()`. The single source of truth for training and retraining the picked decoder. |
| `--window-start` / `--window-end` | ISO8601 | Replay range. Must be timezone-aware (`+00:00`). End-exclusive in the underlying event-spine clock. |
| `--decoder-id` | int | Pick a specific permutation. Mutually exclusive with `--input-from-file`. |
| `--n-decoders` | int (default 1) | Pick the top-N decoders ranked by `backtest_mean_kelly_pct`. |
| `--input-from-file` | path to results.csv | Pick decoders from a Limen `results.csv` filter pool. |
| `--experiment-dir` | path | Reuse an existing UEL run. Default: bts trains on demand into a content-addressed snapshot. |
| `--output-format` | `text` (default) or `json` | Text: human one-liner; JSON: structured report. |
| `--seed` | int | Reproducibility seed; threaded into RNG-using strategy paths. |
| `--maker` | flag | Enable LIMIT-on-signal strategy variant (queue-position / partial-fill maker model). Default off (MARKET-only). |
| `--strict-impact` | flag | REJECT entry-leg BUY orders pre-fill when market-impact bps exceeds `threshold_fraction` of bucket volume. Default off (measure-only). |
| `--atr-k` | float (default 0.5) | R-denominator floor: `stop_distance` must be ≥ `atr_k * ATR(window)`. |
| `--atr-window-seconds` | int (default 900) | ATR Wilder window for the gate. |
| `--check-parity-vs` | path | Assert byte-identical event-spine vs the reference JSONL. |
| `--parity-tolerance` | `strict` (default) or `clock_normalized` | `strict` = full-byte match; `clock_normalized` = strip envelope event_seq + timestamp, keep payload bytes (for cross-runtime cases). |

### Printouts during operation

At verbosity 0, stdout carries the orchestration header lines (decoder pick, training cache hit/miss, seed-price fetch) followed by the per-window summary. At `-v` the same lines plus pipeline INFO. At `-vv` per-tick decisions. At `-vvv` Nexus / Praxis structlog flows through.

### Outputs — `--output-format text` (default)

One headline line per window with the metrics most operators read first:

```
   perm 0     2026-04-20  trades 4    PF 1.42    R̄ +0.31    DD -1.28%   total +2.14%   slip +0.42bp n=8/excl=0  imp +1.10bp n=8/flagged=0  atr_rej=0/uncal=0
```

Followed by one line per BUY→SELL pair:

```
     08:15 → 09:42   BUY     69842.10 → SELL     69987.00   +0.21%   R +0.42
```

Then the book-gap line **only if** at least one STOP/TP fired, and the event-spine path:

```
   book_gap   max=0.183s  p95=0.142s  n_stops=2
bts run         event_spine_jsonl=<work>/event_spine.jsonl  n_events=43
```

If `--check-parity-vs` is set, parity status surfaces via return code (0 pass, 1 fail) plus stderr. JSON mode keeps stdout pure.

### Outputs — `--output-format json`

A single JSON object on stdout. Keys (every key always present unless explicitly noted):

```json
{
  "decoder_id": 0,
  "kelly_pct": "0.05",
  "window_start": "2026-04-20T08:00:00+00:00",
  "window_end": "2026-04-20T16:00:00+00:00",
  "orders": 4,
  "trades": [["coid", "BUY", "0.001", "69842.10", "0.07", "USDT", "2026-04-20T08:15:00+00:00"], ...],
  "declared_stops": {"coid": "69500.00"},
  "book_gap_max_seconds": 0.183,
  "book_gap_n_observed": 2,
  "book_gap_p95_seconds": 0.142,
  "slippage_realised_bps": "0.42",
  "slippage_realised_cost_bps": "0.31",
  "slippage_realised_buy_bps": "0.45",
  "slippage_realised_sell_bps": "-0.39",
  "slippage_predicted_cost_bps": "0.28",
  "slippage_predict_vs_realised_gap_bps": "0.03",
  "slippage_n_samples": 8, "slippage_n_excluded": 0,
  "slippage_n_uncalibrated_predict": 0, "slippage_n_predicted_samples": 8,
  "n_limit_orders_submitted": 0, "n_limit_filled_full": 0,
  "n_limit_filled_partial": 0, "n_limit_filled_zero": 0,
  "n_limit_marketable_taker": 0, "maker_fill_efficiency_p50": null,
  "market_impact_realised_bps": "1.10",
  "market_impact_n_samples": 8, "market_impact_n_flagged": 0,
  "market_impact_n_uncalibrated": 0, "market_impact_n_rejected": 0,
  "atr_k": "0.5", "atr_window_seconds": 900,
  "n_atr_rejected": 0, "n_atr_uncalibrated": 0,
  "event_spine_jsonl": "<work>/event_spine.jsonl",
  "event_spine_n_events": 43
}
```

### Market-relevance commitment

- `trades` are the actual taker fills against the historical trade tape (`walk_trades` walks the same prints a live taker would have hit). No `fill_price` adjustment from the slippage model — adjusting it would double-count the price-discovery effect already in the tape.
- `declared_stops` is the per-trade declared stop captured at reservation time, used downstream for the honest-R metric (no virtual `stop_bps` denominator).
- `slippage_realised_cost_bps` is the side-normalized mean: BUY contributes `+bps`, SELL contributes `-bps`. Positive = paid spread; negative = price improvement. The signed `slippage_realised_bps` is directional, NOT a cost metric.
- `slippage_predict_vs_realised_gap_bps` is the calibration-error signal: realised − predicted. A non-zero gap tells the operator to recalibrate; a zero gap with a non-zero `n_uncalibrated_predict` means coverage is incomplete.
- `market_impact_n_rejected` only counts BUY entries (long-only template) when `--strict-impact` is on; SELL exits never get rejected by the impact gate.
- `n_atr_rejected` counts ENTER+BUY denied for stops tighter than `atr_k * ATR(window)`. `n_atr_uncalibrated` is the warmup signal — operator runs early in the day will see this until enough pre-decision tape accumulates.
- `event_spine_jsonl` is the byte-equivalent dump of the run's `aiosqlite` event spine. The same paper-trade Praxis dumps the same events; with `--check-parity-vs PATH` and `--parity-tolerance strict`, byte equality is enforced.

---

## `bts sweep`

`bts sweep` runs the backtest pipeline over **N decoders × M days**. This is the main pathway. Decoders are picked from a UEL-trained pool (default) or from a `results.csv` filter pool. Each window runs in a fresh Python subprocess for state isolation (asyncio + freezegun bleed across windows otherwise).

### Inputs

```
bts sweep --exp-code PATH
          [--n-decoders INT] [--n-permutations INT]
          [--trading-hours-start HH:MM --trading-hours-end HH:MM]
          [--replay-period-start YYYY-MM-DD --replay-period-end YYYY-MM-DD]
          [--input-from-file PATH] [--trades-q-range LO,HI]
          [--tp-min-q FLOAT] [--fpr-max-q FLOAT]
          [--kelly-min-q FLOAT] [--trade-count-min-q FLOAT]
          [--net-return-min-q FLOAT]
          [--maker] [--strict-impact]
          [--atr-k FLOAT] [--atr-window-seconds INT]
          [--cpcv-n-groups INT] [--cpcv-n-test-groups INT]
          [--cpcv-purge-seconds INT] [--cpcv-embargo-seconds INT]
          [-v|-vv|-vvv]
```

**Pair rule.** Within `(--trading-hours-start, --trading-hours-end)` and `(--replay-period-start, --replay-period-end)` either both values are given or neither. Giving one without the other fails loud — half-specified ranges are operator error and the simulator refuses to guess.

| flag | meaning |
| --- | --- |
| `--exp-code` | UEL-compliant Python file (params + manifest). REQUIRED — bts has no fallback SFD. |
| `--n-decoders` | Top-N decoders to backtest (default 1). |
| `--n-permutations` | UEL parameter samples per decoder pool training run (default 30). |
| `--input-from-file` | Limen `results.csv` filter pool. Overrides UEL training. |
| `--trades-q-range` / `--tp-min-q` / `--fpr-max-q` / `--kelly-min-q` / `--trade-count-min-q` / `--net-return-min-q` | Quantile-range filters applied to `--input-from-file` rows. |
| `--maker` / `--strict-impact` / `--atr-k` / `--atr-window-seconds` | Same semantics as `bts run`. |
| `--cpcv-n-groups` / `--cpcv-n-test-groups` | López de Prado §11 CPCV path construction. Default 4 groups, 2 test. |
| `--cpcv-purge-seconds` / `--cpcv-embargo-seconds` | Purge drops train days adjacent to test boundaries (both sides); embargo drops train days AFTER each test block. |

### Printouts during operation

```
bts sweep        starting (decoders=5 perms=30 days=14)
bts sweep        training pool from /path/to/exp_code.py …
bts sweep        pool ready: 30 perms cached at <work>/op-sfd-cache/_bts_op_<sha16>/
bts sweep        picked decoders: 0, 7, 12, 18, 25
bts sweep        fetching trade tape …  (4 days, 5.2M rows)
   perm 0     2026-04-07  trades 3    PF 0.84    R̄ -0.21    DD -1.10%   total -1.42%   slip +0.39bp n=6/excl=0
   perm 0     2026-04-08  …
   …
bts sweep        signals parity OK n_compared=140
bts sweep        cpcv_pbo skipped: tied IS top-2 on path 3 (logit ambiguity)
bts sweep        dsr=0.43  spa=p≤0.04  pbo=skipped
bts sweep        complete (wall=18m31s, n_runs=70, n_failed=0)
```

The `signals parity` line is mandatory: every per-window `produce_signal` call is captured, and the parent compares the captured `(timestamp, pred)` pairs against the per-decoder `SignalsTable.lookup(t)` value. Mismatch = `ParityViolation`. Empty-capture = also `ParityViolation` (auditor post-v2.0.2 contract: the gate must NEVER silently no-op).

### Outputs

Sweep writes `results_with_backtest.csv` to the experiment dir; each row is one `(decoder, day)` pair with the same metric columns as `bts run --output-format json`. The DSR / SPA / PBO summary lines on stdout are the operator-facing pulse; the CSV is the durable record for cross-run analysis.

### Market-relevance commitment

- Per-window subprocess isolation guarantees that asyncio + freezegun state from window N doesn't leak into window N+1. This matches paper / live where each Praxis run is a fresh process.
- The signals parity assertion enforces the slice's "strategy tested = strategy deployed" contract. The deployed strategy uses `SignalsTable.lookup(t)`; the sweep replay calls Nexus's actual `produce_signal`; both must agree byte-for-byte at every tick boundary. A passing sweep with `n_compared > 0` is the proof that the live decoder will produce the same predictions on the same input.
- The CPCV path construction follows López de Prado §11 (combinatorial purge + embargo). Per-path top-2 IS-tie + OOS rank ambiguity skips with explicit `skipped:<reason>` reasons — the gate refuses to fabricate a logit when the data doesn't support one.

---

## `bts enrich`

`bts enrich` joins `<experiment>/results.csv` with an optional `backtest_results.parquet` into `<experiment>/results_with_backtest.csv`. Does NOT run a backtest sweep — only enriches existing results.

### Inputs

```
bts enrich --experiment PATH
           [--backtest-parquet PATH]
           [--out PATH]
           [-v]
```

### Printouts

```
bts enrich: wrote /path/to/results_with_backtest.csv
```

### Market-relevance commitment

The enrichment is purely metadata join — it does not re-compute any per-fill metric. If `backtest_results.parquet` is stale, the enriched CSV is stale; the operator is responsible for a fresh `bts sweep` before re-enriching.

---

## `bts test`

`bts test` is the operator-mandated pytest entry point. The slice's operational protocol forbids invoking `pytest` directly; everything goes through this wrapper so verbosity, marker filters, and the integration / honesty / cli scope are unified under one tool.

### Inputs

```
bts test [-v|-vv]
         [-k PATTERN]
         [--honesty | --integration | --cli]
         [-- ...args forwarded to pytest verbatim...]
```

### Printouts

Standard pytest output; the wrapper sets `--tb=short` by default.

### Market-relevance commitment

There is no test-only code path that bypasses honesty gates. `bts test --honesty` runs the same gate suite that PR-time `pr_checks_honesty` runs in CI. A test that passes locally must pass in CI; a passing CI run is the merge contract.

---

## `bts lint`

`bts lint` runs `ruff check` on the package + tools + tests. On a clean tree the last line of output is `All checks passed!`.

### Inputs

```
bts lint [-v]
         [--paths PATH ...]
         [--fix]
```

### Printouts

```
All checks passed!
```

…or specific ruff diagnostics on a dirty tree.

### Market-relevance commitment

The lint surface includes the bloat-budget gates (`tools/check_*.py`) — module line counts, test/code SLOC ratio, file-size balance, no-swallowed-violations. These prevent the simulator from drifting into "thousand-line god module" territory that would mask the honesty surface. A passing lint = a debuggable package.

---

## `bts typecheck`

`bts typecheck` runs pyright in strict mode against the package root (matches `pyproject.toml`'s `[tool.pyright].include`).

### Inputs

```
bts typecheck [-v]
              [--paths PATH ...]
```

### Printouts

```
0 errors, 0 warnings, 0 informations
```

### Market-relevance commitment

The typing surface is strict — `reportUnknownMemberType`, `reportUnknownArgumentType`, all `reportUnknown*` are errors. New `Any`, `cast(..., Any)`, `# type: ignore`, `# pyright: ignore`, `# noqa` are all forbidden by AGENTS law 3 and ratcheted by `tools/typing_gate.py`. A simulator that doesn't type-check is a simulator with hidden contracts; bts holds itself to the same typing standard it expects of strategies that will run in paper / live.

---

## `bts gate <name>`

`bts gate <name>` runs a specific CI gate locally. Wraps the `tools/check_*.py` (bloat budgets) and `tools/*_gate.py` (structural gates) so the operator runs the same gate locally that CI runs at PR time.

### Inputs

```
bts gate {lint|typing|honesty|fail_loud|cc|slice|version|ruleset|
          module_budgets|docstrings|file_size_balance|
          test_code_ratio|no_swallowed_violations|all}
          [-v]
```

### Printouts

Each gate prints its own banner: `MODULE BUDGET GATE -- PASS`, `TYPING GATE -- FAIL`, etc. `bts gate all` runs every gate sequentially and stops at the first non-zero exit.

### Market-relevance commitment

Local-vs-CI parity is the contract. If `bts gate lint` passes locally and `pr_checks_lint` fails in CI, that's a bug in the wrapper, not a flaky CI. The same applies to every gate. Operators rely on `bts gate` to predict CI state before pushing.

---

## `bts notebook`

`bts notebook` executes / converts a Jupyter notebook via `jupyter nbconvert`.

### Inputs

```
bts notebook --path PATH
             [--no-execute]
             [--output-format {ipynb,html,script}]
             [-v]
```

### Printouts

`jupyter nbconvert` output on stderr; converted artefact on stdout when `--output-format script`.

### Market-relevance commitment

The notebook surface is for operator analysis only — it does NOT participate in the strategy → paper → live pipeline. A failing notebook does not fail honesty gates; it fails operator UX.

---

## `bts version`

`bts version` prints `bts <pyproject version>` and exits 0.

### Inputs

```
bts version
```

### Printouts

```
bts 2.0.5
```

### Market-relevance commitment

The printed version is `importlib.metadata.version('backtest_simulator')` read from the installed package metadata. The slice asserts this matches `[project].version` in `pyproject.toml`. The version surfaces the contract version: every PR bumps it (AGENTS law 5), so a paper / live system pinning a specific version pins a specific honesty contract.
