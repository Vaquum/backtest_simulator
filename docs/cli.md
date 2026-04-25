# `bts` — backtest_simulator master CLI

The `bts` command is the operator-facing surface of the package. All
debugging, gate invocation, sweep / single-window runs, lint / typing,
and notebook execution flow through `bts <subcommand>`. The CLI is
intentionally the only entry point: nothing else in the project is
meant to be invoked directly.

Verbosity flags `-v`, `-vv`, `-vvv` are accepted on every subcommand
that runs work and progressively unmask logging:

| flag | level | behaviour |
| --- | --- | --- |
| (none) | ERROR | CLI's own per-run summary is the only output on stdout. |
| `-v` | INFO | Pipeline progress + Praxis / Nexus / Limen INFO logs. |
| `-vv` | DEBUG | Adds DEBUG; useful for "why did this fill?" diagnostics. |
| `-vvv` | NOTSET | Everything: tqdm bars, structlog details, asyncio traces. |

---

## run

`bts run` runs **one backtest window for one decoder** and prints either
a human-readable one-line summary plus per-trade detail (`--output-format
text`, default) or a structured JSON report (`--output-format json`).

```
bts run [-v|-vv|-vvv]
        --window-start ISO8601 --window-end ISO8601
        [--decoder-id INT | --n-decoders INT | --input-from-file PATH]
        [--experiment-dir PATH]
        [--output-format {text,json}]
        [--seed INT]
```

The JSON report carries `decoder_id`, `kelly_pct`, `window_start`,
`window_end`, `orders`, `trades`, `declared_stops`, plus the metric
slots `book_gap_max_seconds`, `slippage_realised_bps`, and
`market_impact_realised_bps`. The latter three are populated by tasks
11/12/13 of slice #17; until then they are `null`.

## sweep

`bts sweep` runs the backtest pipeline over **N decoders × M days**.
Decoders are picked from a UEL-trained pool (default) or from a
`results.csv` filter pool (`--input-from-file`). Each window runs in a
fresh Python subprocess for state isolation.

```
bts sweep [-v|-vv|-vvv]
          [--n-decoders INT] [--n-permutations INT]
          [--trading-hours-start HH:MM --trading-hours-end HH:MM]
          [--replay-period-start YYYY-MM-DD --replay-period-end YYYY-MM-DD]
          [--input-from-file PATH]
          [--trades-q-range LO,HI]
          [--tp-min-q FLOAT] [--fpr-max-q FLOAT] [--kelly-min-q FLOAT]
          [--trade-count-min-q FLOAT] [--net-return-min-q FLOAT]
```

Pair rule: within `(--trading-hours-start, --trading-hours-end)` and
`(--replay-period-start, --replay-period-end)` either both values are
given or neither. Giving one without the other fails loud.

## enrich

`bts enrich` joins `<experiment>/results.csv` with an optional
`backtest_results.parquet` into `<experiment>/results_with_backtest.csv`.
Does NOT run a backtest sweep — only enriches existing results.

```
bts enrich --experiment PATH [--backtest-parquet PATH] [--out PATH]
```

## test

`bts test` is the operator-mandated pytest entry point. The slice's
operational protocol forbids invoking `pytest` directly; everything
goes through this wrapper so verbosity, marker filters, and the
integration / honesty / cli scope are unified under one tool.

```
bts test [-v|-vv]
         [-k PATTERN]
         [--honesty | --integration | --cli]
         [-- ...args forwarded to pytest verbatim...]
```

## lint

`bts lint` runs `ruff check` on the package + tools + tests + scripts.
On a clean tree the last line of output is `All checks passed!`.

```
bts lint [-v]
         [--paths PATH ...]
         [--fix]
```

## typecheck

`bts typecheck` runs pyright in strict mode against the package root
(matches `pyproject.toml`'s `[tool.pyright].include`).

```
bts typecheck [-v]
              [--paths PATH ...]
```

## gate

`bts gate <name>` runs a specific CI gate locally. Wraps the existing
`scripts/check_*.py` and `tools/*_gate.py` scripts so the operator runs
the same gate locally that CI runs at PR time.

```
bts gate {lint|typing|honesty|fail_loud|cc|slice|version|ruleset|
          module_budgets|docstrings|file_size_balance|
          test_code_ratio|no_swallowed_violations|all}
```

`bts gate all` runs every gate sequentially and stops at the first
non-zero exit.

## notebook

`bts notebook` executes / converts a Jupyter notebook via
`jupyter nbconvert`.

```
bts notebook --path PATH
             [--no-execute]
             [--output-format {ipynb,html,script}]
```

When `--output-format script` is combined with `--execute`, the converted
script is fed to `python -` for end-to-end execution; this is what the
slice's MVC `[ -f notebooks/sweep_and_analyze.ipynb ] && jupyter nbconvert
--to script --stdout … | python -` predicate calls.

## version

`bts version` prints `bts <pyproject version>` and exits 0.

```
bts version
```

The printed version is `importlib.metadata.version('backtest_simulator')`
read from the installed package metadata. The slice asserts this matches
`[project].version` in `pyproject.toml`.
