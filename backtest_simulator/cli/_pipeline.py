"""Shared pipeline glue for `bts run` / `bts sweep` (preflight, training, picking)."""

# Moved from `/tmp/bts_sweep.py`. Contains:
#   - ClickHouse env-var defaults + tunnel preflight + seed-price lookup;
#   - `ensure_trained` (UEL run unless cached);
#   - `pick_decoders` (quantile-filtered ranking; file-mode retrain);
#   - per-decoder retrain helper.
#
# Operator-mandated workflow path: `bts run`/`bts sweep` import from here
# so the heavy pipeline orchestration is testable in isolation and shared
# between the two subcommands.
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    import polars as pl  # pragma: no cover - type-only import

# ClickHouse tunnel host / port / user / database have safe operator-
# local defaults (the documented `ssh -fN -L 18123:...` shape). The
# password must come from the operator's environment — committing it
# to the package would leak a shared credential. `_require_password`
# below fails loud with the env-var name to set if it's missing.
os.environ.setdefault('CLICKHOUSE_HOST', '127.0.0.1')
os.environ.setdefault('CLICKHOUSE_PORT', '18123')
os.environ.setdefault('CLICKHOUSE_USER', 'default')
os.environ.setdefault('CLICKHOUSE_DATABASE', 'origo')


_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_DOTENV_PATH: Final[Path] = _REPO_ROOT / '.env'


def _hydrate_environ_from_dotenv() -> None:
    """Inject `<repo>/.env` keys into `os.environ` if not already set.

    Why: subprocess workers (`bts sweep`'s per-run forks) re-read
    `os.environ` via `ClickHouseFeed.from_env()` and never see the
    project `.env` on their own. Loading once at module import in the
    parent makes the values inherit naturally to every child fork.
    `setdefault` is used so a real shell-exported value still wins —
    the `.env` is the bottom of the resolution stack.
    """
    if not _DOTENV_PATH.is_file():
        return
    for raw in _DOTENV_PATH.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, _, v = line.partition('=')
        key = k.strip()
        val = v.strip()
        if not key:
            continue
        # Strip a single matching pair of surrounding quotes — `.env`
        # convention so values with whitespace round-trip cleanly.
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
            val = val[1:-1]
        os.environ.setdefault(key, val)


# Run on import so any subprocess forked AFTER this module loads in
# the parent inherits the populated environment.
_hydrate_environ_from_dotenv()


def _require_password() -> str:
    """Return ClickHouse password from environment (which now includes .env).

    `.env` is hydrated into `os.environ` at module import (see
    `_hydrate_environ_from_dotenv`), so a single `os.environ` lookup
    covers shell-exported, dotenv-loaded, and subprocess-inherited
    paths uniformly. Fails loud with both setup paths if neither is
    available — never swallowed: bts sweep / run cannot proceed
    without a real password.
    """
    pw = os.environ.get('CLICKHOUSE_PASSWORD')
    if pw:
        return pw
    msg = (
        'ClickHouse password unavailable. Set CLICKHOUSE_PASSWORD in '
        f'your shell or add `CLICKHOUSE_PASSWORD=...` to {_DOTENV_PATH} '
        '(the project-root .env file is gitignored). Without one of '
        'these, `bts run` / `bts sweep` cannot reach the tdw database.'
    )
    raise RuntimeError(msg)

SYMBOL: Final[str] = 'BTCUSDT'
EXP_DIR: Final[Path] = Path('/tmp/bts_sweep/experiments')
WORK_DIR: Final[Path] = Path('/tmp/bts_sweep/run')

# Hyperparameter columns Limen's logreg_binary cares about.
_PARAM_COLS: Final[tuple[str, ...]] = (
    'C', 'class_weight', 'feature_groups', 'frac_diff_d',
    'max_iter', 'penalty', 'q', 'roc_period',
    'scaler_type', 'shift', 'solver', 'tol',
)

# Numeric columns Limen writes as strings in results.csv. Cast to Float64
# upfront so `filtered.sort` doesn't lex-sort and float `NaN` to the top.
_NUMERIC_COLS: Final[tuple[str, ...]] = (
    'backtest_trades_count',
    'backtest_mean_kelly_pct',
    'backtest_total_return_net_pct',
    'confusion_tp_mean_return_pct',
    'fpr',
)


def preflight_tunnel() -> None:
    """Verify the ClickHouse tunnel by querying for one recent BTCUSDT trade.

    Three failure modes reported separately so the operator knows
    exactly what to fix:

      1. Password missing — the `_require_password()` error surfaces
         unchanged (env-var or `<repo>/.env`).
      2. Tunnel down / port not reachable — narrow socket-level probe
         on `(host, port)` BEFORE invoking `clickhouse_connect`. Emits
         the exact `ssh -fN -L` reopen command. Codex pinned this:
         `clickhouse_connect.get_client()` runs autoconnect server
         queries during init, so wrapping `get_client()` in a generic
         try/except would misclassify auth refusals as tunnel failures.
      3. Database / auth / query failed — `get_client(...)` and the
         probe query raised. Surface the underlying error verbatim
         (auth refused, table missing, etc.) instead of pretending
         the tunnel is down.
    """
    import socket

    import clickhouse_connect

    # Resolve password OUTSIDE the try blocks — its failure is its own
    # error class and must not be repackaged as a tunnel hint.
    password = _require_password()
    host = os.environ['CLICKHOUSE_HOST']
    port = int(os.environ['CLICKHOUSE_PORT'])
    user = os.environ['CLICKHOUSE_USER']
    database = os.environ['CLICKHOUSE_DATABASE']

    # Tunnel-class: narrow socket-level connectivity probe. If the
    # local tunnel port is not bound or the server side is not
    # reachable through it, the socket connect raises here — well
    # before clickhouse_connect's init can confuse the diagnosis.
    try:
        sock = socket.create_connection((host, port), timeout=2.0)
    except OSError as exc:
        msg = (
            f'ClickHouse tunnel unreachable at {host}:{port} ({exc!r}). '
            f'Open the tunnel with: '
            f'ssh -fN -L {port}:127.0.0.1:8123 root@<clickhouse-host>'
        )
        raise RuntimeError(msg) from exc
    sock.close()

    # Auth/query-class: get_client() may issue server queries during
    # init (autoconnect common-settings probe), and the probe query
    # below validates database access. Both raise into the same class.
    try:
        client = clickhouse_connect.get_client(
            host=host, port=port, username=user, password=password,
            database=database,
        )
        rows = client.query(
            'SELECT toString(datetime) FROM origo.binance_daily_spot_trades '
            'ORDER BY datetime DESC LIMIT 1',
        ).result_rows
    except Exception as exc:
        msg = (
            f'ClickHouse query failed against {host}:{port}/{database} '
            f'({exc!r}). The tunnel is reachable (socket probe OK) but '
            f'auth or database access was refused — verify '
            f'CLICKHOUSE_PASSWORD matches what the server expects.'
        )
        raise RuntimeError(msg) from exc
    if not rows:
        msg = (
            f'origo.binance_daily_spot_trades returned zero rows on '
            f'{host}:{port}; the tdw table is empty or filtered out.'
        )
        raise RuntimeError(msg)


def seed_price_at(ts: datetime) -> Decimal:
    """Return the first BTCUSDT trade price at or after `ts` from ClickHouse.

    Used by `bts run` / `bts sweep` to seed strategy `estimated_price`
    at window start. Raises if the tape has no tick at or after `ts`.
    """
    import clickhouse_connect
    client = clickhouse_connect.get_client(
        host=os.environ['CLICKHOUSE_HOST'],
        port=int(os.environ['CLICKHOUSE_PORT']),
        username=os.environ['CLICKHOUSE_USER'],
        password=_require_password(),
        database=os.environ['CLICKHOUSE_DATABASE'],
    )
    rows = client.query(
        'SELECT price FROM origo.binance_daily_spot_trades '
        'WHERE datetime >= %(s)s ORDER BY datetime LIMIT 1',
        parameters={'s': ts.strftime('%Y-%m-%d %H:%M:%S.%f')},
    ).result_rows
    if not rows:
        msg = f'No ClickHouse tick at or after {ts.isoformat()} for seed price.'
        raise RuntimeError(msg)
    return Decimal(str(rows[0][0]))


def ensure_trained(n_permutations: int) -> None:
    """Run UEL unless `results.csv` is already cached at `EXP_DIR`."""
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    exp_file = EXP_DIR / 'exp.py'
    if not exp_file.exists():
        exp_file.write_text(
            'from limen.sfd import logreg_binary as _base\n'
            'def params():\n'
            '    return _base.params()\n'
            'def manifest():\n'
            '    return _base.manifest()\n',
        )
    results_csv = EXP_DIR / 'results.csv'
    if results_csv.is_file():
        return
    from backtest_simulator.pipeline import ExperimentPipeline
    pipe = ExperimentPipeline(experiment_dir=EXP_DIR)
    loaded = pipe.load_from_file(exp_file)
    pipe.run(loaded, experiment_name='sweep', n_permutations=n_permutations, seed=42)


def row_params(row: dict[str, object]) -> dict[str, object]:
    """Extract the param dict for one CSV row, coercing numeric strings."""
    out: dict[str, object] = {}
    for k in _PARAM_COLS:
        v = row.get(k)
        if isinstance(v, str):
            out[k] = _coerce_param_string(v.strip())
        else:
            out[k] = v
    return out


def _coerce_param_string(s: str) -> int | float | str:
    """Coerce a CSV cell to int → float → str (cascading attempt)."""
    try:
        return int(s)
    except (ValueError, InvalidOperation):
        try:
            return float(s)
        except (ValueError, InvalidOperation):
            return s


def train_single_decoder(sub_dir: Path, params: dict[str, object]) -> None:
    """Train one decoder in its own experiment dir with `n_permutations=1`."""
    if (sub_dir / 'results.csv').is_file():
        return
    sub_dir.mkdir(parents=True, exist_ok=True)
    body = 'def params():\n    return {\n'
    for k in _PARAM_COLS:
        body += f'        {k!r}: [{params[k]!r}],\n'
    body += '    }\n'
    body += 'def manifest():\n'
    body += '    from limen.sfd import logreg_binary as _base\n'
    body += '    return _base.manifest()\n'
    exp_file = sub_dir / 'exp.py'
    exp_file.write_text(body)
    from backtest_simulator.pipeline import ExperimentPipeline
    pipe = ExperimentPipeline(experiment_dir=sub_dir)
    loaded = pipe.load_from_file(exp_file)
    pipe.run(loaded, experiment_name='single', n_permutations=1, seed=42)


def pick_decoders(
    n: int, *,
    trades_q_range: tuple[float, float] | None = None,
    tp_min_q: float | None = None,
    fpr_max_q: float | None = None,
    kelly_min_q: float | None = None,
    trade_count_min_q: float | None = None,
    net_return_min_q: float | None = None,
    input_from_file: str | None = None,
) -> tuple[list[tuple[int, Decimal, Path, int]], int]:
    """Quantile-filtered decoder pool, ranked by kelly desc / net-return desc.

    Returns `(picks, candidate_pool_size)` where:
      - `picks` is a list of `(perm_id, kelly_pct, exp_dir, display_id)`.
        In file-mode each picked decoder is retrained in its own
        sub-dir and `perm_id` is `0`; `display_id` is the file's `id`.
      - `candidate_pool_size` is the raw candidate pool count (the
        size of the search space the operator considered before any
        filters). Used as the multiple-testing inflation factor for
        DSR (slice #17 Task 17, codex round 2 P1: visible pick
        count under-deflates when the search space is much larger).
    """
    import polars as pl

    from backtest_simulator.pipeline import ExperimentPipeline

    if input_from_file is not None:
        file_path = Path(input_from_file).expanduser()
        if not file_path.is_file():
            msg = f'--input-from-file: {file_path} does not exist.'
            raise FileNotFoundError(msg)
        results = pl.read_csv(file_path)
        print(f'  loaded filter pool from {file_path.name}: {results.height} rows', flush=True)
    else:
        pipe = ExperimentPipeline(experiment_dir=EXP_DIR)
        results = pipe.read_results()
    casts = [
        pl.col(c).cast(pl.Float64, strict=False).alias(c)
        for c in _NUMERIC_COLS
        if c in results.columns and results[c].dtype == pl.Utf8
    ]
    if casts:
        results = results.with_columns(casts)
    rank_by = ['backtest_mean_kelly_pct', 'backtest_total_return_net_pct']
    clean_cols = [
        c for c in (*rank_by, 'backtest_mean_kelly_pct')
        if c in results.columns
    ]
    before = results.height
    results = results.drop_nulls(subset=clean_cols).filter(
        pl.all_horizontal([pl.col(c).is_not_nan() for c in clean_cols]),
    )
    dropped = before - results.height
    if dropped > 0:
        print(
            f'  dropped {dropped} row(s) with null/NaN in rank+kelly columns '
            f'({results.height} usable)',
            flush=True,
        )
    range_quantiles: dict[str, tuple[float, float]] = {}
    if trades_q_range is not None:
        range_quantiles['backtest_trades_count'] = trades_q_range
    one_sided: list[tuple[str, str, float]] = []
    if tp_min_q is not None:
        one_sided.append(('confusion_tp_mean_return_pct', '>', tp_min_q))
    if fpr_max_q is not None:
        one_sided.append(('fpr', '<=', fpr_max_q))
    if kelly_min_q is not None:
        one_sided.append(('backtest_mean_kelly_pct', '>=', kelly_min_q))
    if trade_count_min_q is not None:
        one_sided.append(('backtest_trades_count', '>=', trade_count_min_q))
    if net_return_min_q is not None:
        one_sided.append(('backtest_total_return_net_pct', '>=', net_return_min_q))

    def _q(col: str, pct: float) -> float:
        series = results[col]
        if series.dtype == pl.Utf8:
            series = series.cast(pl.Float64, strict=False)
        series = series.drop_nulls().drop_nans()
        return float(series.quantile(pct))

    def _numeric_col(col: str) -> pl.Expr:
        dtype = results[col].dtype
        if dtype == pl.Utf8:
            return pl.col(col).cast(pl.Float64, strict=False)
        return pl.col(col)

    expr = pl.lit(True)
    report_lines: list[str] = []
    for col, (lo_pct, hi_pct) in range_quantiles.items():
        lo_val = _q(col, lo_pct)
        hi_val = _q(col, hi_pct)
        axis_expr = (_numeric_col(col) >= lo_val) & (_numeric_col(col) <= hi_val)
        kept = results.filter(axis_expr).height
        expr = expr & axis_expr
        report_lines.append(
            f'    {col} ∈ [q{int(lo_pct*100)}={lo_val:.4g}, q{int(hi_pct*100)}={hi_val:.4g}]'
            f'   {kept}/{results.height}',
        )
    for col, op, q in one_sided:
        val = _q(col, q)
        if op == '>':
            axis_expr = _numeric_col(col) > val
        elif op == '>=':
            axis_expr = _numeric_col(col) >= val
        elif op == '<':
            axis_expr = _numeric_col(col) < val
        elif op == '<=':
            axis_expr = _numeric_col(col) <= val
        else:
            msg = f'unsupported op in one_sided: {op!r}'
            raise ValueError(msg)
        kept = results.filter(axis_expr).height
        expr = expr & axis_expr
        report_lines.append(
            f'    {col} {op} q{int(q*100)}={val:.4g}   {kept}/{results.height}',
        )
    filtered = results.filter(expr)
    if report_lines:
        print('  filter (axis kept / total):', flush=True)
        for line in report_lines:
            print(line, flush=True)
        print(
            f'    AND combined   {filtered.height}/{results.height} decoders',
            flush=True,
        )
    else:
        print(
            f'  no filters given — ranking all {results.height} decoders as-is',
            flush=True,
        )
    if filtered.height == 0:
        msg = (
            'No decoders passed the combined quantile filter. '
            'Loosen --trades-q-range / --tp-min-q / --fpr-max-q, or train '
            'more candidates with --n-permutations and rm -rf the cache.'
        )
        raise RuntimeError(msg)
    ranked = filtered.sort(rank_by, descending=[True, True])
    take = min(n, ranked.height)
    if take < n:
        print(
            f'NOTE: requested {n} decoders, only {take} pass the filter.',
            flush=True,
        )
    picks: list[tuple[int, Decimal, Path, int]] = []
    for i in range(take):
        file_id = int(ranked['id'][i])
        kelly = Decimal(str(ranked['backtest_mean_kelly_pct'][i]))
        if input_from_file is not None:
            row = {k: ranked[k][i] for k in _PARAM_COLS}
            params = row_params(row)
            # Cache key includes source filename stem AND a content
            # hash of the picked row's params. Without this, two
            # different input files (e.g. `min.csv` vs `max.csv`)
            # with the same `id` column silently aliased to the
            # FIRST-cached training, even when the rows had
            # completely different hyperparameters. The hash widens
            # the key so different content produces a different
            # sub_dir; the stem keeps the directory name human-
            # readable. file_path is in scope from the
            # input_from_file resolution above.
            params_hash = hashlib.sha256(
                json.dumps(
                    {k: str(v) for k, v in sorted(params.items())},
                    sort_keys=True,
                ).encode('utf-8'),
            ).hexdigest()
            # Full 64-char SHA-256, NOT truncated to 8 (codex round-1
            # P1: 32-bit truncation = ~65k birthday-paradox collision
            # window; the saving in directory-name length is not
            # worth the silent-alias risk).
            sub_dir = (
                WORK_DIR / 'trained_from_file'
                / f'{file_path.stem}_id_{file_id}_{params_hash}'
            )
            # Surface a short prefix in the operator log so two
            # nearby cache hits / misses are visually distinguishable
            # without making the line unreadable.
            print(
                f'  training file-id {file_id} from {file_path.name} '
                f'(params_hash={params_hash[:8]}...)  kelly={kelly}  ...',
                flush=True,
            )
            train_single_decoder(sub_dir, params)
            picks.append((0, kelly, sub_dir, file_id))
        else:
            picks.append((file_id, kelly, EXP_DIR, file_id))
    return picks, before
