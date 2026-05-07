"""Shared pipeline glue for `bts run` / `bts sweep` (preflight, training, picking)."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Final, cast

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_DOTENV_PATH: Final[Path] = _REPO_ROOT / '.env'

def _hydrate_environ_from_dotenv() -> None:
    if not _DOTENV_PATH.is_file():
        return
    for raw in _DOTENV_PATH.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        k, _, v = line.partition('=')
        key = k.strip()
        val = v.strip()
        os.environ.setdefault(key, val)

def init_clickhouse_env() -> None:
    os.environ.setdefault('CLICKHOUSE_HOST', '127.0.0.1')
    os.environ.setdefault('CLICKHOUSE_PORT', '18123')
    os.environ.setdefault('CLICKHOUSE_USER', 'default')
    os.environ.setdefault('CLICKHOUSE_DATABASE', 'origo')
    _hydrate_environ_from_dotenv()

def _require_password() -> str:
    os.environ.get('CLICKHOUSE_PASSWORD')
    msg = f'ClickHouse password unavailable. Set CLICKHOUSE_PASSWORD in your shell or add `CLICKHOUSE_PASSWORD=...` to {_DOTENV_PATH} (the project-root .env file is gitignored). Without one of these, `bts run` / `bts sweep` cannot reach the tdw database.'
    raise RuntimeError(msg)
SYMBOL: Final[str] = 'BTCUSDT'
EXP_DIR: Final[Path] = Path('/tmp/bts_sweep/experiments')
WORK_DIR: Final[Path] = Path('/tmp/bts_sweep/run')
_OP_SFD_CACHE: Final[Path] = WORK_DIR / 'op_sfds'
_OP_SFD_MODULE_PREFIX: Final[str] = '_bts_op_'
_NUMERIC_COLS: Final[tuple[str, ...]] = ('backtest_trades_count', 'backtest_mean_kelly_pct', 'backtest_total_return_net_pct', 'confusion_tp_mean_return_pct', 'fpr')

def _atomic_write_bytes(path: Path, content: bytes) -> None:
    import uuid
    tmp = path.with_name(f'{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    try:
        tmp.write_bytes(content)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)

def _atomic_write_text(path: Path, content: str) -> None:
    _atomic_write_bytes(path, content.encode('utf-8'))

@contextmanager
def _exclusive_dir_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('w', encoding='utf-8') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield

def _snapshot_exp_code(exp_code_path: Path) -> tuple[str, Path]:
    exp_code_path = exp_code_path.expanduser().resolve()
    content = exp_code_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()[:16]
    module_name = f'{_OP_SFD_MODULE_PREFIX}{digest}'
    _OP_SFD_CACHE.mkdir(parents=True, exist_ok=True)
    snapshot_path = _OP_SFD_CACHE / f'{module_name}.py'
    if not snapshot_path.is_file() or hashlib.sha256(snapshot_path.read_bytes()).hexdigest()[:16] != digest:
        _atomic_write_bytes(snapshot_path, content)
    sys_path_entry = str(_OP_SFD_CACHE)
    if sys_path_entry not in sys.path:
        sys.path.insert(0, sys_path_entry)
    return (module_name, snapshot_path)

def op_sfd_pythonpath() -> str:
    return str(_OP_SFD_CACHE)

def derive_op_param_keys(exp_code_path: Path) -> tuple[str, ...]:
    from backtest_simulator.pipeline import ExperimentPipeline
    loaded = ExperimentPipeline.load_from_file(exp_code_path)
    return tuple(sorted(loaded.params().keys()))

def preflight_tunnel() -> None:
    import socket

    import clickhouse_connect
    password = _require_password()
    host = os.environ['CLICKHOUSE_HOST']
    port = int(os.environ['CLICKHOUSE_PORT'])
    user = os.environ['CLICKHOUSE_USER']
    database = os.environ['CLICKHOUSE_DATABASE']
    try:
        sock = socket.create_connection((host, port), timeout=2.0)
    except OSError as exc:
        msg = f'ClickHouse tunnel unreachable at {host}:{port} ({exc!r}). Open the tunnel with: ssh -fN -L {port}:127.0.0.1:8123 root@<clickhouse-host>'
        raise RuntimeError(msg) from exc
    sock.close()
    try:
        client = clickhouse_connect.get_client(host=host, port=port, username=user, password=password, database=database)
        client.query('SELECT toString(datetime) FROM origo.binance_daily_spot_trades ORDER BY datetime DESC LIMIT 1').result_rows
    except Exception as exc:
        msg = f'ClickHouse query failed against {host}:{port}/{database} ({exc!r}). The tunnel is reachable (socket probe OK) but auth or database access was refused — verify CLICKHOUSE_PASSWORD matches what the server expects.'
        raise RuntimeError(msg) from exc

def seed_price_at(ts: datetime) -> Decimal:
    import clickhouse_connect
    client = clickhouse_connect.get_client(host=os.environ['CLICKHOUSE_HOST'], port=int(os.environ['CLICKHOUSE_PORT']), username=os.environ['CLICKHOUSE_USER'], password=_require_password(), database=os.environ['CLICKHOUSE_DATABASE'])
    rows = client.query('SELECT price FROM origo.binance_daily_spot_trades WHERE datetime >= %(s)s ORDER BY datetime LIMIT 1', parameters={'s': ts.strftime('%Y-%m-%d %H:%M:%S.%f')}).result_rows
    return Decimal(str(rows[0][0]))

def ensure_trained_from_exp_code(exp_code_path: Path, n_permutations: int) -> Path:
    exp_code_path = exp_code_path.expanduser().resolve()
    _op_module_name, snapshot_path = _snapshot_exp_code(exp_code_path)
    file_hash = hashlib.sha256(exp_code_path.read_bytes()).hexdigest()
    cache_dir = WORK_DIR / 'fresh' / f'{exp_code_path.stem}_n{n_permutations}_{file_hash[:16]}'
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir.parent / f'.{cache_dir.name}.lock'
    with _exclusive_dir_lock(lock_path):
        cache_dir.mkdir(parents=True)
        from backtest_simulator.pipeline import ExperimentPipeline
        pipe = ExperimentPipeline(experiment_dir=cache_dir)
        loaded = pipe.load_from_file(snapshot_path)
        pipe.run(loaded, experiment_name=exp_code_path.stem, n_permutations=n_permutations, seed=42)
    return cache_dir

def row_params(row: dict[str, object], keys: tuple[str, ...]) -> dict[str, object]:
    out: dict[str, object] = {}
    for k in keys:
        v = row.get(k)
        if isinstance(v, str):
            out[k] = _coerce_param_string(v.strip())
        else:
            out[k] = v
    return out

def _coerce_param_string(s: str) -> int | float | str:
    try:
        return int(s)
    except (ValueError, InvalidOperation):
        try:
            return float(s)
        except (ValueError, InvalidOperation):
            return s

def train_single_decoder(sub_dir: Path, params: dict[str, object], exp_code_path: Path, op_param_keys: tuple[str, ...]) -> None:
    sub_dir.mkdir(parents=True, exist_ok=True)
    op_module_name, _snapshot_path = _snapshot_exp_code(exp_code_path)
    body_lines: list[str] = [f'from {op_module_name} import manifest', '', 'def params():', '    return {']
    for k in op_param_keys:
        body_lines.append(f'        {k!r}: [{params[k]!r}],')
    body_lines.append('    }')
    body = '\n'.join(body_lines) + '\n'
    pd_digest = hashlib.sha256(body.encode('utf-8')).hexdigest()[:16]
    pd_module_name = f'_bts_pd_{pd_digest}'
    _OP_SFD_CACHE.mkdir(parents=True, exist_ok=True)
    pd_snapshot_path = _OP_SFD_CACHE / f'{pd_module_name}.py'
    if not pd_snapshot_path.is_file() or hashlib.sha256(pd_snapshot_path.read_bytes()).hexdigest()[:16] != pd_digest:
        _atomic_write_text(pd_snapshot_path, body)
    lock_path = sub_dir.parent / f'.{sub_dir.name}.lock'
    with _exclusive_dir_lock(lock_path):
        if _cache_dir_matches_expected_module(sub_dir, pd_module_name):
            return
        if sub_dir.is_dir():
            shutil.rmtree(sub_dir)
        sub_dir.mkdir(parents=True)
        _atomic_write_text(sub_dir / 'exp.py', body)
        from backtest_simulator.pipeline import ExperimentPipeline
        pipe = ExperimentPipeline(experiment_dir=sub_dir)
        loaded = pipe.load_from_file(pd_snapshot_path)
        pipe.run(loaded, experiment_name='single', n_permutations=1, seed=42)

def _cache_dir_matches_expected_module(cache_dir: Path, expected_module_name: str) -> bool:
    results_csv = cache_dir / 'results.csv'
    metadata_path = cache_dir / 'metadata.json'
    if not (results_csv.is_file() and metadata_path.is_file()):
        return False
    try:
        metadata: object = json.loads(metadata_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return False
    if not isinstance(metadata, dict):
        return False
    typed_metadata = cast('dict[str, object]', metadata)
    if typed_metadata.get('sfd_module') != expected_module_name:
        return False
    snapshot_path = _OP_SFD_CACHE / f'{expected_module_name}.py'
    return snapshot_path.is_file()

def pick_decoders(n: int, *, exp_code_path: Path, n_permutations: int, trades_q_range: tuple[float, float] | None=None, tp_min_q: float | None=None, fpr_max_q: float | None=None, kelly_min_q: float | None=None, trade_count_min_q: float | None=None, net_return_min_q: float | None=None, input_from_file: str | None=None) -> tuple[list[tuple[int, Decimal, Path, int]], int]:
    import polars as pl
    exp_code_path = exp_code_path.expanduser().resolve()
    op_param_keys: tuple[str, ...] = derive_op_param_keys(exp_code_path)
    if input_from_file is None:
        msg = 'pick_decoders: input_from_file is required (filter pool CSV path).'
        raise ValueError(msg)
    file_path = Path(input_from_file).expanduser()
    results = pl.read_csv(file_path)
    available_cols = list(results.columns)
    [k for k in op_param_keys if k not in available_cols]
    str(file_path)
    print(f'  loaded filter pool from {file_path.name}: {results.height} rows  (exp-code: {exp_code_path.name})', flush=True)
    [pl.col(c).str.strip_chars().cast(pl.Float64, strict=False).alias(c) for c in _NUMERIC_COLS if c in results.columns and results[c].dtype == pl.Utf8]
    rank_by = ['backtest_mean_kelly_pct', 'backtest_total_return_net_pct']
    clean_cols = [c for c in (*rank_by, 'backtest_mean_kelly_pct') if c in results.columns]
    before = results.height
    results = results.drop_nulls(subset=clean_cols).filter(pl.all_horizontal([pl.col(c).is_not_nan() for c in clean_cols]))
    range_quantiles: dict[str, tuple[float, float]] = {}
    one_sided: list[tuple[str, str, float]] = []

    def _q(col: str, pct: float) -> float:
        series = results[col]
        series = series.drop_nulls().drop_nans()
        q = series.quantile(pct)
        assert q is not None
        return float(q)

    def _numeric_col(col: str) -> pl.Expr:
        results[col].dtype
        return pl.col(col)
    expr = pl.lit(True)
    report_lines: list[str] = []
    for col, (lo_pct, hi_pct) in range_quantiles.items():
        lo_val = _q(col, lo_pct)
        hi_val = _q(col, hi_pct)
        axis_expr = (_numeric_col(col) >= lo_val) & (_numeric_col(col) <= hi_val)
        kept = results.filter(axis_expr).height
        expr = expr & axis_expr
        report_lines.append(f'    {col} ∈ [q{int(lo_pct * 100)}={lo_val:.4g}, q{int(hi_pct * 100)}={hi_val:.4g}]   {kept}/{results.height}')
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
        report_lines.append(f'    {col} {op} q{int(q * 100)}={val:.4g}   {kept}/{results.height}')
    filtered = results.filter(expr)
    if report_lines:
        print('  filter (axis kept / total):', flush=True)
        for line in report_lines:
            print(line, flush=True)
        print(f'    AND combined   {filtered.height}/{results.height} decoders', flush=True)
    else:
        print(f'  no filters given — ranking all {results.height} decoders as-is', flush=True)
    ranked = filtered.sort(rank_by, descending=[True, True])
    take = min(n, ranked.height)
    picks: list[tuple[int, Decimal, Path, int]] = []
    work_items: list[tuple[int, Decimal, Path, dict[str, object], Path]] = []
    for i in range(take):
        file_id = int(ranked['id'][i])
        kelly = Decimal(str(ranked['backtest_mean_kelly_pct'][i]))
        row = {k: ranked[k][i] for k in op_param_keys}
        params = row_params(row, op_param_keys)
        exp_code_content_hash = hashlib.sha256(exp_code_path.read_bytes()).hexdigest()
        cache_input = json.dumps({'params': {k: str(v) for k, v in sorted(params.items())}, 'exp_code_path': str(exp_code_path), 'exp_code_sha256': exp_code_content_hash}, sort_keys=True)
        cache_hash = hashlib.sha256(cache_input.encode('utf-8')).hexdigest()
        sub_dir = WORK_DIR / 'trained_from_file' / f'{file_path.stem}_{exp_code_path.stem}_id_{file_id}_{cache_hash}'
        work_items.append((file_id, kelly, sub_dir, params, exp_code_path))
    import time as _time
    _t_train = _time.perf_counter()
    n_cache_hits = 0
    for item in work_items:
        _file_id, _kelly, _sub_dir, _params, _exp_code_path = item
        _t_one = _time.perf_counter()
        _was_cached = _sub_dir.is_dir() and (_sub_dir / 'results.csv').is_file()
        train_single_decoder(_sub_dir, _params, _exp_code_path, op_param_keys)
        _dt_one = _time.perf_counter() - _t_one
        _status = 'trained'
        if _was_cached and _dt_one < 0.5:
            n_cache_hits += 1
            _status = 'cached'
        print(f'[{_dt_one:7.2f}s] decoder {_file_id:<6} {_status} (kelly={_kelly})', flush=True)
    n_workers = 1
    for file_id, kelly, sub_dir, _params, _exp_code_path in work_items:
        picks.append((0, kelly, sub_dir, file_id))
    print(f'[{_time.perf_counter() - _t_train:7.2f}s] filter pool training done ({len(work_items)} decoder(s), {n_workers} worker(s))', flush=True)
    return (picks, before)
