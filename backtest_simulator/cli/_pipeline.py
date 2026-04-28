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
from typing import TYPE_CHECKING, Final, cast

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

# Operator-supplied `--exp-code` files are snapshotted into this dir
# under content-addressed names (`_bts_op_<sha16>.py`). Limen records
# `module.__name__` in `metadata.json["sfd_module"]` and `Trainer`
# later does `importlib.import_module(...)` on it — possibly in a
# fresh subprocess. The bare-stem path-load (`source_path.stem`) is
# NOT on sys.path, so cross-process reimport fails. The snapshot's
# importable name closes that gap. See `_snapshot_exp_code` below.
_OP_SFD_CACHE: Final[Path] = WORK_DIR / 'op_sfds'
_OP_SFD_MODULE_PREFIX: Final[str] = '_bts_op_'

# Numeric columns Limen writes as strings in results.csv. Cast to Float64
# upfront so `filtered.sort` doesn't lex-sort and float `NaN` to the top.
_NUMERIC_COLS: Final[tuple[str, ...]] = (
    'backtest_trades_count',
    'backtest_mean_kelly_pct',
    'backtest_total_return_net_pct',
    'confusion_tp_mean_return_pct',
    'fpr',
)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    """Write `content` to `path` atomically (write unique tmp, then rename).

    Codex round-5 P1: a non-atomic `path.write_bytes(content)` can
    leave a partial file on the destination if the writer is
    interrupted (operator Ctrl-C, OOM, panic). Subsequent runs
    that key on `path.is_file()` then accept the partial file as
    a valid snapshot — but `import_module(...)` raises
    `SyntaxError` on the truncated source.

    Codex round-6 P1: a deterministic `<path>.tmp` filename also
    races between concurrent writers. Two processes opening the
    same `<path>.tmp` would both write content; the first
    `replace()` consumes the tmp, the second raises
    `FileNotFoundError`. Fix: each call uses a UNIQUE tmp name
    (`<path>.<pid>.<uuid4-hex>.tmp`) inside `path.parent`.
    Atomicity holds: every writer's `replace()` either succeeds
    (its tmp existed, target now points at its content) or fails
    cleanly with no partial-write left at `path`.

    `Path.replace` (POSIX `os.rename`) is atomic on the same
    filesystem: the destination either holds the complete content
    or doesn't exist; there is no observable middle state.
    """
    import uuid
    tmp = path.with_name(
        f'{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp',
    )
    try:
        tmp.write_bytes(content)
        tmp.replace(path)
    finally:
        # If `replace` succeeded, the tmp was consumed (rename
        # IS the atomic step). If it raised, the tmp may still
        # exist — best-effort cleanup so we don't leak `.tmp`
        # files into the cache dir on error paths. Use
        # `missing_ok=True` so the post-success no-op case is
        # silent.
        tmp.unlink(missing_ok=True)


def _atomic_write_text(path: Path, content: str) -> None:
    """Write text via `_atomic_write_bytes` after UTF-8 encoding."""
    _atomic_write_bytes(path, content.encode('utf-8'))


@contextmanager
def _exclusive_dir_lock(lock_path: Path) -> Iterator[None]:
    """Hold an `fcntl.flock` exclusive lock on `lock_path` for the block.

    Codex round-5 P1: `train_single_decoder`'s validate-then-wipe
    sequence has a TOCTOU window between
    `_per_decoder_cache_is_valid(...)` and `shutil.rmtree(...)`.
    Two concurrent sweeps targeting the same `sub_dir` can both
    see "stale", both wipe + retrain, and trample each other.

    Wrapping the whole validate+wipe+retrain block in this
    per-sub_dir lock serializes them. The kernel-level
    `fcntl.flock` is released automatically when the file
    handle closes (via `with open`), so no `LOCK_UN` plumbing
    is needed.

    POSIX-only (macOS + Linux). Windows would need `msvcrt.locking`;
    bts is operator-side macOS / Linux-CI only.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('w', encoding='utf-8') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield


def _snapshot_exp_code(exp_code_path: Path) -> tuple[str, Path]:
    """Snapshot operator's exp-code to a content-addressed sys.path location.

    Codex round-2 P0: Limen persists `module.__name__` to the
    experiment's `metadata.json["sfd_module"]`, and `Trainer.train()`
    later does `importlib.import_module(sfd_module)` — possibly in
    a fresh subprocess. A path-loaded module's `__name__` is the
    bare file stem (e.g. `op_sfd`), which is NOT on sys.path, so
    the reimport fails for every operator file outside the package.

    Fix: copy the operator's file to `_OP_SFD_CACHE` under a
    content-addressed name (`_bts_op_<sha16>.py`). Loading that
    snapshot makes `module.__name__` the importable hash-name —
    reimportable across processes once the cache dir is on
    sys.path / PYTHONPATH.

    Identical content -> identical name -> identical snapshot path
    (cache hit). Different content -> different name (so a stale
    metadata entry can never alias edited code).

    Side effect: ensures the cache dir is on the calling process's
    `sys.path`. Subprocess workers must propagate it via
    PYTHONPATH; use `op_sfd_pythonpath()` to get the entry to
    splice into the child's env.

    Returns `(module_name, snapshot_path)`.
    """
    exp_code_path = exp_code_path.expanduser().resolve()
    if not exp_code_path.is_file():
        msg = (
            f'_snapshot_exp_code: --exp-code file not found: '
            f'{exp_code_path}.'
        )
        raise FileNotFoundError(msg)
    content = exp_code_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()[:16]
    module_name = f'{_OP_SFD_MODULE_PREFIX}{digest}'
    _OP_SFD_CACHE.mkdir(parents=True, exist_ok=True)
    snapshot_path = _OP_SFD_CACHE / f'{module_name}.py'
    # Validate: cache-hit only if the existing file's content
    # actually matches the expected hash. Codex round-5 P1: a
    # partial-write leftover would have the same name but
    # corrupt content. `is_file()` alone accepts it. Verify
    # by re-hashing, and rewrite atomically on mismatch.
    if not snapshot_path.is_file() or hashlib.sha256(
        snapshot_path.read_bytes(),
    ).hexdigest()[:16] != digest:
        _atomic_write_bytes(snapshot_path, content)
    sys_path_entry = str(_OP_SFD_CACHE)
    if sys_path_entry not in sys.path:
        sys.path.insert(0, sys_path_entry)
    return module_name, snapshot_path


def op_sfd_pythonpath() -> str:
    """Return the cache-dir entry callers must splice into subprocess PYTHONPATH.

    Subprocess workers do NOT inherit the parent's runtime
    `sys.path` mutations — only the env. Spawners that invoke
    `python -m ...` must merge this entry into PYTHONPATH so the
    child's `importlib.import_module('_bts_op_<sha16>')` resolves:

        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join(
            filter(None, [op_sfd_pythonpath(), env.get('PYTHONPATH', '')])
        )

    Returns the absolute string path of `_OP_SFD_CACHE`.
    """
    return str(_OP_SFD_CACHE)


def derive_op_param_keys(exp_code_path: Path) -> tuple[str, ...]:
    """Load operator's exp-code and return its `params()` grid keys, sorted.

    Codex round-2 P1: the prior `_PARAM_COLS` constant baked
    logreg_binary's hyperparameter set into `train_single_decoder`
    and `pick_decoders`. Non-logreg SFDs (the `--exp-code` use
    case) declare different keys, so the hardcoded set produced
    misaligned per-decoder `params()` dicts. The operator's own
    `params()` is the only authoritative source.

    Sorted to make the per-decoder `exp.py` byte-stable across
    invocations — same operator file -> same generated body ->
    same Limen artifact hash on the per-decoder side.
    """
    from backtest_simulator.pipeline import ExperimentPipeline
    loaded = ExperimentPipeline.load_from_file(exp_code_path)
    return tuple(sorted(loaded.params().keys()))


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


def ensure_trained_from_exp_code(
    exp_code_path: Path, n_permutations: int,
) -> Path:
    """Run UEL via the operator's `--exp-code` file; cache by file content + n_permutations.

    `exp_code_path` must be a self-contained Python file with
    module-level `params()` and `manifest()` callables (the
    `ExperimentPipeline.load_from_file` contract). Operator
    convention is to define a class (`Round3SFD` etc.) and expose
    its static methods via module-level aliases:

        params = Round3SFD.params
        manifest = Round3SFD.manifest

    Any `uel.run(...)` boilerplate inside the file MUST be guarded
    by `if __name__ == '__main__':` so importing the file has no
    side effects — bts drives uel itself with bts-controlled
    `experiment_name` / `n_permutations`.

    Cache key: `WORK_DIR / 'fresh' / {file_stem}_n{n_permutations}_{file_sha256[:16]}`.
    Different file content -> different cache; same content + same
    n_permutations -> reuses the cached experiment.

    Returns the experiment_dir containing `results.csv`,
    `metadata.json`, `round_data.jsonl`, and per-permutation
    artifacts. Slice #17 / operator-mandated contract: there is
    NO fallback `exp.py`; missing `exp_code_path` raises.
    """
    exp_code_path = exp_code_path.expanduser().resolve()
    if not exp_code_path.is_file():
        msg = (
            f'ensure_trained_from_exp_code: --exp-code file not '
            f'found: {exp_code_path}. The operator must supply a '
            f'self-contained UEL-compliant Python file; bts has no '
            f'fallback code path.'
        )
        raise FileNotFoundError(msg)
    # Snapshot the operator's file FIRST. The snapshot path's stem
    # is `_bts_op_<sha16>` (a content-addressed importable module
    # name); `load_from_file(snapshot_path)` produces a module with
    # `__name__ = _bts_op_<sha16>`, which is what Limen records to
    # `metadata.json["sfd_module"]`. Cross-process reimport via
    # `importlib.import_module(...)` then resolves to the snapshot
    # on `_OP_SFD_CACHE` (which is on sys.path via _snapshot_exp_code,
    # and propagated to subprocess workers via PYTHONPATH —
    # see `op_sfd_pythonpath`).
    _, snapshot_path = _snapshot_exp_code(exp_code_path)
    file_hash = hashlib.sha256(exp_code_path.read_bytes()).hexdigest()
    cache_dir = (
        WORK_DIR / 'fresh'
        / f'{exp_code_path.stem}_n{n_permutations}_{file_hash[:16]}'
    )
    if (cache_dir / 'results.csv').is_file():
        return cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    from backtest_simulator.pipeline import ExperimentPipeline
    pipe = ExperimentPipeline(experiment_dir=cache_dir)
    loaded = pipe.load_from_file(snapshot_path)
    pipe.run(
        loaded, experiment_name=exp_code_path.stem,
        n_permutations=n_permutations, seed=42,
    )
    return cache_dir


def row_params(
    row: dict[str, object], keys: tuple[str, ...],
) -> dict[str, object]:
    """Extract the param dict for one CSV row, coercing numeric strings.

    `keys` comes from the operator's `params()` (via
    `derive_op_param_keys`); see codex round-2 P1.
    """
    out: dict[str, object] = {}
    for k in keys:
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


def train_single_decoder(
    sub_dir: Path,
    params: dict[str, object],
    exp_code_path: Path,
    op_param_keys: tuple[str, ...],
) -> None:
    """Train one decoder in its own experiment dir with `n_permutations=1`.

    The generated per-decoder `exp.py` imports the operator's
    `manifest` by name from the snapshotted exp-code — see
    `_snapshot_exp_code` for the cross-process reimport rationale
    (codex round-2 P0). The `params()` it defines is a single-row
    grid containing the picked CSV row's hyperparameters as
    1-element lists; UEL's MSQ runs exactly that one permutation.

    `op_param_keys` MUST come from `derive_op_param_keys(exp_code_path)`
    (codex round-2 P1: keys come from the operator's `params()`,
    not the prior `_PARAM_COLS` hardcode).
    """
    sub_dir.mkdir(parents=True, exist_ok=True)
    # Snapshot first: ensures `_OP_SFD_CACHE` is on sys.path AND
    # gives us the importable module name. The per-decoder exp.py
    # then does a plain `from <name> import manifest` — no
    # `spec_from_file_location` nesting, no path-loaded module
    # objects whose `__name__` would collide / mis-resolve.
    op_module_name, _snapshot_path = _snapshot_exp_code(exp_code_path)
    body_lines: list[str] = [
        f'from {op_module_name} import manifest',
        '',
        'def params():',
        '    return {',
    ]
    for k in op_param_keys:
        body_lines.append(f'        {k!r}: [{params[k]!r}],')
    body_lines.append('    }')
    body = '\n'.join(body_lines) + '\n'
    # Codex round-3 P0: writing the per-decoder exp.py to `sub_dir`
    # and loading it via `spec_from_file_location` produces a
    # module whose `__name__ == 'exp'` — the bare stem of the
    # `sub_dir/exp.py` write target. Limen records that bare stem
    # in the per-decoder `metadata.json["sfd_module"]`, and a
    # fresh subprocess (`BacktestLauncher` / `Trainer.train()`)
    # later does `importlib.import_module('exp')` which raises
    # `ModuleNotFoundError` because `sub_dir` is not on
    # PYTHONPATH. Same root cause as round-2 P0, just on the
    # generated-exp side.
    #
    # Fix: snapshot the generated body too — into `_OP_SFD_CACHE`
    # under a content-addressed name (`_bts_pd_<sha16>`). The
    # `__name__` becomes the importable hash-name; the cache dir
    # is already on sys.path / PYTHONPATH (added by
    # `_snapshot_exp_code` and propagated to subprocess workers
    # via `op_sfd_pythonpath`). Persist the operator-readable
    # `sub_dir/exp.py` as a no-op symlink-like artifact for
    # traceability — load_from_file uses the snapshot path, not
    # the sub_dir copy.
    pd_digest = hashlib.sha256(body.encode('utf-8')).hexdigest()[:16]
    pd_module_name = f'_bts_pd_{pd_digest}'
    _OP_SFD_CACHE.mkdir(parents=True, exist_ok=True)
    pd_snapshot_path = _OP_SFD_CACHE / f'{pd_module_name}.py'
    # Same validate-by-hash pattern as `_snapshot_exp_code`.
    # Codex round-5 P1: a corrupt partial file would otherwise
    # be accepted. Re-hash and rewrite atomically on mismatch.
    if not pd_snapshot_path.is_file() or hashlib.sha256(
        pd_snapshot_path.read_bytes(),
    ).hexdigest()[:16] != pd_digest:
        _atomic_write_text(pd_snapshot_path, body)
    # Codex round-4 P0 + round-5 P1: validate the cache-hit
    # (sfd_module matches the expected importable name) AND
    # serialize the validate-then-wipe-then-retrain block under
    # a per-sub_dir lock. Without the lock, two concurrent
    # sweeps with the same `sub_dir` can both see "stale", both
    # `shutil.rmtree`, and trample each other.
    lock_path = sub_dir.parent / f'.{sub_dir.name}.lock'
    with _exclusive_dir_lock(lock_path):
        # Re-validate UNDER the lock — a concurrent process may
        # have produced a valid sub_dir while we waited.
        if _per_decoder_cache_is_valid(sub_dir, pd_module_name):
            return
        # Stale (or absent) — clear and retrain. `mkdir(parents=True)`
        # below recreates the dir; `shutil.rmtree` is safe because
        # `sub_dir` is bts-controlled (under WORK_DIR /
        # 'trained_from_file') AND we hold the exclusive lock.
        if sub_dir.is_dir():
            shutil.rmtree(sub_dir)
        sub_dir.mkdir(parents=True)
        # Mirror the snapshot into `sub_dir/exp.py` for operator
        # debuggability — they can `cat sub_dir/exp.py` to see the
        # generated grid; the snapshot at `_OP_SFD_CACHE/...` is
        # what load_from_file actually loads.
        _atomic_write_text(sub_dir / 'exp.py', body)
        from backtest_simulator.pipeline import ExperimentPipeline
        pipe = ExperimentPipeline(experiment_dir=sub_dir)
        loaded = pipe.load_from_file(pd_snapshot_path)
        pipe.run(loaded, experiment_name='single', n_permutations=1, seed=42)


def _per_decoder_cache_is_valid(
    sub_dir: Path, expected_pd_module_name: str,
) -> bool:
    """Return True iff `sub_dir` is a complete + reimportable cache hit.

    Cache hit requires:
      1. `results.csv` exists (Limen finished writing the run)
      2. `metadata.json` exists and is parseable JSON
      3. `metadata['sfd_module']` equals the expected
         `_bts_pd_<sha16>` content-addressed name AND the
         corresponding snapshot file exists in `_OP_SFD_CACHE`

    Any other state -> stale, return False so the caller wipes
    and retrains. Codex round-4 P0: catches sub_dirs left over
    from the round-3 build that recorded `sfd_module='exp'`,
    plus any partial-write / corrupted-metadata leftovers.
    """
    results_csv = sub_dir / 'results.csv'
    metadata_path = sub_dir / 'metadata.json'
    if not (results_csv.is_file() and metadata_path.is_file()):
        return False
    try:
        metadata: object = json.loads(
            metadata_path.read_text(encoding='utf-8'),
        )
    except json.JSONDecodeError:
        # Corrupt metadata is a stale-cache signal, not a fatal
        # error — we wipe + retrain. Per AGENTS law 4 the handler
        # must do real work; the explicit `return False` IS the
        # work (vs. `pass` which would imply the corrupt state is
        # acceptable).
        return False
    if not isinstance(metadata, dict):
        return False
    typed_metadata = cast('dict[str, object]', metadata)
    if typed_metadata.get('sfd_module') != expected_pd_module_name:
        return False
    # Snapshot file MUST also exist (operator could have wiped
    # _OP_SFD_CACHE by hand even though sub_dir survived).
    snapshot_path = _OP_SFD_CACHE / f'{expected_pd_module_name}.py'
    return snapshot_path.is_file()


def pick_decoders(
    n: int, *,
    exp_code_path: Path,
    n_permutations: int,
    trades_q_range: tuple[float, float] | None = None,
    tp_min_q: float | None = None,
    fpr_max_q: float | None = None,
    kelly_min_q: float | None = None,
    trade_count_min_q: float | None = None,
    net_return_min_q: float | None = None,
    input_from_file: str | None = None,
) -> tuple[list[tuple[int, Decimal, Path, int]], int]:
    """Quantile-filtered decoder pool, ranked by kelly desc / net-return desc.

    `exp_code_path` is the operator's `--exp-code` file (REQUIRED;
    no fallback). When `input_from_file` is None, bts runs uel via
    `ensure_trained_from_exp_code(exp_code_path, n_permutations)`
    and reads picks from that experiment_dir. When
    `input_from_file` is set, bts reads picks from that CSV and
    retrains each pick via `train_single_decoder` using the
    operator's manifest from `exp_code_path` (NOT a hardcoded
    logreg_binary).

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

    exp_code_path = exp_code_path.expanduser().resolve()
    if not exp_code_path.is_file():
        msg = (
            f'pick_decoders: --exp-code file not found: {exp_code_path}. '
            f'bts requires a self-contained UEL-compliant Python file '
            f'with module-level `params()` and `manifest()` callables; '
            f'there is no fallback code path.'
        )
        raise FileNotFoundError(msg)

    # `op_param_keys` is the SOLE authoritative source for the
    # per-decoder hyperparameter set (codex round-2 P1). Used in
    # the file-mode retrain path AND to validate that the CSV's
    # columns match what the operator's `params()` declares — a
    # column mismatch raises here, before any training kicks off.
    # This also serves as an early-validation that the exp-code
    # is actually loadable (UEL-compliant module-level
    # `params`/`manifest`); any contract violation surfaces as
    # the same `ValueError` the no-input path would hit later.
    op_param_keys: tuple[str, ...] = derive_op_param_keys(exp_code_path)

    # Hoisted so pyright sees them as bound (Optional) in both
    # branches and downstream code. Mode-specific binding
    # populates exactly one of the two; per-pick code asserts
    # the relevant one is set before use.
    file_path: Path | None = None
    cache_dir: Path | None = None
    source: str

    if input_from_file is not None:
        file_path = Path(input_from_file).expanduser()
        if not file_path.is_file():
            msg = f'--input-from-file: {file_path} does not exist.'
            raise FileNotFoundError(msg)
        results = pl.read_csv(file_path)
        available_cols = cast('list[str]', list(results.columns))
        missing_cols = [k for k in op_param_keys if k not in available_cols]
        if missing_cols:
            msg = (
                f'--input-from-file {file_path.name} is missing columns '
                f'the operator\'s params() declares: {missing_cols}. The '
                f'CSV must have one column per param key. Available '
                f'columns: {available_cols}.'
            )
            raise ValueError(msg)
        source = str(file_path)
        print(
            f'  loaded filter pool from {file_path.name}: {results.height} '
            f'rows  (exp-code: {exp_code_path.name})',
            flush=True,
        )
    else:
        # Mode 1: bts runs uel itself via the operator's exp-code,
        # writing artifacts to a content-hashed cache_dir, then
        # reads results.csv from there.
        cache_dir = ensure_trained_from_exp_code(exp_code_path, n_permutations)
        from backtest_simulator.pipeline import ExperimentPipeline
        pipe = ExperimentPipeline(experiment_dir=cache_dir)
        results = pipe.read_results()
        source = str(cache_dir)
    # Strip leading/trailing whitespace before casting. Some
    # operator-supplied CSVs (e.g. exported from notebooks with
    # `to_csv(..., float_format=' %.3f')`) pad numeric strings
    # with a leading space; polars' `cast(Float64, strict=False)`
    # returns null on `' -0.343'` and the entire pool gets dropped
    # on the null-filter below. Stripping first lets the cast
    # succeed and the operator's filter actually run.
    casts = [
        pl.col(c).str.strip_chars().cast(pl.Float64, strict=False).alias(c)
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
    if results.height == 0:
        # Fail loud BEFORE the quantile machinery would crash with a
        # confusing TypeError. The operator's CSV either has no
        # numeric data in the rank+kelly columns or the cast
        # silently dropped everything (e.g. unrecognised value
        # format). Tell them exactly what to look at. `source` is
        # bound by both modes above — file_path in input mode,
        # cache_dir in fresh-train mode.
        msg = (
            f'pick_decoders: 0 usable rows in {source}. The cast '
            f'to Float64 returned null for every value in '
            f'{clean_cols}. Common causes: the column is non-'
            f'numeric (string labels), the CSV uses an '
            f'unrecognised number format, or the column is '
            f'genuinely all-null. Inspect the first few rows of '
            f'those columns and re-export.'
        )
        raise RuntimeError(msg)
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
            # Narrowing: file_path is bound by the input-mode branch
            # above; pyright doesn't carry the relationship between
            # `input_from_file is not None` and `file_path is not
            # None` across blocks, so re-state it here.
            assert file_path is not None
            row = {k: ranked[k][i] for k in op_param_keys}
            params = row_params(row, op_param_keys)
            # Cache key includes source filename stem AND a content
            # hash of (the picked row's params + the operator's
            # exp-code FILE CONTENT). Codex round-1 P1: hashing
            # only the path lets in-place edits to `sfd.py` reuse
            # the stale cached training. Hashing the file content
            # closes that window — every edit -> new sub_dir.
            exp_code_content_hash = hashlib.sha256(
                exp_code_path.read_bytes(),
            ).hexdigest()
            cache_input = json.dumps({
                'params': {k: str(v) for k, v in sorted(params.items())},
                'exp_code_path': str(exp_code_path),
                'exp_code_sha256': exp_code_content_hash,
            }, sort_keys=True)
            cache_hash = hashlib.sha256(
                cache_input.encode('utf-8'),
            ).hexdigest()
            sub_dir = (
                WORK_DIR / 'trained_from_file'
                / f'{file_path.stem}_{exp_code_path.stem}_id_'
                f'{file_id}_{cache_hash}'
            )
            print(
                f'  training file-id {file_id} from {file_path.name} '
                f'(exp-code: {exp_code_path.name}, '
                f'cache_hash={cache_hash[:8]}...)  kelly={kelly}  ...',
                flush=True,
            )
            train_single_decoder(
                sub_dir, params, exp_code_path, op_param_keys,
            )
            picks.append((0, kelly, sub_dir, file_id))
        else:
            # Mode 1: cache_dir from ensure_trained_from_exp_code
            # is the experiment_dir for all picks. Resolved above
            # before the filter machinery runs. Same narrowing
            # rationale as `assert file_path is not None` above.
            assert cache_dir is not None
            picks.append((file_id, kelly, cache_dir, file_id))
    return picks, before
