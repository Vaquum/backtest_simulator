"""Golden test for canonical `bts sweep` (slice 1; The-Plan.md)."""
# Runs the real CLI as a subprocess twice with the same canonical args
# (2 decoders x 3 days against the locked `bundle.zip` fixture) and
# asserts that, after normalization, both runs produce identical
# stdout / stderr / sweep_per_window.csv / sweep_per_tick.csv AND match
# the saved expected fixtures byte-for-byte.
#
# Slice 0 (#66) made `bts sweep` deterministic by replacing the racy
# `threading.Timer` clock with a schedule-driven replay clock; this
# test is the mechanical proof that the determinism holds for the
# canonical configuration. A regression that re-introduces the timer
# race fails this test.

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO_ROOT / 'tests' / 'fixtures' / 'canonical'
_BUNDLE = _FIXTURES / 'bundle.zip'
_KLINES = _FIXTURES / 'klines.parquet'
_TRADES = _FIXTURES / 'trades.parquet'
_EXPECTED = _FIXTURES / 'expected'
_NORMALIZER = _REPO_ROOT / 'tools' / 'normalize_sweep_outputs.py'

_BTS = _REPO_ROOT / '.venv' / 'bin' / 'bts'
_PYTHON = _REPO_ROOT / '.venv' / 'bin' / 'python'

_REPLAY_START = '2026-04-01'
_REPLAY_END = '2026-04-03'
_N_DECODERS = '2'

# Trades parquet cache filename — `prefetch_sweep_trades` builds this
# from `replay_start - 30min` and `replay_end_of_trading_day + 600s`.
# For the canonical (2026-04-01 to 2026-04-03 trading 00:00-23:59)
# this is the only filename `bts sweep` will read.
_TRADES_CACHE_FILENAME = 'btcusdt_20260331T233000_20260404T000900.parquet'

# Klines cache filename — `commands/sweep.py` builds this from kline
# size (4h = 14400s) and writes / reads from
# `<HOME>/.cache/backtest_simulator/limen_klines/btcusdt_<size>.parquet`.
_KLINES_CACHE_FILENAME = 'btcusdt_14400.parquet'

_DOTENV_PATH = _REPO_ROOT / '.env'

_CANONICAL_ARGS: list[str] = [
    '--bundle', str(_BUNDLE),
    '--n-decoders', _N_DECODERS,
    '--n-permutations', '5000',
    '--trading-hours-start', '00:00',
    '--trading-hours-end', '23:59',
    '--replay-period-start', _REPLAY_START,
    '--replay-period-end', _REPLAY_END,
    '--max-allocation-per-trade-pct', '0.4',
    '--predict-lookback', '1',
    '--cpcv-n-groups', '4',
    '--cpcv-n-test-groups', '2',
    '--cpcv-purge-seconds', '0',
    '--cpcv-embargo-seconds', '0',
]


@pytest.fixture(scope='module')
def _fixture_checksums_match() -> None:
    """Pin every input fixture's bytes via SHA256.

    Verifies bundle.zip + klines.parquet + trades.parquet all match
    the recorded SHAs. A silent regen of any of them would let
    `bts sweep` produce different numbers without the expected/*
    files updating, masking real drift; this guard fails loud.
    """
    text = (_FIXTURES / 'checksums.sha256').read_text(encoding='utf-8')
    expected: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        sha, name = line.split(maxsplit=1)
        expected[name.strip()] = sha
    for name, sha in expected.items():
        path = _FIXTURES / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != sha:
            msg = (
                f'canonical fixture {name} checksum mismatch: '
                f'expected {sha}, got {actual}. The fixture has been '
                f'silently modified; restore it or update the SHA in '
                f'`tests/fixtures/canonical/checksums.sha256` AND the '
                f'expected/* outputs in the same PR.'
            )
            raise AssertionError(msg)


def _read_dotenv() -> dict[str, str]:
    """Read `<repo>/.env` without mutating `os.environ`.

    Mirrors `_pipeline._hydrate_environ_from_dotenv`'s parsing rules
    (including stripping a single matching pair of quotes). The test
    process intentionally does NOT mutate its own environment — the
    parsed values are forwarded to the subprocess via the explicit
    `env=` dict in `_run_canonical_sweep`.
    """
    if not _DOTENV_PATH.is_file():
        return {}
    out: dict[str, str] = {}
    for raw in _DOTENV_PATH.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, _, v = line.partition('=')
        key = k.strip()
        val = v.strip()
        if not key:
            continue
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
            val = val[1:-1]
        out[key] = val
    return out


@pytest.fixture(scope='module')
def _clickhouse_password() -> str:
    """Locate a real ClickHouse password for the subprocess.

    `bts sweep`'s `_pipeline.py` applies safe defaults for
    `CLICKHOUSE_HOST/PORT/USER/DATABASE` (ssh-tunnel-shaped:
    127.0.0.1:18123 / default / origo) on import; the operator only
    has to supply `CLICKHOUSE_PASSWORD`, either in their shell or in
    `<repo>/.env`. CI provides it via repository secrets exposed to
    this workflow. We resolve in the same order the sweep itself
    would; missing fails loud — the gate is meaningless without real
    data-plane access.
    """
    pw = os.environ.get('CLICKHOUSE_PASSWORD')
    if pw:
        return pw
    pw = _read_dotenv().get('CLICKHOUSE_PASSWORD')
    if pw:
        return pw
    msg = (
        'canonical golden test requires a real ClickHouse password. '
        f'Set CLICKHOUSE_PASSWORD in your shell or add it to '
        f'{_DOTENV_PATH}. In CI, expose it via repository secrets.'
    )
    raise AssertionError(msg)


def _build_isolated_env(tmp_home: Path, ch_password: str) -> dict[str, str]:
    """Build a subprocess env that points `bts sweep` at the locked
    fixture parquets via an isolated HOME, while preserving the
    operator's real ClickHouse credentials so `preflight_tunnel()`
    runs against the real data plane.

    The cache lookup in `prefetch_sweep_trades` and Limen's
    `HistoricalData` short-circuits on `path.is_file()`. Pre-placing
    the locked parquets at the canonical cache paths means no actual
    HTTP fetch (HF) or ClickHouse trade query runs during the sweep —
    the only network call is the preflight's one-row sentinel
    `SELECT toString(datetime) FROM origo.binance_daily_spot_trades
    ORDER BY datetime DESC LIMIT 1` which validates the connection
    end-to-end. The fixture parquets are byte-equal to what the
    sweep would have fetched (operator-captured + SHA256-pinned).
    """
    cache_root = tmp_home / '.cache' / 'backtest_simulator'
    trades_dir = cache_root / 'trades'
    klines_dir = cache_root / 'limen_klines'
    trades_dir.mkdir(parents=True, exist_ok=True)
    klines_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_TRADES, trades_dir / _TRADES_CACHE_FILENAME)
    shutil.copyfile(_KLINES, klines_dir / _KLINES_CACHE_FILENAME)
    env = os.environ.copy()
    env['HOME'] = str(tmp_home)
    env['XDG_CACHE_HOME'] = str(tmp_home / '.cache')
    env['HF_HOME'] = str(tmp_home / '.cache' / 'huggingface')
    env['CLICKHOUSE_PASSWORD'] = ch_password
    return env


def _normalize(mode: str, in_path: Path, out_path: Path) -> None:
    subprocess.run(
        [str(_PYTHON), str(_NORMALIZER), '--mode', mode,
         '--in', str(in_path), '--out', str(out_path)],
        check=True,
    )


def _run_canonical_sweep(
    *, session_id: str, work_dir: Path, ch_password: str,
) -> None:
    """Run `bts sweep` once with the canonical args, capturing stdout +
    stderr to files under `work_dir`. Each call gets an isolated
    `HOME` populated with the locked klines + trades fixtures so the
    sweep runs against repo-shipped real data without re-fetching.
    The real ClickHouse password (operator's locally; CI repo secret
    in CI) is forwarded so preflight runs end-to-end against the real
    data plane.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    tmp_home = work_dir / 'home'
    tmp_home.mkdir(exist_ok=True)
    env = _build_isolated_env(tmp_home, ch_password)
    cmd = [str(_BTS), 'sweep', *_CANONICAL_ARGS, '--session-id', session_id]
    with (
        (work_dir / 'stdout.txt').open('wb') as out_fp,
        (work_dir / 'stderr.txt').open('wb') as err_fp,
    ):
        result = subprocess.run(
            cmd, stdout=out_fp, stderr=err_fp,
            check=False, env=env,
        )
    if result.returncode != 0:
        stderr_text = (work_dir / 'stderr.txt').read_text(errors='replace')[-2000:]
        msg = (
            f'canonical bts sweep returned exit={result.returncode}; '
            f'see {work_dir}/stderr.txt\n--- last 2KB ---\n{stderr_text}'
        )
        raise AssertionError(msg)


def _capture_normalized_outputs(
    *, session_id: str, work_dir: Path, ch_password: str,
) -> dict[str, Path]:
    """Run sweep once, normalize all four artefact streams. The
    session output dir lives under the isolated
    `<work_dir>/home/sweep/sessions/` (HOME is overridden inside the
    subprocess), not the caller's actual home — the test is hermetic.
    """
    _run_canonical_sweep(
        session_id=session_id, work_dir=work_dir, ch_password=ch_password,
    )
    session_dir = work_dir / 'home' / 'sweep' / 'sessions' / session_id
    norm = work_dir / 'normalized'
    norm.mkdir(exist_ok=True)
    _normalize('stdout', work_dir / 'stdout.txt', norm / 'stdout.txt')
    _normalize('stderr', work_dir / 'stderr.txt', norm / 'stderr.txt')
    _normalize(
        'csv', session_dir / 'sweep_per_window.csv',
        norm / 'sweep_per_window.csv',
    )
    _normalize(
        'csv', session_dir / 'sweep_per_tick.csv',
        norm / 'sweep_per_tick.csv',
    )
    return {
        'stdout': norm / 'stdout.txt',
        'stderr': norm / 'stderr.txt',
        'per_window': norm / 'sweep_per_window.csv',
        'per_tick': norm / 'sweep_per_tick.csv',
    }


@pytest.fixture(scope='module')
def _two_runs(
    _fixture_checksums_match: None,
    _clickhouse_password: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, Path], dict[str, Path]]:
    """Run the canonical `bts sweep` twice (independent isolated
    HOMEs), return the normalized artefact paths for run A and B."""
    base = tmp_path_factory.mktemp('canonical_sweep_runs')
    run_a = _capture_normalized_outputs(
        session_id=f'golden-A-{os.getpid()}',
        work_dir=base / 'A',
        ch_password=_clickhouse_password,
    )
    run_b = _capture_normalized_outputs(
        session_id=f'golden-B-{os.getpid()}',
        work_dir=base / 'B',
        ch_password=_clickhouse_password,
    )
    return run_a, run_b


class TestCanonicalSweepGolden:

    def test_run_a_per_window_byte_equal_to_fixture(
        self, _two_runs: tuple[dict[str, Path], dict[str, Path]],
    ) -> None:
        run_a, _ = _two_runs
        actual = run_a['per_window'].read_bytes()
        expected = (_EXPECTED / 'sweep_per_window.csv').read_bytes()
        assert actual == expected, (
            f'run A normalized sweep_per_window.csv diverges from fixture; '
            f'see {run_a["per_window"]} vs {_EXPECTED}/sweep_per_window.csv'
        )

    def test_run_a_per_tick_byte_equal_to_fixture(
        self, _two_runs: tuple[dict[str, Path], dict[str, Path]],
    ) -> None:
        run_a, _ = _two_runs
        actual = run_a['per_tick'].read_bytes()
        expected = (_EXPECTED / 'sweep_per_tick.csv').read_bytes()
        assert actual == expected, (
            f'run A normalized sweep_per_tick.csv diverges from fixture; '
            f'see {run_a["per_tick"]} vs {_EXPECTED}/sweep_per_tick.csv'
        )

    def test_run_a_stdout_byte_equal_to_fixture(
        self, _two_runs: tuple[dict[str, Path], dict[str, Path]],
    ) -> None:
        run_a, _ = _two_runs
        actual = run_a['stdout'].read_bytes()
        expected = (_EXPECTED / 'stdout.txt').read_bytes()
        assert actual == expected, (
            f'run A normalized stdout diverges from fixture; '
            f'see {run_a["stdout"]} vs {_EXPECTED}/stdout.txt'
        )

    def test_run_a_stderr_byte_equal_to_fixture(
        self, _two_runs: tuple[dict[str, Path], dict[str, Path]],
    ) -> None:
        run_a, _ = _two_runs
        actual = run_a['stderr'].read_bytes()
        expected = (_EXPECTED / 'stderr.txt').read_bytes()
        assert actual == expected, (
            'run A normalized stderr diverges from fixture'
        )

    def test_run_a_and_run_b_produce_byte_equal_per_window(
        self, _two_runs: tuple[dict[str, Path], dict[str, Path]],
    ) -> None:
        run_a, run_b = _two_runs
        a = run_a['per_window'].read_bytes()
        b = run_b['per_window'].read_bytes()
        assert a == b

    def test_run_a_and_run_b_produce_byte_equal_per_tick(
        self, _two_runs: tuple[dict[str, Path], dict[str, Path]],
    ) -> None:
        run_a, run_b = _two_runs
        a = run_a['per_tick'].read_bytes()
        b = run_b['per_tick'].read_bytes()
        assert a == b

    def test_run_a_and_run_b_produce_byte_equal_stdout(
        self, _two_runs: tuple[dict[str, Path], dict[str, Path]],
    ) -> None:
        run_a, run_b = _two_runs
        a = run_a['stdout'].read_bytes()
        b = run_b['stdout'].read_bytes()
        assert a == b

    def test_run_a_and_run_b_produce_byte_equal_stderr(
        self, _two_runs: tuple[dict[str, Path], dict[str, Path]],
    ) -> None:
        run_a, run_b = _two_runs
        a = run_a['stderr'].read_bytes()
        b = run_b['stderr'].read_bytes()
        assert a == b


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
