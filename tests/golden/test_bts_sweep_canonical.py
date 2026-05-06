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

# Klines cache filename — `commands/sweep.py` builds this from kline
# size (4h = 14400s) and writes / reads from
# `<HOME>/.cache/backtest_simulator/limen_klines/btcusdt_<size>.parquet`.
# We pre-place the locked klines parquet here so Limen's
# `HistoricalData.get_spot_klines` cache-hits and skips the HuggingFace
# fetch entirely — combined with `--trades-tape` (which skips the
# ClickHouse preflight + prefetch + benchmark seed-price lookups) the
# subprocess runs end-to-end without any network access.
_KLINES_CACHE_FILENAME = 'btcusdt_14400.parquet'

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
    '--trades-tape', str(_TRADES),
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


def _build_isolated_env(tmp_home: Path) -> dict[str, str]:
    """Build a subprocess env that lets `bts sweep` run hermetically
    against the locked fixture parquets — no live ClickHouse or
    HuggingFace round-trips.

    The klines parquet is pre-placed at the canonical cache path so
    Limen's `HistoricalData.get_spot_klines` short-circuits and skips
    the HF fetch. The trades parquet is handed to sweep via
    `--trades-tape` (a real CLI feature, not a test hook) so the
    sweep skips the ClickHouse preflight, prefetch, and per-day
    seed-price lookups for the buy-hold benchmark — every CH-touching
    code path is bypassed at the source. Sockets aren't blocked here;
    no code path remains that would open one.
    """
    cache_root = tmp_home / '.cache' / 'backtest_simulator'
    klines_dir = cache_root / 'limen_klines'
    klines_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_KLINES, klines_dir / _KLINES_CACHE_FILENAME)
    env = os.environ.copy()
    env['HOME'] = str(tmp_home)
    env['XDG_CACHE_HOME'] = str(tmp_home / '.cache')
    env['HF_HOME'] = str(tmp_home / '.cache' / 'huggingface')
    # Strip any inherited ClickHouse credentials so an accidental new
    # CH call surfaces immediately as a missing-env fail-loud rather
    # than silently passing on an operator's live tunnel.
    for key in (
        'CLICKHOUSE_HOST', 'CLICKHOUSE_PORT', 'CLICKHOUSE_USER',
        'CLICKHOUSE_PASSWORD', 'CLICKHOUSE_DATABASE',
    ):
        env.pop(key, None)
    return env


def _normalize(mode: str, in_path: Path, out_path: Path) -> None:
    subprocess.run(
        [str(_PYTHON), str(_NORMALIZER), '--mode', mode,
         '--in', str(in_path), '--out', str(out_path)],
        check=True,
    )


def _run_canonical_sweep(*, session_id: str, work_dir: Path) -> None:
    """Run `bts sweep` once with the canonical args, capturing stdout +
    stderr to files under `work_dir`. Each call gets an isolated
    `HOME` populated with the locked klines fixture; the trades tape
    is fed via `--trades-tape`. No live ClickHouse / HuggingFace
    access — drift in the gate's output therefore reflects code drift,
    not data-plane drift.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    tmp_home = work_dir / 'home'
    tmp_home.mkdir(exist_ok=True)
    env = _build_isolated_env(tmp_home)
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
    *, session_id: str, work_dir: Path,
) -> dict[str, Path]:
    """Run sweep once, normalize all four artefact streams. The
    session output dir lives under the isolated
    `<work_dir>/home/sweep/sessions/` (HOME is overridden inside the
    subprocess), not the caller's actual home — the test is hermetic.
    """
    _run_canonical_sweep(session_id=session_id, work_dir=work_dir)
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
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, Path], dict[str, Path]]:
    """Run the canonical `bts sweep` twice (independent isolated
    HOMEs), return the normalized artefact paths for run A and B."""
    base = tmp_path_factory.mktemp('canonical_sweep_runs')
    run_a = _capture_normalized_outputs(
        session_id=f'golden-A-{os.getpid()}',
        work_dir=base / 'A',
    )
    run_b = _capture_normalized_outputs(
        session_id=f'golden-B-{os.getpid()}',
        work_dir=base / 'B',
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
