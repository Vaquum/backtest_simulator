"""Golden test for canonical `bts sweep` (slice 1; The-Plan.md)."""
# Runs the real CLI as a subprocess twice with the same canonical args
# (2 decoders × 3 days against the locked `bundle.zip` fixture) and
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
_EXPECTED = _FIXTURES / 'expected'
_NORMALIZER = _REPO_ROOT / 'tools' / 'normalize_sweep_outputs.py'

_BTS = _REPO_ROOT / '.venv' / 'bin' / 'bts'
_PYTHON = _REPO_ROOT / '.venv' / 'bin' / 'python'

_REPLAY_START = '2026-04-01'
_REPLAY_END = '2026-04-03'
_N_DECODERS = '2'

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
def _bundle_checksum_matches() -> None:
    expected = (_FIXTURES / 'checksums.sha256').read_text(
        encoding='utf-8',
    ).split()[0]
    actual = hashlib.sha256(_BUNDLE.read_bytes()).hexdigest()
    if expected != actual:
        msg = (
            f'canonical bundle.zip checksum mismatch: '
            f'expected {expected}, got {actual}. The bundle fixture has '
            f'been silently modified; restore it or update '
            f'checksums.sha256 deliberately as part of a slice that '
            f'declares the new expected outputs.'
        )
        raise AssertionError(msg)


def _normalize(mode: str, in_path: Path, out_path: Path) -> None:
    subprocess.run(
        [str(_PYTHON), str(_NORMALIZER), '--mode', mode,
         '--in', str(in_path), '--out', str(out_path)],
        check=True,
    )


def _run_canonical_sweep(*, session_id: str, work_dir: Path) -> None:
    """Run `bts sweep` once with the canonical args, capturing stdout +
    stderr to files under `work_dir`. The session output dir under
    `~/sweep/sessions/<session-id>/` is the implicit per-run artefact
    surface; the caller is responsible for picking a unique session_id.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(_BTS), 'sweep', *_CANONICAL_ARGS, '--session-id', session_id]
    with (
        (work_dir / 'stdout.txt').open('wb') as out_fp,
        (work_dir / 'stderr.txt').open('wb') as err_fp,
    ):
        result = subprocess.run(
            cmd, stdout=out_fp, stderr=err_fp,
            check=False, env=os.environ.copy(),
        )
    if result.returncode != 0:
        msg = (
            f'canonical bts sweep returned exit={result.returncode}; '
            f'see {work_dir}/stderr.txt'
        )
        raise AssertionError(msg)


def _capture_normalized_outputs(
    *, session_id: str, work_dir: Path,
) -> dict[str, Path]:
    """Run sweep once, normalize all four artefact streams, return the
    per-stream paths under `work_dir / 'normalized'`."""
    _run_canonical_sweep(session_id=session_id, work_dir=work_dir)
    session_dir = Path.home() / 'sweep' / 'sessions' / session_id
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
def _two_runs(_bundle_checksum_matches: None, tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Path], dict[str, Path]]:
    """Run the canonical `bts sweep` twice, return the normalized
    artefact paths for run A and run B."""
    base = tmp_path_factory.mktemp('canonical_sweep_runs')
    run_a = _capture_normalized_outputs(
        session_id=f'golden-A-{os.getpid()}',
        work_dir=base / 'A',
    )
    run_b = _capture_normalized_outputs(
        session_id=f'golden-B-{os.getpid()}',
        work_dir=base / 'B',
    )
    yield run_a, run_b
    for sid in (f'golden-A-{os.getpid()}', f'golden-B-{os.getpid()}'):
        sd = Path.home() / 'sweep' / 'sessions' / sid
        if sd.exists():
            shutil.rmtree(sd, ignore_errors=True)


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
            f'run A normalized stderr diverges from fixture'
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
