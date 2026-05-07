#!/usr/bin/env python3
"""Sweep-reference gate: the live/empty/dead file classification must match.

Runs the canonical `bts sweep` under coverage, classifies every
`backtest_simulator/**/*.py` file as live / empty / no-executed-lines,
and compares against the committed snapshot at
`tests/fixtures/canonical/sweep_reference.json`. Any drift fails the
gate — the canonical-coverage contract stays pinned.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
SNAPSHOT: Final[Path] = REPO_ROOT / 'tests' / 'fixtures' / 'canonical' / 'sweep_reference.json'
FIXTURES: Final[Path] = REPO_ROOT / 'tests' / 'fixtures' / 'canonical'
PYTHON: Final[Path] = Path(sys.executable)
_BTS_RESOLVED: Final[str | None] = shutil.which('bts')
if _BTS_RESOLVED is None:
    raise RuntimeError(
        'bts binary not found on PATH. Install the package '
        '(`uv pip install -e .[integration]`) before running this gate.'
    )
BTS: Final[Path] = Path(_BTS_RESOLVED)


def _classify(coverage_json: dict) -> dict:
    live: dict[str, dict] = {}
    empty_init: list[str] = []
    no_executed: list[dict] = []
    for path, info in coverage_json['files'].items():
        rel = path.replace(f'{REPO_ROOT}/', '')
        n = info['summary']['num_statements']
        c = info['summary']['covered_lines']
        if n == 0:
            empty_init.append(rel)
        elif c > 0:
            live[rel] = {
                'num_statements': n,
                'covered_lines': c,
                'num_branches': info['summary'].get('num_branches', 0),
                'covered_branches': info['summary'].get('covered_branches', 0),
            }
        else:
            no_executed.append({'path': rel, 'num_statements': n})
    return {
        'live': dict(sorted(live.items())),
        'empty_init': sorted(empty_init),
        'no_executed_lines': sorted(no_executed, key=lambda x: x['path']),
    }


def _run_canonical_sweep_under_coverage(work_dir: Path) -> Path:
    (work_dir / 'home' / '.cache' / 'backtest_simulator' / 'limen_klines').mkdir(
        parents=True, exist_ok=True,
    )
    shutil.copyfile(
        FIXTURES / 'klines.parquet',
        work_dir / 'home' / '.cache' / 'backtest_simulator' / 'limen_klines'
        / 'btcusdt_14400.parquet',
    )
    rcfile = work_dir / '.coveragerc'
    rcfile.write_text(
        '[run]\n'
        'branch = True\n'
        'parallel = True\n'
        'concurrency = multiprocessing,thread\n'
        f'source = backtest_simulator\n'
        f'data_file = {work_dir}/.coverage\n',
        encoding='utf-8',
    )
    env = os.environ.copy()
    for k in ('CLICKHOUSE_HOST', 'CLICKHOUSE_PORT', 'CLICKHOUSE_USER',
              'CLICKHOUSE_PASSWORD', 'CLICKHOUSE_DATABASE'):
        env.pop(k, None)
    env['HOME'] = str(work_dir / 'home')
    env['XDG_CACHE_HOME'] = str(work_dir / 'home' / '.cache')
    env['HF_HOME'] = str(work_dir / 'home' / '.cache' / 'huggingface')
    env['COVERAGE_PROCESS_START'] = str(rcfile)
    cmd = [
        str(PYTHON), '-m', 'coverage', 'run', f'--rcfile={rcfile}',
        str(BTS), 'sweep',
        '--bundle', str(FIXTURES / 'bundle.zip'),
        '--n-decoders', '2',
        '--n-permutations', '5000',
        '--trading-hours-start', '00:00',
        '--trading-hours-end', '23:59',
        '--replay-period-start', '2026-04-01',
        '--replay-period-end', '2026-04-03',
        '--max-allocation-per-trade-pct', '0.4',
        '--predict-lookback', '1',
        '--cpcv-n-groups', '4',
        '--cpcv-n-test-groups', '2',
        '--cpcv-purge-seconds', '0',
        '--cpcv-embargo-seconds', '0',
        '--trades-tape', str(FIXTURES / 'trades.parquet'),
        '--session-id', 'sweep-reference-gate',
    ]
    result = subprocess.run(
        cmd, cwd=work_dir, env=env, capture_output=True, text=True,
        check=False, timeout=300,
    )
    if result.returncode != 0:
        msg = (
            f'canonical bts sweep returned exit={result.returncode}\n'
            f'stderr tail: {result.stderr[-2000:]}'
        )
        raise RuntimeError(msg)
    subprocess.run(
        [str(PYTHON), '-m', 'coverage', 'combine', f'--rcfile={rcfile}'],
        cwd=work_dir, env=env, check=True, capture_output=True,
    )
    json_path = work_dir / 'coverage.json'
    subprocess.run(
        [str(PYTHON), '-m', 'coverage', 'json', f'--rcfile={rcfile}', '-o', str(json_path)],
        cwd=work_dir, env=env, check=True, capture_output=True,
    )
    return json_path


def main() -> int:
    if not SNAPSHOT.is_file():
        print(f'SWEEP REFERENCE GATE -- FAIL: snapshot missing at {SNAPSHOT}', file=sys.stderr)
        return 1
    snapshot = json.loads(SNAPSHOT.read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory(prefix='sweep_reference_gate_') as tmp:
        work = Path(tmp)
        try:
            cov_path = _run_canonical_sweep_under_coverage(work)
        except RuntimeError as exc:
            print(f'SWEEP REFERENCE GATE -- FAIL: {exc}', file=sys.stderr)
            return 1
        cov = json.loads(cov_path.read_text(encoding='utf-8'))
    actual = _classify(cov)
    expected = {
        'live': snapshot['live'],
        'empty_init': snapshot['empty_init'],
        'no_executed_lines': snapshot['no_executed_lines'],
    }
    violations: list[str] = []
    for key in ('live', 'empty_init', 'no_executed_lines'):
        if actual[key] != expected[key]:
            violations.append(key)
    if violations:
        print('SWEEP REFERENCE GATE -- FAIL', file=sys.stderr)
        print(f'  drift in: {violations}', file=sys.stderr)
        for k in violations:
            exp = expected[k]
            got = actual[k]
            print(f'  --- {k} (expected count={len(exp)}, got count={len(got)}) ---', file=sys.stderr)
            if isinstance(exp, dict) and isinstance(got, dict):
                missing = set(exp) - set(got)
                extra = set(got) - set(exp)
                if missing:
                    print(f'    files no longer live: {sorted(missing)}', file=sys.stderr)
                if extra:
                    print(f'    files newly live:     {sorted(extra)}', file=sys.stderr)
            elif isinstance(exp, list) and isinstance(got, list):
                exp_keys = {e if isinstance(e, str) else e['path'] for e in exp}
                got_keys = {g if isinstance(g, str) else g['path'] for g in got}
                missing = exp_keys - got_keys
                extra = got_keys - exp_keys
                if missing:
                    print(f'    missing: {sorted(missing)}', file=sys.stderr)
                if extra:
                    print(f'    extra:   {sorted(extra)}', file=sys.stderr)
        print('Re-run canonical sweep, regenerate sweep_reference.json, '
              'review the diff, commit.', file=sys.stderr)
        return 1
    print(
        f'SWEEP REFERENCE GATE -- PASS '
        f'(live={len(actual["live"])}, '
        f'empty_init={len(actual["empty_init"])}, '
        f'no_executed_lines={len(actual["no_executed_lines"])})'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
