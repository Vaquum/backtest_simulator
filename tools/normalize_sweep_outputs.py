#!/usr/bin/env python3
"""Normalize `bts sweep` outputs for byte-equal comparison across runs."""
# `bts sweep` produces deterministic numerical outputs (after slice 0;
# Vaquum/backtest_simulator#66) but the surrounding stdout / stderr
# carry wall-clock-dependent and parallelism-dependent noise:
#
#   - per-phase durations like `[   4.07s]` shift run-to-run
#   - the session-id and its session path appear in stdout
#   - klines-cache age (`age=X.Xh`) reflects real wall time
#   - per-window completion log lines (`perm N day done (i/M, trades=K)`)
#     arrive in parallel-completion order, not deterministic order
#   - tqdm progress bars in stderr carry per-iteration timing
#   - per_window / per_tick CSV row order is also parallel-driven
#
# The numerical content (probs, preds, trade counts, profits, capital
# allocation) is run-to-run identical. The normalizer scrubs the noise
# and sorts what's parallel-emitted so two normalized runs are
# byte-equal when the underlying replay was deterministic.

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_DURATION_RE = re.compile(r'\[\s*\d+\.\d{2}s\]')
_AGE_RE = re.compile(r'age=\d+\.\d+h')
_SESSION_PATH_RE = re.compile(r'/[^ \t\n]+/sweep/sessions/[^ \t\n]+')
_SESSION_ID_RE = re.compile(r'session [^ \t\n→]+')
# Filter-pool training emits `trained (...)` on first run and
# `cached (...)` on subsequent runs (Limen reads the picked decoders
# from disk). Normalize both forms so the cache-cold and cache-warm
# paths byte-compare equal.
_TRAINED_OR_CACHED_RE = re.compile(r'(decoder \d+\s+)(trained|cached)\b')
# Per-window `done  (N/M, trades=K)` lines: the `N/M` index reflects
# parallel-completion order, which is not deterministic. The
# `trades=K` count IS deterministic and stays.
_COMPLETION_INDEX_RE = re.compile(r'done\s+\(\d+/(\d+),')
# Final summary `done   N run(s) in X.Xs` carries total wall time.
_TOTAL_WALL_TIME_RE = re.compile(r'(done\s+\d+ run\(s\) in )\d+\.\d+s')


def normalize_stdout(text: str) -> str:
    """Strip wall-clock noise, scrub session paths/ids, sort
    parallel-emitted blocks so byte-equal compares survive run-to-run.
    """
    scrubbed = _DURATION_RE.sub('[<dur>]', text)
    scrubbed = _AGE_RE.sub('age=<age>', scrubbed)
    scrubbed = _SESSION_PATH_RE.sub('<session-path>', scrubbed)
    scrubbed = _SESSION_ID_RE.sub('session <session-id>', scrubbed)
    scrubbed = _TRAINED_OR_CACHED_RE.sub(r'\1<trained-or-cached>', scrubbed)
    scrubbed = _COMPLETION_INDEX_RE.sub(r'done  (<i>/\1,', scrubbed)
    scrubbed = _TOTAL_WALL_TIME_RE.sub(r'\1<wall>s', scrubbed)
    # The per-window `done` lines are emitted in parallel-completion
    # order. Sort the whole stdout line-wise so two runs with the same
    # underlying content (different completion order) compare byte-equal.
    lines = scrubbed.splitlines()
    return '\n'.join(sorted(lines)) + '\n'


def normalize_stderr(text: str) -> str:
    """Drop tqdm progress lines (they carry per-iter timings); keep the
    residual lines (typically empty) so we still byte-compare what
    actually produced output."""
    keep: list[str] = []
    for line in text.splitlines():
        if 'it/s]' in line or 'it [' in line:
            continue
        keep.append(line)
    return '\n'.join(sorted(keep)) + '\n'


def normalize_csv(text: str) -> str:
    """Sort CSV rows (preserving the header) so parallel completion
    order does not drive byte-equality."""
    lines = text.splitlines()
    if not lines:
        return ''
    header, rows = lines[0], sorted(lines[1:])
    return header + '\n' + '\n'.join(rows) + '\n'


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', required=True, choices=['stdout', 'stderr', 'csv'])
    p.add_argument('--in', dest='in_path', required=True, type=Path)
    p.add_argument('--out', dest='out_path', required=True, type=Path)
    args = p.parse_args(argv)
    text = args.in_path.read_text(encoding='utf-8')
    if args.mode == 'stdout':
        normalized = normalize_stdout(text)
    elif args.mode == 'stderr':
        normalized = normalize_stderr(text)
    else:
        normalized = normalize_csv(text)
    args.out_path.write_text(normalized, encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
