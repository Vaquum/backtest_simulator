"""`~/sweep/sessions/index.json` read-modify-write helpers.

Extracted from `cli/commands/sweep.py` so the orchestrator stays
within the file-size balance ratio. The two helpers exposed here
are called from `_run`:

  * `atomic_index_update(path, mutate)`: serialises start-of-sweep
    `append` and end-of-sweep `ended_at` updates from concurrent
    `bts sweep` processes against a shared `index.json`. Uses a
    SEPARATE `.lock` file (so the lock survives the rename-into-
    place that the writer performs) + atomic `tmp.replace(path)`
    so readers (the dashboard, anyone polling the manifest) see
    only the OLD or NEW complete file.

  * `finalize_session(path, session_id)`: stamps `ended_at` for
    `session_id`. Registered via `atexit` immediately after the
    session is appended at sweep start so every exit path (clean
    return, `ParityViolation`, `RuntimeError("sweep aborted")`,
    `sys.exit`, `Ctrl-C`) clears the dashboard's `live` indicator.

Re-`re_session_id_pattern()` is also re-exported so the path-
traversal regex stays adjacent to the manifest writer.
"""
from __future__ import annotations

import fcntl
import json as _index_json
import re
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast


def re_session_id_pattern() -> re.Pattern[str]:
    """ASCII alnum + `-_.`, no leading dot. Path-traversal-safe."""
    return re.compile(r'[A-Za-z0-9_\-][A-Za-z0-9_\-.]*')


def atomic_index_update(
    index_path: Path,
    mutate: Callable[[list[dict[str, object]]], None],
) -> None:
    """Read-modify-write `index.json` under a flock + atomic temp+replace.

    Concurrency:
      - flock is taken on a SEPARATE `index.json.lock` file (not on
        the data file itself), so the lock survives the
        rename-into-place that the writer performs.
      - The new JSON is written to `<path>.tmp` first, then
        `Path.replace`d into place. Readers (the dashboard, anyone
        else that opens `index.json` without holding the lock) see
        either the OLD complete file or the NEW complete file —
        never the partial in-place truncate-then-write window of a
        single open file (Copilot P1).

    Serialises start-of-sweep `append` and end-of-sweep `ended_at`
    updates from concurrent `bts sweep` processes against a shared
    `~/sweep/sessions/index.json`. Without the lock the two
    processes' read-modify-write windows can overlap and the
    loser's update is silently lost (bit-mis P1).

    Failure modes are explicit:
      - missing file → seed with `{"sessions": []}` and proceed.
      - malformed JSON or unreadable file → write to stderr and BAIL
        without rewriting (do NOT clobber other operators' sessions
        with `{"sessions": []}`; bit-mis P1 silent-data-loss).
      - mutate raises → propagate; the temp file is left on disk
        for diagnostics but `index_path` is never overwritten.
    """
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if not index_path.is_file():
        index_path.write_text('{"sessions": []}', encoding='utf-8')
    lock_path = index_path.with_suffix(index_path.suffix + '.lock')
    try:
        lock_fp = lock_path.open('w', encoding='utf-8')
    except OSError as exc:
        sys.stderr.write(
            f'bts sweep: cannot open sessions index lock at '
            f'{lock_path} ({exc}); leaving manifest unchanged.\n',
        )
        return
    try:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        try:
            raw = index_path.read_text(encoding='utf-8')
        except OSError as exc:
            sys.stderr.write(
                f'bts sweep: cannot read sessions index at '
                f'{index_path} ({exc}); leaving manifest unchanged.\n',
            )
            return
        if not raw.strip():
            data: dict[str, object] = {'sessions': []}
        else:
            try:
                loaded: object = _index_json.loads(raw)
            except _index_json.JSONDecodeError as exc:
                sys.stderr.write(
                    f'bts sweep: sessions index at {index_path} is '
                    f'malformed ({exc}); leaving manifest unchanged.\n',
                )
                return
            if not isinstance(loaded, dict):
                sys.stderr.write(
                    f'bts sweep: sessions index at {index_path} is not '
                    f'an object; leaving manifest unchanged.\n',
                )
                return
            data = cast('dict[str, object]', loaded)
        raw_sessions = data.get('sessions')
        sessions_list: list[dict[str, object]] = []
        if isinstance(raw_sessions, list):
            sessions_list = [
                cast('dict[str, object]', s)
                for s in cast('list[object]', raw_sessions)
                if isinstance(s, dict)
            ]
        mutate(sessions_list)
        data['sessions'] = sessions_list
        tmp_path = index_path.with_suffix(index_path.suffix + '.tmp')
        tmp_path.write_text(_index_json.dumps(data, indent=2), encoding='utf-8')
        tmp_path.replace(index_path)
    finally:
        lock_fp.close()


def finalize_session(index_path: Path, session_id: str) -> None:
    """Stamp `ended_at` for `session_id` in `index.json`.

    Registered via `atexit` immediately after the session is appended at
    sweep start, so the dashboard's `live` indicator clears whether the
    sweep returned cleanly, raised `ParityViolation`,
    `RuntimeError("sweep aborted")`, or was interrupted (Ctrl-C lets
    `atexit` run; `os._exit` does not — that's only used post-success in
    subprocess children, not the parent).

    Idempotent: only stamps when `ended_at is None`. atexit handlers are
    not re-entered automatically, but the guard makes a manual second
    call harmless too.
    """
    def _stamp(sessions: list[dict[str, object]]) -> None:
        now_iso = datetime.now(UTC).isoformat()
        for entry in sessions:
            if entry.get('id') != session_id:
                continue
            if entry.get('ended_at') is None:
                entry['ended_at'] = now_iso
            return
    atomic_index_update(index_path, _stamp)
