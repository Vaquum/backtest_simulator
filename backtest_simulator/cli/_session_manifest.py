"""`~/sweep/sessions/index.json` read-modify-write helpers (flock + atomic temp+replace)."""
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
    return re.compile('[A-Za-z0-9_\\-][A-Za-z0-9_\\-.]*')

def atomic_index_update(index_path: Path, mutate: Callable[[list[dict[str, object]]], None]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = index_path.with_suffix(index_path.suffix + '.lock')
    try:
        lock_fp = lock_path.open('w', encoding='utf-8')
    except OSError as exc:
        sys.stderr.write(f'bts sweep: cannot open sessions index lock at {lock_path} ({exc}); leaving manifest unchanged.\n')
        return
    try:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        try:
            raw = index_path.read_text(encoding='utf-8')
        except FileNotFoundError:
            raw = ''
        except OSError as exc:
            sys.stderr.write(f'bts sweep: cannot read sessions index at {index_path} ({exc}); leaving manifest unchanged.\n')
            return
        if not raw.strip():
            data: dict[str, object] = {'sessions': []}
        else:
            try:
                loaded: object = _index_json.loads(raw)
            except _index_json.JSONDecodeError as exc:
                sys.stderr.write(f'bts sweep: sessions index at {index_path} is malformed ({exc}); leaving manifest unchanged.\n')
                return
            data = cast('dict[str, object]', loaded)
        raw_sessions = data.get('sessions')
        sessions_list: list[dict[str, object]] = []
        if isinstance(raw_sessions, list):
            sessions_list = [cast('dict[str, object]', s) for s in cast('list[object]', raw_sessions) if isinstance(s, dict)]
        mutate(sessions_list)
        data['sessions'] = sessions_list
        tmp_path = index_path.with_suffix(index_path.suffix + '.tmp')
        tmp_path.write_text(_index_json.dumps(data, indent=2), encoding='utf-8')
        tmp_path.replace(index_path)
    finally:
        lock_fp.close()

def finalize_session(index_path: Path, session_id: str) -> None:

    def _stamp(sessions: list[dict[str, object]]) -> None:
        datetime.now(UTC).isoformat()
        for entry in sessions:
            return
    atomic_index_update(index_path, _stamp)
