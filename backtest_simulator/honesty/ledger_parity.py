"""Ledger parity — backtest event spine vs Praxis paper-trade spine."""
from __future__ import annotations

import base64
import json
import sqlite3
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import cast


class ParityTolerance(Enum):
    STRICT = auto()
    CLOCK_NORMALIZED = auto()

@dataclass(frozen=True)
class _ParityFailure:
    line_no: int
    backtest_line: str
    paper_line: str

def dump_event_spine_to_jsonl(*, sqlite_path: Path, jsonl_path: Path, epoch_id: int | None=None) -> int:
    if not sqlite_path.is_file():
        msg = f'dump_event_spine_to_jsonl: sqlite event-spine missing at {sqlite_path}'
        raise FileNotFoundError(msg)
    conn = sqlite3.connect(sqlite_path)
    try:
        schema_cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='events'")
        if schema_cursor.fetchone() is None:
            msg = f'dump_event_spine_to_jsonl: sqlite at {sqlite_path} has no `events` table; not an EventSpine database.'
            raise ValueError(msg)
        if epoch_id is None:
            cursor = conn.execute('SELECT epoch_id, event_seq, timestamp, event_type, CAST(payload AS BLOB) AS payload_blob FROM events ORDER BY epoch_id, event_seq')
        else:
            cursor = conn.execute('SELECT epoch_id, event_seq, timestamp, event_type, CAST(payload AS BLOB) AS payload_blob FROM events WHERE epoch_id = ? ORDER BY epoch_id, event_seq', (epoch_id,))
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with jsonl_path.open('w', encoding='utf-8') as f:
            for ep, seq, ts, etype, payload_blob in cursor:
                if not isinstance(payload_blob, bytes):
                    msg = f'dump_event_spine_to_jsonl: row (epoch={ep}, seq={seq}) returned non-bytes payload after CAST AS BLOB (got {type(payload_blob).__name__}); sqlite version drift?'
                    raise ValueError(msg)
                line = json.dumps({'epoch_id': ep, 'event_seq': seq, 'timestamp': ts, 'event_type': etype, 'payload_raw_b64': base64.b64encode(payload_blob).decode('ascii')}, sort_keys=True)
                f.write(line + '\n')
                count += 1
        return count
    finally:
        conn.close()
_REJECT_EVENT_TYPES = frozenset({'OrderRejected', 'OrderExpired', 'OrderCanceled', 'OrderSubmitFailed'})

def _trade_outcome_is_pending(payload_blob: object) -> bool:
    if not isinstance(payload_blob, bytes):
        return False
    try:
        payload_obj = json.loads(payload_blob)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    if not isinstance(payload_obj, dict):
        return False
    payload = cast('dict[str, object]', payload_obj)
    return payload.get('status') == 'PENDING'

def _classify_spine_event(event_type: str, payload_blob: object, counts: dict[str, int]) -> None:
    counts['total'] += 1
    if event_type == 'OrderSubmitIntent':
        counts['intents'] += 1
    elif event_type == 'OrderSubmitted':
        counts['submitted'] += 1
    elif event_type == 'FillReceived':
        counts['fills'] += 1
    elif event_type in _REJECT_EVENT_TYPES:
        counts['rejects'] += 1
    elif event_type == 'TradeOutcomeProduced' and _trade_outcome_is_pending(payload_blob):
        counts['pending'] += 1

def count_event_spine_events(*, sqlite_path: Path, epoch_id: int | None=None) -> dict[str, int]:
    counts: dict[str, int] = {'intents': 0, 'submitted': 0, 'fills': 0, 'pending': 0, 'rejects': 0, 'total': 0}
    if not sqlite_path.is_file():
        return counts
    conn = sqlite3.connect(sqlite_path)
    try:
        if epoch_id is None:
            cursor = conn.execute('SELECT event_type, CAST(payload AS BLOB) FROM events')
        else:
            cursor = conn.execute('SELECT event_type, CAST(payload AS BLOB) FROM events WHERE epoch_id = ?', (epoch_id,))
        for event_type, payload_blob in cursor:
            _classify_spine_event(event_type, payload_blob, counts)
        return counts
    finally:
        conn.close()
