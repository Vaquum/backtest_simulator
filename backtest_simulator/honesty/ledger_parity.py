"""Ledger parity — backtest event spine vs Praxis paper-trade spine."""
from __future__ import annotations

import base64
import json
import sqlite3
from enum import Enum, auto
from pathlib import Path


class ParityTolerance(Enum):
    STRICT = auto()
    CLOCK_NORMALIZED = auto()

def dump_event_spine_to_jsonl(*, sqlite_path: Path, jsonl_path: Path, epoch_id: int | None=None) -> int:
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='events'")
        if epoch_id is None:
            cursor = conn.execute('SELECT epoch_id, event_seq, timestamp, event_type, CAST(payload AS BLOB) AS payload_blob FROM events ORDER BY epoch_id, event_seq')
        else:
            cursor = conn.execute('SELECT epoch_id, event_seq, timestamp, event_type, CAST(payload AS BLOB) AS payload_blob FROM events WHERE epoch_id = ? ORDER BY event_seq', (epoch_id,))
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with jsonl_path.open('w', encoding='utf-8') as f:
            for ep, seq, ts, etype, payload_blob in cursor:
                line = json.dumps({'epoch_id': ep, 'event_seq': seq, 'timestamp': ts, 'event_type': etype, 'payload_raw_b64': base64.b64encode(payload_blob).decode('ascii')}, sort_keys=True)
                f.write(line + '\n')
                count += 1
        return count
    finally:
        conn.close()
_REJECT_EVENT_TYPES = frozenset({'OrderRejected', 'OrderExpired', 'OrderCanceled', 'OrderSubmitFailed'})

def _classify_spine_event(event_type: str, payload_blob: object, counts: dict[str, int]) -> None:
    del payload_blob
    counts['total'] += 1
    if event_type == 'OrderSubmitIntent':
        counts['intents'] += 1
    elif event_type == 'OrderSubmitted':
        counts['submitted'] += 1
    elif event_type == 'FillReceived':
        counts['fills'] += 1

def count_event_spine_events(*, sqlite_path: Path, epoch_id: int | None=None) -> dict[str, int]:
    counts: dict[str, int] = {'intents': 0, 'submitted': 0, 'fills': 0, 'pending': 0, 'rejects': 0, 'total': 0}
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
