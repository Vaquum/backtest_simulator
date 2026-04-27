"""Honesty gate: dump EventSpine sqlite -> JSONL preserving raw payload bytes.

Slice #17 Task 18 / auditor round 3 P1: STRICT byte-equality on
the JSONL implies byte-equality on the underlying sqlite payload
bytes. Tests pin the contract.
"""
from __future__ import annotations

import base64
import json
import sqlite3
from pathlib import Path

import pytest

from backtest_simulator.honesty.ledger_parity import (
    dump_event_spine_to_jsonl,
)


def _make_event_spine_sqlite(
    db_path: Path, rows: list[tuple[int, int, str, str, bytes]],
) -> None:
    """Build a minimal sqlite EventSpine fixture.

    Schema mirrors `praxis.infrastructure.event_spine`'s `events`
    table: (epoch_id, event_seq, timestamp, event_type, payload).
    Payload is BLOB; tests stuff arbitrary bytes (incl. invalid
    UTF-8) to exercise the BLOB-cast preservation path.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            'CREATE TABLE events ('
            'epoch_id INTEGER, event_seq INTEGER PRIMARY KEY, '
            'timestamp TEXT, event_type TEXT, payload BLOB)',
        )
        conn.executemany(
            'INSERT INTO events '
            '(epoch_id, event_seq, timestamp, event_type, payload) '
            'VALUES (?, ?, ?, ?, ?)',
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def test_dump_event_spine_to_jsonl_basic(tmp_path: Path) -> None:
    """Dump round-trips epoch_id, event_seq, timestamp, event_type, payload."""
    db = tmp_path / 'spine.sqlite'
    payload = b'{"foo":"bar"}'
    _make_event_spine_sqlite(db, [
        (1, 1, '2024-01-01T00:00:00+00:00', 'StartEvent', payload),
        (1, 2, '2024-01-01T00:00:01+00:00', 'FillEvent', b'{"qty":"1"}'),
    ])
    out = tmp_path / 'spine.jsonl'
    n = dump_event_spine_to_jsonl(sqlite_path=db, jsonl_path=out)
    assert n == 2
    lines = out.read_text(encoding='utf-8').splitlines()
    assert len(lines) == 2
    obj = json.loads(lines[0])
    assert obj['epoch_id'] == 1
    assert obj['event_seq'] == 1
    assert obj['event_type'] == 'StartEvent'
    decoded = base64.b64decode(obj['payload_raw_b64'])
    assert decoded == payload


def test_dump_event_spine_to_jsonl_preserves_blob_payload(
    tmp_path: Path,
) -> None:
    """Non-UTF8 payload bytes round-trip via CAST AS BLOB + base64.

    Codex round 3 P1: `payload.encode('utf-8')` would explode on
    invalid UTF-8 and fail to preserve original bytes. The
    `CAST(payload AS BLOB)` path keeps raw bytes regardless.

    Mutation proof: a binary payload (e.g. orjson with embedded
    bytes) round-trips bit-for-bit; if the dump parsed +
    reserialized, the bytes would shift.
    """
    db = tmp_path / 'spine.sqlite'
    binary_payload = b'\x80\x81\x82\xff\x00\x01'  # invalid UTF-8 prefix
    _make_event_spine_sqlite(db, [
        (1, 1, '2024-01-01T00:00:00+00:00', 'X', binary_payload),
    ])
    out = tmp_path / 'spine.jsonl'
    dump_event_spine_to_jsonl(sqlite_path=db, jsonl_path=out)
    obj = json.loads(out.read_text(encoding='utf-8').strip())
    decoded = base64.b64decode(obj['payload_raw_b64'])
    assert decoded == binary_payload


def test_dump_event_spine_to_jsonl_filters_by_epoch_id(
    tmp_path: Path,
) -> None:
    """`epoch_id` filter dumps only events for that epoch."""
    db = tmp_path / 'spine.sqlite'
    _make_event_spine_sqlite(db, [
        (1, 1, 't1', 'X', b'a'),
        (2, 2, 't2', 'X', b'b'),
        (1, 3, 't3', 'X', b'c'),
    ])
    out = tmp_path / 'spine.jsonl'
    n = dump_event_spine_to_jsonl(sqlite_path=db, jsonl_path=out, epoch_id=1)
    assert n == 2
    lines = out.read_text(encoding='utf-8').splitlines()
    epochs = [json.loads(line)['epoch_id'] for line in lines]
    assert all(ep == 1 for ep in epochs)


def test_dump_event_spine_to_jsonl_orders_by_seq(tmp_path: Path) -> None:
    """Output ordered by (epoch_id, event_seq) ascending."""
    db = tmp_path / 'spine.sqlite'
    # Insert out of order; query must reorder.
    _make_event_spine_sqlite(db, [
        (1, 5, 't5', 'X', b'5'),
        (1, 2, 't2', 'X', b'2'),
        (1, 7, 't7', 'X', b'7'),
        (1, 1, 't1', 'X', b'1'),
    ])
    out = tmp_path / 'spine.jsonl'
    dump_event_spine_to_jsonl(sqlite_path=db, jsonl_path=out)
    lines = out.read_text(encoding='utf-8').splitlines()
    seqs = [json.loads(line)['event_seq'] for line in lines]
    assert seqs == [1, 2, 5, 7]


def test_dump_event_spine_to_jsonl_raises_on_missing_file(
    tmp_path: Path,
) -> None:
    """Missing sqlite file -> FileNotFoundError loud."""
    db = tmp_path / 'nonexistent.sqlite'
    out = tmp_path / 'spine.jsonl'
    with pytest.raises(FileNotFoundError, match='sqlite event-spine missing'):
        dump_event_spine_to_jsonl(sqlite_path=db, jsonl_path=out)


def test_dump_event_spine_to_jsonl_raises_on_missing_table(
    tmp_path: Path,
) -> None:
    """Sqlite without `events` table -> ValueError loud (codex round 1 #5)."""
    db = tmp_path / 'wrong.sqlite'
    conn = sqlite3.connect(db)
    try:
        conn.execute('CREATE TABLE other_table (id INTEGER)')
        conn.commit()
    finally:
        conn.close()
    out = tmp_path / 'spine.jsonl'
    with pytest.raises(ValueError, match='no `events` table'):
        dump_event_spine_to_jsonl(sqlite_path=db, jsonl_path=out)
