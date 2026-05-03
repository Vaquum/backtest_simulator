"""Ledger parity — backtest event spine vs Praxis paper-trade spine."""
from __future__ import annotations

# Slice #17 Task 18. The M3 unlock is "backtest ≡ paper-Praxis ≡
# live, byte-identical event spine on a deterministic scenario".
# `assert_ledger_parity` enforces this contract: given two event-
# spine paths (each a JSON-lines log of the runtime's event
# sequence), they must agree under the chosen tolerance.
#
# `ParityTolerance.STRICT` requires byte-identical spines.
# `ParityTolerance.CLOCK_NORMALIZED` strips the envelope
# `event_seq` (sqlite-assigned) + envelope `timestamp` (wall-
# clock vs frozen) but leaves payload bytes intact. Use it for
# the cross-runtime case where the same scripted Praxis runs at
# different wall-clock times.
import base64
import json
import sqlite3
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import cast

from backtest_simulator.exceptions import ParityViolation


class ParityTolerance(Enum):
    """Comparison strictness for `assert_ledger_parity`."""

    STRICT = auto()
    CLOCK_NORMALIZED = auto()


@dataclass(frozen=True)
class _ParityFailure:
    line_no: int
    backtest_line: str
    paper_line: str


def _read_lines(path: Path) -> list[str]:
    if not path.is_file():
        msg = f'assert_ledger_parity: event-spine file missing: {path}'
        raise FileNotFoundError(msg)
    return path.read_text(encoding='utf-8').splitlines()


def _strict_diff(a: list[str], b: list[str]) -> list[_ParityFailure]:
    failures: list[_ParityFailure] = []
    for i, (al, bl) in enumerate(zip(a, b, strict=False)):
        if al != bl:
            failures.append(_ParityFailure(
                line_no=i + 1, backtest_line=al, paper_line=bl,
            ))
    if len(a) != len(b):
        # Length mismatch surfaces as a tail diff.
        if len(a) > len(b):
            for i, line in enumerate(a[len(b):], start=len(b) + 1):
                failures.append(_ParityFailure(
                    line_no=i, backtest_line=line, paper_line='',
                ))
        else:
            for i, line in enumerate(b[len(a):], start=len(a) + 1):
                failures.append(_ParityFailure(
                    line_no=i, backtest_line='', paper_line=line,
                ))
    return failures


def assert_ledger_parity(
    *,
    backtest_event_spine: Path,
    paper_event_spine: Path,
    tolerance: ParityTolerance = ParityTolerance.STRICT,
) -> None:
    """Assert that two event-spine files agree under `tolerance`.

    `STRICT` requires byte-identical lines.
    `CLOCK_NORMALIZED` strips the envelope `event_seq` (sqlite-
    assigned) + envelope `timestamp` (wall-clock vs frozen) from
    each line before comparison; payload bytes (`payload_raw_b64`)
    stay intact so a payload-content drift still raises.

    Raises `ParityViolation` on the first mismatch (or all of
    them — production may want a delta report; the MVC pin is the
    raise itself).
    """
    a = _read_lines(backtest_event_spine)
    b = _read_lines(paper_event_spine)
    if tolerance == ParityTolerance.CLOCK_NORMALIZED:
        a = [_clock_normalize(line) for line in a]
        b = [_clock_normalize(line) for line in b]
    failures = _strict_diff(a, b)
    if failures:
        first = failures[0]
        msg = (
            f'assert_ledger_parity ({tolerance.name}): backtest spine '
            f'({backtest_event_spine}) diverges from paper spine '
            f'({paper_event_spine}) at line {first.line_no} '
            f'(total {len(failures)} divergent lines). '
            f'backtest={first.backtest_line!r} '
            f'paper={first.paper_line!r}'
        )
        raise ParityViolation(msg)


def _clock_normalize(line: str) -> str:
    """Strip envelope `event_seq` + `timestamp`; keep payload bytes.

    The envelope timestamp is wall-clock vs frozen across runtimes;
    event_seq is sqlite-assigned and cross-DB-volatile. Both
    legitimately differ between bts and paper-Praxis. The payload
    bytes (`payload_raw_b64`) stay — they encode the deterministic
    event content (signal, action, fill data) and any drift there
    is a real divergence.
    """
    obj = json.loads(line)
    obj.pop('event_seq', None)
    obj.pop('timestamp', None)
    return json.dumps(obj, sort_keys=True)


def dump_event_spine_to_jsonl(
    *,
    sqlite_path: Path,
    jsonl_path: Path,
    epoch_id: int | None = None,
) -> int:
    """Dump events from an EventSpine sqlite DB into JSONL.

    Each line is a JSON object with keys: `epoch_id`, `event_seq`,
    `timestamp`, `event_type`, `payload_raw_b64`. The payload is
    stored as base64 of the literal sqlite payload bytes —
    `CAST(payload AS BLOB)` forces the column-affinity coercion at
    the sqlite layer so we get the underlying byte array regardless
    of whether the row was inserted via TEXT or BLOB binding. base64
    is bijective, so `STRICT` line equality on the dumped JSONL
    implies byte-equality on the underlying event row.

    `epoch_id`: optional filter — only dump events for that epoch.
    Returns the number of events dumped.

    Raises:
        FileNotFoundError: sqlite file missing.
        ValueError: events table missing or schema unexpected.
    """
    if not sqlite_path.is_file():
        msg = (
            f'dump_event_spine_to_jsonl: sqlite event-spine missing '
            f'at {sqlite_path}'
        )
        raise FileNotFoundError(msg)
    conn = sqlite3.connect(sqlite_path)
    try:
        # Validate schema loudly — sqlite version drift could change
        # column names/order; better to fail with row context than
        # silently dump garbage.
        schema_cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='events'",
        )
        if schema_cursor.fetchone() is None:
            msg = (
                f'dump_event_spine_to_jsonl: sqlite at {sqlite_path} '
                f'has no `events` table; not an EventSpine database.'
            )
            raise ValueError(msg)
        if epoch_id is None:
            cursor = conn.execute(
                'SELECT epoch_id, event_seq, timestamp, event_type, '
                'CAST(payload AS BLOB) AS payload_blob '
                'FROM events ORDER BY epoch_id, event_seq',
            )
        else:
            cursor = conn.execute(
                'SELECT epoch_id, event_seq, timestamp, event_type, '
                'CAST(payload AS BLOB) AS payload_blob '
                'FROM events WHERE epoch_id = ? '
                'ORDER BY epoch_id, event_seq',
                (epoch_id,),
            )
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with jsonl_path.open('w', encoding='utf-8') as f:
            for ep, seq, ts, etype, payload_blob in cursor:
                if not isinstance(payload_blob, bytes):
                    msg = (
                        f'dump_event_spine_to_jsonl: row '
                        f'(epoch={ep}, seq={seq}) returned non-bytes '
                        f'payload after CAST AS BLOB '
                        f'(got {type(payload_blob).__name__}); '
                        f'sqlite version drift?'
                    )
                    raise ValueError(msg)
                line = json.dumps({
                    'epoch_id': ep, 'event_seq': seq,
                    'timestamp': ts, 'event_type': etype,
                    'payload_raw_b64': base64.b64encode(
                        payload_blob,
                    ).decode('ascii'),
                }, sort_keys=True)
                f.write(line + '\n')
                count += 1
        return count
    finally:
        conn.close()


_REJECT_EVENT_TYPES = frozenset({
    'OrderRejected', 'OrderExpired',
    'OrderCanceled', 'OrderSubmitFailed',
})


def _trade_outcome_is_pending(payload_blob: object) -> bool:
    """`TradeOutcomeProduced` payload encodes `status` as JSON bytes."""
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


def _classify_spine_event(
    event_type: str, payload_blob: object, counts: dict[str, int],
) -> None:
    """Bump the right counter bucket for one event row. Mutates `counts`."""
    counts['total'] += 1
    if event_type == 'OrderSubmitIntent':
        counts['intents'] += 1
    elif event_type == 'OrderSubmitted':
        counts['submitted'] += 1
    elif event_type == 'FillReceived':
        counts['fills'] += 1
    elif event_type in _REJECT_EVENT_TYPES:
        counts['rejects'] += 1
    elif event_type == 'TradeOutcomeProduced' and _trade_outcome_is_pending(
        payload_blob,
    ):
        counts['pending'] += 1


def count_event_spine_events(
    *,
    sqlite_path: Path,
    epoch_id: int | None = None,
) -> dict[str, int]:
    """Count operationally-significant events in an EventSpine sqlite DB.

    Returns a dict with keys: `intents`, `submitted`, `fills`,
    `pending`, `rejects`, `total`. The first five answer the trader's
    five-question scan after each window:
      - intents: did my strategy decide to act? (`OrderSubmitIntent`)
      - submitted: did the venue accept the action? (`OrderSubmitted`)
      - fills: did money move? (`FillReceived`)
      - pending: what's still hanging at window close?
        (`TradeOutcomeProduced` with payload `status == 'PENDING'`)
      - rejects: what got blocked or expired? (`OrderRejected`,
        `OrderExpired`, `OrderCanceled`, `OrderSubmitFailed`)

    `total` is the unfiltered event count for cross-checking against
    `dump_event_spine_to_jsonl`'s return.

    Returns all-zero counts when the file is missing rather than
    raising — the per-run summary should degrade gracefully when a
    window's spine wasn't written (e.g. the launcher crashed before
    flush).
    """
    counts: dict[str, int] = {
        'intents': 0,
        'submitted': 0,
        'fills': 0,
        'pending': 0,
        'rejects': 0,
        'total': 0,
    }
    if not sqlite_path.is_file():
        return counts
    conn = sqlite3.connect(sqlite_path)
    try:
        if epoch_id is None:
            cursor = conn.execute(
                'SELECT event_type, CAST(payload AS BLOB) FROM events',
            )
        else:
            cursor = conn.execute(
                'SELECT event_type, CAST(payload AS BLOB) FROM events '
                'WHERE epoch_id = ?',
                (epoch_id,),
            )
        for event_type, payload_blob in cursor:
            _classify_spine_event(event_type, payload_blob, counts)
        return counts
    finally:
        conn.close()
