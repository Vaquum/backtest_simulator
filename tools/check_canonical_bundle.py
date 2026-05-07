#!/usr/bin/env python3
"""Canonical-bundle gate: locked-fixture bytes must match the pinned SHA256s."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
FIXTURES_DIR: Final[Path] = REPO_ROOT / 'tests' / 'fixtures' / 'canonical'
CHECKSUMS_FILE: Final[Path] = FIXTURES_DIR / 'checksums.sha256'


def _parse_checksums(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            msg = f'malformed checksum line: {line!r}'
            raise ValueError(msg)
        digest, name = parts
        out[name.strip()] = digest.strip()
    return out


def main() -> int:
    if not CHECKSUMS_FILE.is_file():
        print(f'CANONICAL BUNDLE GATE -- FAIL: {CHECKSUMS_FILE} missing', file=sys.stderr)
        return 1
    pinned = _parse_checksums(CHECKSUMS_FILE.read_text(encoding='utf-8'))
    if not pinned:
        print('CANONICAL BUNDLE GATE -- FAIL: empty checksums file', file=sys.stderr)
        return 1
    violations: list[str] = []
    for name, expected in pinned.items():
        path = FIXTURES_DIR / name
        if not path.is_file():
            violations.append(f'{name}: missing at {path}')
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            violations.append(f'{name}: expected={expected} actual={actual}')
    if violations:
        print('CANONICAL BUNDLE GATE -- FAIL', file=sys.stderr)
        for v in violations:
            print(f'  {v}', file=sys.stderr)
        print('Merge blocked.', file=sys.stderr)
        return 1
    print(f'CANONICAL BUNDLE GATE -- PASS ({len(pinned)} fixture(s) match)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
