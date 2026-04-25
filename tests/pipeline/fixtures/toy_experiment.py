"""Minimal fixture: a user experiment file with `params()` and `manifest()`."""
from __future__ import annotations


def params() -> dict[str, list[object]]:
    return {'lookback': [10, 20, 30], 'threshold': [0.55, 0.60]}


def manifest() -> object:
    class _FakeManifest:
        name = 'toy'
    return _FakeManifest()
