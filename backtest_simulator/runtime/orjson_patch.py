"""Monkey-patch orjson to coerce freezegun FakeDatetime via .isoformat()."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import orjson

_PATCHED = False


def apply() -> None:
    """Patch orjson so FakeDatetime (a datetime subclass) doesn't trip its dispatch.

    Praxis's event-spine serializer passes `datetime` subclasses to
    `orjson.dumps`. `orjson` dispatches on exact type, so `FakeDatetime`
    from freezegun raises TypeError. The patch wraps `orjson.dumps` to
    add a `default` callback that coerces datetime subclasses via
    `.isoformat()`. Idempotent — safe to call many times.
    """
    global _PATCHED
    if _PATCHED:
        return
    original_dumps = orjson.dumps

    def _default(obj: Any) -> Any:  # noqa: ANN401 - orjson's `default` protocol is inherently dynamic
        if isinstance(obj, datetime):
            return obj.isoformat()
        msg = f'orjson_patch cannot serialize {type(obj).__name__}'
        raise TypeError(msg)

    def _patched(*args: Any, **kwargs: Any) -> bytes:  # noqa: ANN401 - orjson.dumps signature
        provided = kwargs.get('default')
        if provided is None:
            kwargs['default'] = _default
        return original_dumps(*args, **kwargs)

    orjson.dumps = _patched  # type: ignore[assignment]
    _PATCHED = True
