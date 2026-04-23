"""CLOCK_MONOTONIC wall-clock reader that bypasses freezegun."""
from __future__ import annotations

import ctypes
import ctypes.util
import time
from typing import Final

_CLOCK_MONOTONIC_RAW: Final[int] = 4


class _TimeSpec(ctypes.Structure):
    _fields_ = (('tv_sec', ctypes.c_longlong), ('tv_nsec', ctypes.c_long))


_libname = ctypes.util.find_library('c')
_libc: ctypes.CDLL | None = ctypes.CDLL(_libname) if _libname else None
if _libc is not None:
    _libc.clock_gettime.argtypes = (ctypes.c_int, ctypes.POINTER(_TimeSpec))
    _libc.clock_gettime.restype = ctypes.c_int


def monotonic_seconds() -> float:
    """Return CLOCK_MONOTONIC_RAW seconds. Not patched by freezegun.

    Required for every wall-time measurement inside a freeze_time
    context. time.perf_counter / time.monotonic / time.time are all
    patched. Falls back to time.perf_counter when libc cannot be
    loaded (non-POSIX environments).
    """
    if _libc is None:
        return time.perf_counter()
    ts = _TimeSpec()
    _libc.clock_gettime(_CLOCK_MONOTONIC_RAW, ctypes.byref(ts))
    return ts.tv_sec + ts.tv_nsec * 1e-9
