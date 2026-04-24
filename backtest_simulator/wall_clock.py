"""CLOCK_MONOTONIC wall-clock reader that bypasses freezegun."""
from __future__ import annotations

import ctypes
import ctypes.util
import time
from typing import Final

_CLOCK_MONOTONIC_RAW: Final[int] = 4
_CLOCK_MONOTONIC: Final[int] = 1


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
    loaded (non-POSIX environments) or when clock_gettime rejects
    CLOCK_MONOTONIC_RAW (some older / emulated kernels). Silently
    returning an uninitialised timespec (old behaviour) would leave
    ts.tv_sec=0, producing bogus timings that look like "freeze stuck
    at epoch" — the exact failure mode every wall-time gate in §9
    must fail loud on.
    """
    if _libc is None:
        return time.perf_counter()
    ts = _TimeSpec()
    rc = _libc.clock_gettime(_CLOCK_MONOTONIC_RAW, ctypes.byref(ts))
    if rc != 0:
        # clock_gettime returned non-zero → errno populated. The
        # CLOCK_MONOTONIC (non-RAW) clock is POSIX-mandatory on every
        # supported platform, so a fallback to it is strictly safer
        # than silently using an uninitialised timespec. freezegun
        # does not patch ctypes-dispatched clock_gettime, so the raw
        # clock's anti-patch property is preserved.
        rc = _libc.clock_gettime(_CLOCK_MONOTONIC, ctypes.byref(ts))
        if rc != 0:
            return time.perf_counter()
    return ts.tv_sec + ts.tv_nsec * 1e-9
