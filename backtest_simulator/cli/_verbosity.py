"""Shared verbosity / logging setup for `bts` subcommands."""

# `-v` raises stdlib + structlog log level from ERROR (default) to INFO,
# `-vv` raises to DEBUG, `-vvv` to NOTSET (everything; including Limen
# tqdm and any module-level prints we silence at level 0). The mapping
# is centralised so every subcommand interprets the same flag identically.
from __future__ import annotations

import argparse
import logging
import warnings
from typing import Final

_VERBOSITY_TO_LEVEL: Final[dict[int, int]] = {
    0: logging.ERROR,
    1: logging.INFO,
    2: logging.DEBUG,
    3: logging.NOTSET,
}

_NOISY_LOGGERS: Final[tuple[str, ...]] = (
    'praxis', 'nexus', 'limen', 'urllib3', 'clickhouse_connect',
    'backtest_simulator', 'asyncio', 'binance',
)


def add_verbosity_arg(parser: argparse.ArgumentParser) -> None:
    """Add `-v` / `-vv` / `-vvv` (mutually-counted) to a subparser."""
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help='Verbosity: -v INFO, -vv DEBUG, -vvv unmasked (default ERROR).',
    )


def configure(verbosity: int) -> None:
    """Apply the verbosity level to every chatty subsystem.

    At level 0 (no `-v`), the CLI's own per-run summary is the only
    thing on stdout. Higher levels progressively unmask Praxis / Nexus
    / Limen logs and Limen's tqdm training bar. Calling this more than
    once is safe — each call rebinds the level.
    """
    level = _VERBOSITY_TO_LEVEL.get(verbosity, logging.NOTSET)
    logging.basicConfig(level=level, format='%(levelname)s %(name)s %(message)s')
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(level)
    if verbosity == 0:
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=DeprecationWarning)
    if verbosity < 3:
        _silence_tqdm_and_structlog()


def _silence_tqdm_and_structlog() -> None:
    """Silence Limen's tqdm training bar + Praxis's structlog formatters.

    Both bypass stdlib logging — tqdm writes directly to stderr and
    structlog has its own processor chain — so configuring `logging`
    alone leaves them noisy. Both are guaranteed transitive deps
    (`tqdm` via vaquum_limen / binancial, `structlog` via
    vaquum-nexus / vaquum-praxis), so direct imports are honest
    here — if either is absent, the bts venv is misconfigured and
    we want a loud ImportError, not a silent no-op.
    """
    import structlog
    import tqdm as _tqdm

    class _NoopTqdm:
        # zero-bang post-auditor-4 P1: replaces a bare `iter(())`
        # passthrough that broke tqdm's manual-mode API
        # (`.update()` / `.close()`). Iterates the wrapped iterable;
        # any other attribute (.update, .set_description, .close,
        # .refresh, etc.) returns a no-op via __getattr__. Also a
        # no-op context manager.
        def __init__(self, iterable: object = None, *_a: object, **_kw: object) -> None:
            self._iterable = iterable

        def __iter__(self) -> object:
            return iter(self._iterable) if self._iterable is not None else iter(())

        def __getattr__(self, _name: str) -> object:
            return lambda *_a, **_kw: None

        def __enter__(self) -> object:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    _tqdm.tqdm = _NoopTqdm
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR),
    )
