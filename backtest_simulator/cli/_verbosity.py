"""Shared verbosity / logging setup for `bts` subcommands."""
from __future__ import annotations

import argparse
import logging
import warnings
from typing import Final

_VERBOSITY_TO_LEVEL: Final[dict[int, int]] = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG, 3: logging.NOTSET}
_NOISY_LOGGERS: Final[tuple[str, ...]] = ('praxis', 'nexus', 'limen', 'urllib3', 'clickhouse_connect', 'backtest_simulator', 'asyncio', 'binance')

def add_verbosity_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Verbosity: -v INFO, -vv DEBUG, -vvv unmasked (default ERROR).')

def configure(verbosity: int) -> None:
    level = _VERBOSITY_TO_LEVEL.get(verbosity, logging.NOTSET)
    logging.basicConfig(level=level, format='%(levelname)s %(name)s %(message)s')
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(level)
    if verbosity == 0:
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=DeprecationWarning)
    if verbosity < 3:
        _silence_tqdm_and_structlog(level)

def _silence_tqdm_and_structlog(level: int) -> None:
    from collections.abc import Iterable, Iterator

    import structlog
    import tqdm as _tqdm

    class _NoopTqdm:
        def __init__(self, iterable: Iterable[object] = (), *_: object, **__: object) -> None:
            self._iterable = iterable
        def __iter__(self) -> Iterator[object]:
            return iter(self._iterable)
        def __enter__(self) -> _NoopTqdm:
            return self
        def __exit__(self, *_: object) -> None:
            return None
        def update(self, *_: object, **__: object) -> None:
            return None
        def close(self) -> None:
            return None
        def set_description(self, *_: object, **__: object) -> None:
            return None
        def set_postfix(self, *_: object, **__: object) -> None:
            return None
    _tqdm.tqdm = _NoopTqdm
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(level))
