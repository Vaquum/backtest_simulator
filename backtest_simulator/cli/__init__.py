"""bts master CLI — see docs/cli.md for the operator-facing surface."""

# Subcommands: run, sweep, enrich, test, lint, typecheck, gate, notebook,
# version. Verbosity flags `-v`, `-vv`, `-vvv` are accepted on every
# subcommand and progressively unmask logging from structlog (Praxis /
# Nexus / Limen) and the backtest pipeline. Returns the subcommand's
# exit code; on parse error returns 2.
#
# This is the operator-facing surface of the package. All debugging and
# gate invocation flows through `bts <subcommand>`; nothing else in the
# project is meant to be invoked directly.
from __future__ import annotations

import argparse
import sys
from typing import Final

from backtest_simulator.cli.commands import (
    enrich as _enrich,
)
from backtest_simulator.cli.commands import (
    gate as _gate,
)
from backtest_simulator.cli.commands import (
    lint as _lint,
)
from backtest_simulator.cli.commands import (
    notebook as _notebook,
)
from backtest_simulator.cli.commands import (
    run as _run,
)
from backtest_simulator.cli.commands import (
    sweep as _sweep,
)
from backtest_simulator.cli.commands import (
    test as _test,
)
from backtest_simulator.cli.commands import (
    typecheck as _typecheck,
)
from backtest_simulator.cli.commands import (
    version as _version,
)

SUBCOMMANDS: Final[tuple[str, ...]] = (
    'run', 'sweep', 'enrich', 'test', 'lint',
    'typecheck', 'gate', 'notebook', 'version',
)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the master `bts` parser. Subparsers carry their own args.

    Verbosity (`-v`/`-vv`/`-vvv`) is registered on each subparser, not
    the top-level parser, so `bts -v run ...` is rejected and the only
    valid invocation shape is `bts <subcommand> -v ...`. This keeps the
    CLI surface unambiguous and matches the per-subcommand log-level
    contract documented in `docs/cli.md`.

    Each subcommand module exposes a `register(add_parser)` function
    that takes a closure capable of producing a fresh
    `argparse.ArgumentParser`. The closure shape avoids exporting
    `argparse._SubParsersAction` (private in the stdlib stubs) past
    this module — only this `_build_parser` mentions it via type
    inference, the subcommand modules see only the `add_parser`
    callable.
    """
    ap = argparse.ArgumentParser(prog='bts', description='backtest_simulator master CLI')
    sub = ap.add_subparsers(dest='cmd', required=True)

    def add_parser(name: str, help_: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help_)

    _run.register(add_parser)
    _sweep.register(add_parser)
    _enrich.register(add_parser)
    _test.register(add_parser)
    _lint.register(add_parser)
    _typecheck.register(add_parser)
    _gate.register(add_parser)
    _notebook.register(add_parser)
    _version.register(add_parser)
    return ap


def main(argv: list[str] | None = None) -> int:
    """Master entry point invoked by the `bts` console script."""
    ap = _build_parser()
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == '__main__':  # pragma: no cover - manual invocation path
    sys.exit(main())
