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
    """
    ap = argparse.ArgumentParser(prog='bts', description='backtest_simulator master CLI')
    sub = ap.add_subparsers(dest='cmd', required=True)
    _run.register(sub)
    _sweep.register(sub)
    _enrich.register(sub)
    _test.register(sub)
    _lint.register(sub)
    _typecheck.register(sub)
    _gate.register(sub)
    _notebook.register(sub)
    _version.register(sub)
    return ap


def main(argv: list[str] | None = None) -> int:
    """Master entry point invoked by the `bts` console script."""
    ap = _build_parser()
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == '__main__':  # pragma: no cover - manual invocation path
    sys.exit(main())
