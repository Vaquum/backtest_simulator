"""`bts notebook` — execute / convert a Jupyter notebook via nbconvert."""

# The slice's MVC asserts the operator-facing demo notebook
# `notebooks/sweep_and_analyze.ipynb` runs end-to-end. This subcommand
# wraps `jupyter nbconvert` so that single workflow has a stable entry
# point under `bts`, matching the operator-mandated "CLI is the master
# tool" rule.
from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final

from backtest_simulator.cli._verbosity import add_verbosity_arg, configure

_FORMAT_TO_NBCONVERT: Final[dict[str, list[str]]] = {
    'ipynb': ['--to', 'notebook'],
    'html': ['--to', 'html'],
    'script': ['--to', 'script'],
}


def register(add_parser: Callable[[str, str], argparse.ArgumentParser]) -> None:
    p = add_parser('notebook', 'Execute / convert a Jupyter notebook.')
    p.add_argument('--path', required=True, type=Path,
                   help='Path to the .ipynb file.')
    p.add_argument('--execute', action='store_true', default=True,
                   help='Execute cells (default: true).')
    p.add_argument('--no-execute', action='store_false', dest='execute',
                   help='Skip cell execution (convert only).')
    p.add_argument('--output-format', choices=sorted(_FORMAT_TO_NBCONVERT.keys()),
                   default='ipynb',
                   help='Output format (default: ipynb).')
    add_verbosity_arg(p)
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    configure(args.verbose)
    if not args.path.is_file():
        sys.stderr.write(f'bts notebook: {args.path} not found\n')
        return 2
    cmd: list[str] = [sys.executable, '-m', 'jupyter', 'nbconvert']
    cmd.extend(_FORMAT_TO_NBCONVERT[args.output_format])
    if args.execute:
        cmd.append('--execute')
    if args.output_format == 'script':
        cmd.extend(['--stdout', str(args.path)])
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            sys.stderr.write(result.stderr)
            return result.returncode
        return subprocess.run(
            [sys.executable, '-c', result.stdout], check=False,
        ).returncode
    cmd.append(str(args.path))
    return subprocess.run(cmd, check=False).returncode
