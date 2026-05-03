#!/usr/bin/env bash
# Force-pull latest Praxis / Nexus / Limen into the active Python env.
#
# `pyproject.toml` declares all three as unpinned `git+<url>` so a
# fresh install always tracks the latest sibling main. pip's default
# behaviour is to skip already-installed packages, so subsequent
# `pip install -e '.[integration]'` runs do NOT re-fetch — the local
# .venv silently lags whichever sibling SHA was current at first
# install. This script forces a clean re-fetch.
#
# Usage:
#   tools/refresh_siblings.sh            # uses `python` from PATH
#   PYTHON=~/.venvs/bts/bin/python tools/refresh_siblings.sh
#
# Run this before any `bts sweep` / `bts run` whose correctness
# depends on a sibling-side change merged after your last install.
set -euo pipefail
PYTHON="${PYTHON:-python}"
echo "Refreshing Praxis / Nexus / Limen via $PYTHON"
"$PYTHON" -m pip install --force-reinstall --no-deps \
  "vaquum-praxis @ git+https://github.com/Vaquum/Praxis" \
  "vaquum-nexus @ git+https://github.com/Vaquum/Nexus" \
  "vaquum_limen @ git+https://github.com/Vaquum/Limen"
echo "Done. Versions:"
"$PYTHON" -c "
import importlib.metadata as m
for pkg in ('vaquum-praxis', 'vaquum-nexus', 'vaquum_limen'):
    try:
        print(f'  {pkg} = {m.version(pkg)}')
    except m.PackageNotFoundError:
        print(f'  {pkg} = (not installed)')
"
