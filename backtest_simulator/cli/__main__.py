"""Allow `python -m backtest_simulator.cli ...` as an alternate entry."""
from __future__ import annotations

import sys

from backtest_simulator.cli import main

if __name__ == '__main__':
    sys.exit(main())
