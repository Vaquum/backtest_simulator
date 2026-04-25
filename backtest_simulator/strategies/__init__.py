"""Strategy harnesses for sanity / prescient / shuffle / over-trading tests."""

# Each module in this package implements one Nexus `Strategy` subclass
# used by the M2 sanity / lookahead tests (slice #17 tasks 2-9, 15).
# The strategies are real `nexus.strategy.base.Strategy` implementations
# (not templates) so they can be instantiated directly in unit tests
# without going through ManifestBuilder's template-substitution path.

from backtest_simulator.strategies.buy_and_hold import BuyAndHoldStrategy
from backtest_simulator.strategies.over_trading import OverTradingStrategy
from backtest_simulator.strategies.zero_trade import ZeroTradeStrategy

__all__ = [
    'BuyAndHoldStrategy',
    'OverTradingStrategy',
    'ZeroTradeStrategy',
]
