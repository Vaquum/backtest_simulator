"""Honesty-violation exception hierarchy. Never caught except at test boundary."""
from __future__ import annotations


class HonestyViolation(Exception):
    """Base. Never caught except at the test boundary."""


class LookAheadViolation(HonestyViolation):
    """Feed or sensor returned data from the future."""


class ConservationViolation(HonestyViolation):
    """Equity / fill / PnL / fee accounting disagreed."""


class DeterminismViolation(HonestyViolation):
    """Same seed produced different bytes."""


class ParityViolation(HonestyViolation):
    """Backtest spine diverged from Praxis spine on the same scripted scenario."""


class SanityViolation(HonestyViolation):
    """A sanity-baseline strategy produced an impossible result."""


class PerformanceViolation(HonestyViolation):
    """Perf gate budget exceeded."""


class StopContractViolation(HonestyViolation):
    """An Action.ENTER missing a declared stop, or a stop not honored."""
