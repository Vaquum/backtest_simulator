"""Honesty-violation exception hierarchy. Never caught except at test boundary."""
from __future__ import annotations


class HonestyViolation(Exception):
    pass

class LookAheadViolation(HonestyViolation):
    pass

class ConservationViolation(HonestyViolation):
    pass

class DeterminismViolation(HonestyViolation):
    pass

class ParityViolation(HonestyViolation):
    pass

class SanityViolation(HonestyViolation):
    pass

class PerformanceViolation(HonestyViolation):
    pass

class StopContractViolation(HonestyViolation):
    pass
