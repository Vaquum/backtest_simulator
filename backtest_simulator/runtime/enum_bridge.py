"""Value-match enum translation between Nexus and Praxis enum classes."""
from __future__ import annotations

from enum import Enum
from typing import TypeVar

_PE = TypeVar('_PE', bound=Enum)


def to_praxis(praxis_cls: type[_PE], nexus_enum: Enum) -> _PE:
    """Convert a Nexus enum value to the corresponding Praxis enum instance.

    Nexus and Praxis define parallel enums for order type / side / TIF
    with identical string values but distinct classes. Use `.value` as
    the shared domain.
    """
    return praxis_cls(nexus_enum.value)


def from_praxis(nexus_cls: type[_PE], praxis_enum: Enum) -> _PE:
    """Inverse — Praxis enum back to Nexus."""
    return nexus_cls(praxis_enum.value)
