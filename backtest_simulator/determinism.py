"""Deterministic seed derivation keyed on (manifest_hash, path_id, base_seed)."""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedTriple:
    """Three independent RNG seeds for one job."""

    strategy_seed: int
    venue_seed: int
    numpy_seed: int


def derive(manifest_hash: str, path_id: str, base_seed: int) -> SeedTriple:
    """Derive three 63-bit seeds from the job's identity."""
    stem = f'{manifest_hash}|{path_id}|{base_seed}'.encode()
    strategy = int.from_bytes(hashlib.sha256(b's|' + stem).digest()[:8], 'big') >> 1
    venue = int.from_bytes(hashlib.sha256(b'v|' + stem).digest()[:8], 'big') >> 1
    npseed = int.from_bytes(hashlib.sha256(b'n|' + stem).digest()[:8], 'big') >> 1
    return SeedTriple(strategy_seed=strategy, venue_seed=venue, numpy_seed=npseed)


def install(seeds: SeedTriple) -> tuple[random.Random, np.random.Generator]:
    """Build the per-job RNG handles; caller threads them to consumers."""
    return random.Random(seeds.strategy_seed), np.random.default_rng(seeds.numpy_seed)
