"""Determinism: same seed -> byte-identical outputs."""
from __future__ import annotations

from backtest_simulator.determinism import derive, install


def test_derive_is_pure() -> None:
    a = derive('manifest-abc', 'path-0', 42)
    b = derive('manifest-abc', 'path-0', 42)
    assert a == b


def test_derive_differs_on_manifest_change() -> None:
    a = derive('manifest-a', 'path-0', 42)
    b = derive('manifest-b', 'path-0', 42)
    assert a != b


def test_derive_differs_on_path_change() -> None:
    a = derive('manifest-abc', 'path-0', 42)
    b = derive('manifest-abc', 'path-1', 42)
    assert a != b


def test_derive_differs_on_seed_change() -> None:
    a = derive('manifest-abc', 'path-0', 42)
    b = derive('manifest-abc', 'path-0', 43)
    assert a != b


def test_install_produces_repeatable_streams() -> None:
    seeds = derive('manifest-abc', 'path-0', 42)
    rng_a, np_a = install(seeds)
    rng_b, np_b = install(seeds)
    assert [rng_a.random() for _ in range(5)] == [rng_b.random() for _ in range(5)]
    assert (np_a.random(5) == np_b.random(5)).all()


def test_seeds_are_within_63_bit_range() -> None:
    s = derive('manifest-abc', 'path-0', 42)
    for seed in (s.strategy_seed, s.venue_seed, s.numpy_seed):
        assert 0 <= seed < 2 ** 63
