"""bts run promotes naive --window-start / --window-end to UTC.

`datetime.fromisoformat` returns a naive datetime when the input is
a date-only string (e.g. `"2026-04-07"`). `ManifestBuilder.build`
then rejects the derived `force_flatten_after = window_end -
kline_size` for being effectively-naive, with `ValueError:
StrategyParamsSpec.force_flatten_after must be tz-aware`. `bts
sweep` already constructs windows via
`datetime.combine(day, hours, tzinfo=UTC)`; `_parse_window_arg`
brings `bts run` into parity.
"""
from __future__ import annotations

from datetime import UTC

from backtest_simulator.cli.commands.run import _parse_window_arg


def test_parse_window_arg_promotes_date_only_to_utc() -> None:
    """A bare date string yields a UTC-aware datetime at midnight."""
    parsed = _parse_window_arg('2026-04-07')
    assert parsed.tzinfo is UTC
    assert (parsed.year, parsed.month, parsed.day) == (2026, 4, 7)
    assert (parsed.hour, parsed.minute, parsed.second) == (0, 0, 0)


def test_parse_window_arg_promotes_naive_iso_to_utc() -> None:
    """A naive ISO datetime (no offset suffix) is promoted to UTC."""
    parsed = _parse_window_arg('2026-04-07T13:30:00')
    assert parsed.tzinfo is UTC
    assert (parsed.hour, parsed.minute) == (13, 30)


def test_parse_window_arg_preserves_explicit_offset() -> None:
    """An ISO datetime with an explicit +00:00 offset is left untouched.

    Double-tagging would wrap a UTC tzinfo around an already-UTC
    datetime; comparing such a value against a fresh
    `datetime.now(tz=UTC)` still works, but the equality check
    `tzinfo is UTC` fails on a `datetime.timezone.utc` instance built
    from an offset string. Leaving the value alone keeps the
    parser's "I parse what I'm given" contract honest.
    """
    parsed = _parse_window_arg('2026-04-07T13:30:00+00:00')
    assert parsed.utcoffset() is not None
    assert parsed.utcoffset().total_seconds() == 0
    assert (parsed.hour, parsed.minute) == (13, 30)


def test_parse_window_arg_preserves_non_utc_offset() -> None:
    """A non-UTC offset is preserved (no silent normalisation)."""
    parsed = _parse_window_arg('2026-04-07T13:30:00+02:00')
    offset = parsed.utcoffset()
    assert offset is not None
    assert offset.total_seconds() == 7200, (
        'a +02:00 input must keep its offset; promoting non-UTC '
        'offsets to UTC would silently shift the operator\'s '
        'declared window'
    )
