"""Tests for eigsep_base.time."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from eigsep_base.time import (
    parse_time_from_name,
    to_unix_time,
    _parse_time_from_name,
)


# --- to_unix_time ---------------------------------------------------


def test_numeric_passthrough():
    assert to_unix_time(0) == 0.0
    assert to_unix_time(1e9) == 1e9
    assert to_unix_time(1752732000.5) == 1752732000.5
    for value in (np.int64(42), np.float32(42.0), np.float64(42.0)):
        result = to_unix_time(value)
        assert result == 42.0
        assert isinstance(result, float)


def test_naive_datetime_is_utc():
    """A datetime with no tzinfo must be read as UTC, not local time."""
    naive = datetime(2026, 7, 17, 6, 0, 0)
    aware = datetime(2026, 7, 17, 6, 0, 0, tzinfo=timezone.utc)
    assert to_unix_time(naive) == to_unix_time(aware)
    assert to_unix_time(naive) == aware.timestamp()


def test_aware_datetime_respects_offset():
    utc = datetime(2026, 7, 17, 6, 0, 0, tzinfo=timezone.utc)
    plus2 = datetime(
        2026, 7, 17, 8, 0, 0, tzinfo=timezone(timedelta(hours=2))
    )
    assert to_unix_time(plus2) == to_unix_time(utc)


@pytest.mark.parametrize(
    "text",
    [
        "2026-07-17 06:00:00",
        "2026-07-17T06:00:00",
        "2026-07-17T06:00:00Z",
        "2026-07-17T06:00:00+00:00",
    ],
)
def test_equivalent_string_formats(text):
    expected = datetime(
        2026, 7, 17, 6, 0, 0, tzinfo=timezone.utc
    ).timestamp()
    assert to_unix_time(text) == expected


def test_single_digit_month_and_hour():
    """The documented '2026-7-17 6:00:00' shorthand."""
    expected = datetime(
        2026, 7, 17, 6, 0, 0, tzinfo=timezone.utc
    ).timestamp()
    assert to_unix_time("2026-7-17 6:00:00") == expected


def test_minute_and_date_only_formats():
    assert to_unix_time("2026-07-17 06:00") == datetime(
        2026, 7, 17, 6, 0, tzinfo=timezone.utc
    ).timestamp()
    assert to_unix_time("2026-07-17") == datetime(
        2026, 7, 17, tzinfo=timezone.utc
    ).timestamp()


def test_surrounding_whitespace_is_stripped():
    assert to_unix_time("  2026-07-17 06:00:00  ") == to_unix_time(
        "2026-07-17 06:00:00"
    )


@pytest.mark.parametrize(
    "bad", ["not a time", "17/07/2026", "", "2026-13-01 00:00:00"]
)
def test_unparseable_raises_value_error(bad):
    with pytest.raises(ValueError, match="Could not interpret time"):
        to_unix_time(bad)


# --- parse_time_from_name -------------------------------------------


@pytest.mark.parametrize(
    "fname,expected",
    [
        ("corr_20250922_160500.h5", datetime(2025, 9, 22, 16, 5, 0)),
        # UTC marker suffix
        ("corr_20260715_172825Z.h5", datetime(2026, 7, 15, 17, 28, 25)),
        # same-second disambiguating suffix
        ("corr_20260712_235712Z-1.h5", datetime(2026, 7, 12, 23, 57, 12)),
        # full path, not just a basename
        (
            "/data/marjum/corr_20260718_032248Z.h5",
            datetime(2026, 7, 18, 3, 22, 48),
        ),
        # non-corr prefixes use the same stamp convention
        ("ants11_20260715_172825Z.h5", datetime(2026, 7, 15, 17, 28, 25)),
        ("metadata_20260715_172825.h5", datetime(2026, 7, 15, 17, 28, 25)),
    ],
)
def test_parses_deployment_filename_variants(fname, expected):
    assert parse_time_from_name(fname) == expected


def test_returns_naive_datetime():
    """The stamp is UTC wallclock but carries no tzinfo, matching the
    original in eigsep_data.data."""
    assert parse_time_from_name("corr_20250922_160500.h5").tzinfo is None


@pytest.mark.parametrize(
    "bad", ["corr.h5", "corr_2025_1605.h5", "nothing_here.txt"]
)
def test_unparseable_name_raises(bad):
    with pytest.raises(ValueError, match="Could not parse a timestamp"):
        parse_time_from_name(bad)


def test_private_alias_is_the_same_function():
    """eigsep_data imported the underscore name; keep it working."""
    assert _parse_time_from_name is parse_time_from_name


def test_round_trip_through_to_unix_time():
    dt = parse_time_from_name("corr_20260715_172825Z.h5")
    unix = to_unix_time(dt)
    assert datetime.fromtimestamp(unix, tz=timezone.utc).replace(
        tzinfo=None
    ) == dt
