"""Tests for eigsep_base.time."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from eigsep_base.time import (
    _parse_time_from_name,
    parse_time_from_name,
    to_unix_time,
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
    plus2 = datetime(2026, 7, 17, 8, 0, 0, tzinfo=timezone(timedelta(hours=2)))
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
    expected = datetime(2026, 7, 17, 6, 0, 0, tzinfo=timezone.utc).timestamp()
    assert to_unix_time(text) == expected


def test_single_digit_month_and_hour():
    """The documented '2026-7-17 6:00:00' shorthand."""
    expected = datetime(2026, 7, 17, 6, 0, 0, tzinfo=timezone.utc).timestamp()
    assert to_unix_time("2026-7-17 6:00:00") == expected


def test_minute_and_date_only_formats():
    assert (
        to_unix_time("2026-07-17 06:00")
        == datetime(2026, 7, 17, 6, 0, tzinfo=timezone.utc).timestamp()
    )
    assert (
        to_unix_time("2026-07-17")
        == datetime(2026, 7, 17, tzinfo=timezone.utc).timestamp()
    )


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
    assert (
        datetime.fromtimestamp(unix, tz=timezone.utc).replace(tzinfo=None)
        == dt
    )


# --- format_time / parse_filename_time / filename_unix ---------------
# Ported verbatim from eigsep_data/tests/test_clock.py when
# eigsep_data.clock's implementation moved here (2026-09-17), so the
# behaviour the metadata index relies on is pinned at its new home.

from eigsep_base.time import (
    filename_unix,
    format_time,
    parse_filename_time,
)


class TestFormatTime:
    def test_renders_utc_by_default(self):
        # 1784321280.0 == 2026-07-17 20:48:00Z (verified, not assumed).
        assert format_time(1784321280.0) == "2026-07-17 20:48:00"

    def test_renders_mountain_for_field_notes(self):
        # The legacy +3600 existed so timestamps matched watches in
        # Utah. That is a rendering job, not a correction to the data.
        # MDT is UTC-6, so 20:48Z is 14:48 local.
        assert (
            format_time(1784321280.0, tz="America/Denver")
            == "2026-07-17 14:48:00"
        )

    def test_accepts_an_array(self):
        out = format_time(np.array([1784321280.0, 1784321340.0]))
        assert out == ["2026-07-17 20:48:00", "2026-07-17 20:49:00"]


class TestParseFilenameTime:
    def test_z_suffix_is_utc(self):
        t = parse_filename_time("corr_20260717_204800Z.h5")
        assert t.tzinfo is not None
        assert t.timestamp() == 1784321280.0

    def test_z_suffix_wins_over_an_explicit_tz(self):
        # The suffix is the producer's own statement of the zone; a
        # caller's guess must not override it.
        t = parse_filename_time(
            "corr_20260717_204800Z.h5", tz="America/Los_Angeles"
        )
        assert t.timestamp() == 1784321280.0

    def test_no_suffix_defaults_to_pacific(self):
        # Before eigsep_observing c4ef1ee the writer used a naive
        # datetime.now(), so deployment 1-4 filenames are Pacific wall
        # clock. September is PDT, UTC-7.
        t = parse_filename_time("corr_20250922_160500.h5")
        assert t.utcoffset().total_seconds() == -7 * 3600
        assert (t.hour, t.minute) == (16, 5)

    def test_explicit_tz_is_honoured_without_suffix(self):
        t = parse_filename_time("corr_20250922_160500.h5", tz="America/Denver")
        assert t.utcoffset().total_seconds() == -6 * 3600

    def test_disambiguating_suffix_still_parses(self):
        t = parse_filename_time("corr_20260712_235712Z-1.h5")
        assert t.tzinfo == timezone.utc

    def test_unparseable_name_raises(self):
        with pytest.raises(ValueError):
            parse_filename_time("not_a_corr_file.h5")

    def test_naive_counterpart_agrees_on_the_stamp(self):
        # parse_time_from_name is the same stamp without a zone; the
        # two must never disagree about the wall-clock digits.
        naive = parse_time_from_name("corr_20260717_204800Z.h5")
        aware = parse_filename_time("corr_20260717_204800Z.h5")
        assert naive.replace(tzinfo=timezone.utc) == aware


class TestFilenameUnix:
    def test_matches_parse(self):
        assert filename_unix("corr_20260717_204800Z.h5") == 1784321280.0

    def test_nan_on_garbage(self):
        # The metadata index calls this on every file it globs; a stray
        # file must yield NaN, not an exception.
        assert np.isnan(filename_unix("notes.txt"))
