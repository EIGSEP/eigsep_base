"""
Time handling shared across EIGSEP packages: epoch in, rendering out.

Every timestamp a corr file stores is Unix epoch, and this package
never shifts it. Timezones enter in exactly two places: rendering an
epoch for a human (:func:`format_time`), and parsing a *filename*
stamp, whose zone depends on the deployment
(:func:`parse_filename_time`).

Originally two separate copies -- ``to_unix_time``/``parse_time_from_name``
extracted here from ``eigsep_data.data``, and a parallel
``eigsep_data.clock`` grown independently on the metadata-index branch
with the zone-aware filename handling. Consolidated here 2026-09-17
(Aaron's decision) so there is one implementation; ``eigsep_data.clock``
is now a thin re-export of this module.
"""

from datetime import datetime, timezone
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import numpy as np

__all__ = [
    "to_unix_time",
    "parse_time_from_name",
    "parse_filename_time",
    "filename_unix",
    "format_time",
    "FILENAME_TZ",
    "LEGACY_FILENAME_TZ",
]

#: Timezone the *filename* of each deployment was stamped in. Before
#: eigsep_observing commit c4ef1ee (2026-05-15, "stamp auto-generated
#: h5 filenames in UTC with Z suffix") the writer used a naive
#: datetime.now(), so deployment 1-4 filenames are Pacific wall clock;
#: from deployment 5 they are UTC and carry the Z suffix. Kept as
#: documentation -- parse_filename_time keys off the suffix itself.
FILENAME_TZ = {
    "deployment4": "America/Los_Angeles",
    "deployment5": "UTC",
}

#: Zone assumed for a filename stamp that has no Z suffix -- the
#: deployment 1-4 convention, so it is that table's entry rather than
#: a second copy of the string.
LEGACY_FILENAME_TZ = FILENAME_TZ["deployment4"]

_STAMP = re.compile(r"(\d{8})_(\d{6})(Z?)")


def to_unix_time(value):
    """
    Convert a datetime string, datetime object, or Unix timestamp to Unix
    seconds (float).

    Strings without a timezone are interpreted as UTC.

    Accepted formats:
        "2026-07-17 06:00:00"
        "2026-7-17 6:00:00"
        "2026-07-17T06:00:00Z"
    """
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(
                    f"Could not interpret time {value!r}. "
                    "Use a format such as '2026-07-17 06:00:00'."
                )

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def parse_time_from_name(fname: str) -> datetime:
    """
    Parse datetime from a correlator filename.

    Handles the naming variants seen across deployments, e.g.
    'corr_20250922_160500.h5', 'corr_20260715_172825Z.h5' (UTC marker)
    and 'corr_20260712_235712Z-1.h5' (disambiguating suffix for files
    closed within the same second).

    Note this is the file *close* time, which lags the integrations
    inside it -- by up to ~17 min on the Marjum Pass 2026-07 data, and
    by far more on the ~10% of files written before the clock synced.
    Use header["times"] whenever the actual integration time matters.
    """
    stem = Path(fname).stem
    match = re.search(r"(\d{8})_(\d{6})", stem)
    if match is None:
        raise ValueError(f"Could not parse a timestamp from {fname!r}.")
    return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")


# The original lived in eigsep_data.data as a private helper. Keep the
# underscore name importable so existing call sites keep working.
_parse_time_from_name = parse_time_from_name


def format_time(t, tz="UTC", fmt="%Y-%m-%d %H:%M:%S"):
    """
    Render Unix epoch seconds in a named timezone.

    Deployment timestamps are epoch and always correct; only their
    *display* is zone-dependent. Use ``tz="America/Denver"`` to match
    what watches read in the field, which is what the old
    ``pacific_to_mountain`` flag was really for -- it shifted the data
    to fake a rendering.

    Parameters
    ----------
    t : float or array_like
        Unix seconds.
    tz : str
        IANA zone name.
    fmt : str
        ``strftime`` format.

    Returns
    -------
    str or list of str
    """
    zone = ZoneInfo(tz)
    values = np.atleast_1d(np.asarray(t, dtype=float))
    out = [
        datetime.fromtimestamp(float(v), tz=zone).strftime(fmt) for v in values
    ]
    return out[0] if np.ndim(t) == 0 else out


def parse_filename_time(fname, tz=None):
    """
    Parse a timezone-aware datetime from a correlator filename.

    A ``Z`` suffix on the stamp (``corr_20260715_172825Z.h5``) means
    UTC and wins over *tz*. Without the suffix the stamp is local wall
    clock: *tz* if given, else ``America/Los_Angeles``, the convention
    for deployments 1-4 (see :data:`FILENAME_TZ`). Handles the
    ``-1`` disambiguating suffix for files closed within one second.

    This is the zone-aware counterpart to
    :func:`parse_time_from_name`, which returns the same stamp as a
    naive datetime and is kept for the call sites that want that.

    Note this is the file *close* time, which lags the integrations
    inside it -- by up to ~16 min on deployment-5 data when the writer
    backlogs. It is, however, always stamped from a correct clock,
    which header ``times`` are not on ~12 % of deployment-5 files.

    Parameters
    ----------
    fname : str or Path
    tz : str or None
        IANA zone for a stamp with no ``Z`` suffix.

    Returns
    -------
    datetime
        Timezone-aware.
    """
    match = _STAMP.search(Path(fname).stem)
    if match is None:
        raise ValueError(f"Could not parse a timestamp from {fname!r}.")
    naive = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
    if match.group(3):
        return naive.replace(tzinfo=timezone.utc)
    return naive.replace(tzinfo=ZoneInfo(tz or LEGACY_FILENAME_TZ))


def filename_unix(fname, tz=None):
    """Unix seconds from a corr filename, or NaN when it has no stamp."""
    try:
        return parse_filename_time(fname, tz=tz).timestamp()
    except ValueError:
        return np.nan
