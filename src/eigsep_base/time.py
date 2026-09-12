"""
Time parsing utilities shared across EIGSEP packages.

Extracted verbatim from ``eigsep_data.data`` so that any package can
turn a user-supplied time (string, datetime, or Unix float) into Unix
seconds, and can recover the wallclock stamp embedded in a correlator
filename, without depending on the data-analysis package.
"""

from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np

__all__ = ["to_unix_time", "parse_time_from_name"]


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
