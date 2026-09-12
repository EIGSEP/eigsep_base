"""
Non-JAX coordinate utilities.

Two groups of functions live here:

1. Pure-numpy vector/angle helpers ported from ``eigsep_sim.coord``
   (itself a copy of ``aipy.coord``). Vectors are 3-dimensional with
   ``x, y, z`` along the *first* axis, and angles are in radians.

2. An astropy-based replacement for ``aipy.coord.convert_m``, which
   needed ``pyephem``. :func:`convert_m` returns the same 3x3 frame
   rotation matrices, letting eigsep drop the aipy dependency.

JAX versions of the rotation machinery (``rot_m``, ``xyz2thphi``,
``thphi2xyz``, ``eq2top_m``) live in ``healjax.coord``; this module
deliberately does not duplicate them.
"""

import numpy as np
from astropy.coordinates import (
    BarycentricTrueEcliptic,
    FK5,
    Galactic,
    ICRS,
    SkyCoord,
)
from astropy import units as u
from astropy.time import Time

__all__ = [
    "xyz2thphi",
    "thphi2xyz",
    "eq2radec",
    "radec2eq",
    "latlong2xyz",
    "top2azalt",
    "azalt2top",
    "eq2top_m",
    "top2eq_m",
    "convert_m",
    "convert",
    "eq2ga_m",
    "ga2eq_m",
    "wrap_angle",
    "wrap_2pi",
    "wrap_pi",
    "angle_diff",
]


# ---------------------------------------------------------------------
# Pure-numpy vector <-> angle helpers
# ---------------------------------------------------------------------


def xyz2thphi(xyz):
    """Convert xyz vectors (x,y,z along first axis) into angles theta
    (from z axis), phi (counter-clockwise around z, 0 at x axis)."""
    x, y, z = xyz
    phi = np.arctan2(y, x)
    th = np.arctan2(np.sqrt(x**2 + y**2), z)
    return np.array([th, phi], dtype=np.double)


def thphi2xyz(th_phi):
    """Convert angles theta (from z axis), phi (counter-clockwise around
    z, 0 at x axis) into xyz vectors (x,y,z along first axis)."""
    th, phi = th_phi
    z = np.cos(th)
    r = np.sin(th)
    x, y = r * np.cos(phi), r * np.sin(phi)
    return np.array([x, y, z], dtype=np.double)


def eq2radec(xyz):
    """Convert equatorial xyz vectors (x,y,z along first axis) into
    angles ra (counter-clockwise around z = north, 0 at x axis), dec
    (from equator)."""
    th, phi = xyz2thphi(xyz)
    dec = np.pi / 2 - th
    ra = np.where(phi < 0, phi + 2 * np.pi, phi)
    return np.array([ra, dec], dtype=np.double)


def radec2eq(ra_dec):
    """Convert angles ra (counter-clockwise around z = north, 0 at x
    axis), dec (from equator) into equatorial xyz vectors (x,y,z along
    first axis)."""
    phi, th = ra_dec
    return thphi2xyz((np.pi / 2 - th, phi))


def latlong2xyz(lat_long):
    """Convert angles lat (from equator), long (counter-clockwise around
    z = north, 0 at x axis) into xyz vectors (x,y,z along first
    axis)."""
    lat, long = lat_long
    return radec2eq((long, lat))


def top2azalt(xyz):
    """Convert topocentric xyz vectors (x,y,z along first axis) into
    angles az (clockwise around z = up, 0 at y axis), alt (from
    horizon).

    Note the topocentric basis is ``x = east, y = north, z = up``, so
    azimuth is zero along +y and 90 degrees along +x. aipy's docstring
    for this function said "0 at x axis = north", which contradicts both
    its own arithmetic and the module docstring; the behavior here is
    aipy's, with the description corrected.
    """
    th, phi = xyz2thphi(xyz)
    alt = np.pi / 2 - th
    az = np.pi / 2 - phi
    az = np.where(az < 0, az + 2 * np.pi, az)
    return np.array([az, alt], dtype=np.double)


def azalt2top(az_alt):
    """Convert angles az (clockwise around z = up, 0 at y axis = north),
    alt (from horizon) into topocentric xyz vectors (x,y,z along first
    axis). Inverse of :func:`top2azalt`."""
    az, alt = az_alt
    return thphi2xyz((np.pi / 2 - alt, np.pi / 2 - az))


def eq2top_m(ha, dec):
    """Return the 3x3 matrix converting equatorial coordinates to
    topocentric at the given hour angle (ha) and declination (dec).

    ``ha`` and ``dec`` may be scalars or arrays; arrays are broadcast
    against each other and the result has shape ``(..., 3, 3)``. aipy's
    version only tolerated a varying ``ha`` against a scalar ``dec`` on
    old numpy -- on numpy >= 1.24 the ragged ``np.array`` build raises,
    so the entries are broadcast explicitly here.
    """
    ha, dec = np.broadcast_arrays(
        np.asarray(ha, dtype=np.double), np.asarray(dec, dtype=np.double)
    )
    sin_H, cos_H = np.sin(ha), np.cos(ha)
    sin_d, cos_d = np.sin(dec), np.cos(dec)
    zero = np.zeros_like(ha)
    m = np.array(
        [
            [sin_H, cos_H, zero],
            [-sin_d * cos_H, sin_d * sin_H, cos_d],
            [cos_d * cos_H, -cos_d * sin_H, sin_d],
        ]
    )
    # m is (3, 3, *batch); move the batch axes to the front.
    if m.ndim > 2:
        return np.moveaxis(m, (0, 1), (-2, -1))
    return m


def top2eq_m(ha, dec):
    """Return the 3x3 matrix converting topocentric coordinates to
    equatorial at the given hour angle (ha) and declination (dec)."""
    # np.linalg.inv broadcasts over leading axes, so this covers both
    # the single-matrix and the stacked (ntimes, 3, 3) cases that
    # eq2top_m returns.
    return np.linalg.inv(eq2top_m(ha, dec))


# ---------------------------------------------------------------------
# Frame conversions (astropy replacement for aipy.coord.convert_m)
# ---------------------------------------------------------------------

#: Coordinate-system codes accepted by :func:`convert` and
#: :func:`convert_m`, matching the two-letter aipy/pyephem names.
#: ``'eq'`` is ICRS at the J2000 epoch (aipy's ``ephem.Equatorial``
#: default); pass an explicit epoch to get the FK5 equinox of date.
SYS_NAMES = {"eq": "equatorial", "ec": "ecliptic", "ga": "galactic"}


def _frame(sys, epoch=None):
    """Return the astropy frame for a two-letter system code.

    ``epoch`` is any :class:`astropy.time.Time`-compatible value (e.g.
    ``"J2000"``, ``"B1950"``, a ``Time`` object). ``None`` means J2000,
    for which the equatorial frame is ICRS -- the same convention aipy
    used and within milliarcseconds of FK5 J2000.
    """
    key = str(sys)[:2].lower()
    if key not in SYS_NAMES:
        raise ValueError(
            f"Unknown coordinate system {sys!r}; expected one of "
            f"{sorted(SYS_NAMES)}"
        )
    if key == "ga":
        # Galactic coordinates are defined by a fixed rotation from
        # ICRS and carry no equinox, so the epoch is irrelevant.
        return Galactic()
    if epoch is None:
        if key == "eq":
            return ICRS()
        return BarycentricTrueEcliptic(equinox=Time("J2000"))
    equinox = epoch if isinstance(epoch, Time) else Time(epoch)
    if key == "eq":
        return FK5(equinox=equinox)
    return BarycentricTrueEcliptic(equinox=equinox)


def convert(crd, isys, osys, iepoch=None, oepoch=None):
    """Convert ``crd`` from coordinate system ``isys`` to ``osys``,
    including epoch precession.

    Valid coordinate systems are ``'ec'`` (ecliptic), ``'eq'``
    (equatorial), and ``'ga'`` (galactic). Epochs may be date strings
    (``"J2000"``, ``"B1950"``) or :class:`astropy.time.Time` objects;
    ``None`` means J2000.

    Parameters
    ----------
    crd : array_like
        Either a length-2 ``(lon, lat)`` angle pair in radians or a
        length-3 ``(x, y, z)`` unit vector, matching aipy's behavior.
    isys, osys : str
        Two-letter input and output system codes.
    iepoch, oepoch : str, Time, or None
        Input and output epochs.

    Returns
    -------
    ndarray
        Length-2 ``(lon, lat)`` angle pair in radians in ``osys``.
    """
    crd = np.asarray(crd, dtype=np.double)
    if crd.shape[0] == 3:
        crd = eq2radec(crd)
    lon, lat = crd
    sky = SkyCoord(
        lon * u.rad,
        lat * u.rad,
        frame=_frame(isys, iepoch),
    )
    out = sky.transform_to(_frame(osys, oepoch))
    lon_out, lat_out = out.spherical.lon.rad, out.spherical.lat.rad
    return np.array([lon_out, lat_out], dtype=np.double)


def convert_m(isys, osys, iepoch=None, oepoch=None):
    """Return the 3x3 matrix for a coordinate-system/precession
    transformation (see :func:`convert`).

    This is the astropy-backed replacement for
    ``aipy.coord.convert_m``, constructed the same way (column ``i`` is
    the image of input basis vector ``e_i``), so existing call sites
    such as ``convert_m('eq', 'ga')`` in ``eigsep_data.sim`` keep their
    argument order.

    The matrix maps component vectors *out* of ``isys`` and *into*
    ``osys``::

        v_osys = convert_m(isys, osys) @ v_isys

    So to rotate a vector known in equatorial coordinates into
    galactic coordinates, use ``convert_m('eq', 'ga')``. The
    ``'eq' -> 'ga'`` result is verified in the test suite against the
    IAU-defined north galactic pole and galactic center directions.

    .. note::
       aipy's own docstring tells the caller to reverse ``isys``/
       ``osys`` "to obtain a transformation matrix to dot with". That
       note contradicts the relation above and the naming at aipy's
       eigsep call sites; the behavior documented here is what this
       implementation is tested against.

    Parameters
    ----------
    isys, osys : str
        Two-letter input and output system codes.
    iepoch, oepoch : str, Time, or None
        Input and output epochs. ``None`` means J2000.

    Returns
    -------
    ndarray
        3x3 rotation matrix, float64.
    """
    m = np.eye(3, dtype=np.double)
    for i in range(3):
        c = convert(m[:, i], isys, osys, iepoch=iepoch, oepoch=oepoch)
        m[:, i] = radec2eq(c)
    return m


def eq2ga_m(epoch=None):
    """3x3 matrix rotating equatorial component vectors to galactic.

    Shorthand for ``convert_m('eq', 'ga', iepoch=epoch)`` -- the drop-in
    replacement for the ``aipy.coord.convert_m('eq', 'ga')`` call in
    ``eigsep_data.sim``.
    """
    return convert_m("eq", "ga", iepoch=epoch)


def ga2eq_m(epoch=None):
    """3x3 matrix rotating galactic component vectors to equatorial."""
    return convert_m("ga", "eq", oepoch=epoch)


# ---------------------------------------------------------------------
# Angle wrapping
# ---------------------------------------------------------------------


def wrap_angle(angles, center=0.0, period=2 * np.pi):
    """Wrap angles into the half-open interval centered on ``center``.

    Parameters
    ----------
    angles : array_like
        Angles to wrap, in the same units as ``period``.
    center : float
        Center of the output interval. The result lies in
        ``[center - period/2, center + period/2)``.
    period : float
        Full period. Use ``2*np.pi`` for radians (default) or ``360``
        for degrees.

    Returns
    -------
    ndarray
        Wrapped angles, float64, same shape as ``angles``.
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    angles = np.asarray(angles, dtype=np.double)
    lo = center - period / 2
    return (angles - lo) % period + lo


def wrap_2pi(angles):
    """Wrap angles in radians into ``[0, 2*pi)``."""
    return wrap_angle(angles, center=np.pi, period=2 * np.pi)


def wrap_pi(angles):
    """Wrap angles in radians into ``[-pi, pi)``."""
    return wrap_angle(angles, center=0.0, period=2 * np.pi)


def angle_diff(a, b, period=2 * np.pi):
    """Smallest signed difference ``a - b``, wrapped to half a period.

    Result lies in ``[-period/2, period/2)``, so differences across the
    wrap point (e.g. 359 deg vs 1 deg) come out small.
    """
    diff = np.asarray(a, dtype=np.double) - np.asarray(b, dtype=np.double)
    return wrap_angle(diff, center=0.0, period=period)
