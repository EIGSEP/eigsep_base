"""Tests for eigsep_base.coord."""

import numpy as np
import pytest

from eigsep_base import coord


# Reference directions from the IAU definition of galactic coordinates,
# in ICRS degrees. These are what aipy's ephem-backed convert_m encoded;
# reproducing them is how we know the astropy replacement is correct.
NGP_RA_DEG, NGP_DEC_DEG = 192.85948, 27.12825
GAL_CENTER_RA_DEG, GAL_CENTER_DEC_DEG = 266.40510, -28.93617


# --- vector/angle helpers -------------------------------------------


def test_thphi2xyz_known_directions():
    # +z axis
    assert np.allclose(coord.thphi2xyz((0.0, 0.0)), [0, 0, 1])
    # +x axis
    assert np.allclose(coord.thphi2xyz((np.pi / 2, 0.0)), [1, 0, 0])
    # +y axis
    assert np.allclose(coord.thphi2xyz((np.pi / 2, np.pi / 2)), [0, 1, 0])
    # -z axis
    assert np.allclose(coord.thphi2xyz((np.pi, 0.0)), [0, 0, -1])


def test_xyz2thphi_inverts_thphi2xyz():
    rng = np.random.default_rng(0)
    th = rng.uniform(0.01, np.pi - 0.01, 50)
    phi = rng.uniform(-np.pi, np.pi, 50)
    xyz = coord.thphi2xyz((th, phi))
    th2, phi2 = coord.xyz2thphi(xyz)
    assert np.allclose(th, th2)
    assert np.allclose(np.cos(phi), np.cos(phi2))
    assert np.allclose(np.sin(phi), np.sin(phi2))


def test_thphi2xyz_gives_unit_vectors():
    rng = np.random.default_rng(1)
    th = rng.uniform(0, np.pi, 100)
    phi = rng.uniform(-np.pi, np.pi, 100)
    xyz = coord.thphi2xyz((th, phi))
    assert np.allclose(np.sum(xyz**2, axis=0), 1.0)


def test_eq2radec_ra_is_in_zero_to_2pi():
    """Unlike phi, right ascension must not come back negative."""
    xyz = coord.thphi2xyz((np.pi / 2, -np.pi / 4))  # phi = -45 deg
    ra, dec = coord.eq2radec(xyz)
    assert ra == pytest.approx(7 * np.pi / 4)
    assert dec == pytest.approx(0.0, abs=1e-12)


def test_radec2eq_inverts_eq2radec():
    rng = np.random.default_rng(2)
    ra = rng.uniform(0, 2 * np.pi, 50)
    dec = rng.uniform(-np.pi / 2 + 0.01, np.pi / 2 - 0.01, 50)
    xyz = coord.radec2eq((ra, dec))
    ra2, dec2 = coord.eq2radec(xyz)
    assert np.allclose(ra, ra2)
    assert np.allclose(dec, dec2)


def test_radec2eq_north_pole():
    assert np.allclose(coord.radec2eq((0.0, np.pi / 2)), [0, 0, 1])


def test_latlong2xyz_matches_radec2eq_with_swapped_args():
    lat, long = np.deg2rad(35.0), np.deg2rad(-120.0)
    assert np.allclose(
        coord.latlong2xyz((lat, long)), coord.radec2eq((long, lat))
    )


def test_latlong2xyz_equator_prime_meridian():
    assert np.allclose(coord.latlong2xyz((0.0, 0.0)), [1, 0, 0])


def test_top2azalt_uses_x_east_y_north():
    """Topocentric basis is x=east, y=north, z=up, so azimuth is 0 at
    +y and 90 deg at +x. This pins down the convention aipy's own
    docstring got backwards."""
    az, alt = coord.top2azalt(np.array([0.0, 1.0, 0.0]))  # north
    assert az == pytest.approx(0.0)
    assert alt == pytest.approx(0.0, abs=1e-12)
    az, alt = coord.top2azalt(np.array([1.0, 0.0, 0.0]))  # east
    assert az == pytest.approx(np.pi / 2)
    assert alt == pytest.approx(0.0, abs=1e-12)
    az, _ = coord.top2azalt(np.array([-1.0, 0.0, 0.0]))  # west
    assert az == pytest.approx(3 * np.pi / 2)


def test_azalt2top_zenith():
    assert np.allclose(coord.azalt2top((0.0, np.pi / 2)), [0, 0, 1])
    # azimuth 0 = north = +y
    assert np.allclose(coord.azalt2top((0.0, 0.0)), [0, 1, 0], atol=1e-12)


def test_azalt2top_inverts_top2azalt():
    rng = np.random.default_rng(3)
    az = rng.uniform(0, 2 * np.pi, 50)
    alt = rng.uniform(-np.pi / 2 + 0.01, np.pi / 2 - 0.01, 50)
    xyz = coord.azalt2top((az, alt))
    az2, alt2 = coord.top2azalt(xyz)
    assert np.allclose(az, az2)
    assert np.allclose(alt, alt2)


# --- rotation matrices ----------------------------------------------


def test_eq2top_m_is_orthonormal():
    for ha in np.linspace(0, 2 * np.pi, 7):
        for dec in np.linspace(-1.4, 1.4, 5):
            m = coord.eq2top_m(ha, dec)
            assert np.allclose(m @ m.T, np.eye(3), atol=1e-12)


def test_eq2top_m_sends_source_at_zenith_to_up():
    """A source at ha=0 and dec=lat is straight overhead: +z."""
    dec = np.deg2rad(39.25)
    m = coord.eq2top_m(0.0, dec)
    src = coord.radec2eq((0.0, dec))
    assert np.allclose(m @ src, [0, 0, 1], atol=1e-12)


def test_eq2top_m_vectorizes_over_hour_angle():
    ha = np.linspace(0, 1, 4)
    m = coord.eq2top_m(ha, 0.5)
    assert m.shape == (4, 3, 3)
    for i, h in enumerate(ha):
        assert np.allclose(m[i], coord.eq2top_m(h, 0.5))


def test_top2eq_m_inverts_eq2top_m():
    m = coord.eq2top_m(0.7, 0.3)
    assert np.allclose(coord.top2eq_m(0.7, 0.3) @ m, np.eye(3))


def test_top2eq_m_vectorized():
    ha = np.linspace(0, 1, 4)
    inv = coord.top2eq_m(ha, 0.3)
    fwd = coord.eq2top_m(ha, 0.3)
    assert inv.shape == (4, 3, 3)
    assert np.allclose(inv @ fwd, np.broadcast_to(np.eye(3), (4, 3, 3)))


# --- frame conversion -----------------------------------------------


def test_convert_m_eq_to_ga_is_a_rotation():
    m = coord.convert_m("eq", "ga")
    assert m.shape == (3, 3)
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-10)
    assert np.linalg.det(m) == pytest.approx(1.0)


def test_convert_m_maps_ngp_to_z():
    """convert_m('eq','ga') @ v_eq == v_ga, checked on the north
    galactic pole -- the defining direction of the galactic frame."""
    m = coord.convert_m("eq", "ga")
    ngp_eq = coord.radec2eq(
        (np.deg2rad(NGP_RA_DEG), np.deg2rad(NGP_DEC_DEG))
    )
    assert np.allclose(m @ ngp_eq, [0, 0, 1], atol=1e-5)


def test_convert_m_maps_galactic_center_to_x():
    m = coord.convert_m("eq", "ga")
    gc_eq = coord.radec2eq(
        (
            np.deg2rad(GAL_CENTER_RA_DEG),
            np.deg2rad(GAL_CENTER_DEC_DEG),
        )
    )
    assert np.allclose(m @ gc_eq, [1, 0, 0], atol=1e-5)


def test_convert_m_round_trip():
    fwd = coord.convert_m("eq", "ga")
    back = coord.convert_m("ga", "eq")
    assert np.allclose(back @ fwd, np.eye(3), atol=1e-10)


def test_convert_m_matches_known_icrs_to_galactic_matrix():
    """Compare against the published ICRS->galactic rotation."""
    expected = np.array(
        [
            [-0.05487566, -0.87343705, -0.48383507],
            [+0.49410944, -0.44482972, +0.74698218],
            [-0.86766614, -0.19807634, +0.45598381],
        ]
    )
    assert np.allclose(coord.convert_m("eq", "ga"), expected, atol=1e-6)


def test_eq2ga_m_and_ga2eq_m_shorthands():
    assert np.allclose(coord.eq2ga_m(), coord.convert_m("eq", "ga"))
    assert np.allclose(coord.ga2eq_m(), coord.convert_m("ga", "eq"))
    assert np.allclose(coord.eq2ga_m() @ coord.ga2eq_m(), np.eye(3))


def test_convert_m_ecliptic_tilt_is_the_obliquity():
    """eq->ec tilts the pole by the obliquity, ~23.44 deg.

    The rotation axis is the equinox, within ~14 arcsec of the ICRS
    +x axis -- astropy's ecliptic frame uses the *true* equinox
    (including nutation) while 'eq' with no epoch is ICRS.
    """
    m = coord.convert_m("eq", "ec")
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-10)
    assert np.allclose(m[:, 0], [1, 0, 0], atol=1e-4)
    # Angle between the equatorial pole and the ecliptic pole.
    eq_pole_in_ec = m @ np.array([0.0, 0.0, 1.0])
    obliquity = np.rad2deg(np.arccos(eq_pole_in_ec[2]))
    assert obliquity == pytest.approx(23.4393, abs=1e-3)


def test_convert_accepts_angles_and_vectors():
    lon, lat = np.deg2rad(NGP_RA_DEG), np.deg2rad(NGP_DEC_DEG)
    from_angles = coord.convert((lon, lat), "eq", "ga")
    from_vector = coord.convert(coord.radec2eq((lon, lat)), "eq", "ga")
    assert np.allclose(from_angles, from_vector, atol=1e-8)
    # Galactic latitude of the NGP is +90 deg.
    assert np.rad2deg(from_angles[1]) == pytest.approx(90.0, abs=1e-4)


def test_convert_identity_transform():
    lon, lat = 1.0, 0.5
    out = coord.convert((lon, lat), "eq", "eq")
    assert out[0] == pytest.approx(lon, abs=1e-10)
    assert out[1] == pytest.approx(lat, abs=1e-10)


def test_precession_between_b1950_and_j2000_is_about_half_a_degree():
    """FK5 B1950 -> J2000 shifts coordinates by ~0.64 deg over 50 yr."""
    m = coord.convert_m("eq", "eq", iepoch="B1950", oepoch="J2000")
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-10)
    angle = np.rad2deg(np.arccos((np.trace(m) - 1) / 2))
    assert 0.3 < angle < 1.0


def test_galactic_frame_ignores_epoch():
    """Galactic coordinates carry no equinox."""
    assert np.allclose(
        coord.convert_m("eq", "ga", oepoch="B1950"),
        coord.convert_m("eq", "ga"),
    )


@pytest.mark.parametrize("bad", ["xx", "", "topocentric"])
def test_unknown_system_raises(bad):
    with pytest.raises(ValueError, match="Unknown coordinate system"):
        coord.convert_m(bad, "ga")


@pytest.mark.parametrize(
    "spelling", ["ga", "GA", "Ga", "gal", "galactic", "Galactic"]
)
def test_system_codes_match_on_the_first_two_letters(spelling):
    """aipy keyed on ``sys[:2].lower()``, so spelled-out names work."""
    assert np.allclose(
        coord.convert_m("eq", spelling), coord.convert_m("eq", "ga")
    )


# --- angle wrapping -------------------------------------------------


def test_wrap_pi_basic():
    assert coord.wrap_pi(0.0) == pytest.approx(0.0)
    assert coord.wrap_pi(3 * np.pi) == pytest.approx(-np.pi)
    assert coord.wrap_pi(-3 * np.pi) == pytest.approx(-np.pi)
    assert coord.wrap_pi(np.pi / 2) == pytest.approx(np.pi / 2)


def test_wrap_pi_interval_is_half_open_at_plus_pi():
    """[-pi, pi): exactly +pi wraps to -pi."""
    assert coord.wrap_pi(np.pi) == pytest.approx(-np.pi)


def test_wrap_2pi_basic():
    assert coord.wrap_2pi(0.0) == pytest.approx(0.0)
    assert coord.wrap_2pi(-0.1) == pytest.approx(2 * np.pi - 0.1)
    assert coord.wrap_2pi(2 * np.pi) == pytest.approx(0.0, abs=1e-12)


def test_wrap_preserves_direction():
    rng = np.random.default_rng(4)
    angles = rng.uniform(-30, 30, 200)
    for wrapped in (coord.wrap_pi(angles), coord.wrap_2pi(angles)):
        assert np.allclose(np.cos(wrapped), np.cos(angles))
        assert np.allclose(np.sin(wrapped), np.sin(angles))


def test_wrap_angle_in_degrees():
    assert coord.wrap_angle(370.0, center=180.0, period=360.0) == (
        pytest.approx(10.0)
    )
    assert coord.wrap_angle(-10.0, center=180.0, period=360.0) == (
        pytest.approx(350.0)
    )
    assert coord.wrap_angle(350.0, center=0.0, period=360.0) == (
        pytest.approx(-10.0)
    )


def test_wrap_angle_preserves_shape_and_dtype():
    angles = np.arange(12, dtype=np.int64).reshape(3, 4)
    out = coord.wrap_angle(angles)
    assert out.shape == (3, 4)
    assert out.dtype == np.float64


def test_wrap_angle_rejects_bad_period():
    with pytest.raises(ValueError, match="period must be positive"):
        coord.wrap_angle(1.0, period=0.0)


def test_angle_diff_across_the_wrap_point():
    """359 deg and 1 deg are 2 deg apart, not 358."""
    d = coord.angle_diff(
        np.deg2rad(1.0), np.deg2rad(359.0), period=2 * np.pi
    )
    assert np.rad2deg(d) == pytest.approx(2.0)


def test_angle_diff_sign():
    assert coord.angle_diff(0.5, 0.2) == pytest.approx(0.3)
    assert coord.angle_diff(0.2, 0.5) == pytest.approx(-0.3)


def test_angle_diff_in_degrees():
    assert coord.angle_diff(10.0, 350.0, period=360.0) == (
        pytest.approx(20.0)
    )
