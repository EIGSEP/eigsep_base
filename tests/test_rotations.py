import numpy as np
import pytest

from eigsep_base.rotations import (
    body_to_enu,
    enu_to_body,
    mount_rotation,
    rx,
    rz,
    spherical_basis,
    vector_to_spherical,
)

PSI_MARJUM = 142.164  # Marjum 2026-07 highline, deg ccw from East


def compass_bearing(v):
    return np.degrees(np.arctan2(v[..., 0], v[..., 1])) % 360


def test_el_zero_points_boresight_to_zenith_for_any_roll():
    R = mount_rotation(np.linspace(-180, 180, 13), 0.0, PSI_MARJUM)
    np.testing.assert_allclose(
        body_to_enu([0, 0, 1], R), np.tile([0, 0, 1.0], (13, 1)), atol=1e-12
    )


def test_el_90_points_to_the_marjum_sweep_plane_and_180_to_nadir():
    b90 = mount_rotation(0, 90, PSI_MARJUM) @ [0, 0, 1]
    assert abs(b90[2]) < 1e-12
    assert compass_bearing(b90) == pytest.approx(37.836, abs=1e-9)
    np.testing.assert_allclose(
        mount_rotation(0, 180, PSI_MARJUM) @ [0, 0, 1], [0, 0, -1], atol=1e-12
    )


def test_roll_never_moves_the_boresight():
    az = np.linspace(-180, 180, 25)
    for el in (30.0, 90.0, 150.0):
        b = body_to_enu([0, 0, 1], mount_rotation(az, el, PSI_MARJUM))
        np.testing.assert_allclose(
            b, np.broadcast_to(b[0], b.shape), atol=1e-12
        )


def test_differs_from_alt_az():
    # Alt-az (tilt, then turn about the vertical) swings the tilted boresight around the compass.
    az = np.array([0.0, 90.0, 180.0])
    alt_az = body_to_enu([0, 0, 1], rz(az) @ rx(45.0))
    assert np.ptp(compass_bearing(alt_az)) > 90
    # The same two matrices in the roll order agree only when the boresight is vertical.
    np.testing.assert_allclose(
        rz(33.0) @ rx(0.0), rx(0.0) @ rz(33.0), atol=1e-12
    )
    assert not np.allclose(rz(33.0) @ rx(45.0), rx(45.0) @ rz(33.0))


def test_arm_lies_along_the_axle_at_zero_roll_and_roll_is_right_handed():
    arm = mount_rotation(0, 0, PSI_MARJUM) @ [1, 0, 0]
    assert np.degrees(np.arctan2(arm[1], arm[0])) % 360 == pytest.approx(
        PSI_MARJUM
    )
    np.testing.assert_allclose(
        mount_rotation(0, 57, PSI_MARJUM) @ [1, 0, 0], arm, atol=1e-12
    )
    np.testing.assert_allclose(
        mount_rotation(90, 0, 0.0) @ [1, 0, 0], [0, 1, 0], atol=1e-12
    )


def test_rotations_are_proper_orthonormal_and_broadcast():
    rng = np.random.default_rng(1)
    az, el = rng.uniform(-180, 180, (4, 5)), rng.uniform(-180, 180, (4, 5))
    R = mount_rotation(az, el, PSI_MARJUM)
    assert R.shape == (4, 5, 3, 3)
    np.testing.assert_allclose(
        R @ np.swapaxes(R, -1, -2),
        np.broadcast_to(np.eye(3), R.shape),
        atol=1e-12,
    )
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)
    v = rng.normal(size=(4, 5, 3))
    np.testing.assert_allclose(
        enu_to_body(body_to_enu(v, R), R), v, atol=1e-12
    )


def test_spherical_helpers_round_trip():
    rng = np.random.default_rng(2)
    v = rng.normal(size=(50, 3))
    th, ph = vector_to_spherical(v)
    r, thhat, phhat = spherical_basis(th, ph)
    np.testing.assert_allclose(
        r, v / np.linalg.norm(v, axis=1, keepdims=True), atol=1e-12
    )
    np.testing.assert_allclose(np.sum(r * thhat, 1), 0, atol=1e-12)
    np.testing.assert_allclose(np.sum(thhat * phhat, 1), 0, atol=1e-12)
