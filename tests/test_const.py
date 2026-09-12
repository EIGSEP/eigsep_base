"""Tests for eigsep_base.const."""

import sys

import numpy as np
import pytest

from eigsep_base import const


def test_no_jax_dependency():
    """const must not drag JAX into the process."""
    assert "jax" not in sys.modules


def test_fundamental_constants_si():
    # Spot-check against CODATA values in SI units.
    assert const.c == pytest.approx(2.99792458e8)
    assert const.k_B == pytest.approx(1.380649e-23)
    assert const.h == pytest.approx(6.62607015e-34)
    assert const.e == pytest.approx(1.602176634e-19)
    assert const.G == pytest.approx(6.674e-11, rel=1e-3)
    assert const.sigma_sb == pytest.approx(5.670374e-8, rel=1e-5)
    assert const.m_e == pytest.approx(9.1093837e-31, rel=1e-6)
    assert const.m_p == pytest.approx(1.67262192e-27, rel=1e-6)


def test_astronomical_quantities():
    assert const.au == pytest.approx(1.495978707e11)
    assert const.pc == pytest.approx(3.0857e16, rel=1e-4)
    assert const.R_EARTH == pytest.approx(6.3781e6, rel=1e-4)
    assert const.R_MOON == 1_737_400.0
    assert const.R_SUN == const.r_sun


def test_time_quantities():
    assert const.s_per_day == pytest.approx(86400.0)
    assert const.s_per_yr == pytest.approx(3.1557600e7)
    # A sidereal day is ~236 s shorter than a solar day.
    assert const.s_per_day - const.sidereal_day == pytest.approx(236, abs=1)


def test_derived_quantities():
    assert const.len_ns == pytest.approx(0.299792458)
    assert const.deg == pytest.approx(np.pi / 180)
    assert const.arcmin == pytest.approx(const.deg / 60)
    assert const.arcsec == pytest.approx(const.arcmin / 60)
    assert const.sq_deg == pytest.approx(const.deg**2)
    assert const.ft == pytest.approx(0.3048)
    assert const.Jy == pytest.approx(1e-26)
    assert const.eta_0 == pytest.approx(376.73)


def test_everything_is_a_plain_float():
    """No astropy Quantity leakage -- callers expect bare floats."""
    for name in ("c", "k_B", "au", "R_EARTH", "Jy", "s_per_day"):
        value = getattr(const, name)
        assert isinstance(value, float), f"{name} is {type(value)}"


def test_dtype_default_is_numpy_not_jax():
    assert const.DTYPE_R_NPY is np.float64
    assert not hasattr(const, "DTYPE_R_JAX")


def test_frequency_grid():
    assert const.FREQS.shape == (const.NCHAN,)
    assert const.FREQS[0] == 0.0
    assert const.DFREQ == const.SAMPLE_RATE / (2 * const.NCHAN)
    # Real sampling: the grid spans DC to the Nyquist frequency.
    assert const.FREQS[-1] == pytest.approx(
        const.SAMPLE_RATE / 2 - const.DFREQ
    )
    assert np.allclose(np.diff(const.FREQS), const.DFREQ)


def test_freqs_matches_calc_freqs_dfreq():
    """The bundled grid must agree with the runtime header-based one."""
    from eigsep_base.io import calc_freqs_dfreq

    freqs, dfreq = calc_freqs_dfreq(const.SAMPLE_RATE, const.NCHAN)
    assert np.array_equal(freqs, const.FREQS)
    assert dfreq == const.DFREQ


def test_site_locations():
    lat, lon, alt = const.MARJUM_PASS
    # Marjum Pass, Utah: northern hemisphere, western longitude, high
    # desert.
    assert 39 < lat < 40
    assert -114 < lon < -113
    assert 1000 < alt < 3000
    assert const.SITES["marjum"] == const.MARJUM_PASS


def test_description_keys_exist():
    desc = const.description()
    for key in desc:
        assert hasattr(const, key), f"description() names missing {key}"
