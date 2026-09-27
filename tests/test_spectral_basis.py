import numpy as np
import pytest

from eigsep_base.spectral_basis import SpectralBasis

FREQS = np.linspace(50.0, 250.0, 51)


def smooth_samples(n=300, complex_=False, seed=0):
    """Spectra that are exact combinations of three smooth functions of frequency."""
    rng = np.random.default_rng(seed)
    x = (FREQS - 150.0) / 100.0
    funcs = np.stack([np.ones_like(x), x, np.cos(3 * x)])
    c = rng.normal(size=(n, 3))
    if complex_:
        c = c + 1j * rng.normal(size=(n, 3))
    return c @ funcs


@pytest.mark.parametrize("complex_", [False, True])
def test_svd_basis_spans_the_samples(complex_):
    X = smooth_samples(complex_=complex_)
    b = SpectralBasis.from_samples(FREQS, X, n_modes=3)
    assert b.A.shape == (51, 3) and b.is_complex == complex_
    np.testing.assert_allclose(b.A.conj().T @ b.A, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(b.deproject(b.project(X)), X, atol=1e-10)


def test_mode_count_from_the_error_threshold():
    X = smooth_samples(complex_=True) + 1e-4 * np.random.default_rng(1).normal(
        size=(300, 51)
    )
    b = SpectralBasis.from_samples(FREQS, X, max_relative_error=1e-2)
    assert b.nmodes == 3
    assert [r["modes"] for r in b.records] == [1, 2, 3]
    assert (
        b.records[-1]["worst_relative_error"]
        <= 1e-2
        < b.records[-2]["worst_relative_error"]
    )
    with pytest.raises(ValueError):
        SpectralBasis.from_samples(
            FREQS, X, max_relative_error=1e-12, max_modes=5
        )
    with pytest.raises(ValueError):
        SpectralBasis.from_samples(FREQS, X)


def test_evaluate_interpolates_and_guards_the_edges():
    b = SpectralBasis.from_samples(FREQS, smooth_samples(), n_modes=3)
    np.testing.assert_allclose(b.evaluate(FREQS[5:9]), b.A[5:9])
    mid = 0.5 * (FREQS[10] + FREQS[11])
    np.testing.assert_allclose(b.evaluate([mid])[0], 0.5 * (b.A[10] + b.A[11]))
    with pytest.raises(ValueError):
        b.evaluate([260.0])
    assert np.all(b.evaluate([260.0], fill_value=0.0) == 0)
    r = b.resample(FREQS[::2])
    assert r.nfreq == 26 and np.allclose(r.A, b.A[::2])


def test_save_load_round_trip(tmp_path):
    b = SpectralBasis.from_samples(
        FREQS, smooth_samples(complex_=True), n_modes=3
    )
    b.save(tmp_path / "basis.npz")
    c = SpectralBasis.load(tmp_path / "basis.npz")
    np.testing.assert_array_equal(c.A, b.A)
    np.testing.assert_array_equal(c.freqs, b.freqs)
    np.testing.assert_array_equal(c.singular_values, b.singular_values)
    d = SpectralBasis.load(tmp_path / "basis.npz", freqs=FREQS[:10])
    np.testing.assert_allclose(d.A, b.A[:10])
