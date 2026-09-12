"""Tests for eigsep_base.noise."""

import numpy as np
import pytest

from eigsep_base import noise


# --- interleaved_noise_variance_estimate -----------------------------


def test_kernel_must_sum_to_zero():
    vis = np.zeros((8, 8), dtype=complex)
    with pytest.raises(AssertionError, match="must sum to zero"):
        noise.interleaved_noise_variance_estimate(vis, kernel=[[1, 1]])


def test_kernel_must_be_2d():
    vis = np.zeros((8, 8), dtype=complex)
    with pytest.raises(AssertionError, match="must be 2D"):
        noise.interleaved_noise_variance_estimate(vis, kernel=[1, -2, 1])


def test_default_kernel_sums_to_zero():
    assert np.sum(noise.DEFAULT_KERNEL) == 0


def test_output_shape_matches_input():
    vis = np.zeros((16, 32), dtype=complex)
    out = noise.interleaved_noise_variance_estimate(vis)
    assert out.shape == vis.shape


def test_smooth_data_has_near_zero_estimated_variance():
    """A signal the kernel annihilates -- here a constant -- must
    report no noise. That is the whole point of a zero-sum kernel."""
    vis = np.full((32, 32), 5.0 + 3.0j)
    out = noise.interleaved_noise_variance_estimate(vis)
    assert np.allclose(out, 0.0, atol=1e-20)


def test_linear_ramp_is_also_annihilated():
    """[1,-2,1] is a second difference, so it kills linear trends."""
    t = np.arange(32)[:, None]
    f = np.arange(32)[None, :]
    vis = (2.0 * t + 3.0 * f).astype(complex)
    out = noise.interleaved_noise_variance_estimate(vis)
    # Interior pixels only: the wrap-around boundary sees a jump.
    assert np.allclose(out[2:-2, 2:-2], 0.0, atol=1e-18)


def test_recovers_white_noise_variance():
    """On pure white noise the estimator is unbiased."""
    rng = np.random.default_rng(42)
    sigma = 3.0
    n = 400
    vis = rng.normal(0, sigma / np.sqrt(2), (n, n)) + 1j * rng.normal(
        0, sigma / np.sqrt(2), (n, n)
    )
    out = noise.interleaved_noise_variance_estimate(vis)
    assert np.mean(out) == pytest.approx(sigma**2, rel=0.02)


def test_recovers_variance_on_top_of_smooth_signal():
    """A bright smooth foreground must not inflate the estimate."""
    rng = np.random.default_rng(7)
    sigma = 2.0
    n = 300
    t = np.arange(n)[:, None]
    f = np.arange(n)[None, :]
    smooth = 1000.0 + 5.0 * t + 2.0 * f
    noise_real = rng.normal(0, sigma / np.sqrt(2), (n, n))
    noise_imag = rng.normal(0, sigma / np.sqrt(2), (n, n))
    vis = smooth + noise_real + 1j * noise_imag
    out = noise.interleaved_noise_variance_estimate(vis)
    assert np.mean(out[2:-2, 2:-2]) == pytest.approx(sigma**2, rel=0.03)


def test_scaling_data_scales_variance_quadratically():
    rng = np.random.default_rng(3)
    vis = rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))
    base = noise.interleaved_noise_variance_estimate(vis)
    scaled = noise.interleaved_noise_variance_estimate(3 * vis)
    assert np.allclose(scaled, 9 * base)


def test_frequency_only_kernel():
    """A (1, 3) kernel differences in frequency alone."""
    rng = np.random.default_rng(11)
    sigma = 1.5
    vis = rng.normal(0, sigma / np.sqrt(2), (64, 512)) + 1j * rng.normal(
        0, sigma / np.sqrt(2), (64, 512)
    )
    out = noise.interleaved_noise_variance_estimate(
        vis, kernel=[[1, -2, 1]]
    )
    assert out.shape == vis.shape
    assert np.mean(out) == pytest.approx(sigma**2, rel=0.05)


def test_works_on_real_valued_input():
    rng = np.random.default_rng(5)
    vis = rng.normal(0, 1.0, (128, 128))
    out = noise.interleaved_noise_variance_estimate(vis)
    assert np.all(out >= 0)
    assert np.mean(out) == pytest.approx(1.0, rel=0.05)


# --- infer_dt ---------------------------------------------------------


def test_infer_dt_regular_grid():
    assert noise.infer_dt(np.arange(10) * 0.25) == pytest.approx(0.25)


def test_infer_dt_is_robust_to_a_gap():
    """A dropped integration must not double the inferred cadence."""
    times = np.array([0.0, 0.25, 0.5, 1.5, 1.75, 2.0])
    assert noise.infer_dt(times) == pytest.approx(0.25)


def test_infer_dt_needs_two_samples():
    with pytest.raises(ValueError, match="need at least 2"):
        noise.infer_dt(np.array([1.0]))


# --- predict_noise_variance_from_autos -------------------------------


def test_radiometer_equation():
    """sigma^2 = Vii * Vjj / (dt * df)."""
    auto1 = np.full((4, 8), 100.0)
    auto2 = np.full((4, 8), 400.0)
    var = noise.predict_noise_variance_from_autos(
        auto1, auto2, dt=10.0, df=1e5
    )
    assert np.allclose(var, 100.0 * 400.0 / 10.0 / 1e5)


def test_single_auto_is_used_for_both_antennas():
    auto = np.full((2, 4), 7.0)
    both = noise.predict_noise_variance_from_autos(auto, auto, dt=1, df=1)
    one = noise.predict_noise_variance_from_autos(auto, dt=1, df=1)
    assert np.allclose(one, both)
    assert np.allclose(one, 49.0)


def test_longer_integration_lowers_the_variance():
    auto = np.full((2, 4), 10.0)
    short = noise.predict_noise_variance_from_autos(auto, dt=1.0, df=1e5)
    long = noise.predict_noise_variance_from_autos(auto, dt=10.0, df=1e5)
    assert np.allclose(long, short / 10)


def test_nsamples_divides_the_variance():
    auto = np.full((2, 4), 10.0)
    base = noise.predict_noise_variance_from_autos(auto, dt=1.0, df=1.0)
    averaged = noise.predict_noise_variance_from_autos(
        auto, dt=1.0, df=1.0, nsamples=np.full((2, 4), 4.0)
    )
    assert np.allclose(averaged, base / 4)


def test_dt_and_df_inferred_from_times_and_freqs():
    auto = np.full((5, 6), 3.0)
    times = np.arange(5) * 0.5
    freqs = np.arange(6) * 1e5
    inferred = noise.predict_noise_variance_from_autos(
        auto, times=times, freqs=freqs
    )
    explicit = noise.predict_noise_variance_from_autos(
        auto, dt=0.5, df=1e5
    )
    assert np.allclose(inferred, explicit)


def test_missing_dt_and_times_raises():
    auto = np.ones((2, 2))
    with pytest.raises(ValueError, match="either dt or times"):
        noise.predict_noise_variance_from_autos(auto, df=1.0)


def test_missing_df_and_freqs_raises():
    auto = np.ones((2, 2))
    with pytest.raises(ValueError, match="either df or freqs"):
        noise.predict_noise_variance_from_autos(auto, dt=1.0)


def test_single_frequency_cannot_infer_channel_width():
    auto = np.ones((2, 1))
    with pytest.raises(ValueError, match="fewer than 2 frequencies"):
        noise.predict_noise_variance_from_autos(
            auto, dt=1.0, freqs=np.array([1e8])
        )


def test_result_is_always_non_negative():
    """Complex or negative-valued autos still give a real variance."""
    auto1 = np.array([[-4.0, 2.0], [3.0, -1.0]])
    auto2 = np.array([[2.0, -5.0], [-2.0, 6.0]])
    var = noise.predict_noise_variance_from_autos(
        auto1, auto2, dt=1.0, df=1.0
    )
    assert np.all(var >= 0)


def test_prediction_matches_interleaved_estimate_on_simulated_data():
    """The two independent routes to a noise level must agree.

    Build a visibility whose noise is set by the radiometer equation
    from two known autos, then check the difference-based estimator
    recovers the predicted variance.
    """
    rng = np.random.default_rng(1234)
    ntimes, nfreqs = 400, 400
    dt, df = 10.0, 1e5
    auto1 = np.full((ntimes, nfreqs), 150.0)
    auto2 = np.full((ntimes, nfreqs), 220.0)
    predicted = noise.predict_noise_variance_from_autos(
        auto1, auto2, dt=dt, df=df
    )
    sigma = np.sqrt(predicted / 2)  # per real/imaginary component
    vis = rng.normal(0, sigma) + 1j * rng.normal(0, sigma)
    estimated = noise.interleaved_noise_variance_estimate(vis)
    assert np.mean(estimated) == pytest.approx(
        np.mean(predicted), rel=0.02
    )
