"""Tests for eigsep_base.utils."""

import numpy as np
import pytest

from eigsep_base import utils


# --- correlator bookkeeping ------------------------------------------


def test_calc_freqs_dfreq_real_sampling():
    freqs, dfreq = utils.calc_freqs_dfreq(500e6, 1024)
    assert dfreq == pytest.approx(500e6 / 2048)
    assert freqs.shape == (1024,)
    assert freqs[0] == 0.0
    # Real sampling: spectrum spans DC to Nyquist = sample_rate / 2.
    assert freqs[-1] == pytest.approx(500e6 / 2 - dfreq)
    assert np.allclose(np.diff(freqs), dfreq)


def test_calc_freqs_dfreq_scales_with_nchan():
    _, dfreq_1k = utils.calc_freqs_dfreq(500e6, 1024)
    _, dfreq_2k = utils.calc_freqs_dfreq(500e6, 2048)
    assert dfreq_1k == pytest.approx(2 * dfreq_2k)


def test_calc_inttime_default_matches_adc_demux():
    """eigsep_observing called this with two args, relying on
    ADC_DEMUX; eigsep_corr passed acc_bins explicitly. They agree."""
    assert utils.calc_inttime(500e6, 2**28) == utils.calc_inttime(
        500e6, 2**28, acc_bins=utils.ADC_DEMUX
    )
    assert utils.ADC_DEMUX == 2


def test_calc_inttime_value():
    # 500 Msps, acc_len clocks at 250 MHz fabric rate.
    assert utils.calc_inttime(500e6, 1000) == pytest.approx(4e-6)
    assert utils.calc_inttime(500e6, 2**26) == pytest.approx(0.2684354)


def test_calc_inttime_scales_linearly_with_acc_len():
    assert utils.calc_inttime(500e6, 2000) == pytest.approx(
        2 * utils.calc_inttime(500e6, 1000)
    )


def test_calc_times():
    acc_cnt = np.arange(5)
    times = utils.calc_times(acc_cnt, 0.25, 1_700_000_000.0)
    assert np.allclose(
        times, 1_700_000_000.0 + np.array([0, 0.25, 0.5, 0.75, 1.0])
    )


def test_calc_times_accepts_per_sample_sync_times():
    acc_cnt = np.array([0, 1, 2])
    sync = np.array([100.0, 100.0, 200.0])
    assert np.allclose(
        utils.calc_times(acc_cnt, 2.0, sync), [100.0, 102.0, 204.0]
    )


def test_calc_integration_len_autos_and_crosses():
    # 4-byte samples, 2 acc bins, 1024 chans.
    auto_only = utils.calc_integration_len(4, 2, 1024, ["0", "1"])
    assert auto_only == 4 * 2 * 1024 * 2
    # A cross costs twice an auto (real + imaginary).
    one_cross = utils.calc_integration_len(4, 2, 1024, ["02"])
    assert one_cross == 2 * utils.calc_integration_len(4, 2, 1024, ["0"])
    mixed = utils.calc_integration_len(4, 2, 1024, ["0", "1", "02"])
    assert mixed == auto_only + one_cross


# --- interp_peak ------------------------------------------------------


def _tone(n, freq_bin, phase=0.0, amp=1.0):
    """A complex tone whose FFT peaks at (possibly fractional)
    ``freq_bin``."""
    t = np.arange(n)
    return amp * np.exp(2j * np.pi * freq_bin * t / n + 1j * phase)


def test_interp_peak_finds_integer_bin():
    n = 64
    data = np.fft.fft(_tone(n, 7))[None, :]
    indices, shifts, peaks, new_peaks = utils.interp_peak(data)
    assert indices[0] == 7
    assert shifts[0] == pytest.approx(0.0, abs=1e-6)
    assert peaks[0] == pytest.approx(n, rel=1e-6)


def test_interp_peak_recovers_fractional_bin():
    """Quinn's method should resolve a peak between bins."""
    n = 128
    true_bin = 20.3
    data = np.fft.fft(_tone(n, true_bin))[None, :]
    indices, shifts, _, _ = utils.interp_peak(data, method="quinn")
    assert indices[0] + shifts[0] == pytest.approx(true_bin, abs=0.05)


def test_interp_peak_quadratic_on_real_data():
    n = 128
    true_bin = 20.3
    data = np.abs(np.fft.fft(_tone(n, true_bin)))[None, :]
    indices, shifts, _, _ = utils.interp_peak(data, method="quadratic")
    assert indices[0] + shifts[0] == pytest.approx(true_bin, abs=0.3)


def test_interp_peak_reshapes_1d_input():
    n = 64
    indices, shifts, peaks, new_peaks = utils.interp_peak(
        np.fft.fft(_tone(n, 5))
    )
    assert indices.shape == (1,)
    assert indices[0] == 5


def test_interp_peak_handles_multiple_rows_independently():
    n = 64
    rows = np.array(
        [np.fft.fft(_tone(n, b)) for b in (3, 11, 25)]
    )
    indices, _, _, _ = utils.interp_peak(rows)
    assert list(indices) == [3, 11, 25]


def test_interp_peak_rejects_unknown_method():
    with pytest.raises(ValueError, match="not a recognized peak"):
        utils.interp_peak(np.fft.fft(_tone(32, 4)), method="bogus")


def test_interp_peak_reject_edges_runs():
    n = 64
    data = np.fft.fft(_tone(n, 30))[None, :]
    indices, _, _, _ = utils.interp_peak(data, reject_edges=True)
    assert indices.shape == (1,)


def test_quinn_tau_is_finite_and_odd_ish():
    x = np.linspace(0.01, 5, 50)
    assert np.all(np.isfinite(utils.quinn_tau(x)))


# --- fft_dly ----------------------------------------------------------


def _delayed_gain(ntimes, nfreqs, df, delay_s, offset=0.0, f0=0.0):
    """Complex gains with a pure linear phase slope of ``delay_s``."""
    freqs = np.arange(nfreqs) * df + f0
    phase = 2 * np.pi * delay_s * freqs + offset
    return np.tile(np.exp(1j * phase), (ntimes, 1))


def test_fft_dly_recovers_a_known_delay():
    nfreqs, df = 1024, 100e3  # 100 kHz channels
    true_delay = 50e-9  # 50 ns
    data = _delayed_gain(4, nfreqs, df, true_delay)
    dlys, offset = fft_dly_call(data, df)
    assert dlys.shape == (4, 1)
    assert np.allclose(dlys, true_delay, atol=2e-10)


def test_fft_dly_recovers_a_negative_delay():
    nfreqs, df = 1024, 100e3
    true_delay = -120e-9
    dlys, _ = fft_dly_call(_delayed_gain(2, nfreqs, df, true_delay), df)
    assert np.allclose(dlys, true_delay, atol=2e-10)


def test_fft_dly_recovers_the_phase_offset():
    nfreqs, df = 512, 200e3
    true_delay, true_offset = 30e-9, 0.7
    data = _delayed_gain(3, nfreqs, df, true_delay, offset=true_offset)
    dlys, offset = fft_dly_call(data, df)
    assert offset.shape == (3, 1)
    assert np.allclose(offset, true_offset, atol=1e-2)


def test_fft_dly_zero_delay_gives_zero():
    nfreqs, df = 256, 100e3
    dlys, offset = fft_dly_call(_delayed_gain(2, nfreqs, df, 0.0), df)
    assert np.allclose(dlys, 0.0, atol=1e-12)
    assert np.allclose(offset, 0.0, atol=1e-6)


def test_fft_dly_distinct_delays_per_time():
    nfreqs, df = 512, 100e3
    freqs = np.arange(nfreqs) * df
    delays = np.array([10e-9, -40e-9, 75e-9])
    data = np.exp(2j * np.pi * delays[:, None] * freqs[None, :])
    dlys, _ = fft_dly_call(data, df)
    assert np.allclose(dlys[:, 0], delays, atol=5e-10)


def test_fft_dly_ignores_flagged_channels_via_wgts():
    nfreqs, df = 1024, 100e3
    true_delay = 60e-9
    data = _delayed_gain(1, nfreqs, df, true_delay).astype(np.complex128)
    wgts = np.ones_like(data, dtype=np.float64)
    # Corrupt a block of channels and down-weight it to zero.
    data[:, 100:200] = 50.0 + 0j
    wgts[:, 100:200] = 0.0
    dlys, _ = fft_dly_call(data, df, wgts=wgts)
    # Zeroing 10% of the band broadens the transform peak, so Quinn's
    # sub-bin estimate picks up a ~0.5 ns bias. The point is that the
    # corrupted channels are excluded rather than dominating: without
    # the weights the constant block pulls the answer to ~0 delay.
    assert np.allclose(dlys, true_delay, atol=1e-9)
    unweighted, _ = fft_dly_call(data, df)
    assert abs(unweighted[0, 0] - true_delay) > 1e-9


def test_fft_dly_does_not_mutate_input_when_medfilt():
    nfreqs, df = 128, 100e3
    data = _delayed_gain(2, nfreqs, df, 25e-9).astype(np.complex128)
    original = data.copy()
    fft_dly_call(data, df, medfilt=True, kernel=(1, 5))
    assert np.array_equal(data, original)


def test_fft_dly_edge_cut_rejects_too_large_a_cut():
    nfreqs, df = 64, 100e3
    data = _delayed_gain(1, nfreqs, df, 10e-9)
    with pytest.raises(AssertionError, match="edge_cut cannot be"):
        fft_dly_call(data, df, edge_cut=nfreqs)


def test_fft_dly_with_edge_cut_still_finds_the_delay():
    nfreqs, df = 1024, 100e3
    true_delay = 45e-9
    data = _delayed_gain(1, nfreqs, df, true_delay)
    dlys, _ = fft_dly_call(data, df, edge_cut=16)
    assert np.allclose(dlys, true_delay, atol=2e-9)


def fft_dly_call(data, df, **kwargs):
    """fft_dly writes into its input via ``dw[np.isnan(dw)] = 0``;
    hand it a fresh copy so tests stay independent."""
    return utils.fft_dly(np.array(data, dtype=np.complex128), df, **kwargs)
