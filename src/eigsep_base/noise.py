"""
Noise estimation on visibility/spectrum waterfalls.

Cherry-picked from ``hera_cal.noise`` and stripped of its pyuvdata /
``DataContainer`` machinery. Both functions operate on plain numpy
arrays:

- :func:`interleaved_noise_variance_estimate` estimates noise from the
  data itself by differencing neighboring time/frequency samples.
- :func:`predict_noise_variance_from_autos` predicts noise from
  autocorrelations via the radiometer equation.
"""

import numpy as np
import scipy.signal

__all__ = [
    "interleaved_noise_variance_estimate",
    "predict_noise_variance_from_autos",
    "infer_dt",
]

#: Default second-difference kernel: the outer product of [1, -2, 1]
#: with itself. Sums to zero, so a smooth signal cancels and only noise
#: survives.
DEFAULT_KERNEL = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]


def interleaved_noise_variance_estimate(vis, kernel=DEFAULT_KERNEL):
    """Estimate the noise on a visibility per frequency and time using
    weighted differencing of neighboring frequency and time samples.

    Arguments:
        vis: complex visibility waterfall, usually a numpy array of size
            (Ntimes, Nfreqs)
        kernel: differencing kernel for how to weight each visibility
            relative to its neighbors in time and frequency. Must sum to
            zero and must be 2D (either dimension could be length 1)

    Returns:
        variance: estimate of the noise variance on the input visibility
            with the same shape
    """
    assert (
        np.sum(kernel) == 0
    ), "The kernal must sum to zero for difference-based noise estimation."
    assert np.array(kernel).ndim == 2, "The kernel must be 2D."
    variance = (
        np.abs(
            scipy.signal.convolve2d(
                vis, kernel, mode="same", boundary="wrap"
            )
        )
        ** 2
    )
    variance /= np.sum(np.array(kernel) ** 2)
    return variance


def infer_dt(times):
    """Infer the integration time from an array of sample times.

    Arguments:
        times: 1D array of sample times. Units are arbitrary (JD and
            Unix seconds are both common in EIGSEP); the return value
            carries the same units.

    Returns:
        dt: float, median spacing between consecutive times.

    Raises:
        ValueError: if fewer than two times are given, so no spacing
            can be measured.

    Note:
        ``hera_cal``'s version of this took a ``times_by_bl`` dict and
        could borrow a spacing from another baseline. EIGSEP has a
        single time axis per file, so this takes the time array
        directly.
    """
    times = np.asarray(times)
    if times.size < 2:
        raise ValueError(
            f"Cannot infer dt from {times.size} time(s); need at least 2. "
            "Pass dt explicitly."
        )
    return float(np.median(np.ediff1d(times)))


def predict_noise_variance_from_autos(
    auto1, auto2=None, dt=None, df=None, times=None, freqs=None, nsamples=None
):
    """Predict the noise variance on a baseline using autocorrelation
    data using the formula sigma^2 = Vii * Vjj / Delta t / Delta nu.

    Arguments:
        auto1: autocorrelation waterfall for the first antenna, usually
            shape (Ntimes, Nfreqs).
        auto2: autocorrelation waterfall for the second antenna. If
            None, ``auto1`` is used for both -- i.e. the noise on the
            autocorrelation itself.
        dt: integration time in seconds. If None, inferred from
            ``times``, which must then be given in seconds.
        df: channel width in Hz. If None, inferred from ``freqs``,
            which must then be given in Hz.
        times: 1D array of sample times in seconds, used only when
            ``dt`` is None.
        freqs: 1D array of channel frequencies in Hz, used only when
            ``df`` is None.
        nsamples: optional array, broadcastable against the autos, of
            the number of integrations averaged into each sample. The
            variance is divided by it.

    Returns:
        Noise variance predicted on the baseline, in units of the
        autocorrelation data squared.

    Note:
        ``hera_cal``'s version took a baseline tuple and a
        ``DataContainer`` and looked the autos up by polarization. Here
        the caller passes the two auto waterfalls directly, so no
        polarization-string parsing or container type is needed.
    """
    if auto2 is None:
        auto2 = auto1
    if dt is None:
        if times is None:
            raise ValueError("Must provide either dt or times.")
        dt = infer_dt(times)
    if df is None:
        if freqs is None:
            raise ValueError("Must provide either df or freqs.")
        freqs = np.asarray(freqs)
        if freqs.size < 2:
            raise ValueError(
                "Cannot infer channel width from fewer than 2 frequencies. "
                "Pass df explicitly."
            )
        df = float(np.median(np.ediff1d(freqs)))

    var = np.abs(np.asarray(auto1) * np.asarray(auto2) / dt / df)
    if nsamples is not None:
        return var / np.asarray(nsamples)
    return var
