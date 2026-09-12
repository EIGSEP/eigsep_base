"""
Small shared utilities.

Two groups:

1. Correlator bookkeeping (``calc_freqs_dfreq``, ``calc_inttime``,
   ``calc_times``, ``calc_integration_len``) shared by the HDF5 and
   legacy ``.eig`` readers in :mod:`eigsep_base.io`. These were
   duplicated between ``eigsep_observing.utils`` and
   ``eigsep_corr.utils``.

2. FFT-based delay estimation (``fft_dly``, ``interp_peak``,
   ``quinn_tau``) cherry-picked from ``hera_cal.utils``. Pure
   numpy/scipy -- the pyuvdata/hera_cal machinery around them is not
   needed to estimate a delay from a complex waterfall.
"""

import copy
import warnings

import numpy as np
from scipy import signal

__all__ = [
    "calc_freqs_dfreq",
    "calc_inttime",
    "calc_times",
    "calc_integration_len",
    "fft_dly",
    "interp_peak",
    "quinn_tau",
]


# ---------------------------------------------------------------------
# Correlator bookkeeping
# ---------------------------------------------------------------------

# The SNAP ADC delivers ADC_DEMUX samples per FPGA fabric clock
# (demux-2: 500 Msps ADC, 250 MHz fabric). Registers that count
# "clocks" (corr_acc_len, sync uptime) tick at sample_rate / ADC_DEMUX.
ADC_DEMUX = 2


def calc_freqs_dfreq(sample_rate_Hz, nchan):
    """Return frequencies and delta between frequencies for real-sampled
    spectra from the SNAP spectrometer/correlator."""
    dfreq = sample_rate_Hz / (2 * nchan)  # assumes real sampling
    freqs = np.arange(nchan) * dfreq
    return freqs, dfreq


def calc_inttime(sample_rate_Hz, acc_len, acc_bins=ADC_DEMUX):
    """
    Calculate time per integration [s] from sample_freq and acc_len.

    ``acc_len`` (the ``corr_acc_len`` register) sets the vacc dump
    period in FPGA fabric clocks, which run at
    ``sample_rate / ADC_DEMUX``. The number of accumulator bins the
    firmware emits per dump (even/odd in v2.3, one spectrum in v2.4)
    does not enter the timing: v2.4 accumulates the same window into
    one full-duty bin instead of two half-duty banks.

    The ``acc_bins`` keyword exists because the legacy ``.eig`` reader
    passed the header's ``acc_bins`` positionally. It is the demux
    factor, and on every firmware EIGSEP has flown it equals
    ``ADC_DEMUX``, so both call conventions agree.
    """
    inttime = 1 / sample_rate_Hz * acc_len * acc_bins
    return inttime


def calc_times(acc_cnt, inttime, sync_time):
    """Calculate integration times [s] from acc_cnt using sync time."""
    times = acc_cnt * inttime + sync_time
    return times


def calc_integration_len(itemsize, acc_bins, nchan, pairs):
    """
    Calculate the number of bytes for an integration of ``acc_bins``
    bins. Cross-correlations have double length since there's a real
    and imaginary part.

    Parameters
    ----------
    itemsize : int
        Size of data type in bytes.
    acc_bins : int
        Number of accumulations per integration.
    nchan : int
        Number of frequency channels per spectrum.
    pairs : list of str
        List of correlation pairs. Length 1 for autos, 2 for cross.

    Returns
    -------
    int_len : int
        Number of bytes for an integration of ``acc_bins`` bins.

    """
    n_auto = len([p for p in pairs if len(p) == 1])
    n_cross = len(pairs) - n_auto
    return itemsize * acc_bins * nchan * (n_auto + 2 * n_cross)


# ---------------------------------------------------------------------
# FFT delay estimation (from hera_cal.utils)
# ---------------------------------------------------------------------


def fft_dly(
    data, df, wgts=None, f0=0.0, medfilt=False, kernel=(1, 11), edge_cut=0
):
    """Get delay of visibility across band using FFT and Quinn's Second
    Method to fit the delay and phase offset.

    Arguments:
        data : ndarray of complex data (e.g. gains or visibilities) of
            shape (Ntimes, Nfreqs)
        df : frequency channel width in Hz
        wgts : multiplicative wgts of the same shape as the data
        f0 : float lowest frequency channel. Optional parameter used in
            getting the offset correct.
        medfilt : boolean, median filter data before fft
        kernel : size of median filter kernel along (time, freq) axes
        edge_cut : int, number of channels to exclude at each band edge
            of data in FFT window

    Returns:
        dlys : (Ntimes, 1) ndarray containing delay for each integration
        offset : (Ntimes, 1) ndarray containing estimated
            frequency-independent phases
    """
    # setup
    Ntimes, Nfreqs = data.shape
    if wgts is None:
        wgts = np.ones_like(data, dtype=np.float32)

    # smooth via median filter
    if medfilt:
        # this prevents filtering of the original input data
        data = copy.deepcopy(data)
        data.real = signal.medfilt(data.real, kernel_size=kernel)
        data.imag = signal.medfilt(data.imag, kernel_size=kernel)

    # fft w/ wgts
    dw = data * wgts
    if edge_cut > 0:
        assert 2 * edge_cut < Nfreqs - 1, "edge_cut cannot be >= Nfreqs/2 - 1"
        dw = dw[:, edge_cut:(-edge_cut + 1)]
    dw[np.isnan(dw)] = 0
    fftfreqs = np.fft.fftfreq(dw.shape[1], df)
    dtau = fftfreqs[1] - fftfreqs[0]
    vfft = np.fft.fft(dw, axis=1)

    # get interpolated peak and indices
    inds, bin_shifts, peaks, interp_peaks = interp_peak(vfft)
    dlys = (fftfreqs[inds] + bin_shifts * dtau).reshape(-1, 1)

    # Now that we know the slope, estimate the remaining phase offset
    freqs = np.arange(Nfreqs, dtype=data.dtype) * df + f0
    fSlice = slice(edge_cut, len(freqs) - edge_cut)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "invalid value encountered in divide"
        )
        offset = np.angle(
            np.sum(
                wgts[:, fSlice]
                * data[:, fSlice]
                * np.exp(
                    -np.complex64(2j * np.pi)
                    * dlys
                    * freqs[fSlice].reshape(1, -1)
                ),
                axis=1,
                keepdims=True,
            )
            / np.sum(wgts[:, fSlice], axis=1, keepdims=True)
        )

    return dlys, offset


def quinn_tau(x):
    """Quinn subgrid interpolation parameter
    (see https://ieeexplore.ieee.org/document/558515)"""
    t = 0.25 * np.log(3 * x**2 + 6 * x + 1)
    t -= (
        6**0.5
        / 24
        * np.log((x + 1 - (2 / 3) ** 0.5) / (x + 1 + (2 / 3) ** 0.5))
    )
    return t


def interp_peak(data, method="quinn", reject_edges=False):
    """
    Spectral interpolation for finding peak and amplitude of data along
    last axis.

    Args:
        data : complex 2d ndarray in Fourier space.
            If fed as 1d array will reshape into [1, N] array.
            Quinn's method usually operates on complex data (eg. fft'ed
            data) while the quadratic method operates on real-valued
            data (generally absolute values).
        method : either 'quinn'
            (see https://ieeexplore.ieee.org/document/558515) or
            'quadratic' (see
            https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html).
        reject_edges : bool, if True, reject solution if it isn't a true
            "peak", in other words if it is along the axis edges

    Returns:
        indices : index array holding argmax of data along last axis
        bin_shifts : estimated peak bin shift value [-1, 1] from indices
        peaks : argmax of data corresponding to indices
        new_peaks : estimated peak value at indices + bin_shifts
    """
    # get properties
    if data.ndim == 1:
        data = data[None, :]
    N1, N2 = data.shape

    # get abs
    dabs = np.abs(data)

    # ensure edge cases are handled is requested
    if reject_edges:
        # scroll through diffs and set monotonically decreasing edges
        # to zero
        forw_diff = dabs[:, 1:] - dabs[:, :-1]
        for i, fd in enumerate(forw_diff):
            ncut = np.argmax(fd > 0)
            if ncut > 0:
                dabs[i, :ncut] = 0.0
            ncut = N2 - np.argmax(fd[::-1] < 0)
            if ncut > 0:
                dabs[i, ncut:] = 0.0

    # get argmaxes along last axis
    if method not in ("quinn", "quadratic"):
        raise ValueError(
            "'{}' is not a recognized peak interpolation method.".format(
                method
            )
        )

    indices = np.argmax(dabs, axis=-1)
    peaks = data[range(N1), indices]

    # calculate shifted peak for sub-bin resolution
    k0 = data[range(N1), indices - 1]
    k1 = data[range(N1), indices]
    k2 = data[range(N1), (indices + 1) % N2]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", r"invalid value encountered in divide"
        )
        if method == "quinn":
            alpha1 = (k0 / k1).real
            alpha2 = (k2 / k1).real
            delta1 = alpha1 / (1 - alpha1)
            delta2 = -alpha2 / (1 - alpha2)
            d = (
                (delta1 + delta2) / 2
                + quinn_tau(delta1**2)
                - quinn_tau(delta2**2)
            )
            d[~np.isfinite(d)] = 0.0

            numerator_ck = np.exp(2.0j * np.pi * d) - 1
            ck = np.array(
                [
                    np.true_divide(
                        numerator_ck, 2.0j * np.pi * (d - k), where=~(d == 0)
                    )
                    for k in [-1, 0, 1]
                ]
            )
            rho = np.abs(k0 * ck[0] + k1 * ck[1] + k2 * ck[2]) / np.abs(
                np.sum(ck**2)
            )
            rho[d == 0] = np.abs(k1[d == 0])
            return indices, d, np.abs(peaks), rho

        elif method == "quadratic":
            denom = k0 - 2 * k1 + k2
            bin_shifts = 0.5 * np.true_divide(
                (k0 - k2), denom, where=~np.isclose(denom, 0.0)
            )
            new_peaks = k1 - 0.25 * (k0 - k2) * bin_shifts
            return indices, bin_shifts, peaks, new_peaks
