"""Spectral bases: compact frequency dependence for beams and skies.

A :class:`SpectralBasis` stores a matrix ``A`` of shape ``(nfreq, nmodes)``
whose columns are functions of frequency on a grid ``freqs``. A quantity
``x(nu)`` with any leading shape (pixels, field components, ...) is represented
as ``x ~ c @ A.T`` with coefficients ``c`` of shape ``(..., nmodes)``.

``A`` may be real (sky and beam *power* spectra) or complex (beam *E-field*
spectra, where power follows only after the coefficients are combined). For a
basis built by SVD the columns are orthonormal, so :meth:`project` is exact
least squares; for other bases it is the orthogonal projection only if the
columns are orthonormal.

Frequencies are in whatever unit the caller chooses (eigsep_sim uses Hz,
beam mapping uses MHz); the class never converts them.

Public API
----------
SpectralBasis
"""

import numpy as np
from scipy.interpolate import interp1d

__all__ = ["SpectralBasis"]


class SpectralBasis:
    """Frequency basis ``A`` (nfreq, nmodes) on the grid ``freqs``.

    Parameters
    ----------
    A : array (nfreq, nmodes), real or complex
    freqs : array (nfreq,), optional
        Frequencies of the rows of ``A``. Needed for :meth:`evaluate`.
    singular_values : array (nmodes,), optional
        Singular values when ``A`` came from an SVD.
    records : list of dict, optional
        Diagnostics from mode selection (see :meth:`from_samples`).
    """

    def __init__(self, A, freqs=None, singular_values=None, records=None):
        A = np.asarray(A)
        if A.ndim != 2:
            raise ValueError(f"A must be 2-D, got shape {A.shape}")
        self.A = A
        self.freqs = (
            None if freqs is None else np.asarray(freqs, dtype=np.float64)
        )
        if self.freqs is not None and self.freqs.shape != (A.shape[0],):
            raise ValueError("freqs must have one entry per row of A")
        self.singular_values = (
            None if singular_values is None else np.asarray(singular_values)
        )
        self.records = records

    @property
    def nfreq(self):
        return self.A.shape[0]

    @property
    def nmodes(self):
        return self.A.shape[1]

    @property
    def is_complex(self):
        return np.iscomplexobj(self.A)

    def project(self, data):
        """Coefficients of ``data`` (..., nfreq): ``data @ conj(A)`` -> (..., nmodes)."""
        return np.matmul(data, self.A.conj())

    def deproject(self, coeffs):
        """Reconstruct (..., nfreq) from coefficients (..., nmodes): ``coeffs @ A.T``."""
        return np.matmul(coeffs, self.A.T)

    def evaluate(self, freqs, fill_value=None):
        """The basis at new frequencies, (len(freqs), nmodes), by linear interpolation.

        Outside the grid: an error by default, or ``fill_value`` if given (for
        example 0.0).
        """
        if self.freqs is None:
            raise ValueError("basis has no frequency grid")
        kw = (
            {"bounds_error": True}
            if fill_value is None
            else {"bounds_error": False, "fill_value": fill_value}
        )
        return interp1d(self.freqs, self.A, axis=0, kind="linear", **kw)(
            np.asarray(freqs, dtype=np.float64)
        )

    def resample(self, freqs, fill_value=None):
        """A new basis on ``freqs`` (see :meth:`evaluate`)."""
        return type(self)(
            self.evaluate(freqs, fill_value),
            freqs=freqs,
            singular_values=self.singular_values,
        )

    @classmethod
    def from_samples(
        cls,
        freqs,
        samples,
        n_modes=None,
        max_relative_error=None,
        max_modes=None,
    ):
        """Build a basis from samples by SVD.

        ``samples`` is (n_samples, nfreq): each row is one spectrum (a pixel, a
        field component at a pixel, ...), real or complex. It is not centred:
        the basis spans the spectra themselves. Give exactly one of:

        * ``n_modes``: keep that many leading modes.
        * ``max_relative_error``: keep the fewest modes for which, at every
          frequency, ``||x_f - reconstruction_f|| / ||x_f|| <= max_relative_error``
          over the samples. ``records`` lists the error for each count tried
          (up to ``max_modes``, default all).
        """
        if (n_modes is None) == (max_relative_error is None):
            raise ValueError(
                "give exactly one of n_modes and max_relative_error"
            )
        freqs = np.asarray(freqs, dtype=np.float64)
        X = np.asarray(samples)
        _, s, vh = np.linalg.svd(X, full_matrices=False)
        if n_modes is not None:
            k = int(n_modes)
            return cls(vh[:k].T, freqs=freqs, singular_values=s[:k])
        norm = np.linalg.norm(X, axis=0)
        records = []
        limit = len(s) if max_modes is None else min(int(max_modes), len(s))
        for k in range(1, limit + 1):
            A = vh[:k].T
            error = np.linalg.norm(
                X - (X @ A.conj()) @ A.T, axis=0
            ) / np.maximum(norm, 1e-300)
            records.append(
                {
                    "modes": k,
                    "worst_relative_error": float(error.max()),
                    "median_relative_error": float(np.median(error)),
                    "energy_fraction": float(
                        np.sum(s[:k] ** 2) / np.sum(s**2)
                    ),
                }
            )
            if error.max() <= max_relative_error:
                return cls(
                    A, freqs=freqs, singular_values=s[:k], records=records
                )
        raise ValueError(
            f"no mode count up to {limit} reaches max_relative_error={max_relative_error}"
        )

    def save(self, path):
        """Write ``A``, ``freqs`` and ``singular_values`` (where present) to an npz."""
        arrays = {"A": self.A}
        if self.freqs is not None:
            arrays["freqs"] = self.freqs
        if self.singular_values is not None:
            arrays["singular_values"] = self.singular_values
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path, freqs=None, fill_value=None):
        """Read a basis written by :meth:`save`, optionally resampled to ``freqs``."""
        with np.load(path, allow_pickle=False) as z:
            basis = cls(
                z["A"],
                freqs=z.get("freqs"),
                singular_values=z.get("singular_values"),
            )
        return basis if freqs is None else basis.resample(freqs, fill_value)
