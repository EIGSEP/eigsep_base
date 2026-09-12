"""
eigsep_base -- the JAX-free foundation of the EIGSEP software stack.

Everything every other eigsep package may need: the file-format
contract, physical constants, non-JAX coordinate transforms, time
parsing, noise estimation, caching, and small shared numerics.

Deliberately depends only on numpy, h5py, scipy, and astropy, so it
installs on hardware-control machines with no GPU and no JAX toolchain.
Everything returns plain numpy arrays and dicts -- no JAX arrays, no
HPM objects.

Submodules
----------
io
    Read/write schema for all EIGSEP data formats: correlator HDF5,
    VNA/S11, standalone metadata, and the legacy ``.eig`` binary.
coord
    Non-JAX coordinate utilities, including an astropy replacement for
    ``aipy.coord.convert_m``. JAX versions live in ``healjax.coord``.
const
    Physical constants and EIGSEP instrument/site parameters.
time
    Time parsing (``to_unix_time``, ``parse_time_from_name``).
noise
    Noise variance estimation and radiometer-equation prediction.
cache
    Content-hash NPZ caching with config fingerprinting.
utils
    Correlator bookkeeping plus FFT delay estimation (``fft_dly``).
"""

__version__ = "0.1.0"

from . import cache
from . import const
from . import coord
from . import io
from . import noise
from . import time
from . import utils

# The handful of names that are useful at the top level. Everything
# else is reached through its submodule, which keeps it obvious where a
# function came from when it turns up in someone else's code.
from .io import read_hdf5, write_hdf5
from .time import to_unix_time, parse_time_from_name

__all__ = [
    "__version__",
    "cache",
    "const",
    "coord",
    "io",
    "noise",
    "time",
    "utils",
    "read_hdf5",
    "write_hdf5",
    "to_unix_time",
    "parse_time_from_name",
]
