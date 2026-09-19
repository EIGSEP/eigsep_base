"""
eigsep_base -- the JAX-free foundation of the EIGSEP software stack.

Everything every other eigsep package may need: the file-format
contract, physical constants, and time parsing.

Deliberately depends only on numpy, h5py, scipy, and astropy, so it
installs on hardware-control machines with no GPU and no JAX toolchain.
Everything returns plain numpy arrays and dicts -- no JAX arrays, no
HPM objects.

Submodules
----------
io
    Read/write schema for all EIGSEP data formats: correlator HDF5,
    VNA/S11, standalone metadata, and the legacy ``.eig`` binary.
const
    Physical constants and EIGSEP instrument/site parameters.
time
    Time handling (``to_unix_time``, ``parse_time_from_name``,
    ``parse_filename_time``, ``filename_unix``, ``format_time``).
"""

__version__ = "0.1.0"

from . import const, io, time

# The handful of names that are useful at the top level. Everything
# else is reached through its submodule, which keeps it obvious where a
# function came from when it turns up in someone else's code.
from .io import read_hdf5, write_hdf5
from .time import (
    format_time,
    parse_filename_time,
    parse_time_from_name,
    to_unix_time,
)

__all__ = [
    "__version__",
    "const",
    "format_time",
    "io",
    "parse_filename_time",
    "parse_time_from_name",
    "read_hdf5",
    "time",
    "to_unix_time",
    "write_hdf5",
]
