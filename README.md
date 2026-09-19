# eigsep_base

The JAX-free foundation of the EIGSEP software stack: the file-format
contract, physical constants, non-JAX coordinate transforms, time
parsing, noise estimation, caching, and small shared numerics.

Every other eigsep package may depend on this one. It depends only on
`numpy`, `h5py`, `scipy`, and `astropy`, so it installs on
hardware-control machines with no GPU and no JAX toolchain.

Everything returns plain numpy arrays and dicts — no JAX arrays, no
`HPM` objects.

## Install

```bash
pip install -e .          # plus [dev] for pytest and ruff
```

## Modules

| Module | Contents | Extracted from |
|---|---|---|
| `io` | Read/write schema for **all** EIGSEP formats: correlator HDF5, VNA/S11, standalone metadata, legacy `.eig` binary | `eigsep_observing/io.py`, `eigsep_corr/io.py` |
| `coord` | Non-JAX coordinate utilities + astropy replacement for `aipy.coord.convert_m`, angle wrapping | `eigsep_sim/coord.py` |
| `const` | Physical constants, frequency grid, site locations | `eigsep_sim/const.py` |
| `time` | `to_unix_time`, `parse_time_from_name` | `eigsep_data/data.py` |
| `noise` | `interleaved_noise_variance_estimate`, `predict_noise_variance_from_autos` | `hera_cal/noise.py` |
| `cache` | Content-hash NPZ caching with config fingerprinting | `bloom21cm/sim_cache.py` |
| `utils` | Correlator bookkeeping, `fft_dly`, `interp_peak` | `eigsep_observing/utils.py`, `hera_cal/utils.py` |

## The format contract

`eigsep_base.io` is the single definition of how EIGSEP data is laid
out on disk. `eigsep_observing` uses it to **write**; `eigsep_data`
uses it to **read**. Neither needs to depend on the other, and the
format is a first-class shared API rather than an implementation
detail of the hardware package.

```python
from eigsep_base import io

data, header, metadata = io.read_hdf5("corr_20260715_172825Z.h5")
data, cal, header, metadata = io.read_s11_file("ants11_20260715.h5")
header, data = io.read_eig_file("legacy.eig")  # pre-HDF5 archive
```

## Deviations from the originals

Function signatures match the originals except where noted. The
extraction is verified against `eigsep_observing.io` directly:
`reshape_data` agrees across all four `acc_bins`/`avg_even_odd` modes,
`avg_metadata` and `append_corr_header` agree, all five schema
constants compare equal, and `write_hdf5` produces **byte-identical**
files.

- **`io.File` stays in `eigsep_observing`.** The double-buffered,
  threaded writer that backs the observing loop is instrument
  machinery, not a format definition. It calls `write_hdf5` here.
- **`append_corr_header` takes the linear-range loader as an argument**
  (`load_linear_range`, `validate_operating_point`). The calibration
  product is packaged data inside `eigsep_observing`, so the loader is
  injected rather than imported. Omitting it skips the linear-range
  block; passing a loader without a validator raises, since bounds must
  never be written without an operating-point check.
- **`.eig` readers gained unambiguous aliases.** `read_file`,
  `write_file`, and `read_header` keep their `eigsep_corr` names so
  ported call sites work unchanged, but `read_eig_file`,
  `write_eig_file`, and `read_eig_header` say which format they mean.
- **`predict_noise_variance_from_autos` takes arrays, not a
  `DataContainer`.** The caller passes the two autocorrelation
  waterfalls directly; no baseline tuples or polarization parsing.
  `infer_dt` likewise takes a time array instead of a `times_by_bl`
  dict.
- **`cache` is config-agnostic.** The BLOOM-specific `config_payload`
  and the per-result-kind save/load pairs are replaced by
  `NpzCache.save`/`load`/`try_load` over any JSON-able config.
  Fingerprints use truncated sha256 rather than md5, which keeps it
  working on FIPS-restricted hosts.
- **`rot_m` is deliberately absent** from `coord`. Per the
  reorganization plan there is one implementation, in `healjax.coord`,
  alongside the other JAX rotation helpers.
- **`const` drops the JAX pieces**: no `DTYPE_R_JAX`, and no
  `jax.config.update("jax_enable_x64", True)` side effect at import.
  Those stay in `eigsep_sim`.
- **`eq2top_m` broadcasts `ha` against `dec`.** aipy's ragged
  `np.array` build raises on numpy >= 1.24; scalar behavior is
  unchanged.

Two aipy docstrings turned out to contradict their own arithmetic, and
the corrected behavior is pinned by tests:

- `convert_m(isys, osys)` maps `v_osys = M @ v_isys`, verified against
  the IAU north galactic pole and galactic center. aipy's docstring
  told callers to reverse the arguments, which disagrees with its own
  eigsep call sites.
- `top2azalt` uses `x = east, y = north`, so azimuth is 0 at `+y`.
  aipy's per-function docstring said "0 at x axis = north", which
  contradicts both its arithmetic and its module docstring.

## Tests

```bash
pytest                                    # 294 tests
pytest --cov=eigsep_base                  # 97% coverage
```

The `.eig` and S11 readers are additionally exercised against real
archival files in the monorepo (`terrain/s11_data/*.h5`,
`eigsep_corr/transmitter/*.eig`).
