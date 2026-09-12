"""
Content-hash NPZ caching with config fingerprinting.

Generalized from ``bloom21cm.sim_cache``. The original hard-coded the
BLOOM mission config (antenna arm lengths, orbit normals, ...) into its
fingerprint and exposed one save/load pair per result kind
(``setup``, ``multifreq``, ``pointing_error``). This version keeps the
mechanism -- fingerprint the parameters that define a computation, key
the cache file by that fingerprint, and refuse to load a file whose
fingerprint disagrees -- but takes the config as an arbitrary
JSON-able payload, so any expensive computation can use it.

Why fingerprint at all: a cached array that silently came from a
different config is worse than no cache. Changing any fingerprinted
parameter changes the filename, so stale results are simply a cache
miss rather than a wrong answer, with no manual cleanup.

Typical use::

    from eigsep_base.cache import NpzCache

    cache = NpzCache("~/.cache/eigsep/beam")
    fp = cache.fingerprint({"nside": 64, "freqs_mhz": [50, 100]})

    hit = cache.try_load("beam", fp)
    if hit is None:
        beam = expensive_beam_computation()
        cache.save("beam", fp, beam=beam)
    else:
        beam = hit["beam"]
"""

from collections.abc import Mapping, Sequence, Set
import dataclasses
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "NpzCache",
    "canonicalize",
    "config_fingerprint",
    "cache_path",
    "save_result",
    "load_result",
    "try_load_result",
    "DEFAULT_CACHE_DIR",
    "FINGERPRINT_LENGTH",
]

#: Default cache directory, overridable with ``EIGSEP_CACHE_DIR``.
DEFAULT_CACHE_DIR = os.environ.get(
    "EIGSEP_CACHE_DIR", os.path.join("~", ".cache", "eigsep")
)

#: Number of hex characters kept from the digest. Eight gives ~4e9
#: distinct values -- ample for the handful of configs a single
#: analysis sweeps over, and short enough to keep filenames readable.
FINGERPRINT_LENGTH = 8


# ---------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------


def canonicalize(payload):
    """Convert ``payload`` into JSON-serializable, order-stable values.

    Handles the types that show up in eigsep configs: dataclasses,
    mappings, sequences, sets, numpy arrays and scalars, ``Path``,
    ``datetime``, and anything with an astropy-``Time``-style ``isot``
    attribute.

    Raises
    ------
    TypeError
        On a type it does not recognize. This is deliberate: falling
        back to ``repr`` would fold object identity (memory addresses)
        into the fingerprint, so two runs with identical configs would
        never share a cache entry, and a custom object's ``repr`` might
        omit the very field that matters.
    """
    if payload is None or isinstance(payload, (bool, int, float, str)):
        return payload
    if isinstance(payload, (np.bool_, np.integer, np.floating)):
        return payload.item()
    if isinstance(payload, np.ndarray):
        return payload.tolist()
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, (datetime.datetime, datetime.date)):
        return payload.isoformat()
    if dataclasses.is_dataclass(payload) and not isinstance(payload, type):
        return canonicalize(dataclasses.asdict(payload))
    if isinstance(payload, Mapping):
        # Keys are coerced to str so that {1: x} and {"1": x} cannot
        # collide differently across json round-trips.
        return {str(k): canonicalize(v) for k, v in sorted(
            payload.items(), key=lambda kv: str(kv[0])
        )}
    if isinstance(payload, Set):
        return sorted(canonicalize(v) for v in payload)
    if isinstance(payload, Sequence):  # str already handled above
        return [canonicalize(v) for v in payload]
    # astropy Time and friends expose an unambiguous ISO string.
    isot = getattr(payload, "isot", None)
    if isot is not None:
        return str(isot)
    raise TypeError(
        f"Cannot fingerprint object of type {type(payload).__name__}. "
        "Convert it to a plain value (float, str, list, dict) first."
    )


def config_fingerprint(config, length=FINGERPRINT_LENGTH):
    """
    Return a short hex fingerprint of the parameters in ``config``.

    Any change to a value reachable from ``config`` changes the
    fingerprint, which changes the cache filename and so invalidates
    cached results without manual cleanup.

    Parameters
    ----------
    config : any
        Anything :func:`canonicalize` accepts -- typically a dict of
        the simulation-defining parameters.
    length : int
        Number of hex characters to keep.

    Returns
    -------
    str
        Lowercase hex fingerprint of ``length`` characters.
    """
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")
    blob = json.dumps(canonicalize(config), sort_keys=True).encode()
    # Not a security boundary -- sha256 truncated for a short, stable
    # filename component. (bloom21cm used md5; sha256 avoids trouble on
    # FIPS-restricted hosts, where md5 is unavailable.)
    return hashlib.sha256(blob).hexdigest()[:length]


def _json_bytes(payload):
    return np.bytes_(json.dumps(payload, sort_keys=True))


def _load_json_bytes(value):
    item = value.item()
    if hasattr(item, "decode"):
        item = item.decode()
    return json.loads(str(item))


def _stored_fp(data):
    item = data["config_fp"].item()
    return item.decode() if hasattr(item, "decode") else str(item)


# ---------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------


class NpzCache:
    """A directory of fingerprinted ``.npz`` result files.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory holding the cache files. ``~`` is expanded. Defaults
        to :data:`DEFAULT_CACHE_DIR` (``$EIGSEP_CACHE_DIR`` if set).
        The directory is created on first write, not on construction.
    """

    def __init__(self, cache_dir=None):
        self.cache_dir = Path(
            os.path.expanduser(
                str(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
            )
        )

    def __repr__(self):
        return f"{type(self).__name__}({str(self.cache_dir)!r})"

    # -- naming ------------------------------------------------------

    @staticmethod
    def fingerprint(config, length=FINGERPRINT_LENGTH):
        """Alias for :func:`config_fingerprint`."""
        return config_fingerprint(config, length=length)

    @staticmethod
    def filename(name, fp, tag=None):
        """Return the cache filename for a result.

        ``tag`` is an extra qualifier for results that vary by a
        parameter not captured by the fingerprint (e.g. an NSIDE sweep
        over one config).
        """
        suffix = f"_{tag}" if tag else ""
        return f"{name}_{fp}{suffix}.npz"

    def path(self, name, fp, tag=None, create=True):
        """Full path for a result file, creating the directory."""
        if create:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / self.filename(name, fp, tag)

    # -- save / load -------------------------------------------------

    def save(self, name, fp, config=None, tag=None, **arrays):
        """
        Save an arbitrary named, fingerprinted result.

        Parameters
        ----------
        name : str
            Result kind, e.g. ``"gsm_recovery"``. Filename prefix.
        fp : str
            Config fingerprint from :meth:`fingerprint`.
        config : any, optional
            The config itself. When given, its canonical form is stored
            alongside the arrays so :meth:`load` can detect a
            fingerprint collision, and so a stale file is
            self-describing.
        tag : str, optional
            Extra filename qualifier.
        **arrays
            Named arrays/scalars, passed to
            :func:`numpy.savez_compressed`. The names ``config_fp`` and
            ``config_payload`` are reserved.

        Returns
        -------
        Path
            The path written.
        """
        reserved = {"config_fp", "config_payload"} & set(arrays)
        if reserved:
            raise ValueError(
                f"Array names {sorted(reserved)} are reserved for cache "
                "bookkeeping; rename them."
            )
        p = self.path(name, fp, tag)
        payload = canonicalize(config) if config is not None else {}
        np.savez_compressed(
            p,
            config_fp=np.bytes_(fp),
            config_payload=_json_bytes(payload),
            **arrays,
        )
        logger.info("saved %s -> %s", name, p.name)
        return p

    def load(self, name, fp, config=None, tag=None):
        """
        Load a named result saved by :meth:`save`.

        Returns
        -------
        numpy.lib.npyio.NpzFile
            Index it like a dict for the arrays passed to :meth:`save`
            (plus ``config_fp`` / ``config_payload``).

        Raises
        ------
        FileNotFoundError
            If the cache file does not exist.
        ValueError
            If the stored fingerprint or config payload disagrees with
            the one requested.
        """
        p = self.path(name, fp, tag, create=False)
        data = np.load(p)
        payload = canonicalize(config) if config is not None else None
        self._validate(data, fp, payload=payload, label=f"{name} cache")
        logger.info("loaded %s from %s", name, p.name)
        return data

    def try_load(self, name, fp, config=None, tag=None):
        """Return the loaded result from :meth:`load`, or None on miss.

        A miss is any missing file or any fingerprint/config mismatch --
        exactly the cases where recomputing is the right answer.
        """
        try:
            return self.load(name, fp, config=config, tag=tag)
        except (FileNotFoundError, ValueError) as e:
            logger.info("%s cache miss: %s", name, e)
            return None

    @staticmethod
    def _validate(data, fp, payload=None, label="cache"):
        if "config_fp" not in data.files:
            raise ValueError(f"{label} has no stored fingerprint")
        stored_fp = _stored_fp(data)
        if stored_fp != fp:
            raise ValueError(
                f"{label} fingerprint mismatch: file={stored_fp!r}, "
                f"expected={fp!r}"
            )
        if payload is not None and "config_payload" in data.files:
            stored_payload = _load_json_bytes(data["config_payload"])
            # An empty stored payload means the writer did not pass a
            # config; there is nothing to contradict.
            if stored_payload and stored_payload != payload:
                raise ValueError(f"{label} config payload mismatch")


# ---------------------------------------------------------------------
# Module-level convenience wrappers over the default cache directory
# ---------------------------------------------------------------------


def cache_path(filename, cache_dir=None):
    """Return the full path for a cache file, creating the directory."""
    d = Path(
        os.path.expanduser(
            str(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        )
    )
    d.mkdir(parents=True, exist_ok=True)
    return d / filename


def save_result(name, fp, cache_dir=None, config=None, tag=None, **arrays):
    """Save a named, fingerprinted result. See :meth:`NpzCache.save`."""
    return NpzCache(cache_dir).save(
        name, fp, config=config, tag=tag, **arrays
    )


def load_result(name, fp, cache_dir=None, config=None, tag=None):
    """Load a named result. See :meth:`NpzCache.load`."""
    return NpzCache(cache_dir).load(name, fp, config=config, tag=tag)


def try_load_result(name, fp, cache_dir=None, config=None, tag=None):
    """Load a named result, or return None on a miss."""
    return NpzCache(cache_dir).try_load(name, fp, config=config, tag=tag)
