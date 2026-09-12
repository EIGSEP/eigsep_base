"""Tests for eigsep_base.cache."""

import dataclasses
import datetime
from pathlib import Path

import numpy as np
import pytest

from eigsep_base import cache


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "cache"


@pytest.fixture
def npz(cache_dir):
    return cache.NpzCache(cache_dir)


CONFIG = {"nside": 64, "freqs_mhz": [50.0, 100.0], "model": "dipole"}


# --- canonicalize -----------------------------------------------------


def test_canonicalize_passes_through_plain_values():
    assert cache.canonicalize(None) is None
    assert cache.canonicalize(True) is True
    assert cache.canonicalize(3) == 3
    assert cache.canonicalize(2.5) == 2.5
    assert cache.canonicalize("x") == "x"


def test_canonicalize_numpy_scalars_and_arrays():
    assert cache.canonicalize(np.int64(4)) == 4
    assert isinstance(cache.canonicalize(np.int64(4)), int)
    assert cache.canonicalize(np.float64(1.5)) == 1.5
    assert cache.canonicalize(np.bool_(True)) is True
    assert cache.canonicalize(np.arange(3)) == [0, 1, 2]
    assert cache.canonicalize(np.zeros((2, 2))) == [[0.0, 0.0], [0.0, 0.0]]


def test_canonicalize_paths_and_datetimes():
    assert cache.canonicalize(Path("/a/b")) == "/a/b"
    assert cache.canonicalize(datetime.date(2026, 7, 17)) == "2026-07-17"
    assert (
        cache.canonicalize(datetime.datetime(2026, 7, 17, 6, 0))
        == "2026-07-17T06:00:00"
    )


def test_canonicalize_sorts_mappings_and_sets():
    assert list(cache.canonicalize({"b": 1, "a": 2})) == ["a", "b"]
    assert cache.canonicalize({3, 1, 2}) == [1, 2, 3]


def test_canonicalize_mapping_keys_become_strings():
    assert cache.canonicalize({1: "x"}) == {"1": "x"}


def test_canonicalize_nested_structures():
    payload = {"a": [np.int64(1), {"b": np.arange(2)}], "c": (1.0, 2.0)}
    assert cache.canonicalize(payload) == {
        "a": [1, {"b": [0, 1]}],
        "c": [1.0, 2.0],
    }


def test_canonicalize_dataclass():
    @dataclasses.dataclass
    class Cfg:
        nside: int
        name: str

    assert cache.canonicalize(Cfg(64, "x")) == {"nside": 64, "name": "x"}


def test_canonicalize_object_with_isot():
    class FakeTime:
        isot = "2026-07-17T06:00:00.000"

    assert cache.canonicalize(FakeTime()) == "2026-07-17T06:00:00.000"


def test_canonicalize_rejects_unknown_types():
    """Refusing beats folding an object's memory address into the
    fingerprint, which would make every run a cache miss."""

    class Opaque:
        pass

    with pytest.raises(TypeError, match="Cannot fingerprint"):
        cache.canonicalize(Opaque())


# --- fingerprinting ---------------------------------------------------


def test_fingerprint_is_deterministic():
    assert cache.config_fingerprint(CONFIG) == cache.config_fingerprint(
        dict(CONFIG)
    )


def test_fingerprint_length():
    assert len(cache.config_fingerprint(CONFIG)) == 8
    assert len(cache.config_fingerprint(CONFIG, length=16)) == 16
    assert cache.config_fingerprint(CONFIG, length=16).startswith(
        cache.config_fingerprint(CONFIG)
    )


def test_fingerprint_rejects_bad_length():
    with pytest.raises(ValueError, match="length must be >= 1"):
        cache.config_fingerprint(CONFIG, length=0)


def test_fingerprint_ignores_key_order():
    a = {"x": 1, "y": 2}
    b = {"y": 2, "x": 1}
    assert cache.config_fingerprint(a) == cache.config_fingerprint(b)


@pytest.mark.parametrize(
    "changed",
    [
        {"nside": 128, "freqs_mhz": [50.0, 100.0], "model": "dipole"},
        {"nside": 64, "freqs_mhz": [50.0, 101.0], "model": "dipole"},
        {"nside": 64, "freqs_mhz": [50.0, 100.0], "model": "vivaldi"},
        {"nside": 64, "freqs_mhz": [50.0], "model": "dipole"},
    ],
)
def test_any_parameter_change_changes_the_fingerprint(changed):
    assert cache.config_fingerprint(changed) != cache.config_fingerprint(
        CONFIG
    )


def test_fingerprint_distinguishes_numeric_types_by_value_not_type():
    """int 1 and float 1.0 are the same parameter value."""
    assert cache.config_fingerprint({"n": 1}) == cache.config_fingerprint(
        {"n": np.int64(1)}
    )


def test_numpy_array_and_list_fingerprint_alike():
    assert cache.config_fingerprint(
        {"f": np.array([1.0, 2.0])}
    ) == cache.config_fingerprint({"f": [1.0, 2.0]})


def test_npzcache_fingerprint_is_the_module_function():
    assert cache.NpzCache.fingerprint(CONFIG) == cache.config_fingerprint(
        CONFIG
    )


# --- naming -----------------------------------------------------------


def test_filename_layout():
    assert cache.NpzCache.filename("beam", "abc12345") == (
        "beam_abc12345.npz"
    )
    assert cache.NpzCache.filename("beam", "abc12345", tag="ns64") == (
        "beam_abc12345_ns64.npz"
    )


def test_path_creates_the_directory(npz, cache_dir):
    assert not cache_dir.exists()
    p = npz.path("beam", "abc12345")
    assert cache_dir.is_dir()
    assert p.name == "beam_abc12345.npz"


def test_construction_does_not_touch_the_filesystem(cache_dir):
    cache.NpzCache(cache_dir)
    assert not cache_dir.exists()


def test_cache_dir_expands_user():
    c = cache.NpzCache("~/somewhere")
    assert "~" not in str(c.cache_dir)
    assert c.cache_dir.is_absolute()


def test_repr_shows_the_directory(npz, cache_dir):
    assert str(cache_dir) in repr(npz)


# --- save / load round trip -------------------------------------------


def test_round_trip(npz):
    fp = npz.fingerprint(CONFIG)
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    npz.save("beam", fp, config=CONFIG, beam=arr, scale=np.float64(2.5))
    loaded = npz.load("beam", fp, config=CONFIG)
    assert np.array_equal(loaded["beam"], arr)
    assert loaded["scale"] == 2.5


def test_save_returns_the_path_written(npz):
    fp = npz.fingerprint(CONFIG)
    p = npz.save("beam", fp, x=np.zeros(2))
    assert p.exists()
    assert p.name == "beam_" + fp + ".npz"


def test_load_without_config_still_validates_fingerprint(npz):
    fp = npz.fingerprint(CONFIG)
    npz.save("beam", fp, config=CONFIG, x=np.zeros(2))
    assert npz.load("beam", fp) is not None


def test_tag_separates_results_from_the_same_config(npz):
    fp = npz.fingerprint(CONFIG)
    npz.save("beam", fp, tag="ns64", x=np.array([1.0]))
    npz.save("beam", fp, tag="ns128", x=np.array([2.0]))
    assert npz.load("beam", fp, tag="ns64")["x"] == 1.0
    assert npz.load("beam", fp, tag="ns128")["x"] == 2.0


def test_different_names_do_not_collide(npz):
    fp = npz.fingerprint(CONFIG)
    npz.save("beam", fp, x=np.array([1.0]))
    npz.save("sky", fp, x=np.array([2.0]))
    assert npz.load("beam", fp)["x"] == 1.0
    assert npz.load("sky", fp)["x"] == 2.0


def test_reserved_array_names_are_rejected(npz):
    fp = npz.fingerprint(CONFIG)
    with pytest.raises(ValueError, match="reserved"):
        npz.save("beam", fp, config_fp=np.array([1]))
    with pytest.raises(ValueError, match="reserved"):
        npz.save("beam", fp, config_payload=np.array([1]))


# --- staleness --------------------------------------------------------


def test_changed_config_is_a_miss_not_a_wrong_answer(npz):
    """The central guarantee: a cached array never leaks into a run
    with a different config."""
    fp = npz.fingerprint(CONFIG)
    npz.save("beam", fp, config=CONFIG, beam=np.array([1.0, 2.0]))

    new_config = dict(CONFIG, nside=128)
    new_fp = npz.fingerprint(new_config)
    assert new_fp != fp
    assert npz.try_load("beam", new_fp, config=new_config) is None


def test_missing_file_is_a_miss(npz):
    assert npz.try_load("beam", "deadbeef") is None


def test_load_missing_file_raises(npz):
    with pytest.raises(FileNotFoundError):
        npz.load("beam", "deadbeef")


def test_fingerprint_mismatch_raises(npz):
    """A file whose stored fingerprint disagrees with the requested one
    -- i.e. a hash collision or a hand-renamed file."""
    fp = npz.fingerprint(CONFIG)
    npz.save("beam", fp, config=CONFIG, x=np.zeros(2))
    stale = npz.path("beam", fp).rename(npz.path("beam", "0000beef"))
    assert stale.exists()
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        npz.load("beam", "0000beef")


def test_config_payload_mismatch_raises(npz):
    """Same fingerprint, different config: caught by the stored
    payload."""
    fp = "fixedfp0"
    npz.save("beam", fp, config=CONFIG, x=np.zeros(2))
    other = dict(CONFIG, model="vivaldi")
    with pytest.raises(ValueError, match="config payload mismatch"):
        npz.load("beam", fp, config=other)


def test_payload_written_without_config_does_not_block_a_load(npz):
    """A file saved with no config has nothing to contradict."""
    fp = npz.fingerprint(CONFIG)
    npz.save("beam", fp, x=np.zeros(2))
    assert npz.load("beam", fp, config=CONFIG) is not None


def test_try_load_swallows_a_mismatch(npz):
    fp = "fixedfp0"
    npz.save("beam", fp, config=CONFIG, x=np.zeros(2))
    other = dict(CONFIG, model="vivaldi")
    assert npz.try_load("beam", fp, config=other) is None


def test_file_without_fingerprint_is_rejected(npz, cache_dir):
    """A hand-made npz dropped into the cache directory must not be
    trusted as a cache hit."""
    cache_dir.mkdir(parents=True)
    np.savez(cache_dir / "beam_abc12345.npz", x=np.zeros(2))
    with pytest.raises(ValueError, match="no stored fingerprint"):
        npz.load("beam", "abc12345")


# --- module-level wrappers --------------------------------------------


def test_module_level_round_trip(cache_dir):
    fp = cache.config_fingerprint(CONFIG)
    cache.save_result(
        "beam", fp, cache_dir=cache_dir, config=CONFIG, x=np.array([3.0])
    )
    loaded = cache.load_result(
        "beam", fp, cache_dir=cache_dir, config=CONFIG
    )
    assert loaded["x"] == 3.0


def test_module_level_try_load_miss(cache_dir):
    assert (
        cache.try_load_result("beam", "deadbeef", cache_dir=cache_dir)
        is None
    )


def test_cache_path_creates_directory(cache_dir):
    p = cache.cache_path("x.npz", cache_dir=cache_dir)
    assert cache_dir.is_dir()
    assert p == cache_dir / "x.npz"


def test_no_cache_dir_falls_back_to_the_module_default():
    """EIGSEP_CACHE_DIR is read at import time into DEFAULT_CACHE_DIR;
    that is what callers get when they pass no directory."""
    assert cache.NpzCache().cache_dir == Path(
        cache.DEFAULT_CACHE_DIR
    ).expanduser()
    assert cache.NpzCache(None).cache_dir == cache.NpzCache().cache_dir


# --- realistic usage --------------------------------------------------


def test_expensive_computation_runs_once_then_hits(npz):
    calls = []

    def expensive():
        calls.append(1)
        return np.arange(100, dtype=np.float64)

    def get():
        fp = npz.fingerprint(CONFIG)
        hit = npz.try_load("result", fp, config=CONFIG)
        if hit is not None:
            return hit["value"]
        value = expensive()
        npz.save("result", fp, config=CONFIG, value=value)
        return value

    first, second = get(), get()
    assert len(calls) == 1
    assert np.array_equal(first, second)


def test_changing_the_config_recomputes(npz):
    calls = []

    def get(config):
        fp = npz.fingerprint(config)
        hit = npz.try_load("result", fp, config=config)
        if hit is not None:
            return hit["value"]
        calls.append(config["nside"])
        value = np.full(4, float(config["nside"]))
        npz.save("result", fp, config=config, value=value)
        return value

    get(CONFIG)
    get(CONFIG)
    get(dict(CONFIG, nside=128))
    get(dict(CONFIG, nside=128))
    assert calls == [64, 128]
