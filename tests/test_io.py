"""Tests for eigsep_base.io -- the EIGSEP file-format contract."""

import datetime
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import pytest

from eigsep_base import io

# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def corr_header():
    """A minimal header satisfying CORR_HEADER_SCHEMA."""
    return {
        "acc_bins": 2,
        "avg_even_odd": True,
        "nchan": 16,
        "dtype": ">i4",
        "integration_time": 0.25,
        "sample_rate": 500e6,
        "wiring": {"ants": {"antA": {"snap": {"input": 0}}}},
    }


@pytest.fixture
def eig_header():
    """A legacy .eig header. ``pairs`` mixes autos and one cross."""
    nchan, acc_bins = 8, 2
    return {
        "dtype": ">i4",
        "nchan": nchan,
        "acc_bins": acc_bins,
        "pairs": ["0", "1", "01"],
        "sample_rate": 500e6,
        "corr_acc_len": 2**20,
        "sync_time": 1_700_000_000.0,
        "pam_atten": {"0": 8, "1": 8},
        "acc_cnt": [0, 1, 2],
    }


@pytest.fixture
def eig_data(eig_header):
    """Deterministic per-pair arrays matching ``eig_header``."""
    h = eig_header
    ntimes, nchan, acc_bins = 3, h["nchan"], h["acc_bins"]
    rng = np.random.default_rng(0)
    data = {}
    for p in h["pairs"]:
        ncomp = 1 if len(p) == 1 else 2
        data[p] = rng.integers(
            0, 1000, size=(ntimes, acc_bins, nchan, ncomp)
        ).astype(">i4")
    return data


def sensor_entry(name, **fields):
    """One well-formed sensor reading for the given stream."""
    schema = io.SENSOR_SCHEMAS[name]
    defaults = {"sensor_name": name, "status": "update"}
    entry = {}
    for key, typ in schema.items():
        if key in defaults:
            entry[key] = defaults[key]
        elif typ is float:
            entry[key] = 0.0
        elif typ is bool:
            entry[key] = False
        elif typ is int:
            entry[key] = 1
        else:
            entry[key] = "x"
    entry.update(fields)
    return entry


# =====================================================================
# data_shape / reshape_data
# =====================================================================


def test_data_shape_auto_and_cross():
    assert io.data_shape(10, 2, 1024) == (10, 2048)
    # A cross carries real and imaginary parts, so it is twice as long.
    assert io.data_shape(10, 2, 1024, cross=True) == (10, 4096)
    assert io.data_shape(10, 1, 1024) == (10, 1024)


def test_reshape_auto_averages_even_odd():
    # even spectrum then odd spectrum, Fortran-interleaved
    raw = {"0": np.array([[10, 20, 30, 40]], dtype=np.int32)}
    out = io.reshape_data(raw, acc_bins=2, avg_even_odd=True)["0"]
    assert out.shape == (1, 2)
    assert out.dtype == np.int32
    # reshape(-1, 2, order='F') gives [[10, 30], [20, 40]]
    assert np.array_equal(out, [[20, 30]])


def test_reshape_auto_uses_bankers_rounding():
    """np.rint rounds halves to even, so crosses pick up no bias."""
    raw = {"0": np.array([[0, 2, 1, 3]], dtype=np.int32)}
    out = io.reshape_data(raw, acc_bins=2, avg_even_odd=True)["0"]
    # pairs are (0, 1) -> 0.5 -> 0, and (2, 3) -> 2.5 -> 2
    assert np.array_equal(out, [[0, 2]])


def test_reshape_auto_keeps_even_odd_axis_when_not_averaging():
    raw = {"0": np.array([[10, 20, 30, 40]], dtype=np.int32)}
    out = io.reshape_data(raw, acc_bins=2, avg_even_odd=False)["0"]
    assert out.shape == (1, 2, 2)


def test_reshape_cross_splits_real_and_imaginary():
    raw = {"01": np.arange(8, dtype=np.int32)[None, :]}
    out = io.reshape_data(raw, acc_bins=2, avg_even_odd=True)["01"]
    assert out.shape[-1] == 2
    assert out.dtype == np.int32


def test_reshape_cross_returns_complex_on_the_legacy_path():
    raw = {"01": np.arange(8, dtype=np.int32)[None, :]}
    out = io.reshape_data(raw, acc_bins=2, avg_even_odd=False)["01"]
    assert out.dtype.kind == "c"


def test_reshape_acc_bins_one_passes_autos_through():
    raw = {"0": np.arange(4, dtype=np.int32)[None, :]}
    out = io.reshape_data(raw, acc_bins=1)["0"]
    assert np.array_equal(out, [[0, 1, 2, 3]])
    assert out.dtype == np.int32


def test_reshape_acc_bins_one_ignores_avg_even_odd():
    raw = {"0": np.arange(4, dtype=np.int32)[None, :]}
    a = io.reshape_data(raw, acc_bins=1, avg_even_odd=True)["0"]
    b = io.reshape_data(raw, acc_bins=1, avg_even_odd=False)["0"]
    assert np.array_equal(a, b)


def test_reshape_acc_bins_one_splits_cross():
    raw = {"01": np.array([[1, 2, 3, 4]], dtype=np.int32)}
    out = io.reshape_data(raw, acc_bins=1)["01"]
    assert out.shape == (1, 2, 2)
    assert np.array_equal(out[..., 0], [[1, 3]])  # real
    assert np.array_equal(out[..., 1], [[2, 4]])  # imag


def test_reshape_promotes_1d_input_to_2d():
    raw = {"0": np.array([10, 20, 30, 40], dtype=np.int32)}
    assert io.reshape_data(raw, acc_bins=2)["0"].shape == (1, 2)


def test_reshape_preserves_time_axis():
    raw = {"0": np.arange(4 * 5, dtype=np.int32).reshape(5, 4)}
    assert io.reshape_data(raw, acc_bins=2)["0"].shape == (5, 2)


# =====================================================================
# Antenna labelling
# =====================================================================


def test_effective_input_to_ant_plain_wiring():
    wiring = {
        "ants": {
            "a": {"snap": {"input": 0}},
            "b": {"snap": {"input": 1}},
        }
    }
    assert io.effective_input_to_ant(wiring, 0) == {"0": "a", "1": "b"}


def test_effective_input_to_ant_mux_copies_even_into_odd():
    wiring = {
        "ants": {
            "a": {"snap": {"input": 0}},
            "b": {"snap": {"input": 1}},
        }
    }
    # bit0 routes input 0's antenna into input 1
    assert io.effective_input_to_ant(wiring, 0b001) == {"0": "a", "1": "a"}


def test_effective_input_to_ant_mux_bits_map_to_the_right_pairs():
    wiring = {
        "ants": {
            "a": {"snap": {"input": 0}},
            "c": {"snap": {"input": 2}},
            "e": {"snap": {"input": 4}},
        }
    }
    assert io.effective_input_to_ant(wiring, 0b010)["3"] == "c"
    assert io.effective_input_to_ant(wiring, 0b100)["5"] == "e"
    assert io.effective_input_to_ant(wiring, 0b111) == {
        "0": "a",
        "1": "a",
        "2": "c",
        "3": "c",
        "4": "e",
        "5": "e",
    }


def test_effective_input_to_ant_omits_unwired_inputs():
    """A sparse result lets the renderer fall back to the raw key
    instead of inventing an antenna."""
    wiring = {"ants": {"a": {"snap": {"input": 0}}}}
    assert io.effective_input_to_ant(wiring, 0) == {"0": "a"}


def test_effective_input_to_ant_copy_from_unwired_source_is_omitted():
    wiring = {"ants": {"b": {"snap": {"input": 1}}}}
    # bit0 says input 1 copies input 0, but input 0 is un-wired.
    assert "1" not in io.effective_input_to_ant(wiring, 0b001)


@pytest.mark.parametrize("wiring", [None, {}, {"ants": None}, {"ants": {}}])
def test_effective_input_to_ant_empty_wiring(wiring):
    assert io.effective_input_to_ant(wiring, 0) == {}


def test_pair_label_auto_and_cross():
    mapping = {"0": "antA", "2": "antB"}
    assert io.pair_label("0", mapping) == "antA [0]"
    assert io.pair_label("02", mapping) == "antA / antB [02]"


def test_pair_label_returns_none_when_unmapped():
    assert io.pair_label("0", {}) is None
    assert io.pair_label("02", {"0": "antA"}) is None
    assert io.pair_label("012", {"0": "a", "1": "b", "2": "c"}) is None


def test_corr_pair_labels_prefers_the_written_map():
    header = {"input_to_ant": {0: "antA"}, "wiring": {"ants": {}}}
    assert io.corr_pair_labels(header, ["0"]) == {"0": "antA [0]"}


def test_corr_pair_labels_falls_back_to_wiring():
    header = {
        "wiring": {"ants": {"antA": {"snap": {"input": 0}}}},
        "adc_mux_sel": 0,
    }
    assert io.corr_pair_labels(header, ["0"]) == {"0": "antA [0]"}


def test_corr_pair_labels_handles_empty_header():
    assert io.corr_pair_labels(None, ["0"]) == {"0": None}


# =====================================================================
# append_corr_header
# =====================================================================


def test_append_corr_header_computes_times_and_freqs(corr_header):
    acc_cnts = np.arange(3)
    out = io.append_corr_header(corr_header, acc_cnts, 1_700_000_000.0)
    assert np.array_equal(out["acc_cnt"], acc_cnts)
    assert np.allclose(
        out["times"], 1_700_000_000.0 + np.array([0, 0.25, 0.5])
    )
    assert out["freqs"].shape == (16,)
    assert out["dfreq"] == pytest.approx(500e6 / 32)


def test_append_corr_header_does_not_mutate_the_input(corr_header):
    before = dict(corr_header)
    io.append_corr_header(corr_header, np.arange(2), 0.0)
    assert corr_header == before


def test_append_corr_header_survives_a_missing_field(corr_header, caplog):
    """Corr data is sacred: a broken header must not raise."""
    del corr_header["integration_time"]
    with caplog.at_level(logging.ERROR):
        out = io.append_corr_header(corr_header, np.arange(2), 0.0)
    assert "times" not in out
    assert "freqs" in out  # the other computed field still lands
    assert "cannot compute 'times'" in caplog.text


def test_append_corr_header_survives_a_malformed_sample_rate(
    corr_header, caplog
):
    corr_header["sample_rate"] = "not a number"
    with caplog.at_level(logging.ERROR):
        out = io.append_corr_header(corr_header, np.arange(2), 0.0)
    assert "freqs" not in out and "dfreq" not in out
    assert "times" in out
    assert "cannot compute 'freqs'" in caplog.text


def test_append_corr_header_skips_linear_range_without_a_loader(
    corr_header,
):
    """Default: no loader injected, so no bounds are written."""
    corr_header["linear_range_file"] = "product.npz"
    out = io.append_corr_header(corr_header, np.arange(2), 0.0)
    assert "linear_range_min" not in out


def test_append_corr_header_injects_linear_range_bounds(corr_header):
    corr_header["linear_range_file"] = "product.npz"
    product = {
        "header": {"nchan": 16},
        "linear_min": np.zeros(16),
        "linear_max": np.full(16, 1e6),
    }
    out = io.append_corr_header(
        corr_header,
        np.arange(2),
        0.0,
        load_linear_range=lambda f: product,
        validate_operating_point=lambda ph, lh: [],
    )
    assert np.array_equal(out["linear_range_min"], product["linear_min"])
    assert np.array_equal(out["linear_range_max"], product["linear_max"])


def test_append_corr_header_omits_bounds_on_operating_point_mismatch(
    corr_header, caplog
):
    """Bounds measured at another operating point are junk."""
    corr_header["linear_range_file"] = "product.npz"
    product = {
        "header": {"nchan": 32},
        "linear_min": np.zeros(16),
        "linear_max": np.full(16, 1e6),
    }
    with caplog.at_level(logging.ERROR):
        out = io.append_corr_header(
            corr_header,
            np.arange(2),
            0.0,
            load_linear_range=lambda f: product,
            validate_operating_point=lambda ph, lh: ["nchan 32 != 16"],
        )
    assert "linear_range_min" not in out
    assert "operating-point mismatch" in caplog.text


def test_append_corr_header_omits_bounds_when_the_loader_fails(
    corr_header, caplog
):
    corr_header["linear_range_file"] = "missing.npz"

    def boom(fname):
        raise ValueError("no such product")

    with caplog.at_level(logging.ERROR):
        out = io.append_corr_header(
            corr_header,
            np.arange(2),
            0.0,
            load_linear_range=boom,
            validate_operating_point=lambda ph, lh: [],
        )
    assert "linear_range_min" not in out
    assert "Linear-range contract violation" in caplog.text


def test_append_corr_header_requires_a_validator_with_a_loader(
    corr_header,
):
    """Writing bounds without an operating-point check is never OK."""
    corr_header["linear_range_file"] = "product.npz"
    with pytest.raises(ValueError, match="validate_operating_point"):
        io.append_corr_header(
            corr_header,
            np.arange(2),
            0.0,
            load_linear_range=lambda f: {},
        )


# =====================================================================
# HDF5 round trips
# =====================================================================


def test_write_read_round_trip(tmp_path, corr_header):
    fname = tmp_path / "corr.h5"
    data = {"0": np.arange(32, dtype=np.int32).reshape(2, 16)}
    io.write_hdf5(fname, data, corr_header)
    rd, rh, rm = io.read_hdf5(fname)
    assert np.array_equal(rd["0"], data["0"])
    assert rh["nchan"] == 16
    assert rh["dtype"] == ">i4"
    assert rh["integration_time"] == pytest.approx(0.25)
    assert rm == {}


def test_header_scalar_types_survive_the_round_trip(tmp_path):
    fname = tmp_path / "f.h5"
    header = {
        "a_bool": True,
        "a_false": False,
        "an_int": 7,
        "a_float": 1.5,
        "a_str": "hello",
    }
    io.write_hdf5(fname, {"0": np.zeros(2)}, header)
    _, rh, _ = io.read_hdf5(fname)
    assert bool(rh["a_bool"]) is True
    assert bool(rh["a_false"]) is False
    assert rh["an_int"] == 7
    assert rh["a_float"] == 1.5
    assert rh["a_str"] == "hello"


def test_bool_is_stored_as_bool_not_int(tmp_path):
    """Python bool subclasses int; the writer must check bool first."""
    fname = tmp_path / "f.h5"
    io.write_hdf5(fname, {"0": np.zeros(2)}, {"flag": True})
    with h5py.File(fname, "r") as f:
        assert f["header"].attrs["flag"].dtype == np.bool_


def test_numpy_scalars_are_layout_equivalent_to_python_natives(tmp_path):
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    io.write_hdf5(a, {"0": np.zeros(1)}, {"n": 5, "x": 1.5, "f": True})
    io.write_hdf5(
        b,
        {"0": np.zeros(1)},
        {"n": np.int32(5), "x": np.float32(1.5), "f": np.bool_(True)},
    )
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        for key in ("n", "x", "f"):
            assert fa["header"].attrs[key].dtype == (
                fb["header"].attrs[key].dtype
            )


def test_header_containers_round_trip(tmp_path):
    fname = tmp_path / "f.h5"
    header = {
        "a_list": [1, 2, 3],
        "a_dict": {"x": 1, "y": "two"},
        "an_array": np.arange(4),
        "a_tuple": (1, 2),
    }
    io.write_hdf5(fname, {"0": np.zeros(2)}, header)
    _, rh, _ = io.read_hdf5(fname)
    assert list(rh["a_list"]) == [1, 2, 3]
    assert rh["a_dict"] == {"x": 1, "y": "two"}
    assert np.array_equal(rh["an_array"], np.arange(4))
    assert list(rh["a_tuple"]) == [1, 2]


def test_header_special_types_are_coerced(tmp_path):
    fname = tmp_path / "f.h5"
    header = {
        "path": Path("/data/x.h5"),
        "when": datetime.datetime(2026, 7, 17, 6, 0),
        "items": {"b", "a"},
    }
    io.write_hdf5(fname, {"0": np.zeros(2)}, header)
    _, rh, _ = io.read_hdf5(fname)
    assert rh["path"] == "/data/x.h5"
    assert rh["when"] == "2026-07-17T06:00:00"
    assert list(rh["items"]) == ["a", "b"]  # set -> sorted list


def test_header_complex_round_trips(tmp_path):
    fname = tmp_path / "f.h5"
    io.write_hdf5(fname, {"0": np.zeros(2)}, {"z": 1.0 + 2.0j})
    _, rh, _ = io.read_hdf5(fname)
    assert rh["z"] == 1.0 + 2.0j


def test_nested_wiring_dict_round_trips(tmp_path, corr_header):
    fname = tmp_path / "f.h5"
    io.write_hdf5(fname, {"0": np.zeros(2)}, corr_header)
    _, rh, _ = io.read_hdf5(fname)
    assert rh["wiring"] == corr_header["wiring"]


def test_data_is_written_before_a_bad_header_field(tmp_path, caplog):
    """Corr data is sacred: an unserializable header field is logged
    and skipped, and the data still lands."""
    fname = tmp_path / "f.h5"
    data = {"0": np.arange(4, dtype=np.int32)}
    header = {"good": 1, "bad": object()}
    with caplog.at_level(logging.ERROR):
        io.write_hdf5(fname, data, header)
    rd, rh, _ = io.read_hdf5(fname)
    assert np.array_equal(rd["0"], data["0"])
    assert rh["good"] == 1
    assert "bad" not in rh
    assert "Header contract violation" in caplog.text


def test_a_bad_metadata_field_is_skipped(tmp_path, caplog):
    fname = tmp_path / "f.h5"
    with caplog.at_level(logging.ERROR):
        io.write_hdf5(
            fname,
            {"0": np.zeros(2)},
            {},
            metadata={"good": [1, 2], "bad": object()},
        )
    _, _, rm = io.read_hdf5(fname)
    assert list(rm["good"]) == [1, 2]
    assert "bad" not in rm
    assert "Metadata contract violation" in caplog.text


def test_cross_data_is_rebuilt_as_complex_on_read(tmp_path):
    """Crosses are stored as int32 (re, im); readers want complex."""
    fname = tmp_path / "f.h5"
    stored = np.array([[[1, 2], [3, 4]]], dtype=np.int32)  # (1, 2, 2)
    io.write_hdf5(fname, {"01": stored}, {})
    rd, _, _ = io.read_hdf5(fname)
    assert rd["01"].dtype.kind == "c"
    assert np.array_equal(rd["01"], [[1 + 2j, 3 + 4j]])


def test_legacy_complex_cross_storage_is_returned_as_is(tmp_path):
    fname = tmp_path / "f.h5"
    stored = np.array([[1 + 2j, 3 + 4j]])
    io.write_hdf5(fname, {"01": stored}, {})
    rd, _, _ = io.read_hdf5(fname)
    assert np.array_equal(rd["01"], stored)


def test_autos_are_not_mistaken_for_crosses(tmp_path):
    """An auto that happens to have 2 channels must stay real."""
    fname = tmp_path / "f.h5"
    io.write_hdf5(fname, {"0": np.array([[1, 2]], dtype=np.float64)}, {})
    rd, _, _ = io.read_hdf5(fname)
    assert rd["0"].dtype.kind == "f"


def test_metadata_round_trips_with_none_preserved(tmp_path):
    """A dropped sensor reading must read back as None, not 0."""
    fname = tmp_path / "f.h5"
    metadata = {"imu_el": [{"yaw": 1.0}, {"yaw": None}]}
    io.write_hdf5(fname, {"0": np.zeros(2)}, {}, metadata=metadata)
    _, _, rm = io.read_hdf5(fname)
    assert rm["imu_el"] == [{"yaw": 1.0}, {"yaw": None}]


# =====================================================================
# Standalone metadata files
# =====================================================================


def test_metadata_file_round_trip(tmp_path):
    fname = tmp_path / "metadata.h5"
    metadata = {
        "imu_el": [
            {"yaw": 1.0, "_ts_unix": 100.0},
            {"yaw": None, "_ts_unix": 101.0},
        ],
        "lidar": [{"distance_m": 2.5, "_ts_unix": 100.5}],
    }
    io.write_metadata_hdf5(fname, metadata)
    assert io.read_metadata_hdf5(fname) == metadata


def test_metadata_file_matches_the_corr_file_metadata_group(tmp_path):
    """Same on-disk shape, so one reader works for both."""
    metadata = {"lidar": [{"distance_m": 2.5}]}
    standalone, corr = tmp_path / "m.h5", tmp_path / "c.h5"
    io.write_metadata_hdf5(standalone, metadata)
    io.write_hdf5(corr, {"0": np.zeros(2)}, {}, metadata=metadata)
    _, _, from_corr = io.read_hdf5(corr)
    assert io.read_metadata_hdf5(standalone) == from_corr


def test_read_metadata_file_with_no_group(tmp_path):
    fname = tmp_path / "empty.h5"
    with h5py.File(fname, "w") as f:
        f.create_group("data")
    assert io.read_metadata_hdf5(fname) == {}


def test_metadata_file_skips_a_bad_stream(tmp_path, caplog):
    fname = tmp_path / "m.h5"
    with caplog.at_level(logging.ERROR):
        io.write_metadata_hdf5(fname, {"good": [{"x": 1}], "bad": object()})
    out = io.read_metadata_hdf5(fname)
    assert out["good"] == [{"x": 1}]
    assert "bad" not in out


# =====================================================================
# S11 files
# =====================================================================


def test_s11_round_trip_separates_cal_data(tmp_path):
    data = {"ant": np.array([1 + 1j, 2 + 2j])}
    cal = {
        "VNAO": np.array([1 + 0j, 1 + 0j]),
        "VNAS": np.array([-1 + 0j, -1 + 0j]),
        "VNAL": np.array([0 + 0j, 0 + 0j]),
    }
    header = {"fstart": 1e6, "fstop": 2e8, "npoints": 2}
    io.write_s11_file(
        data, header, cal_data=cal, fname="ants11.h5", save_dir=tmp_path
    )
    rd, rcal, rh, _ = io.read_s11_file(tmp_path / "ants11.h5")
    assert set(rd) == {"ant"}
    assert np.array_equal(rd["ant"], data["ant"])
    assert set(rcal) == {"VNAO", "VNAS", "VNAL"}
    assert np.array_equal(rcal["VNAO"], cal["VNAO"])
    assert rh["npoints"] == 2


def test_s11_without_cal_data_gives_an_empty_cal_dict(tmp_path):
    io.write_s11_file(
        {"rec": np.array([1 + 0j])},
        {"mode": "rec"},
        fname="recs11.h5",
        save_dir=tmp_path,
    )
    rd, rcal, _, _ = io.read_s11_file(tmp_path / "recs11.h5")
    assert rcal == {}
    assert set(rd) == {"rec"}


def test_s11_autogenerated_name_encodes_the_mode(tmp_path):
    io.write_s11_file({"ant": np.array([1 + 0j])}, {}, save_dir=tmp_path)
    names = [p.name for p in tmp_path.glob("*.h5")]
    assert len(names) == 1
    assert names[0].startswith("ants11_")

    io.write_s11_file({"rec": np.array([1 + 0j])}, {}, save_dir=tmp_path)
    assert any(p.name.startswith("recs11_") for p in tmp_path.glob("*.h5"))


def test_s11_autogenerated_names_do_not_collide(tmp_path):
    """Two writes in the same second must not clobber each other."""
    for _ in range(3):
        io.write_s11_file({"ant": np.array([1 + 0j])}, {}, save_dir=tmp_path)
    assert len(list(tmp_path.glob("ants11_*.h5"))) == 3


def test_s11_absolute_fname_ignores_save_dir(tmp_path):
    target = tmp_path / "explicit.h5"
    other = tmp_path / "elsewhere"
    other.mkdir()
    io.write_s11_file(
        {"ant": np.array([1 + 0j])}, {}, fname=target, save_dir=other
    )
    assert target.exists()
    assert list(other.glob("*.h5")) == []


def test_s11_cal_prefix_is_the_disk_convention(tmp_path):
    io.write_s11_file(
        {"ant": np.array([1 + 0j])},
        {},
        cal_data={"VNAO": np.array([1 + 0j])},
        fname="x.h5",
        save_dir=tmp_path,
    )
    with h5py.File(tmp_path / "x.h5", "r") as f:
        assert "cal:VNAO" in f["data"]


# =====================================================================
# Schema validation
# =====================================================================


def test_valid_corr_header_has_no_violations(corr_header):
    assert io.validate_corr_header(corr_header) == []


def test_corr_header_missing_key(corr_header):
    del corr_header["nchan"]
    violations = io.validate_corr_header(corr_header)
    assert any("missing key 'nchan'" in v for v in violations)


def test_corr_header_wrong_type(corr_header):
    corr_header["nchan"] = "sixteen"
    violations = io.validate_corr_header(corr_header)
    assert any("expected int, got str" in v for v in violations)


def test_corr_header_bool_is_not_an_int(corr_header):
    """bool subclasses int, so acc_bins=True must be rejected."""
    corr_header["acc_bins"] = True
    assert any("acc_bins" in v for v in io.validate_corr_header(corr_header))


def test_corr_header_accepts_numpy_scalars_from_a_round_trip(corr_header):
    corr_header["nchan"] = np.int64(16)
    corr_header["avg_even_odd"] = np.bool_(True)
    corr_header["sample_rate"] = np.float64(500e6)
    assert io.validate_corr_header(corr_header) == []


def test_corr_header_accepts_an_int_for_a_float_field(corr_header):
    corr_header["integration_time"] = 1
    assert io.validate_corr_header(corr_header) == []


def test_corr_header_dtype_must_parse_as_numpy(corr_header):
    corr_header["dtype"] = "not_a_dtype"
    violations = io.validate_corr_header(corr_header)
    assert any("cannot parse" in v for v in violations)


def test_corr_header_survives_a_round_trip_through_hdf5(tmp_path, corr_header):
    """The schema must still pass after h5py type coercion."""
    fname = tmp_path / "f.h5"
    io.write_hdf5(fname, {"0": np.zeros(2)}, corr_header)
    _, rh, _ = io.read_hdf5(fname)
    assert io.validate_corr_header(rh) == []


def test_sensor_schema_validation_accepts_a_good_entry():
    assert (
        io.validate_metadata(sensor_entry("lidar"), io.SENSOR_SCHEMAS["lidar"])
        == []
    )


def test_sensor_schema_reports_missing_and_extra_keys():
    entry = sensor_entry("lidar")
    del entry["distance_m"]
    entry["surprise"] = 1
    violations = io.validate_metadata(entry, io.SENSOR_SCHEMAS["lidar"])
    assert any("missing keys" in v for v in violations)
    assert any("extra keys" in v for v in violations)


def test_sensor_schema_allows_none_for_any_field():
    entry = sensor_entry("lidar", distance_m=None)
    assert io.validate_metadata(entry, io.SENSOR_SCHEMAS["lidar"]) == []


def test_sensor_schema_float_field_rejects_an_int():
    """Strict, so contract drift surfaces here rather than as a silent
    None at write time."""
    entry = sensor_entry("lidar", distance_m=2)
    violations = io.validate_metadata(entry, io.SENSOR_SCHEMAS["lidar"])
    assert any("expected float, got int" in v for v in violations)


def test_adc_stats_schema_covers_all_twelve_cores():
    schema = io.SENSOR_SCHEMAS["adc_stats"]
    for n in range(6):
        for c in range(2):
            for stat in ("mean", "power", "rms"):
                assert f"input{n}_core{c}_{stat}" in schema
    assert len(schema) == 2 + 36


def test_vna_s11_header_schema_accepts_a_good_header():
    header = {
        "fstart": 1e6,
        "fstop": 2e8,
        "npoints": 3,
        "ifbw": 100.0,
        "power_dBm": -10.0,
        "mode": "ant",
        "metadata_snapshot_unix": 1_700_000_000.0,
        "freqs": np.linspace(1e6, 2e8, 3),
    }
    assert io.validate_vna_s11_header(header) == []


def test_vna_s11_header_freqs_length_must_match_npoints():
    header = {
        "fstart": 1e6,
        "fstop": 2e8,
        "npoints": 5,
        "ifbw": 100.0,
        "power_dBm": -10.0,
        "mode": "ant",
        "metadata_snapshot_unix": 1.0,
        "freqs": np.linspace(1e6, 2e8, 3),
    }
    violations = io.validate_vna_s11_header(header)
    assert any("does not match npoints" in v for v in violations)


def test_vna_s11_header_accepts_freqs_as_a_list():
    """freqs comes back as a list after a JSON round trip."""
    header = {
        "fstart": 1e6,
        "fstop": 2e8,
        "npoints": 2,
        "ifbw": 100.0,
        "power_dBm": -10.0,
        "mode": "rec",
        "metadata_snapshot_unix": 1.0,
        "freqs": [1e6, 2e8],
    }
    assert io.validate_vna_s11_header(header) == []


def test_vna_s11_header_rejects_an_unknown_mode():
    header = {
        "fstart": 1e6,
        "fstop": 2e8,
        "npoints": 1,
        "ifbw": 100.0,
        "power_dBm": -10.0,
        "mode": "bogus",
        "metadata_snapshot_unix": 1.0,
        "freqs": [1e6],
    }
    violations = io.validate_vna_s11_header(header)
    assert any("key 'mode'" in v for v in violations)


def test_vna_s11_data_requires_the_per_mode_duts():
    data = {k: np.array([1 + 0j]) for k in io.VNA_S11_CAL_KEYS}
    data["rec"] = np.array([1 + 0j])
    assert io.validate_vna_s11_data(data, "rec") == []


def test_vna_s11_ant_mode_requires_all_six_duts():
    data = {k: np.array([1 + 0j]) for k in io.VNA_S11_CAL_KEYS}
    for k in io.VNA_S11_MODE_DATA_KEYS["ant"]:
        data[k] = np.array([1 + 0j])
    assert io.validate_vna_s11_data(data, "ant") == []


def test_vna_s11_data_reports_missing_cal_standards():
    violations = io.validate_vna_s11_data({"rec": np.array([1 + 0j])}, "rec")
    assert any("missing keys" in v for v in violations)


def test_vna_s11_data_requires_complex_arrays():
    data = {k: np.array([1 + 0j]) for k in io.VNA_S11_CAL_KEYS}
    data["rec"] = np.array([1.0])  # real, not complex
    violations = io.validate_vna_s11_data(data, "rec")
    assert any("expected complex dtype" in v for v in violations)


def test_vna_s11_data_rejects_an_unknown_mode():
    violations = io.validate_vna_s11_data({}, "bogus")
    assert any("unknown mode" in v for v in violations)


def test_vna_s11_cal_keys_use_the_disk_prefix():
    assert all(k.startswith("cal:") for k in io.VNA_S11_CAL_KEYS)


# =====================================================================
# Metadata reduction
# =====================================================================


def test_avg_metadata_empty_or_malformed():
    assert io.avg_metadata([]) is None
    assert io.avg_metadata(["not a dict"]) is None


def test_avg_metadata_averages_floats():
    entries = [
        sensor_entry("lidar", distance_m=1.0),
        sensor_entry("lidar", distance_m=3.0),
    ]
    assert io.avg_metadata(entries)["distance_m"] == pytest.approx(2.0)


def test_avg_metadata_ints_reduce_by_min():
    entries = [
        sensor_entry("motor", boot_id=5),
        sensor_entry("motor", boot_id=9),
    ]
    assert io.avg_metadata(entries)["boot_id"] == 5


def test_reject_counters_reduce_by_max():
    """min would wash a mid-integration burst back to zero."""
    entries = [
        sensor_entry("tempctrl_load", sensor_rejects=0),
        sensor_entry("tempctrl_load", sensor_rejects=7),
        sensor_entry("tempctrl_load", sensor_rejects=0),
    ]
    assert io.avg_metadata(entries)["sensor_rejects"] == 7


def test_bools_reduce_by_any():
    entries = [
        sensor_entry("tempctrl_load", stall_tripped=False),
        sensor_entry("tempctrl_load", stall_tripped=True),
    ]
    assert io.avg_metadata(entries)["stall_tripped"] is True


def test_strings_collapse_to_unknown_when_they_disagree():
    entries = [
        sensor_entry("potmon", sp1_term_name="SHORT"),
        sensor_entry("potmon", sp1_term_name="OPEN"),
    ]
    assert io.avg_metadata(entries)["sp1_term_name"] == "UNKNOWN"


def test_unanimous_strings_pass_through():
    entries = [
        sensor_entry("potmon", sp1_term_name="SHORT"),
        sensor_entry("potmon", sp1_term_name="SHORT"),
    ]
    assert io.avg_metadata(entries)["sp1_term_name"] == "SHORT"


def test_status_collapses_to_error_if_any_sample_errored():
    """The integration-level fault flag downstream keys on."""
    entries = [
        sensor_entry("lidar", distance_m=1.0),
        sensor_entry("lidar", status="error", distance_m=999.0),
    ]
    avg = io.avg_metadata(entries)
    assert avg["status"] == "error"
    # The errored sample's value must not pollute the average.
    assert avg["distance_m"] == pytest.approx(1.0)


def test_all_none_reduces_to_none():
    entries = [
        sensor_entry("lidar", distance_m=None),
        sensor_entry("lidar", distance_m=None),
    ]
    assert io.avg_metadata(entries)["distance_m"] is None


def test_all_errored_reduces_values_to_none():
    entries = [sensor_entry("lidar", status="error", distance_m=1.0)]
    avg = io.avg_metadata(entries)
    assert avg["status"] == "error"
    assert avg["distance_m"] is None


def test_rfswitch_returns_the_state_name():
    entries = [
        sensor_entry("rfswitch", sw_state_name="RFANT"),
        sensor_entry("rfswitch", sw_state_name="RFANT"),
    ]
    assert io.avg_metadata(entries) == "RFANT"


def test_rfswitch_returns_unknown_when_the_state_changes():
    entries = [
        sensor_entry("rfswitch", sw_state_name="RFANT"),
        sensor_entry("rfswitch", sw_state_name="VNAO"),
    ]
    assert io.avg_metadata(entries) == "UNKNOWN"


def test_rfswitch_returns_unknown_on_an_error():
    entries = [sensor_entry("rfswitch", status="error")]
    assert io.avg_metadata(entries) == "UNKNOWN"


def test_invariant_disagreement_is_logged(caplog):
    io._last_invariant_log.clear()
    entries = [
        sensor_entry("motor", boot_id=1),
        sensor_entry("motor", boot_id=2),
    ]
    with caplog.at_level(logging.ERROR):
        io.avg_metadata(entries)
    assert "boot_id" in caplog.text
    assert "power-cycled" in caplog.text


def test_producer_invariant_disagreement_gets_a_different_diagnosis(
    caplog,
):
    io._last_invariant_log.clear()
    entries = [
        sensor_entry("motor", app_id=3),
        sensor_entry("motor", app_id=6),
    ]
    with caplog.at_level(logging.ERROR):
        io.avg_metadata(entries)
    assert "check the producer" in caplog.text


def test_invariant_logging_is_throttled(caplog):
    io._last_invariant_log.clear()
    entries = [
        sensor_entry("motor", boot_id=1),
        sensor_entry("motor", boot_id=2),
    ]
    with caplog.at_level(logging.ERROR):
        for _ in range(5):
            io.avg_metadata(entries)
    assert caplog.text.count("Invariant metadata field") == 1


def test_unknown_sensor_warns_but_still_reduces(caplog):
    entries = [
        {"sensor_name": "mystery", "status": "update", "v": 1.0},
        {"sensor_name": "mystery", "status": "update", "v": 3.0},
    ]
    with caplog.at_level(logging.WARNING):
        avg = io.avg_metadata(entries)
    assert "No schema for sensor 'mystery'" in caplog.text
    assert avg["v"] == pytest.approx(2.0)


def test_schemaless_reduction_sniffs_types():
    entries = [
        {
            "sensor_name": "m",
            "status": "update",
            "f": 1.0,
            "b": False,
            "i": 4,
            "s": "a",
        },
        {
            "sensor_name": "m",
            "status": "update",
            "f": 3.0,
            "b": True,
            "i": 2,
            "s": "b",
        },
    ]
    avg = io.avg_metadata(entries)
    assert avg["f"] == pytest.approx(2.0)
    assert avg["b"] is True
    assert avg["i"] == 2
    assert avg["s"] == "UNKNOWN"


def test_schema_violation_is_warned_but_not_fatal(caplog):
    entries = [sensor_entry("lidar", distance_m=2)]  # int, not float
    with caplog.at_level(logging.WARNING):
        avg = io.avg_metadata(entries)
    assert "Metadata contract violation" in caplog.text
    assert avg is not None


# =====================================================================
# Legacy .eig format
# =====================================================================


def test_build_dtype_from_string_and_pair():
    assert io.build_dtype(">i4") == np.dtype(">i4")
    assert io.build_dtype("int32", ">") == np.dtype(">i4")
    assert io.build_dtype("int32", "<") == np.dtype("<i4")


def test_eig_round_trip(tmp_path, eig_header, eig_data):
    fname = tmp_path / "data.eig"
    io.write_file(fname, eig_header, eig_data)
    header, data = io.read_file(fname)
    assert set(data) == set(eig_data)
    for p in eig_data:
        assert np.array_equal(data[p], eig_data[p])
    assert header["nchan"] == eig_header["nchan"]
    assert header["pairs"] == eig_header["pairs"]


def test_eig_header_augmented_with_computed_fields(
    tmp_path, eig_header, eig_data
):
    fname = tmp_path / "data.eig"
    io.write_file(fname, eig_header, eig_data)
    header = io.read_header(fname)
    assert header["nspec"] == 3
    assert header["freqs"].shape == (eig_header["nchan"],)
    assert header["dfreq"] == pytest.approx(500e6 / (2 * eig_header["nchan"]))
    assert header["inttime"] == pytest.approx(2**20 * 2 / 500e6)
    assert np.allclose(
        header["times"],
        eig_header["sync_time"] + np.arange(3) * header["inttime"],
    )
    assert header["filesize"] > 0
    assert Path(header["filename"]).name == "data.eig"


def test_eig_pam_atten_keys_become_ints(tmp_path, eig_header, eig_data):
    fname = tmp_path / "data.eig"
    io.write_file(fname, eig_header, eig_data)
    header = io.read_header(fname)
    assert set(header["pam_atten"]) == {0, 1}


def test_eig_autos_and_crosses_have_the_right_shapes(
    tmp_path, eig_header, eig_data
):
    fname = tmp_path / "data.eig"
    io.write_file(fname, eig_header, eig_data)
    _, data = io.read_file(fname)
    nchan, acc_bins = eig_header["nchan"], eig_header["acc_bins"]
    assert data["0"].shape == (3, acc_bins, nchan, 1)
    assert data["01"].shape == (3, acc_bins, nchan, 2)


def test_eig_skip_and_nspec(tmp_path, eig_header, eig_data):
    fname = tmp_path / "data.eig"
    io.write_file(fname, eig_header, eig_data)
    _, data = io.read_file(fname, nspec=1, skip=1)
    assert data["0"].shape[0] == 1
    assert np.array_equal(data["0"][0], eig_data["0"][1])


def test_eig_pair_offsets_account_for_cross_width():
    offsets = io.calc_pair_offsets(["0", "01"], 2, 8, ">i4")
    itemsize = 4
    auto_len = itemsize * 2 * 8
    assert list(offsets) == [0, auto_len, auto_len + 2 * auto_len]


def test_eig_header_pack_unpack_round_trip(eig_header):
    packed = io.pack_raw_header(eig_header)
    unpacked = io.unpack_raw_header(packed)
    assert unpacked["nchan"] == eig_header["nchan"]
    assert np.array_equal(unpacked["acc_cnt"], np.array(eig_header["acc_cnt"]))


def test_eig_pack_raw_header_does_not_mutate_its_input(eig_header):
    before = list(eig_header["acc_cnt"])
    io.pack_raw_header(eig_header)
    assert eig_header["acc_cnt"] == before


def test_eig_unpack_raw_data_auto_vs_cross():
    raw = np.arange(2 * 4, dtype=">i4").tobytes()
    auto = io.unpack_raw_data(raw, "0", acc_bins=2, nchan=4)
    assert auto.shape == (1, 2, 4, 1)
    cross = io.unpack_raw_data(raw, "01", acc_bins=1, nchan=4)
    assert cross.shape == (1, 1, 4, 2)


def test_eig_write_accepts_prepacked_bytes(tmp_path, eig_header, eig_data):
    a, b = tmp_path / "a.eig", tmp_path / "b.eig"
    io.write_file(a, eig_header, eig_data)
    io.write_file(b, eig_header, io.pack_data(eig_data, eig_header))
    assert a.read_bytes() == b.read_bytes()


def test_eig_pack_corr_data_concatenates_in_pair_order():
    header = {"pairs": ["0", "1"]}
    frames = [{"0": b"aa", "1": b"bb"}, {"0": b"cc", "1": b"dd"}]
    assert io.pack_corr_data(frames, header) == b"aabbccdd"


def test_eig_read_file_accepts_a_precomputed_header(
    tmp_path, eig_header, eig_data
):
    fname = tmp_path / "data.eig"
    io.write_file(fname, eig_header, eig_data)
    header = io.read_header(fname)
    _, data = io.read_file(fname, header=header)
    assert np.array_equal(data["0"], eig_data["0"])


def test_eig_aliases_point_at_the_same_functions():
    assert io.read_eig_file is io.read_file
    assert io.write_eig_file is io.write_file
    assert io.read_eig_header is io.read_header


def test_eig_header_is_valid_json(tmp_path, eig_header, eig_data):
    """The header stays human-inspectable."""
    fname = tmp_path / "data.eig"
    io.write_file(fname, eig_header, eig_data)
    with open(fname, "rb") as fh:
        size = io._read_header_size(fh)
        blob = fh.read(size).decode("utf-8")
    assert json.loads(blob)["nchan"] == eig_header["nchan"]


# =====================================================================
# Package-level guarantees
# =====================================================================


def test_io_imports_no_jax():
    import sys

    assert "jax" not in sys.modules


def test_read_hdf5_returns_plain_numpy(tmp_path, corr_header):
    """No domain objects: dicts of ndarrays and scalars only."""
    fname = tmp_path / "f.h5"
    io.write_hdf5(fname, {"0": np.zeros((2, 4), dtype=np.int32)}, corr_header)
    data, header, metadata = io.read_hdf5(fname)
    assert isinstance(data, dict)
    assert isinstance(header, dict)
    assert isinstance(metadata, dict)
    assert all(isinstance(v, np.ndarray) for v in data.values())
