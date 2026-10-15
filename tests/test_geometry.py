"""Synthetic sweeps with awkward geometry: written, read back, compared with
the geometry computed directly from the imgCIF axes."""

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pytest

from cbf2nxmx.compare import goniometer_matrices
from cbf2nxmx.frames import compose
from cbf2nxmx.geometry import GeometryError, build_sweep
from cbf2nxmx.nxmx import WriteOptions, panel_in_imgcif, write_nxmx
from cbf2nxmx.readback import read_models

from synth import make_cbf

KAPPA = [0.0, 0.6428, 0.766]  # unnormalised on purpose: |v| != 1


def kappa_axes():
    return [
        {"id": "GON_OMEGA", "type": "rotation", "equipment": "goniometer", "vector": (1, 0, 0)},
        {"id": "GON_KAPPA", "type": "rotation", "equipment": "goniometer",
         "depends_on": "GON_OMEGA", "vector": KAPPA},
        {"id": "GON_PHI", "type": "rotation", "equipment": "goniometer",
         "depends_on": "GON_KAPPA", "vector": (1, 0, 0)},
        {"id": "SOURCE", "type": "general", "equipment": "source", "vector": (0, 0, 1)},
        {"id": "GRAVITY", "type": "general", "equipment": "gravity", "vector": (0, -1, 0)},
        {"id": "DET_2THETA", "type": "rotation", "equipment": "detector", "vector": (0.6, 0.8, 0)},
        {"id": "DET_Z", "type": "translation", "equipment": "detector",
         "depends_on": "DET_2THETA", "vector": (0, 0, -1)},
        {"id": "DET_Y", "type": "translation", "equipment": "detector",
         "depends_on": "DET_Z", "vector": (0, 1, 0)},
        {"id": "DET_X", "type": "translation", "equipment": "detector",
         "depends_on": "DET_Y", "vector": (1, 0, 0)},
        {"id": "ELEMENT_X", "type": "translation", "equipment": "detector",
         "depends_on": "DET_X", "vector": (0, 1, 0), "offset": (-10.5, 12.25, 3.0)},
        {"id": "ELEMENT_Y", "type": "translation", "equipment": "detector",
         "depends_on": "ELEMENT_X", "vector": (1, 0, 0), "offset": (0.5, 0, 0)},
    ]


def write_sweep(tmp_path, n=4, scan="GON_PHI", values=None, **kw):
    values = dict(values or {"GON_OMEGA": 12.0, "GON_KAPPA": -33.0, "GON_PHI": 100.0,
                             "DET_2THETA": 17.5, "DET_Z": 123.4, "DET_Y": 2.5, "DET_X": -1.25})
    rng = np.random.default_rng(0)
    files = []
    for i in range(n):
        v = dict(values)
        v[scan] = values[scan] + 0.25 * i
        data = rng.integers(0, 60000, (40, 30)).astype(np.int32)
        data[5, :] = -1
        data[7, 3] = -2
        p = tmp_path / f"s_{i + 1:05d}.cbf"
        make_cbf(p, data, kw.get("axes", kappa_axes()), v, {scan: 0.25}, **{
            k: x for k, x in kw.items() if k != "axes"})
        files.append(p)
    return files


def direct_goniometer(sweep):
    names = sweep.goniometer_chain
    axes = [sweep.axes[k].vector for k in names]
    angles = [sweep.values[k][0] for k in names]
    return {"axes": axes, "angles": angles, "scan_axis": names.index(sweep.scan_axis)}


def test_kappa_two_theta_roundtrip(tmp_path):
    files = write_sweep(tmp_path, cutoff=65000)
    sweep = build_sweep(files)
    assert sweep.goniometer_chain == ["GON_PHI", "GON_KAPPA", "GON_OMEGA"]
    assert sweep.detector_chain == ["DET_X", "DET_Y", "DET_Z", "DET_2THETA"]
    out = tmp_path / "s.nxs"
    write_nxmx(sweep, out, WriteOptions(nproc=1, quiet=True))
    m = read_models(str(out))
    origin, fast, slow = panel_in_imgcif(sweep)
    p = m["detector"][0]["panels"][0]
    assert np.allclose(p["origin"], origin, atol=1e-9)
    assert np.allclose(p["fast_axis"], fast / np.linalg.norm(fast), atol=1e-12)
    assert np.allclose(p["slow_axis"], slow / np.linalg.norm(slow), atol=1e-12)
    assert p["image_size"] == [30, 40]
    a = goniometer_matrices(direct_goniometer(sweep))
    b = goniometer_matrices(m["goniometer"][0])
    for x, y in zip(a, b):
        assert np.allclose(x, y, atol=1e-12)
    assert np.allclose(m["goniometer"][0]["axes"][1], KAPPA)  # not normalised
    assert m["scan"][0]["oscillation"] == pytest.approx([100.0, 0.25])
    with h5py.File(out) as f:
        d = f["/entry/data/data"]
        assert d.dtype == np.uint16 and d.chunks == (1, 40, 30)
        assert d[0][5, 0] == 65535 and d[0][7, 3] == 65534
        assert f["/entry/instrument/detector/pixel_mask"][5, 0] == 1
        assert f["/entry/instrument/detector/pixel_mask"][7, 3] == 2
        assert f["/entry/instrument/detector/saturation_value"][()] == 64999


def test_origin_by_hand(tmp_path):
    """The module origin, worked out independently of compose()."""
    files = write_sweep(tmp_path, n=2)
    sweep = build_sweep(files)
    origin, fast, slow = panel_in_imgcif(sweep)
    from cbf2nxmx.frames import rotation

    R = rotation((0.6, 0.8, 0), 17.5)
    local = np.array([-10.5, 12.25, 3.0]) + np.array([0.5, 0, 0])  # element offsets
    local = local + np.array([-1.25, 0, 0]) + np.array([0, 2.5, 0]) + np.array([0, 0, -123.4])
    assert np.allclose(origin, R @ local)
    assert np.allclose(fast, R @ [0, 1, 0])


def test_vds_split(tmp_path):
    files = write_sweep(tmp_path, n=5)
    sweep = build_sweep(files)
    out = tmp_path / "v_master.h5"
    rep = write_nxmx(sweep, out, WriteOptions(nproc=2, quiet=True, frames_per_file=2,
                                               dtype="uint32", compression="lz4"))
    assert [p.name for p in rep.data_files] == [
        "v_data_000001.h5", "v_data_000002.h5", "v_data_000003.h5"]
    with h5py.File(out) as f:
        d = f["/entry/data/data"]
        assert d.is_virtual and d.shape == (5, 40, 30) and d.dtype == np.uint32
        assert d[4][5, 0] == 2**32 - 1
        assert f["/entry/data/data_000003"].shape == (1, 40, 30)


def test_moving_detector_rejected(tmp_path):
    files = write_sweep(tmp_path, n=2)
    files += write_sweep(tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path,
                         n=1, values={"GON_OMEGA": 12.0, "GON_KAPPA": -33.0,
                                      "GON_PHI": 100.5, "DET_2THETA": 17.5,
                                      "DET_Z": 150.0, "DET_Y": 2.5, "DET_X": -1.25})
    with pytest.raises(GeometryError):
        build_sweep([files[0], files[1], files[2]])


def test_uneven_scan_rejected(tmp_path):
    files = write_sweep(tmp_path, n=3)
    # Overwrite the third image with a jump in phi.
    rng = np.random.default_rng(1)
    make_cbf(files[2], rng.integers(0, 10, (40, 30)).astype(np.int32), kappa_axes(),
             {"GON_OMEGA": 12.0, "GON_KAPPA": -33.0, "GON_PHI": 101.0, "DET_2THETA": 17.5,
              "DET_Z": 123.4, "DET_Y": 2.5, "DET_X": -1.25}, {"GON_PHI": 0.25})
    with pytest.raises(GeometryError, match="step"):
        build_sweep(files)


def test_md5_checked(tmp_path):
    from cbf2nxmx.cbf import CbfError, read_header, read_pixels

    (f,) = write_sweep(tmp_path, n=1)
    raw = bytearray(f.read_bytes())
    c = read_header(f)
    raw[c.binary.data_offset + 10] ^= 0x01
    f.write_bytes(bytes(raw))
    with pytest.raises(CbfError, match="MD5"):
        read_pixels(read_header(f))
