"""Writing a Sweep as NXmx HDF5.

The layout is chosen so that dials.import (dxtbx's FormatNXmx) and mxi_import
read the same models as dials.import does from the CBF:

* Every imgCIF vector v goes into the file as C v, with C = diag(-1, 1, -1):
  dxtbx and mxi take McStas back to imgCIF with the same half turn. That
  makes the round trip numerically exact.
* Vectors are written as the CBF gives them, not normalised (dxtbx scales
  rotation angles by |v|), except for the pixel directions.
* The goniometer chain is the CBF's, innermost first from
  /entry/sample/depends_on, with one value per image on every axis.
* The detector chain is the CBF's. The element axes fold into one
  module_offset plus fast_ and slow_pixel_direction, both depending on
  module_offset.
* Frames are one chunk each, (1, slow, fast): bitshuffle/LZ4 by default, LZ4
  or none on request (mxi reads exactly these three).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from . import __version__
from .cbf import read_header, read_pixels
from .frames import beam_centre, compose, to_mcstas
from .geometry import Sweep
from .pixels import BAD_BIT, GAP_BIT, Policy, choose_dtype, convert

__all__ = ["WriteOptions", "write_nxmx", "nx_name"]

DETECTOR = "/entry/instrument/detector"
DET_T = DETECTOR + "/transformations"
MODULE = DETECTOR + "/module"
SAMPLE_T = "/entry/sample/transformations"


@dataclass
class WriteOptions:
    dtype: str = "auto"  # auto | uint16 | uint32
    compression: str = "bslz4"  # bslz4 | lz4 | none
    overflow: str = "error"  # error | clip
    nproc: int = 0  # 0: all cores
    frames_per_file: int = 0  # 0: frames in the master file
    polarization: bool = False  # write the CBF's polarisation into NXbeam
    check_md5: bool = True
    quiet: bool = False


@dataclass
class WriteReport:
    path: Path
    data_files: list[Path] = field(default_factory=list)
    dtype: str = ""
    saturation_value: int = 0
    n_images: int = 0
    max_count: int = 0
    n_overflow: int = 0
    n_other_negative: int = 0
    masked_gap: int = 0
    masked_bad: int = 0
    notes: list[str] = field(default_factory=list)
    seconds: float = 0.0


def nx_name(axis_id: str) -> str:
    s = axis_id.lower()
    if s.startswith("gon_"):
        s = s[4:]
    s = s.replace("2theta", "two_theta")
    if s == "det_two_theta":
        s = "two_theta"
    return s


def _s(text: str) -> np.bytes_:
    return np.bytes_(text.encode("utf-8"))


def _attrs(obj, **kw):
    for k, v in kw.items():
        if isinstance(v, str):
            obj.attrs[k] = _s(v)
        else:
            obj.attrs[k] = v


def _group(parent, name, nx_class):
    g = parent.require_group(name)
    _attrs(g, NX_class=nx_class)
    return g


def _scalar(group, name, value, units=None, dtype=None):
    if isinstance(value, str):
        ds = group.create_dataset(name, data=_s(value))
    else:
        ds = group.create_dataset(name, data=np.asarray(value, dtype=dtype))
    if units:
        _attrs(ds, units=units)
    return ds


def _transformation(group, name, values, kind, vector, offset, depends_on):
    ds = group.create_dataset(name, data=np.asarray(values, dtype=np.float64))
    _attrs(
        ds,
        transformation_type=kind,
        units="deg" if kind == "rotation" else "mm",
        depends_on=depends_on,
        offset_units="mm",
    )
    ds.attrs["vector"] = np.asarray(vector, dtype=np.float64)
    ds.attrs["offset"] = np.asarray(offset, dtype=np.float64)
    return ds


def _compression(name: str) -> dict:
    if name == "none":
        return {}
    import hdf5plugin  # registers the filters with HDF5

    if name == "bslz4":
        return dict(hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
    if name == "lz4":
        return dict(hdf5plugin.LZ4())
    raise ValueError(f"compression {name!r}: choose bslz4, lz4 or none")


def _end_time(date: str | None, seconds: float | None) -> str | None:
    if date is None:
        return None
    try:
        t = _dt.datetime.fromisoformat(date)
    except ValueError:
        return date
    return (t + _dt.timedelta(seconds=seconds or 0.0)).isoformat(timespec="milliseconds")


def panel_in_imgcif(sweep: Sweep):
    """The panel's origin, fast and slow at the first image, imgCIF frame."""
    m = sweep.module
    steps = [
        (sweep.axes[k].type, sweep.axes[k].vector, sweep.axes[k].offset, sweep.values[k][0])
        for k in sweep.detector_chain
    ]
    R, t = compose(steps)
    return R @ m.offset_mm + t, R @ m.fast_vector, R @ m.slow_vector


def _write_metadata(f: h5py.File, sweep: Sweep, policy: Policy, opts: WriteOptions,
                    saturation: int):
    n = sweep.n_images
    _attrs(
        f,
        NX_class="NXroot",
        default="entry",
        file_name=os.path.basename(f.filename),
        file_time=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        creator=f"cbf2nxmx {__version__}",
        HDF5_Version=h5py.version.hdf5_version,
    )
    entry = _group(f, "entry", "NXentry")
    _attrs(entry, default="data")
    _scalar(entry, "definition", "NXmx")
    if sweep.dates[0]:
        _scalar(entry, "start_time", sweep.dates[0])
        end = _end_time(sweep.dates[-1], sweep.frame_time_s)
        if end:
            _scalar(entry, "end_time_estimated", end)

    data = _group(entry, "data", "NXdata")
    _attrs(data, signal="data")

    instrument = _group(entry, "instrument", "NXinstrument")
    if sweep.source_name:
        _scalar(instrument, "name", sweep.source_name)
    source = _group(entry, "source", "NXsource")
    _scalar(source, "name", sweep.source_name or "unknown")
    if (sweep.source_type or "").lower() == "synchrotron":
        _scalar(source, "type", "Synchrotron X-ray Source")

    beam = _group(instrument, "beam", "NXbeam")
    _scalar(beam, "incident_wavelength", sweep.wavelength_a, "angstrom")
    if opts.polarization and sweep.polarization_ratio is not None:
        # Linear polarisation in the horizontal (McStas x) plane.
        p = sweep.polarization_ratio
        _scalar(beam, "incident_polarization_stokes", [1.0, p, 0.0, 0.0])

    # --- detector -----------------------------------------------------------
    det = _group(instrument, "detector", "NXdetector")
    if sweep.description:
        _scalar(det, "description", sweep.description)
    if sweep.serial:
        _scalar(det, "detector_number", sweep.serial)
    _scalar(det, "type", "Pixel")
    if sweep.sensor_material:
        _scalar(det, "sensor_material", sweep.sensor_material)
    if sweep.sensor_thickness_mm:
        _scalar(det, "sensor_thickness", sweep.sensor_thickness_mm, "mm")
    m = sweep.module
    _scalar(det, "x_pixel_size", m.pixel_fast_mm, "mm")
    _scalar(det, "y_pixel_size", m.pixel_slow_mm, "mm")
    origin, fast, slow = panel_in_imgcif(sweep)
    bc = beam_centre(origin, fast, slow, m.pixel_fast_mm, m.pixel_slow_mm)
    if bc is not None:
        _scalar(det, "beam_center_x", bc[0], "pixel")
        _scalar(det, "beam_center_y", bc[1], "pixel")
        _scalar(det, "distance", bc[2], "mm")
    if sweep.count_time_s is not None:
        _scalar(det, "count_time", sweep.count_time_s, "s")
    if sweep.frame_time_s is not None:
        _scalar(det, "frame_time", sweep.frame_time_s, "s")
    if sweep.threshold_ev is not None:
        _scalar(det, "threshold_energy", sweep.threshold_ev, "eV")
    bits = 8 * np.dtype(policy.dtype).itemsize
    _scalar(det, "bit_depth_image", bits, dtype=np.int32)
    _scalar(det, "saturation_value", saturation, dtype=np.int64)
    if sweep.first.header is not None and sweep.first.header.gain_setting:
        _scalar(det, "gain_setting", sweep.first.header.gain_setting)

    dt = _group(det, "transformations", "NXtransformations")
    for k in sweep.detector_chain:
        a = sweep.axes[k]
        dep = f"{DET_T}/{nx_name(a.depends_on)}" if a.depends_on else "."
        _transformation(
            dt, nx_name(k), [sweep.values[k][0]], a.type, to_mcstas(a.vector),
            to_mcstas(a.offset), dep,
        )
    parent = f"{DET_T}/{nx_name(m.parent)}" if m.parent else "."
    _scalar(det, "depends_on", parent)

    mod = _group(det, "module", "NXdetector_module")
    _scalar(mod, "data_origin", [0, 0], dtype=np.int32)
    _scalar(mod, "data_size", [m.n_slow, m.n_fast], dtype=np.int32)
    _scalar(mod, "data_stride", [1, 1], dtype=np.int32)
    off = to_mcstas(m.offset_mm)
    norm = float(np.linalg.norm(off))
    _transformation(
        mod, "module_offset", norm, "translation",
        off / norm if norm > 0 else [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], parent,
    )
    for name, vec, size in (
        ("fast_pixel_direction", m.fast_vector, m.pixel_fast_mm),
        ("slow_pixel_direction", m.slow_vector, m.pixel_slow_mm),
    ):
        v = to_mcstas(vec)
        _transformation(
            mod, name, size, "translation", v / np.linalg.norm(v), [0.0, 0.0, 0.0],
            f"{MODULE}/module_offset",
        )

    # --- sample and goniometer ---------------------------------------------
    sample = _group(entry, "sample", "NXsample")
    if sweep.sample_name:
        _scalar(sample, "name", sweep.sample_name)
    sample["beam"] = beam  # a hard link: dxtbx and mxi may look in either place
    st = _group(sample, "transformations", "NXtransformations")
    for k in sweep.goniometer_chain:
        a = sweep.axes[k]
        dep = f"{SAMPLE_T}/{nx_name(a.depends_on)}" if a.depends_on else "."
        if k == sweep.scan_axis:
            values = sweep.scan_values()
        else:
            values = np.full(n, sweep.values[k][0])
        _transformation(
            st, nx_name(k), values, "rotation", to_mcstas(a.vector),
            to_mcstas(a.offset), dep,
        )
        if k == sweep.scan_axis:
            _scalar(st, nx_name(k) + "_increment_set", sweep.scan_increment, "deg")
            _scalar(st, nx_name(k) + "_end", values + sweep.scan_increment, "deg")
    _scalar(sample, "depends_on", f"{SAMPLE_T}/{nx_name(sweep.goniometer_chain[0])}")


def _provenance(f, sweep: Sweep, report: WriteReport, opts: WriteOptions):
    note = _group(f["entry"], "cbf2nxmx", "NXnote")
    info = {
        "converter": f"cbf2nxmx {__version__}",
        "first_image": sweep.files[0].name,
        "last_image": sweep.files[-1].name,
        "n_images": sweep.n_images,
        "dtype": report.dtype,
        "saturation_value": report.saturation_value,
        "compression": opts.compression,
        "overflow": opts.overflow,
        "max_count": report.max_count,
        "n_overflow": report.n_overflow,
        "n_other_negative": report.n_other_negative,
        "masked_gap": report.masked_gap,
        "masked_bad": report.masked_bad,
        "notes": sweep.notes + report.notes,
    }
    _scalar(note, "type", "application/json")
    _scalar(note, "description", "Conversion from CBF by cbf2nxmx")
    _scalar(note, "data", json.dumps(info, indent=1))
    _scalar(note, "cbf_header", sweep.first.cif_text)


def _decode(args):
    path, policy, check_md5 = args
    c = read_header(path)
    a = read_pixels(c, check_md5=check_md5)
    return convert(a, policy, where=str(path))


def _frames(sweep: Sweep, policy: Policy, opts: WriteOptions):
    """Converted frames in order, decoded in parallel with a bounded queue."""
    jobs = [(p, policy, opts.check_md5) for p in sweep.files]
    nproc = opts.nproc or os.cpu_count() or 1
    if nproc == 1:
        for j in jobs:
            yield _decode(j)
        return
    with ProcessPoolExecutor(max_workers=nproc) as pool:
        pending = deque()
        it = iter(jobs)
        for j in it:
            pending.append(pool.submit(_decode, j))
            if len(pending) >= 2 * nproc:
                break
        while pending:
            result = pending.popleft().result()
            nxt = next(it, None)
            if nxt is not None:
                pending.append(pool.submit(_decode, nxt))
            yield result


def write_nxmx(sweep: Sweep, out: str | Path, opts: WriteOptions | None = None) -> WriteReport:
    opts = opts or WriteOptions()
    t0 = time.time()
    out = Path(out)
    dtype = choose_dtype(opts.dtype, sweep.overload)
    policy = Policy(dtype=dtype, overflow=opts.overflow)
    saturation = policy.saturation_value(sweep.overload)
    report = WriteReport(path=out, dtype=np.dtype(dtype).name, saturation_value=saturation,
                         n_images=sweep.n_images)
    if sweep.overload is not None and sweep.overload - 1 > saturation:
        report.notes.append(
            f"the CBF's trusted maximum {sweep.overload - 1} is above what "
            f"{report.dtype} holds below its markers; saturation_value is {saturation}"
        )
    n = sweep.n_images
    shape = (n, sweep.module.n_slow, sweep.module.n_fast)
    comp = _compression(opts.compression)

    with h5py.File(out, "w", libver=("v110", "latest") if opts.frames_per_file else "earliest") as f:
        _write_metadata(f, sweep, policy, opts, saturation)
        f.flush()

        # Where each frame goes: (dataset, index within it).
        targets = []
        data_files = []
        if opts.frames_per_file:
            per = opts.frames_per_file
            stem = out.name[: -len(out.suffix)] if out.suffix else out.name
            if stem.endswith("_master"):
                stem = stem[: -len("_master")]
            for k, lo in enumerate(range(0, n, per)):
                hi = min(lo + per, n)
                path = out.with_name(f"{stem}_data_{k + 1:06d}.h5")
                df = h5py.File(path, "w")
                e = _group(df, "entry", "NXentry")
                d = _group(e, "data", "NXdata")
                ds = d.create_dataset(
                    "data", shape=(hi - lo,) + shape[1:], dtype=dtype,
                    chunks=(1,) + shape[1:], **comp,
                )
                _attrs(ds, image_nr_low=np.int32(lo + 1), image_nr_high=np.int32(hi))
                data_files.append((df, ds, lo, hi, path))
                targets.extend((ds, i) for i in range(hi - lo))
        else:
            ds = f["entry/data"].create_dataset(
                "data", shape=shape, dtype=dtype, chunks=(1,) + shape[1:], **comp
            )
            targets = [(ds, i) for i in range(n)]

        mask = np.zeros(shape[1:], dtype=np.uint32)
        first_counts = None
        varied = False
        for i, conv in enumerate(_frames(sweep, policy, opts)):
            ds, j = targets[i]
            ds[j] = conv.data
            gap = conv.data == policy.gap_value
            bad = conv.data == policy.bad_value
            mask[gap] |= GAP_BIT
            mask[bad] |= BAD_BIT
            counts = (int(gap.sum()), int(bad.sum()))
            if first_counts is None:
                first_counts = counts
            elif counts != first_counts and not varied:
                varied = True
                report.notes.append(
                    f"image {i + 1} has {counts} gap/bad pixels, image 1 {first_counts}: "
                    "pixel_mask is their union"
                )
            report.max_count = max(report.max_count, conv.maximum)
            report.n_overflow += conv.n_overflow
            report.n_other_negative += conv.n_other_negative
            if not opts.quiet and ((i + 1) % 100 == 0 or i + 1 == n):
                print(f"  {i + 1}/{n} images", file=sys.stderr, flush=True)

        if report.n_overflow:
            report.notes.append(
                f"{report.n_overflow} pixels clipped to {policy.top}, above saturation_value"
            )
        if report.n_other_negative:
            report.notes.append(
                f"{report.n_other_negative} pixels with negative values other than "
                "-1 and -2 marked bad"
            )

        if data_files:
            layout = h5py.VirtualLayout(shape=shape, dtype=dtype)
            for k, (df, dsk, lo, hi, path) in enumerate(data_files):
                layout[lo:hi] = h5py.VirtualSource(
                    path.name, "entry/data/data", shape=(hi - lo,) + shape[1:]
                )
                f["entry/data"][f"data_{k + 1:06d}"] = h5py.ExternalLink(
                    path.name, "entry/data/data"
                )
                df.close()
            f["entry/data"].create_virtual_dataset("data", layout, fillvalue=0)
            report.data_files = [d[4] for d in data_files]

        det = f[DETECTOR]
        det.create_dataset("pixel_mask", data=mask, compression="gzip",
                           compression_opts=4, chunks=mask.shape)
        report.masked_gap = int(np.count_nonzero(mask & GAP_BIT))
        report.masked_bad = int(np.count_nonzero(mask & BAD_BIT))
        report.seconds = time.time() - t0
        _provenance(f, sweep, report, opts)
    return report
