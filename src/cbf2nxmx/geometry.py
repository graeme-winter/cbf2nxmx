"""A sweep's geometry and metadata from its CBF headers, in the imgCIF frame.

The imgCIF axis tables are the authority for geometry. The PILATUS header
supplies what imgCIF leaves out: sensor material and thickness, the count
cutoff (if _array_intensities.overload is missing), exposure and the
detector's name.

Everything here is in the imgCIF laboratory frame, the frame dxtbx uses, with
lengths in mm and angles in degrees. Converting to NeXus's McStas frame
happens in the writer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .cbf import CbfFile, read_header
from .cif import CifBlock

__all__ = ["Axis", "Module", "Sweep", "build_sweep", "GeometryError"]


class GeometryError(RuntimeError):
    pass


@dataclass
class Axis:
    id: str
    type: str  # "rotation" | "translation" | "general"
    equipment: str
    depends_on: str | None
    vector: np.ndarray  # as the file gives it, not normalised
    offset: np.ndarray  # mm


@dataclass
class Module:
    """The detector element: the corner of the first pixel and the directions
    of increasing fast and slow index, relative to the detector axis it hangs
    from (`parent`)."""

    parent: str | None
    n_fast: int
    n_slow: int
    fast_vector: np.ndarray
    slow_vector: np.ndarray
    pixel_fast_mm: float
    pixel_slow_mm: float
    offset_mm: np.ndarray
    fast_axis_id: str
    slow_axis_id: str


@dataclass
class Sweep:
    files: list[Path]
    first: CbfFile
    axes: dict[str, Axis]
    values: dict[str, np.ndarray]  # per frame, deg or mm, every positioner
    scan_axis: str
    scan_start: float
    scan_increment: float
    goniometer_chain: list[str]  # innermost first
    detector_chain: list[str]  # from the module's parent outwards
    module: Module
    wavelength_a: float
    overload: int | None
    undefined_value: int | None
    gain: float | None
    sensor_material: str | None
    sensor_thickness_mm: float | None
    count_time_s: float | None
    frame_time_s: float | None
    dates: list[str | None]
    description: str | None
    serial: str | None
    threshold_ev: float | None
    source_name: str | None
    source_type: str | None
    sample_name: str | None
    polarization_ratio: float | None
    notes: list[str] = field(default_factory=list)

    @property
    def n_images(self) -> int:
        return len(self.files)

    def scan_values(self) -> np.ndarray:
        return self.scan_start + self.scan_increment * np.arange(self.n_images)


def _f(x: str | None, default: float = 0.0) -> float:
    if x is None:
        return default
    return float(x)


def _vec(row: dict, name: str) -> np.ndarray:
    return np.array([_f(row.get(f"{name}[{k}]")) for k in (1, 2, 3)], dtype=float)


def _axes(block: CifBlock) -> dict[str, Axis]:
    axes = {}
    for row in block.table("axis"):
        a = Axis(
            id=row["id"],
            type=(row.get("type") or "general").lower(),
            equipment=(row.get("equipment") or "general").lower(),
            depends_on=row.get("depends_on"),
            vector=_vec(row, "vector"),
            offset=_vec(row, "offset"),
        )
        if a.id in axes:
            raise GeometryError(f"axis {a.id} defined twice")
        axes[a.id] = a
    if not axes:
        raise GeometryError("no _axis table: not a full imgCIF header")
    return axes


def _frame_values(block: CifBlock, axes: dict[str, Axis]) -> dict[str, float]:
    """Each positioner's value for this frame: the frame's own table, then the
    scan's start, then 0."""
    out: dict[str, float] = {}
    for row in block.table("diffrn_scan_axis"):
        a = axes.get(row["axis_id"])
        if a is None:
            continue
        key = "angle_start" if a.type == "rotation" else "displacement_start"
        out[a.id] = _f(row.get(key))
    for row in block.table("diffrn_scan_frame_axis"):
        a = axes.get(row["axis_id"])
        if a is None:
            continue
        key = "angle" if a.type == "rotation" else "displacement"
        out[a.id] = _f(row.get(key))
    for a in axes.values():
        if a.type in ("rotation", "translation"):
            out.setdefault(a.id, 0.0)
    return out


_VARYING_TABLES = {
    "diffrn_scan_frame",
    "diffrn_scan_frame_axis",
    "diffrn_scan_axis",
    "diffrn_scan",
    "array_data",
}


def _signature(c: CbfFile) -> dict:
    """What must not change from one image of a sweep to the next."""
    sig = {k: v for k, v in c.block.tables.items() if k not in _VARYING_TABLES}
    if c.header is not None:
        h = {
            k: v
            for k, v in c.header.fields.items()
            if k not in c.header.VARYING
        }
        sig["__pilatus__"] = h
    b = c.binary
    sig["__binary__"] = (b.fast, b.slow, b.element_type, b.conversions)
    return sig


def _diff(a: dict, b: dict) -> list[str]:
    out = []
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            if isinstance(a.get(k), dict) and isinstance(b.get(k), dict):
                for kk in sorted(set(a[k]) | set(b[k])):
                    if a[k].get(kk) != b[k].get(kk):
                        out.append(f"{k}.{kk}: {a[k].get(kk)!r} -> {b[k].get(kk)!r}")
            else:
                out.append(f"{k}: {a.get(k)!r} -> {b.get(k)!r}")
    return out


def _module(block: CifBlock, axes: dict[str, Axis], c: CbfFile, notes) -> Module:
    rows = block.table("array_structure_list")
    if not rows:
        raise GeometryError("no _array_structure_list table")
    arrays = {r.get("array_id") for r in rows}
    if len(arrays) != 1:
        raise GeometryError(f"several arrays {sorted(map(str, arrays))}: unsupported")
    by_prec = sorted(rows, key=lambda r: int(r["precedence"]))
    if len(by_prec) != 2:
        raise GeometryError(f"{len(by_prec)}-dimensional array: unsupported")
    axis_rows = block.table("array_structure_list_axis")

    def element(r):
        set_id = r["axis_set_id"]
        members = [a for a in axis_rows if a["axis_set_id"] == set_id]
        if len(members) != 1:
            raise GeometryError(
                f"axis set {set_id} has {len(members)} axes: unsupported"
            )
        m = members[0]
        a = axes.get(m["axis_id"])
        if a is None:
            raise GeometryError(f"array axis {m['axis_id']} not in _axis")
        if a.type != "translation":
            raise GeometryError(f"array axis {a.id} is a {a.type}: unsupported")
        disp = _f(m.get("displacement"))
        inc = _f(m.get("displacement_increment"))
        if inc == 0:
            raise GeometryError(f"array axis {a.id} has no displacement_increment")
        direction = (r.get("direction") or "increasing").lower()
        sign = np.sign(inc) * (1.0 if direction == "increasing" else -1.0)
        if direction != "increasing":
            notes.append(
                f"array axis {a.id} is '{direction}': the pixel direction is "
                "reversed (untested on real data)"
            )
        return a, int(r["dimension"]), disp, abs(inc), sign

    fa, nf, df, pf, sf = element(by_prec[0])
    sa, ns, ds, ps, ss = element(by_prec[1])
    if (nf, ns) != (c.binary.fast, c.binary.slow):
        raise GeometryError(
            f"array dimensions {nf} x {ns} differ from the binary's "
            f"{c.binary.fast} x {c.binary.slow}"
        )
    # The two element translations hang one from the other, then from the rest
    # of the detector.
    if sa.depends_on == fa.id:
        parent = fa.depends_on
    elif fa.depends_on == sa.id:
        parent = sa.depends_on
    else:
        raise GeometryError(
            f"array axes {fa.id} and {sa.id} do not depend on one another"
        )
    if df or ds:
        notes.append(
            f"array axes have displacements {df}, {ds} mm: the module origin "
            "includes them (untested against dxtbx)"
        )
    offset = fa.offset + sa.offset + df * fa.vector + ds * sa.vector
    sizes = block.table("array_element_size")
    for r in sizes:
        idx = int(r["index"])
        expect = pf if idx == int(by_prec[0]["index"]) else ps
        got = _f(r.get("size")) * 1000.0
        if abs(got - expect) > 1e-6:
            notes.append(
                f"_array_element_size index {idx} is {got} mm but the axis "
                f"increment is {expect} mm: the increment is used"
            )
    return Module(
        parent=parent,
        n_fast=nf,
        n_slow=ns,
        fast_vector=sf * fa.vector,
        slow_vector=ss * sa.vector,
        pixel_fast_mm=pf,
        pixel_slow_mm=ps,
        offset_mm=offset,
        fast_axis_id=fa.id,
        slow_axis_id=sa.id,
    )


def _chain(axes: dict[str, Axis], start: str | None) -> list[str]:
    out = []
    at = start
    while at is not None:
        if at in out:
            raise GeometryError(f"axis dependencies loop at {at}")
        if at not in axes:
            raise GeometryError(f"axis {at} depended on but not defined")
        out.append(at)
        at = axes[at].depends_on
    return out


def frame_number(path: Path) -> int | None:
    m = re.search(r"(\d+)\.cbf(?:\.gz|\.bz2)?$", path.name)
    return int(m.group(1)) if m else None


def build_sweep(paths: Sequence[str | Path], check_consistency: bool = True) -> Sweep:
    files = [Path(p) for p in paths]
    if not files:
        raise GeometryError("no images")
    nums = [frame_number(p) for p in files]
    if None not in nums:
        order = np.argsort(nums)
        files = [files[i] for i in order]
        nums = sorted(nums)
        gaps = [b for a, b in zip(nums, nums[1:]) if b != a + 1]
        if gaps:
            raise GeometryError(f"images not consecutive: missing before {gaps[:5]}")

    notes: list[str] = []
    headers = [read_header(p) for p in files]
    first = headers[0]
    block = first.block
    axes = _axes(block)

    if check_consistency:
        sig0 = _signature(first)
        for c in headers[1:]:
            d = _diff(sig0, _signature(c))
            if d:
                raise GeometryError(
                    f"{c.path.name} differs from {first.path.name}: " + "; ".join(d[:5])
                )

    per_frame = [_frame_values(c.block, axes) for c in headers]
    values = {
        k: np.array([pf[k] for pf in per_frame], dtype=float) for k in per_frame[0]
    }

    # The scan axis: the one rotation that moves.
    moving = [
        k for k, v in values.items() if v.size > 1 and np.ptp(v) > 1e-9
    ]
    for k in moving:
        if axes[k].type != "rotation":
            raise GeometryError(f"translation {k} moves during the sweep: unsupported")
    if len(moving) > 1:
        raise GeometryError(f"several axes move: {moving}")
    increments = {}
    for row in block.table("diffrn_scan_axis"):
        a = axes.get(row["axis_id"])
        if a is not None and a.type == "rotation":
            increments[a.id] = _f(row.get("angle_increment"))
    if moving:
        scan_axis = moving[0]
    else:
        nonzero = [k for k, v in increments.items() if v]
        if len(nonzero) != 1:
            raise GeometryError(
                "cannot tell the scan axis: nothing moves and "
                f"{len(nonzero)} axes have an angle increment"
            )
        scan_axis = nonzero[0]
    inc = increments.get(scan_axis)
    v = values[scan_axis]
    if v.size > 1:
        steps = np.diff(v)
        if inc is None or inc == 0:
            inc = float(np.median(steps))
        bad = np.flatnonzero(np.abs(steps - inc) > 1e-3)
        if bad.size:
            raise GeometryError(
                f"{scan_axis} does not step evenly by {inc}: e.g. frame "
                f"{bad[0] + 1} to {bad[0] + 2} steps {steps[bad[0]]}"
            )
    if not inc:
        raise GeometryError(f"scan axis {scan_axis} has no increment")
    scan_start = float(v[0])

    # The goniometer: one chain, innermost first.
    gon = [a for a in axes.values() if a.equipment == "goniometer"]
    depended = {a.depends_on for a in gon}
    inner = [a.id for a in gon if a.id not in depended]
    if len(inner) != 1:
        raise GeometryError(f"goniometer axes form {len(inner)} chains: {inner}")
    gchain = _chain(axes, inner[0])
    for k in gchain:
        if axes[k].type != "rotation":
            raise GeometryError(f"goniometer axis {k} is a {axes[k].type}")
        if np.any(axes[k].offset):
            notes.append(f"goniometer axis {k} has an offset {axes[k].offset}")
    if scan_axis not in gchain:
        raise GeometryError(f"scan axis {scan_axis} is not on the goniometer")
    unused = [a.id for a in gon if a.id not in gchain]
    if unused:
        notes.append(f"goniometer axes not on the sample's chain: {unused}")

    module = _module(block, axes, first, notes)
    dchain = _chain(axes, module.parent)
    for k in dchain:
        if axes[k].type not in ("rotation", "translation"):
            raise GeometryError(f"detector axis {k} is a {axes[k].type}")

    for a in axes.values():
        if a.equipment == "source" and a.type == "general":
            if not np.allclose(a.vector, [0, 0, 1]):
                raise GeometryError(
                    f"source axis {a.vector}: only a beam along -z is supported"
                )
        if a.equipment == "gravity" and a.type == "general":
            if not np.allclose(a.vector, [0, -1, 0]):
                notes.append(
                    f"gravity is {a.vector} in the imgCIF frame, not (0,-1,0): "
                    "NeXus's y will not be up"
                )

    hdr = first.header
    wl = block.value("diffrn_radiation_wavelength", "wavelength")
    wavelength = _f(wl) if wl is not None else (hdr.wavelength_a if hdr else None)
    if not wavelength:
        raise GeometryError("no wavelength")

    overload = block.value("array_intensities", "overload")
    overload = int(float(overload)) if overload is not None else None
    if overload is None and hdr and hdr.count_cutoff:
        overload = hdr.count_cutoff
    undefined = block.value("array_intensities", "undefined_value")
    gain = block.value("array_intensities", "gain")

    count_time = hdr.exposure_time_s if hdr else None
    frame_time = hdr.exposure_period_s if hdr else None
    if count_time is None:
        x = block.value("diffrn_scan_frame", "integration_time")
        count_time = _f(x) if x is not None else None
    if frame_time is None:
        x = block.value("diffrn_scan_frame", "exposure_time")
        frame_time = _f(x) if x is not None else None

    dates = []
    for c in headers:
        d = c.block.value("diffrn_scan_frame", "date")
        if d is None and c.header is not None:
            d = c.header.date
        dates.append(d)

    pol = block.value("diffrn_radiation", "polarizn_source_ratio")

    sensor_material = hdr.sensor_material if hdr else None
    thickness = hdr.sensor_thickness_m * 1000.0 if hdr and hdr.sensor_thickness_m else None
    if sensor_material is None:
        notes.append("no sensor material in the header: none written")

    return Sweep(
        files=files,
        first=first,
        axes=axes,
        values=values,
        scan_axis=scan_axis,
        scan_start=scan_start,
        scan_increment=float(inc),
        goniometer_chain=gchain,
        detector_chain=dchain,
        module=module,
        wavelength_a=float(wavelength),
        overload=overload,
        undefined_value=int(float(undefined)) if undefined is not None else None,
        gain=_f(gain) if gain is not None else None,
        sensor_material=sensor_material,
        sensor_thickness_mm=thickness,
        count_time_s=count_time,
        frame_time_s=frame_time,
        dates=dates,
        description=hdr.model if hdr else block.value("diffrn_detector", "type"),
        serial=hdr.serial if hdr else block.value("diffrn_detector", "id"),
        threshold_ev=hdr.threshold_ev if hdr else None,
        source_name=block.value("diffrn_source", "type"),
        source_type=block.value("diffrn_source", "source"),
        sample_name=block.value("diffrn", "crystal_id"),
        polarization_ratio=_f(pol) if pol is not None else None,
        notes=notes,
    )
