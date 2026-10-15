"""Read an NXmx file back into DIALS-style experiment models, without DIALS.

This follows how dxtbx's FormatNXmx (and mxi_import) read a file:
transformation chains composed innermost first, vectors as the file gives
them, then McStas -> imgCIF by a half turn about y. It exists to check the
writer offline. The real test is dials.import and mxi_import themselves (see
cbf2nxmx.compare).

The output is a dict shaped like a DIALS .expt file (beam, detector,
goniometer, scan lists), so cbf2nxmx.compare can read it.
"""

from __future__ import annotations

import posixpath

import h5py
import numpy as np

from .frames import MCSTAS_TO_IMGCIF, compose

_MM = {"mm": 1.0, "m": 1000.0, "um": 1e-3, "micron": 1e-3, "nm": 1e-6, "cm": 10.0}
_DEG = {"deg": 1.0, "degree": 1.0, "degrees": 1.0, "rad": 180.0 / np.pi}


def _str(x) -> str:
    if isinstance(x, bytes):
        return x.decode()
    if isinstance(x, np.ndarray):
        return _str(x[()] if x.shape == () else x.ravel()[0])
    return str(x)


def _step(f: h5py.File, path: str):
    d = f[path]
    kind = _str(d.attrs["transformation_type"]).lower()
    units = _str(d.attrs.get("units", "mm" if kind == "translation" else "deg")).lower()
    k = _MM[units] if kind == "translation" else _DEG[units]
    values = np.atleast_1d(d[()]).astype(float) * k
    vector = np.asarray(d.attrs["vector"], dtype=float)
    offset = np.asarray(d.attrs.get("offset", np.zeros(3)), dtype=float)
    ou = _str(d.attrs.get("offset_units", units if kind == "translation" else "mm")).lower()
    offset = offset * _MM[ou]
    dep = _str(d.attrs.get("depends_on", "."))
    if dep != "." and not dep.startswith("/"):
        dep = posixpath.normpath(posixpath.join(posixpath.dirname(path), dep))
    return kind, vector, offset, values, dep


def chain(f: h5py.File, start: str):
    """[(name, kind, vector, offset, values)], innermost first."""
    out = []
    at = start
    while at and at != ".":
        kind, vector, offset, values, dep = _step(f, at)
        out.append((posixpath.basename(at), kind, vector, offset, values))
        at = dep
    return out


def read_models(path: str) -> dict:
    C = MCSTAS_TO_IMGCIF
    with h5py.File(path, "r") as f:
        beam = f.get("/entry/instrument/beam") or f.get("/entry/sample/beam")
        wavelength = float(beam["incident_wavelength"][()])

        det = f["/entry/instrument/detector"]
        panels = []
        for name, g in det.items():
            if not isinstance(g, h5py.Group) or _str(g.attrs.get("NX_class", "")) != "NXdetector_module":
                continue
            fp = g.name + "/fast_pixel_direction"
            sp = g.name + "/slow_pixel_direction"
            kf, vf, of, valf, depf = _step(f, fp)
            ks, vs, os_, vals, deps = _step(f, sp)
            steps = chain(f, depf)
            R, t = compose((k, v, o, val[0]) for _, k, v, o, val in steps)
            origin = C @ (R @ of + t)
            fast = C @ (R @ (vf / np.linalg.norm(vf)))
            slow = C @ (R @ (vs / np.linalg.norm(vs)))
            size = np.asarray(g["data_size"][()])
            sat = det.get("saturation_value")
            panels.append(
                {
                    "name": g.name,
                    "fast_axis": fast.tolist(),
                    "slow_axis": slow.tolist(),
                    "origin": origin.tolist(),
                    "image_size": [int(size[1]), int(size[0])],
                    "pixel_size": [float(valf[0]), float(vals[0])],
                    "trusted_range": [0.0, float(sat[()]) if sat is not None else 2**31 - 1],
                    "thickness": float(det["sensor_thickness"][()]) * _MM[
                        _str(det["sensor_thickness"].attrs.get("units", "mm")).lower()
                    ] if "sensor_thickness" in det else 0.0,
                    "material": _str(det["sensor_material"][()]) if "sensor_material" in det else "",
                }
            )

        sample = f["/entry/sample"]
        gsteps = chain(f, _str(sample["depends_on"][()]))
        names = [s[0] for s in gsteps]
        axes = [(C @ s[2]).tolist() for s in gsteps]
        moving = [i for i, s in enumerate(gsteps) if s[4].size > 1 and np.ptp(s[4]) > 0]
        if len(moving) != 1:
            raise ValueError(f"{len(moving)} goniometer axes move")
        scan_axis = moving[0]
        angles = [float(s[4][0]) for s in gsteps]
        scan_values = gsteps[scan_axis][4]
        n = int(scan_values.size)
        osc = [float(scan_values[0]), float(scan_values[1] - scan_values[0])]
        frame_time = float(det["frame_time"][()]) if "frame_time" in det else 0.0

    return {
        "beam": [
            {
                "direction": [0.0, 0.0, 1.0],
                "wavelength": wavelength,
                "polarization_normal": [0.0, 1.0, 0.0],
                "polarization_fraction": 0.999,
            }
        ],
        "detector": [{"panels": panels}],
        "goniometer": [
            {"axes": axes, "angles": angles, "names": names, "scan_axis": scan_axis}
        ],
        "scan": [
            {
                "image_range": [1, n],
                "oscillation": osc,
                "properties": {
                    "exposure_time": [frame_time] * n,
                    "epochs": [frame_time * i for i in range(n)],
                },
            }
        ],
    }
