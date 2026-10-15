"""Compare two experiment lists' models: the CBF's import against the NXmx's.

    cbf2nxmx.compare reference.expt other.expt [--strict]

Either file may be a DIALS .expt (from dials.import or mxi_import -o x.expt)
or an NXmx file, which is read with cbf2nxmx.readback. Geometry must agree
within tolerances. Some things are expected to differ and are only reported:

* exposure_time and epochs: from a CBF, dxtbx takes the integration time and
  the frame dates; from NXmx, frame_time and multiples of it from 0.
* the panel's mask list: dxtbx adds PILATUS module gaps to a CBF panel; in
  NXmx they are in pixel_mask instead.
* identifiers, axis names, and the scan axis's own angle in the goniometer's
  angle list.

--strict treats those as failures too.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np

from .frames import rotation

TOL_MM = 1e-6
TOL_DEG = 1e-6
TOL_REL = 1e-9


def load(path: str) -> dict:
    with open(path, "rb") as fh:
        magic = fh.read(8)
    if magic == b"\x89HDF\r\n\x1a\n":
        from .readback import read_models

        return read_models(path)
    with open(path) as fh:
        return json.load(fh)


def _angle(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def goniometer_matrices(g: dict):
    """(rotation axis datum, F, S) as dxtbx defines them."""
    if "axes" in g:
        axes = [np.asarray(a, float) for a in g["axes"]]
        angles = g["angles"]
        k = g["scan_axis"]
        F = np.eye(3)
        for i in range(k):
            F = rotation(axes[i], angles[i]) @ F
        S = np.eye(3)
        for i in range(k + 1, len(axes)):
            S = rotation(axes[i], angles[i]) @ S
        return axes[k], F, S
    F = np.asarray(g.get("fixed_rotation", np.eye(3).ravel()), float).reshape(3, 3)
    S = np.asarray(g.get("setting_rotation", np.eye(3).ravel()), float).reshape(3, 3)
    return np.asarray(g["rotation_axis"], float), F, S


def _scan(s: dict):
    props = s.get("properties", {})
    if isinstance(props, str):
        import ast

        props = ast.literal_eval(props)
    if "oscillation" in s:
        osc = list(s["oscillation"])
    else:
        o = props["oscillation"]
        osc = [o[0], o[1] - o[0]] if len(o) > 1 else [o[0], props.get("oscillation_width", [0])[0]]
    return s["image_range"], osc, props.get("exposure_time", []), props.get("epochs", [])


def _norm_name(n: str) -> str:
    n = n.lower()
    n = re.sub(r"^gon_", "", n)
    return n


class Report:
    def __init__(self):
        self.rows = []
        self.failures = 0
        self.expected = 0

    def add(self, what: str, ok: bool, detail: str = "", expected: bool = False):
        if ok:
            status = "same"
        elif expected:
            status = "differs (expected)"
            self.expected += 1
        else:
            status = "DIFFERS"
            self.failures += 1
        self.rows.append((what, status, detail))

    def print(self, out=sys.stdout):
        w = max(len(r[0]) for r in self.rows)
        for what, status, detail in self.rows:
            print(f"  {what:<{w}}  {status}{'  ' + detail if detail else ''}", file=out)


def compare(a: dict, b: dict, i: int = 0) -> Report:
    r = Report()
    ba, bb = a["beam"][i], b["beam"][i]
    d = _angle(ba["direction"], bb["direction"])
    r.add("beam direction", d < TOL_DEG, f"{d:.2e} deg")
    wl = abs(ba["wavelength"] - bb["wavelength"]) / ba["wavelength"]
    r.add("wavelength", wl < TOL_REL, f"{ba['wavelength']} / {bb['wavelength']}")
    pf = abs(ba.get("polarization_fraction", 0.999) - bb.get("polarization_fraction", 0.999))
    r.add("polarization fraction", pf < 1e-9,
          f"{ba.get('polarization_fraction')} / {bb.get('polarization_fraction')}")
    pn = _angle(ba.get("polarization_normal", [0, 1, 0]), bb.get("polarization_normal", [0, 1, 0]))
    r.add("polarization normal", pn < TOL_DEG, f"{pn:.2e} deg")

    pa, pb = a["detector"][i]["panels"], b["detector"][i]["panels"]
    r.add("panel count", len(pa) == len(pb), f"{len(pa)} / {len(pb)}")
    for k, (x, y) in enumerate(zip(pa, pb)):
        p = f"panel {k} "
        o = float(np.linalg.norm(np.subtract(x["origin"], y["origin"])))
        r.add(p + "origin", o < TOL_MM, f"{o:.2e} mm")
        for ax in ("fast_axis", "slow_axis"):
            d = _angle(x[ax], y[ax])
            r.add(p + ax, d < TOL_DEG, f"{d:.2e} deg")
        r.add(p + "pixel size", np.allclose(x["pixel_size"], y["pixel_size"], atol=1e-9),
              f"{x['pixel_size']} / {y['pixel_size']}")
        r.add(p + "image size", list(x["image_size"]) == list(y["image_size"]),
              f"{x['image_size']} / {y['image_size']}")
        r.add(p + "trusted range", np.allclose(x["trusted_range"], y["trusted_range"]),
              f"{x['trusted_range']} / {y['trusted_range']}")
        r.add(p + "thickness", abs(x.get("thickness", 0) - y.get("thickness", 0)) < 1e-9,
              f"{x.get('thickness')} / {y.get('thickness')}")
        r.add(p + "material", x.get("material") == y.get("material"),
              f"{x.get('material')} / {y.get('material')}")
        if "mu" in x and "mu" in y:
            r.add(p + "mu", abs(x["mu"] - y["mu"]) <= 1e-6 * abs(x["mu"]),
                  f"{x['mu']} / {y['mu']}")
        if "px_mm_strategy" in x and "px_mm_strategy" in y:
            r.add(p + "px_mm strategy", x["px_mm_strategy"] == y["px_mm_strategy"],
                  f"{x['px_mm_strategy']} / {y['px_mm_strategy']}")
        if "mask" in x or "mask" in y:
            r.add(p + "mask list", x.get("mask", []) == y.get("mask", []),
                  f"{len(x.get('mask', []))} / {len(y.get('mask', []))} rectangles "
                  "(compare effective masks with cbf2nxmx.verify)", expected=True)
        if "identifier" in x or "identifier" in y:
            r.add(p + "identifier", x.get("identifier") == y.get("identifier"),
                  f"{x.get('identifier')!r} / {y.get('identifier')!r}", expected=True)

    ga, gb = a["goniometer"][i], b["goniometer"][i]
    axa, Fa, Sa = goniometer_matrices(ga)
    axb, Fb, Sb = goniometer_matrices(gb)
    d = _angle(axa, axb)
    r.add("rotation axis", d < TOL_DEG, f"{d:.2e} deg")
    r.add("fixed rotation", np.allclose(Fa, Fb, atol=1e-9), f"max {np.abs(Fa - Fb).max():.1e}")
    r.add("setting rotation", np.allclose(Sa, Sb, atol=1e-9), f"max {np.abs(Sa - Sb).max():.1e}")
    if "axes" in ga and "axes" in gb:
        r.add("goniometer axes", len(ga["axes"]) == len(gb["axes"])
              and all(_angle(u, v) < TOL_DEG and abs(np.linalg.norm(u) - np.linalg.norm(v)) < 1e-9
                      for u, v in zip(ga["axes"], gb["axes"])),
              f"{len(ga['axes'])} / {len(gb['axes'])}")
        na = [_norm_name(n) for n in ga.get("names", [])]
        nb = [_norm_name(n) for n in gb.get("names", [])]
        r.add("axis names", na == nb, f"{ga.get('names')} / {gb.get('names')}", expected=True)
        r.add("scan axis index", ga["scan_axis"] == gb["scan_axis"],
              f"{ga['scan_axis']} / {gb['scan_axis']}")

    ra, oa, ea, pa_ = _scan(a["scan"][i])
    rb, ob, eb, pb_ = _scan(b["scan"][i])
    r.add("image range", list(ra) == list(rb), f"{ra} / {rb}")
    r.add("oscillation start", abs(oa[0] - ob[0]) < TOL_DEG, f"{oa[0]} / {ob[0]}")
    r.add("oscillation width", abs(oa[1] - ob[1]) < TOL_DEG, f"{oa[1]} / {ob[1]}")
    if ea and eb:
        r.add("exposure time", abs(ea[0] - eb[0]) < 1e-9, f"{ea[0]} / {eb[0]}", expected=True)
    if pa_ and pb_ and len(pa_) > 1 and len(pb_) > 1:
        same = abs(pa_[0] - pb_[0]) < 1e-6 and abs((pa_[1] - pa_[0]) - (pb_[1] - pb_[0])) < 1e-6
        r.add("epochs", same, f"{pa_[0]:.3f}+{pa_[1]-pa_[0]:.4f}n / {pb_[0]:.3f}+{pb_[1]-pb_[0]:.4f}n",
              expected=True)
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cbf2nxmx.compare", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference")
    ap.add_argument("other")
    ap.add_argument("--strict", action="store_true", help="expected differences fail too")
    args = ap.parse_args(argv)
    a, b = load(args.reference), load(args.other)
    report = compare(a, b)
    print(f"{args.reference} vs {args.other}")
    report.print()
    fails = report.failures + (report.expected if args.strict else 0)
    print(f"{report.failures} differences, {report.expected} expected differences")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
