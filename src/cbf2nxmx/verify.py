"""Check an NXmx file's pixels against the CBF images it came from.

    cbf2nxmx.verify out.nxs IMAGES... [--every N]
    python3 -m cbf2nxmx.verify out.nxs imported.expt --dxtbx [--every N]

Without --dxtbx, each CBF is decoded here and compared with the HDF5 frame:
counts must be equal, -1 must be the gap marker (2^n-1), -2 the bad marker
(2^n-2), and pixel_mask must flag both.

With --dxtbx, both are read through dxtbx, as DIALS reads them. The CBF side
is an .expt from dials.import or the image files. The effective masks
(imageset.get_mask, which includes the trusted range) must be identical, and
so must the raw counts wherever they are unmasked.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def verify_direct(nxs: str, files, every: int = 1) -> int:
    import h5py
    import hdf5plugin  # noqa: F401  registers the filters

    from .cbf import read_header, read_pixels
    from .cli import expand

    files = expand(files)
    bad = 0
    with h5py.File(nxs, "r") as f:
        data = f["/entry/data/data"]
        mask = f["/entry/instrument/detector/pixel_mask"][()]
        if data.shape[0] != len(files):
            print(f"{data.shape[0]} frames in {nxs}, {len(files)} images given")
            return 1
        from .pixels import Policy

        policy = Policy(data.dtype.type, "clip")
        sat = int(f["/entry/instrument/detector/saturation_value"][()])
        for i in range(0, len(files), every):
            a = read_pixels(read_header(files[i])).astype(np.int64)
            b = data[i].astype(np.int64)
            ok = a >= 0
            problems = []
            clipped = ok & (a > policy.top)
            if np.any(a[ok & ~clipped] != b[ok & ~clipped]):
                problems.append(f"{int(np.count_nonzero(a[ok & ~clipped] != b[ok & ~clipped]))} counts differ")
            if np.any(b[clipped] <= sat):
                problems.append("clipped counts not above saturation_value")
            if np.any(b[a == -1] != policy.gap_value) or np.any((mask[a == -1] & 1) == 0):
                problems.append("gap pixels not marked")
            if np.any(b[a == -2] != policy.bad_value) or np.any((mask[a == -2] & 2) == 0):
                problems.append("bad pixels not marked")
            if problems:
                bad += 1
                print(f"  image {i + 1} ({files[i].name}): " + "; ".join(problems))
        n = len(range(0, len(files), every))
        print(f"{n - bad} of {n} images checked agree")
    return 1 if bad else 0


def _imageset(arg: list[str]):
    from dxtbx.model.experiment_list import ExperimentList, ExperimentListFactory

    if len(arg) == 1 and arg[0].endswith((".expt", ".json")):
        return ExperimentList.from_file(arg[0])[0].imageset
    return ExperimentListFactory.from_filenames(arg)[0].imageset


def verify_dxtbx(nxs: str, cbf_args, every: int = 1) -> int:
    ia = _imageset(list(cbf_args))
    ib = _imageset([nxs])
    if len(ia) != len(ib):
        print(f"{len(ia)} CBF images, {len(ib)} NXmx images")
        return 1
    bad = 0
    for i in range(0, len(ia), every):
        ma = [m.as_numpy_array() for m in ia.get_mask(i)]
        mb = [m.as_numpy_array() for m in ib.get_mask(i)]
        ra = [r.as_numpy_array() for r in ia.get_raw_data(i)]
        rb = [r.as_numpy_array() for r in ib.get_raw_data(i)]
        problems = []
        for p, (xa, xb, da, db) in enumerate(zip(ma, mb, ra, rb)):
            if xa.shape != xb.shape:
                problems.append(f"panel {p} shape {xa.shape} / {xb.shape}")
                continue
            if np.any(xa != xb):
                problems.append(
                    f"panel {p}: masks differ at {int(np.count_nonzero(xa != xb))} pixels "
                    f"({int(np.count_nonzero(xa & ~xb))} only CBF-valid, "
                    f"{int(np.count_nonzero(~xa & xb))} only NXmx-valid)"
                )
            both = xa & xb
            diff = np.count_nonzero(da[both].astype(np.int64) != db[both].astype(np.int64))
            if diff:
                problems.append(f"panel {p}: {diff} unmasked counts differ")
        if problems:
            bad += 1
            print(f"  image {i + 1}: " + "; ".join(problems))
    n = len(range(0, len(ia), every))
    print(f"{n - bad} of {n} images checked agree (dxtbx masks and counts)")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cbf2nxmx.verify", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nxs")
    ap.add_argument("cbf", nargs="+", help="the CBF images, or (with --dxtbx) an .expt")
    ap.add_argument("--every", type=int, default=1, help="check every Nth image")
    ap.add_argument("--dxtbx", action="store_true", help="read both through dxtbx")
    args = ap.parse_args(argv)
    if args.dxtbx:
        return verify_dxtbx(args.nxs, args.cbf, args.every)
    return verify_direct(args.nxs, args.cbf, args.every)


if __name__ == "__main__":
    sys.exit(main())
