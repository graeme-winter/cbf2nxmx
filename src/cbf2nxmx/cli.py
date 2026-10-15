"""cbf2nxmx command line."""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

from . import __version__
from .geometry import GeometryError, build_sweep, frame_number
from .nxmx import WriteOptions, write_nxmx

EPILOG = """
IMAGES may be a list of files, a glob ('x_*.cbf'), a template with #
standing for digits ('x_#####.cbf'), or the first image alone, in which
case every image numbered after it with the same prefix is taken.

Pixel types: uint16 or int32. In uint16, -1 (module gap) becomes 65535 and
-2 (bad pixel) 65534. In int32 they stay -1 and -2, the same bits as the
unsigned markers; uint32 is not offered because dxtbx cannot read it. Both
are also recorded in pixel_mask. 'auto' picks uint16 when the detector's
trusted maximum (count cutoff - 1) fits below 65534, otherwise int32.
"""


def expand(images: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in images:
        if "#" in item:
            m = re.search(r"#+", item)
            pattern = re.escape(item[: m.start()]) + r"(\d{%d})" % (m.end() - m.start())
            pattern += re.escape(item[m.end():]) + "$"
            parent = Path(item).parent
            rx = re.compile(pattern)
            hits = sorted(
                str(p) for p in parent.iterdir() if rx.match(str(p)) or rx.match(str(parent / p.name))
            )
            out.extend(Path(h) for h in hits)
        elif any(c in item for c in "*?["):
            out.extend(Path(h) for h in sorted(glob.glob(item)))
        else:
            out.append(Path(item))
    if len(images) == 1 and len(out) == 1 and "#" not in images[0]:
        p = out[0]
        n = frame_number(p)
        if n is not None:
            m = re.search(r"(\d+)(\.cbf(?:\.gz|\.bz2)?)$", p.name)
            width = len(m.group(1))
            prefix = p.name[: m.start(1)]
            suffix = m.group(2)
            seq = []
            k = n
            while True:
                q = p.with_name(f"{prefix}{k:0{width}d}{suffix}")
                if not q.exists():
                    break
                seq.append(q)
                k += 1
            if len(seq) > 1:
                out = seq
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="cbf2nxmx",
        description="Convert a sweep of CBF images (full imgCIF headers) to NXmx HDF5.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("images", nargs="+")
    ap.add_argument("-o", "--output", required=True, help="the NXmx file to write")
    ap.add_argument("--dtype", default="auto", choices=["auto", "uint16", "int32"])
    ap.add_argument("--compression", default="bslz4", choices=["bslz4", "lz4", "none"])
    ap.add_argument(
        "--overflow", default="error", choices=["error", "clip"],
        help="counts too large for the type: fail, or clip and lower saturation_value below them",
    )
    ap.add_argument("--image-range", help="first,last (1-based, inclusive) of the images found")
    ap.add_argument("--frames-per-file", type=int, default=0,
                    help="write data files of this many frames, joined by a VDS in the master")
    ap.add_argument("--nproc", type=int, default=0, help="decoding processes (0: all cores)")
    ap.add_argument("--polarization", action="store_true",
                    help="write the CBF's polarisation ratio (dxtbx ignores it in CBF)")
    ap.add_argument("--no-md5", action="store_true", help="skip the Content-MD5 check")
    ap.add_argument("--no-consistency-check", action="store_true",
                    help="do not insist that every header agrees with the first")
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--version", action="version", version=f"cbf2nxmx {__version__}")
    args = ap.parse_args(argv)

    files = expand(args.images)
    if not files:
        print("cbf2nxmx: no images found", file=sys.stderr)
        return 2
    if args.image_range:
        a, b = (int(x) for x in args.image_range.split(","))
        files = files[a - 1 : b]
    try:
        sweep = build_sweep(files, check_consistency=not args.no_consistency_check)
    except GeometryError as e:
        print(f"cbf2nxmx: {e}", file=sys.stderr)
        return 1
    if not args.quiet:
        print(
            f"cbf2nxmx: {sweep.n_images} images {files[0].name} .. {files[-1].name}; "
            f"scan {sweep.scan_axis} from {sweep.scan_start} by {sweep.scan_increment} deg",
            file=sys.stderr,
        )
    opts = WriteOptions(
        dtype=args.dtype, compression=args.compression, overflow=args.overflow,
        nproc=args.nproc, frames_per_file=args.frames_per_file,
        polarization=args.polarization, check_md5=not args.no_md5, quiet=args.quiet,
    )
    try:
        report = write_nxmx(sweep, args.output, opts)
    except Exception as e:  # noqa: BLE001 - report and fail cleanly
        print(f"cbf2nxmx: {e}", file=sys.stderr)
        out = Path(args.output)
        stem = out.name[: -len(out.suffix)] if out.suffix else out.name
        stem = stem[: -len("_master")] if stem.endswith("_master") else stem
        for p in [out, *out.parent.glob(f"{stem}_data_[0-9][0-9][0-9][0-9][0-9][0-9].h5")]:
            p.unlink(missing_ok=True)
        return 1
    print(
        f"cbf2nxmx: wrote {report.path} ({report.dtype}, saturation_value "
        f"{report.saturation_value}, max count {report.max_count}, "
        f"{report.masked_gap} gap + {report.masked_bad} bad pixels masked) "
        f"in {report.seconds:.1f} s"
    )
    for note in sweep.notes + report.notes:
        print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
