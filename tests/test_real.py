"""The two Diamond example sweeps, if CBF2NXMX_TEST_DATA points at them.

The directory holds CIX1_1_000NN.cbf and l-cyst_01_000NN.cbf (any number of
leading images) and the dials.import results cix/imported.expt and
l-cys/imported.expt made from the whole sweeps.
"""

import os
from pathlib import Path

import pytest

from cbf2nxmx.compare import compare, load
from cbf2nxmx.geometry import build_sweep
from cbf2nxmx.nxmx import WriteOptions, write_nxmx
from cbf2nxmx.verify import verify_direct

DATA = os.environ.get("CBF2NXMX_TEST_DATA")
pytestmark = pytest.mark.skipif(not DATA, reason="CBF2NXMX_TEST_DATA not set")

CASES = [("CIX1_1_", "cix"), ("l-cyst_01_", "l-cys")]


@pytest.mark.parametrize("prefix,ref", CASES)
def test_against_dials_import(tmp_path, prefix, ref):
    files = sorted(Path(DATA).glob(f"{prefix}0000[1-5].cbf"))
    assert files
    sweep = build_sweep(files)
    out = tmp_path / "x.nxs"
    write_nxmx(sweep, out, WriteOptions(quiet=True, nproc=2, dtype="uint32"))
    reference = load(str(Path(DATA) / ref / "imported.expt"))
    report = compare(reference, load(str(out)))
    failures = [r for r in report.rows if r[1] == "DIFFERS" and r[0] != "image range"]
    assert not failures, failures
    assert verify_direct(str(out), [str(f) for f in files]) == 0
