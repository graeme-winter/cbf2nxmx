"""CBF int32 pixels to the HDF5 type written: uint16 or int32.

PILATUS CBF use -1 for pixels in gaps between modules and -2 for bad pixels.

* int32 (the 32-bit choice): the values pass through unchanged. -1 and -2
  have the same bit patterns as the DECTRIS/mxi unsigned markers 2**32-1
  (tile join) and 2**32-2 (bad), and dxtbx cannot read uint32 data at all.
  No real count comes near 2**31.
* uint16: -1 becomes 65535 and -2 becomes 65534, the markers, so the largest
  real count is 65533.

Either way both kinds of pixel are recorded in pixel_mask as well.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DTYPES = {"uint16": np.uint16, "int32": np.int32}

GAP_BIT = 1  # pixel_mask bit 0: gap between modules
BAD_BIT = 2  # pixel_mask bit 1: dead / bad pixel


class OverflowError_(ValueError):
    pass


@dataclass
class Policy:
    dtype: type
    overflow: str  # "error" | "clip"

    @property
    def signed(self) -> bool:
        return np.issubdtype(self.dtype, np.signedinteger)

    @property
    def gap_value(self) -> int:
        return -1 if self.signed else int(np.iinfo(self.dtype).max)

    @property
    def bad_value(self) -> int:
        return -2 if self.signed else int(np.iinfo(self.dtype).max) - 1

    @property
    def top(self) -> int:
        """The largest count the type holds (below its markers if unsigned)."""
        m = int(np.iinfo(self.dtype).max)
        return m if self.signed else m - 2

    def saturation_value(self, overload: int | None) -> int:
        """The largest trusted count: the CBF's overload - 1 (as dxtbx reads
        the CBF), never into the markers, and below any clipped value."""
        limit = self.top - 1 if self.overflow == "clip" else self.top
        if overload is None:
            return limit
        return min(overload - 1, limit)


def choose_dtype(name: str, overload: int | None) -> type:
    if name == "auto":
        if overload is not None and overload - 1 <= np.iinfo(np.uint16).max - 2:
            return np.uint16
        return np.int32
    if name == "uint32":
        raise ValueError("uint32: dxtbx cannot read it; use int32, whose -1 and -2 "
                         "are the same bits as the unsigned markers")
    if name not in DTYPES:
        raise ValueError(f"dtype {name!r}: choose from auto, {', '.join(DTYPES)}")
    return DTYPES[name]


@dataclass
class Converted:
    data: np.ndarray
    n_overflow: int
    n_other_negative: int
    maximum: int


def convert(a: np.ndarray, policy: Policy, where: str = "") -> Converted:
    top = policy.top
    over = a > top
    n_over = int(np.count_nonzero(over))
    if n_over and policy.overflow == "error":
        raise OverflowError_(
            f"{where}: {n_over} pixels above {top}, the largest count "
            f"{np.dtype(policy.dtype).name} holds below its markers (max "
            f"{int(a.max())}); use int32 or --overflow clip"
        )
    other = (a < -2)
    src = np.minimum(a, top) if n_over else a  # top < 2**31 whenever n_over
    out = src.astype(policy.dtype)  # negatives wrap for unsigned; set below
    out[a == -1] = policy.gap_value
    out[a == -2] = policy.bad_value
    n_other = int(np.count_nonzero(other))
    if n_other:
        out[other] = policy.bad_value
    return Converted(out, n_over, n_other, int(a.max()))
