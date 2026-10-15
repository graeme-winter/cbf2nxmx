"""CBF int32 pixels to an unsigned HDF5 type, DECTRIS-style.

PILATUS CBF use -1 for pixels in gaps between modules and -2 for bad pixels.
In an unsigned image the two largest values are the markers, as DECTRIS and
mxi have it: 2**n - 1 for a gap (a tile join) and 2**n - 2 for a bad pixel.
For int32 that is exactly the two's-complement bit pattern. The largest real
count is therefore 2**n - 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DTYPES = {"uint16": np.uint16, "uint32": np.uint32}

GAP_BIT = 1  # pixel_mask bit 0: gap between modules
BAD_BIT = 2  # pixel_mask bit 1: dead / bad pixel


class OverflowError_(ValueError):
    pass


@dataclass
class Policy:
    dtype: type
    overflow: str  # "error" | "clip"

    @property
    def maximum(self) -> int:
        return int(np.iinfo(self.dtype).max)

    @property
    def gap_value(self) -> int:
        return self.maximum

    @property
    def bad_value(self) -> int:
        return self.maximum - 1

    @property
    def top(self) -> int:
        """The largest count the type can hold below its markers."""
        return self.maximum - 2

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
        return np.uint32
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
            f"{int(a.max())}); use a wider type or --overflow clip"
        )
    neg = a < 0
    other = neg & (a != -1) & (a != -2)
    src = np.minimum(a, top) if n_over else a  # top < 2**31 whenever n_over
    out = src.astype(policy.dtype)  # negatives wrap; set explicitly below
    out[a == -1] = policy.gap_value
    out[a == -2] = policy.bad_value
    n_other = int(np.count_nonzero(other))
    if n_other:
        out[other] = policy.bad_value
    return Converted(out, n_over, n_other, int(a.max()))
