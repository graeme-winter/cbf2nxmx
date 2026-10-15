"""Frame changes and transformation chains, shared by writer and reader.

A chain is applied innermost first. Each step maps a point p to
    R p + offset          for a rotation (angle = value * |vector|), or
    p + offset + value * vector   for a translation.
This is how dxtbx reads both imgCIF axes and NeXus transformations.
"""

from __future__ import annotations

import numpy as np

# NeXus/McStas (beam along +z, y up) <-> imgCIF/dxtbx (beam along -z): a half
# turn about y. It is its own inverse.
MCSTAS_TO_IMGCIF = np.diag([-1.0, 1.0, -1.0])


def to_mcstas(v) -> np.ndarray:
    return MCSTAS_TO_IMGCIF @ np.asarray(v, dtype=float)


def to_imgcif(v) -> np.ndarray:
    return MCSTAS_TO_IMGCIF @ np.asarray(v, dtype=float)


def rotation(axis, degrees: float) -> np.ndarray:
    """Right-handed rotation by `degrees` about `axis` (normalised here)."""
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n == 0:
        return np.eye(3)
    x, y, z = a / n
    t = np.radians(degrees)
    c, s = np.cos(t), np.sin(t)
    C = 1 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def compose(steps) -> tuple[np.ndarray, np.ndarray]:
    """steps: iterable of (type, vector, offset, value), innermost first.
    Returns (R, t) with x_outer = R x_inner + t."""
    R = np.eye(3)
    t = np.zeros(3)
    for kind, vector, offset, value in steps:
        vector = np.asarray(vector, dtype=float)
        offset = np.asarray(offset, dtype=float)
        if kind == "rotation":
            r = rotation(vector, value * np.linalg.norm(vector))
            R = r @ R
            t = r @ t + offset
        elif kind == "translation":
            t = t + offset + value * vector
        else:
            raise ValueError(f"transformation type {kind!r}")
    return R, t


def beam_centre(origin, fast, slow, pixel_fast, pixel_slow):
    """Where a beam from the sample travelling along -z (imgCIF frame) meets
    the panel's plane: (x px, y px, distance mm), or None if it never does."""
    f = np.asarray(fast, float) / np.linalg.norm(fast)
    s = np.asarray(slow, float) / np.linalg.norm(slow)
    M = np.column_stack([f, s, [0.0, 0.0, 1.0]])
    if abs(np.linalg.det(M)) < 1e-12:
        return None
    a, b, lam = np.linalg.solve(M, -np.asarray(origin, float))
    if lam <= 0:
        return None
    return a / pixel_fast, b / pixel_slow, lam
