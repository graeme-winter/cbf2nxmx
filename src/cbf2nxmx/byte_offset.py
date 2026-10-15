"""The CBF byte-offset compression (x-CBF_BYTE_OFFSET), decoded with numpy.

Each element is stored as the difference from the one before it, the first
from zero. A difference in -127..127 takes one byte; otherwise the byte 0x80
escapes to a little-endian int16, whose value -32768 escapes to an int32,
whose value -2**31 escapes to an int64.

Decoding is sequential in principle, but escapes are rare. So only the 0x80
bytes are walked in Python, which is enough to know which bytes belong to
escapes. Everything else is numpy: a cumulative sum over the differences.
"""

from __future__ import annotations

import numpy as np

__all__ = ["decode", "encode"]

_I16_ESCAPE = -32768
_I32_ESCAPE = -(2**31)


def decode(data: bytes, n_elements: int) -> np.ndarray:
    """Decode a byte-offset stream to `n_elements` int32 values.

    `data` is the binary section exactly as X-Binary-Size gives it.
    """
    buf = bytes(data)
    b = np.frombuffer(buf, dtype=np.uint8)
    size = b.size

    candidates = np.flatnonzero(b == 0x80)
    starts: list[int] = []
    lengths: list[int] = []
    values: list[int] = []
    next_free = 0
    from_bytes = int.from_bytes
    for p in candidates.tolist():
        if p < next_free:
            continue  # a 0x80 inside an earlier escape's payload
        if p + 3 > size:
            raise ValueError(f"byte-offset data truncated in an escape at {p}")
        v = from_bytes(buf[p + 1 : p + 3], "little", signed=True)
        length = 3
        if v == _I16_ESCAPE:
            if p + 7 > size:
                raise ValueError(f"byte-offset data truncated in an escape at {p}")
            v = from_bytes(buf[p + 3 : p + 7], "little", signed=True)
            length = 7
            if v == _I32_ESCAPE:
                if p + 15 > size:
                    raise ValueError(
                        f"byte-offset data truncated in an escape at {p}"
                    )
                v = from_bytes(buf[p + 7 : p + 15], "little", signed=True)
                length = 15
        starts.append(p)
        lengths.append(length)
        values.append(v)
        next_free = p + length

    deltas = b.view(np.int8).astype(np.int64)
    if starts:
        s = np.asarray(starts, dtype=np.int64)
        ln = np.asarray(lengths, dtype=np.int64)
        # Bytes p+1 .. p+length-1 of each escape are payload, not differences.
        edges = np.zeros(size + 1, dtype=np.int32)
        edges[s + 1] += 1
        edges[s + ln] -= 1
        payload = np.cumsum(edges[:size]) > 0
        deltas[s] = np.asarray(values, dtype=np.int64)
        deltas = deltas[~payload]

    if deltas.size < n_elements:
        raise ValueError(
            f"byte-offset data hold {deltas.size} elements, expected {n_elements}"
        )
    out = np.cumsum(deltas[:n_elements])
    if out.size and (out.min() < -(2**31) or out.max() > 2**31 - 1):
        raise ValueError("byte-offset values outside the int32 range")
    return out.astype(np.int32)


def encode(values: np.ndarray) -> bytes:
    """Encode integer values as a byte-offset stream (for tests and writers)."""
    v = np.asarray(values, dtype=np.int64).ravel()
    d = np.diff(v, prepend=np.int64(0))
    one = (d >= -127) & (d <= 127)
    two = ~one & (d >= -32767) & (d <= 32767)
    four = ~one & ~two & (d > _I32_ESCAPE) & (d <= 2**31 - 1)
    eight = ~(one | two | four)
    sizes = np.where(one, 1, np.where(two, 3, np.where(four, 7, 15)))
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    out = np.zeros(int(sizes.sum()), dtype=np.uint8)

    o = offsets[one]
    out[o] = d[one].astype(np.int8).view(np.uint8)

    def put(where, at, nbytes, x):
        raw = x.astype(f"<i{nbytes}").view(np.uint8).reshape(-1, nbytes)
        for k in range(nbytes):
            out[at + k] = raw[:, k]

    for mask in (two, four, eight):
        o = offsets[mask]
        out[o] = 0x80
    put(two, offsets[two] + 1, 2, d[two])
    for mask, lvl in ((four, 4), (eight, 8)):
        o = offsets[mask]
        put(mask, o + 1, 2, np.full(o.size, _I16_ESCAPE))
        if lvl == 4:
            put(mask, o + 3, 4, d[mask])
        else:
            put(mask, o + 3, 4, np.full(o.size, _I32_ESCAPE))
            put(mask, o + 7, 8, d[mask])
    return out.tobytes()
