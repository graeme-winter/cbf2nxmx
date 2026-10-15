"""Reading a CBF file: its CIF text, its MIME binary header and its pixels."""

from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import byte_offset
from .cif import CifBlock, parse
from .pilatus import PilatusHeader, parse_header

__all__ = ["CbfFile", "read_header", "read_pixels", "CbfError"]

_MARKER = b"\x0c\x1a\x04\xd5"
_SECTION = b"--CIF-BINARY-FORMAT-SECTION--"


class CbfError(RuntimeError):
    pass


@dataclass
class BinaryInfo:
    conversions: str
    size: int
    element_type: str
    byte_order: str
    md5: str | None
    n_elements: int
    fast: int
    slow: int
    data_offset: int  # into the file, of the first binary byte


@dataclass
class CbfFile:
    path: Path
    block: CifBlock
    header: PilatusHeader | None
    binary: BinaryInfo
    cif_text: str


def _read_until_marker(path: Path, chunk: int = 16384) -> bytes:
    with open(path, "rb") as f:
        data = b""
        while True:
            more = f.read(chunk)
            if not more:
                raise CbfError(f"{path}: no binary section marker found")
            data += more
            i = data.find(_MARKER)
            if i >= 0:
                return data[: i + len(_MARKER)]


def _mime(text: str, path: Path, data_offset: int) -> BinaryInfo:
    def get(key: str, required: bool = True) -> str | None:
        m = re.search(rf"^\s*{re.escape(key)}\s*[:=]\s*(.+?)\s*$", text, re.M | re.I)
        if not m:
            if required:
                raise CbfError(f"{path}: no {key} in the binary header")
            return None
        return m.group(1).strip().strip('"').strip(";")

    conv = re.search(r'conversions\s*=\s*"([^"]+)"', text)
    conversions = conv.group(1) if conv else "x-CBF_NONE"
    return BinaryInfo(
        conversions=conversions,
        size=int(get("X-Binary-Size")),
        element_type=get("X-Binary-Element-Type"),
        byte_order=get("X-Binary-Element-Byte-Order", False) or "LITTLE_ENDIAN",
        md5=get("Content-MD5", False),
        n_elements=int(get("X-Binary-Number-of-Elements")),
        fast=int(get("X-Binary-Size-Fastest-Dimension")),
        slow=int(get("X-Binary-Size-Second-Dimension")),
        data_offset=data_offset,
    )


def read_header(path: str | Path) -> CbfFile:
    """Everything but the pixels: the CIF tables, the PILATUS header and the
    binary section's description."""
    path = Path(path)
    head = _read_until_marker(path)
    s = head.find(_SECTION)
    if s < 0:
        raise CbfError(f"{path}: no {_SECTION.decode()}")
    if head.find(_SECTION, s + 1) >= 0:
        raise CbfError(f"{path}: more than one binary section is not supported")
    text = head[:s].decode("latin-1")
    # The CIF text ends at the '_array_data.data' whose ';' field holds the
    # binary section.
    k = text.rfind("_array_data.data")
    cif_text = text[:k] if k >= 0 else text
    blocks = parse(cif_text)
    if not blocks:
        raise CbfError(f"{path}: no CIF data block")
    if len(blocks) > 1:
        raise CbfError(f"{path}: more than one data block is not supported")
    block = blocks[0]
    contents = block.value("array_data", "header_contents")
    pil = parse_header(contents) if contents else None
    mime = head[s : len(head) - len(_MARKER)].decode("latin-1")
    binary = _mime(mime, path, len(head))
    return CbfFile(path=path, block=block, header=pil, binary=binary, cif_text=cif_text)


_TYPES = {
    "signed 32-bit integer": np.int32,
    "unsigned 32-bit integer": np.uint32,
    "signed 16-bit integer": np.int16,
    "unsigned 16-bit integer": np.uint16,
    "signed 8-bit integer": np.int8,
    "unsigned 8-bit integer": np.uint8,
}


def read_pixels(cbf: CbfFile, check_md5: bool = True) -> np.ndarray:
    """The image as int64-safe int32 (slow, fast)."""
    b = cbf.binary
    with open(cbf.path, "rb") as f:
        f.seek(b.data_offset)
        data = f.read(b.size)
    if len(data) != b.size:
        raise CbfError(f"{cbf.path}: binary section truncated")
    if check_md5 and b.md5:
        digest = base64.b64encode(hashlib.md5(data).digest()).decode()
        if digest != b.md5:
            raise CbfError(f"{cbf.path}: MD5 mismatch ({digest} != {b.md5})")
    if b.n_elements != b.fast * b.slow:
        raise CbfError(f"{cbf.path}: {b.n_elements} elements != {b.fast} x {b.slow}")
    conv = b.conversions.lower()
    if conv == "x-cbf_byte_offset":
        values = byte_offset.decode(data, b.n_elements)
    elif conv == "x-cbf_none":
        t = _TYPES.get(b.element_type.lower())
        if t is None:
            raise CbfError(f"{cbf.path}: element type {b.element_type!r} unsupported")
        order = "<" if b.byte_order.upper().startswith("LITTLE") else ">"
        values = np.frombuffer(data, dtype=np.dtype(t).newbyteorder(order))
        values = values[: b.n_elements].astype(np.int32)
    else:
        raise CbfError(f"{cbf.path}: compression {b.conversions!r} unsupported")
    return values.reshape(b.slow, b.fast)
