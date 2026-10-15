"""Synthetic full-imgCIF CBF files for tests."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import numpy as np

from cbf2nxmx.byte_offset import encode

HEADER = """\
# Detector: PILATUS3 2M, S/N 24-0000
# 2026-10-15T12:00:00.000
# Pixel_size 172e-6 m x 172e-6 m
# Silicon sensor, thickness 0.000450 m
# Exposure_time 0.0450000 s
# Exposure_period 0.0500000 s
# Count_cutoff {cutoff} counts
# Threshold_setting: 6000 eV
# Wavelength {wavelength} A
# Start_angle {angle:.4f} deg.
"""


def axis_rows(axes):
    out = []
    for a in axes:
        vec = " ".join(f"{x:.10g}" for x in a["vector"])
        off = " ".join(f"{x:.10g}" for x in a.get("offset", (0, 0, 0)))
        if a["type"] == "rotation" and not any(a.get("offset", (0, 0, 0))):
            off = ". . ."
        out.append(f"{a['id']} {a['type']} {a['equipment']} {a.get('depends_on') or '.'} {vec} {off}")
    return "\n".join(out)


def make_cbf(path: Path, data: np.ndarray, axes: list[dict], frame_values: dict,
             increments: dict, wavelength=0.9763, cutoff=1048500, fast_axis="ELEMENT_X",
             slow_axis="ELEMENT_Y", pixel=0.172, date="2026-10-15T12:00:00.000",
             fast_increment=None, slow_increment=None):
    slow, fast = data.shape
    fi = pixel if fast_increment is None else fast_increment
    si = pixel if slow_increment is None else slow_increment
    scan_rows = []
    frame_rows = []
    by_id = {a["id"]: a for a in axes}
    for k, v in frame_values.items():
        a = by_id[k]
        inc = increments.get(k, 0.0)
        if a["type"] == "rotation":
            scan_rows.append(f"SCAN1 {k} {v:.4f} {inc:.4f} {inc:.4f} 0.0 0.0 0.0")
            frame_rows.append(f"FRAME1 {k} {v:.4f} 0.0")
        else:
            scan_rows.append(f"SCAN1 {k} 0.0 0.0 0.0 {v:.4f} 0.0 0.0")
            frame_rows.append(f"FRAME1 {k} 0.0 {v:.4f}")
    angle = next(iter(frame_values.values()))
    header = HEADER.format(cutoff=cutoff, wavelength=wavelength, angle=angle)
    text = f"""###CBF: VERSION 1.5, synthetic

data_synthetic

_array_data.header_convention "PILATUS_1.2"
_array_data.header_contents
;
{header};

_diffrn.id TEST
_diffrn.crystal_id xtal001

loop_
_diffrn_source.diffrn_id
_diffrn_source.source
_diffrn_source.type
TEST synchrotron 'Test Beamline'

loop_
_diffrn_radiation_wavelength.id
_diffrn_radiation_wavelength.wavelength
_diffrn_radiation_wavelength.wt
WAVELENGTH1 {wavelength} 1

loop_
_diffrn_scan_axis.scan_id
_diffrn_scan_axis.axis_id
_diffrn_scan_axis.angle_start
_diffrn_scan_axis.angle_range
_diffrn_scan_axis.angle_increment
_diffrn_scan_axis.displacement_start
_diffrn_scan_axis.displacement_range
_diffrn_scan_axis.displacement_increment
{chr(10).join(scan_rows)}

loop_
_diffrn_scan_frame.frame_id
_diffrn_scan_frame.frame_number
_diffrn_scan_frame.integration_time
_diffrn_scan_frame.exposure_time
_diffrn_scan_frame.scan_id
_diffrn_scan_frame.date
FRAME1 1 0.045 0.050 SCAN1 {date}

loop_
_diffrn_scan_frame_axis.frame_id
_diffrn_scan_frame_axis.axis_id
_diffrn_scan_frame_axis.angle
_diffrn_scan_frame_axis.displacement
{chr(10).join(frame_rows)}

loop_
_axis.id
_axis.type
_axis.equipment
_axis.depends_on
_axis.vector[1] _axis.vector[2] _axis.vector[3]
_axis.offset[1] _axis.offset[2] _axis.offset[3]
{axis_rows(axes)}

loop_
_array_structure_list.array_id
_array_structure_list.index
_array_structure_list.dimension
_array_structure_list.precedence
_array_structure_list.direction
_array_structure_list.axis_set_id
ARRAY1 1 {fast} 1 increasing {fast_axis}
ARRAY1 2 {slow} 2 increasing {slow_axis}

loop_
_array_structure_list_axis.axis_set_id
_array_structure_list_axis.axis_id
_array_structure_list_axis.displacement
_array_structure_list_axis.displacement_increment
{fast_axis} {fast_axis} 0.0 {fi}
{slow_axis} {slow_axis} 0.0 {si}

loop_
_array_intensities.array_id
_array_intensities.binary_id
_array_intensities.linearity
_array_intensities.gain
_array_intensities.gain_esd
_array_intensities.overload
_array_intensities.undefined_value
ARRAY1 1 linear 1.0 . {cutoff} -1

_array_data.data
;
"""
    binary = encode(data)
    md5 = base64.b64encode(hashlib.md5(binary).digest()).decode()
    mime = (
        "--CIF-BINARY-FORMAT-SECTION--\r\n"
        'Content-Type: application/octet-stream;\r\n     conversions="x-CBF_BYTE_OFFSET"\r\n'
        "Content-Transfer-Encoding: BINARY\r\n"
        f"X-Binary-Size: {len(binary)}\r\n"
        "X-Binary-ID: 1\r\n"
        'X-Binary-Element-Type: "signed 32-bit integer"\r\n'
        "X-Binary-Element-Byte-Order: LITTLE_ENDIAN\r\n"
        f"Content-MD5: {md5}\r\n"
        f"X-Binary-Number-of-Elements: {data.size}\r\n"
        f"X-Binary-Size-Fastest-Dimension: {fast}\r\n"
        f"X-Binary-Size-Second-Dimension: {slow}\r\n"
        "X-Binary-Size-Padding: 4095\r\n\r\n"
    )
    tail = b"\x00" * 4095 + b"\r\n--CIF-BINARY-FORMAT-SECTION----\r\n;\r\n\r\n"
    Path(path).write_bytes(
        text.encode("latin-1") + mime.encode("latin-1") + b"\x0c\x1a\x04\xd5" + binary + tail
    )
