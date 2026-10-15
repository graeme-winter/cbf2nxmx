"""The PILATUS_1.2 ("mini-CBF") header held in _array_data.header_contents.

Each line is '# Key value [unit]'. The keys used here are parsed to numbers
in SI-ish units stated in each field's name. Every line is also kept as text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["PilatusHeader", "parse_header"]

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_DATE = re.compile(r"^#\s*(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?)\s*$")
_DATE_SLASH = re.compile(r"^#\s*(\d{4}/\w{3}/\d\d \d\d:\d\d:\d\d(?:\.\d+)?)\s*$")


@dataclass
class PilatusHeader:
    lines: list[str] = field(default_factory=list)
    fields: dict[str, str] = field(default_factory=dict)
    date: str | None = None
    detector: str | None = None  # the whole 'Detector:' text
    model: str | None = None  # e.g. 'PILATUS3 6M'
    serial: str | None = None  # e.g. '60-0119'
    pixel_size_m: tuple[float, float] | None = None
    sensor_material: str | None = None  # 'Si', 'CdTe', ...
    sensor_thickness_m: float | None = None
    exposure_time_s: float | None = None
    exposure_period_s: float | None = None
    tau_s: float | None = None
    count_cutoff: int | None = None
    threshold_ev: float | None = None
    gain_setting: str | None = None
    wavelength_a: float | None = None
    detector_distance_m: float | None = None
    beam_xy_px: tuple[float, float] | None = None
    flux: float | None = None
    filter_transmission: float | None = None
    polarization: float | None = None
    angles_deg: dict[str, float] = field(default_factory=dict)
    oscillation_axis: str | None = None

    # The keys whose values change from one image to the next in a sweep.
    VARYING = ("Start_angle", "Omega", "Phi", "Chi", "Kappa", "Detector_2theta")


_MATERIALS = {"silicon": "Si", "cdte": "CdTe", "gaas": "GaAs", "germanium": "Ge"}


def _first_number(s: str) -> float | None:
    m = re.search(_NUM, s)
    return float(m.group(0)) if m else None


def parse_header(text: str) -> PilatusHeader:
    h = PilatusHeader()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        h.lines.append(line)
        m = _DATE.match(line) or _DATE_SLASH.match(line)
        if m:
            h.date = m.group(1)
            continue
        body = line.lstrip("#").strip()
        m = re.match(r"^([A-Za-z][A-Za-z0-9_]*)\s*[:=]?\s*(.*)$", body)
        if not m:
            continue
        key, rest = m.group(1), m.group(2).strip()

        if key == "Detector":
            h.detector = rest
            parts = rest.split(",", 1)
            h.model = parts[0].strip()
            sm = re.search(r"S/N\s+(.*)$", rest)
            if sm:
                h.serial = sm.group(1).strip()
            h.fields[key] = rest
            continue
        if body.lower().split()[1:2] == ["sensor,"] or re.match(
            r"^\w+ sensor, thickness", body
        ):
            mat, _, thick = body.partition(" sensor, thickness")
            h.sensor_material = _MATERIALS.get(mat.strip().lower(), mat.strip())
            h.sensor_thickness_m = _first_number(thick)
            h.fields["Sensor"] = body
            continue

        h.fields[key] = rest
        num = _first_number(rest)
        if key == "Pixel_size":
            nums = re.findall(_NUM, rest)
            if len(nums) >= 2:
                h.pixel_size_m = (float(nums[0]), float(nums[1]))
        elif key == "Exposure_time":
            h.exposure_time_s = num
        elif key == "Exposure_period":
            h.exposure_period_s = num
        elif key == "Tau":
            h.tau_s = num
        elif key == "Count_cutoff":
            h.count_cutoff = int(num) if num is not None else None
        elif key == "Threshold_setting":
            h.threshold_ev = num
        elif key == "Gain_setting":
            h.gain_setting = rest
        elif key == "Wavelength":
            h.wavelength_a = num
        elif key == "Detector_distance":
            h.detector_distance_m = num
        elif key == "Beam_xy":
            nums = re.findall(_NUM, rest)
            if len(nums) >= 2:
                h.beam_xy_px = (float(nums[0]), float(nums[1]))
        elif key == "Flux":
            h.flux = num
        elif key == "Filter_transmission":
            h.filter_transmission = num
        elif key == "Polarization":
            h.polarization = num
        elif key == "Oscillation_axis":
            h.oscillation_axis = rest
        elif key in (
            "Start_angle",
            "Angle_increment",
            "Detector_2theta",
            "Alpha",
            "Kappa",
            "Phi",
            "Phi_increment",
            "Omega",
            "Omega_increment",
            "Chi",
            "Chi_increment",
        ):
            if num is not None:
                h.angles_deg[key] = num
    return h
