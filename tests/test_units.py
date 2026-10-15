import numpy as np
import pytest

from cbf2nxmx.byte_offset import decode, encode
from cbf2nxmx.cif import parse
from cbf2nxmx.frames import compose, rotation, to_mcstas
from cbf2nxmx.nxmx import nx_name
from cbf2nxmx.pilatus import parse_header
from cbf2nxmx.pixels import OverflowError_, Policy, choose_dtype, convert


def reference_decode(bb: bytes, n: int) -> np.ndarray:
    out = np.empty(n, dtype=np.int64)
    p = v = 0
    for k in range(n):
        c = bb[p]
        if c != 0x80:
            v += c - 256 if c > 127 else c
            p += 1
        else:
            s = int.from_bytes(bb[p + 1 : p + 3], "little", signed=True)
            if s != -32768:
                v += s
                p += 3
            else:
                s = int.from_bytes(bb[p + 3 : p + 7], "little", signed=True)
                if s != -(2**31):
                    v += s
                    p += 7
                else:
                    v += int.from_bytes(bb[p + 7 : p + 15], "little", signed=True)
                    p += 15
        out[k] = v
    return out


@pytest.mark.parametrize("seed", range(10))
def test_byte_offset_roundtrip(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 5000))
    v = np.concatenate([
        rng.integers(-5, 300, n),
        rng.integers(-(2**31) + 1, 2**31 - 1, n // 7),
        [-1, -2, 127, -127, 128, -128, 32767, -32767, 32768, 2**31 - 1, -(2**31) + 1, 0x80],
    ])
    rng.shuffle(v)
    e = encode(v)
    assert (reference_decode(e, v.size) == v).all()
    assert (decode(e, v.size) == v).all()


def test_byte_offset_trailing_bytes_ignored():
    v = np.arange(100)
    assert (decode(encode(v) + b"\x00" * 50, 100) == v).all()


def test_byte_offset_short():
    with pytest.raises(ValueError):
        decode(encode(np.arange(10)), 11)


def test_cif_parse():
    text = """data_x
_a.b 'quoted value'
_a.c "x y"
_a.d .
# comment
loop_
_t.id
_t.v
p1 1.5
'p 2' ?
_s.text
;
line one
line two
;
"""
    (b,) = parse(text)
    assert b.name == "x"
    assert b.value("a", "b") == "quoted value"
    assert b.value("a", "c") == "x y"
    assert b.value("a", "d") is None
    assert b.table("t") == [{"id": "p1", "v": "1.5"}, {"id": "p 2", "v": None}]
    assert b.value("s", "text") == "line one\nline two"


def test_cif_quote_inside_word():
    (b,) = parse("data_x\n_a.b it's\n_a.c 'don't stop'\n")
    assert b.value("a", "b") == "it's"
    assert b.value("a", "c") == "don't stop"


def test_pilatus_header():
    h = parse_header(
        """# Detector: PILATUS 2M, S/N 24-0107 Diamond
# 2016-03-05T09:01:45.613
# Pixel_size 172e-6 m x 172e-6 m
# CdTe sensor, thickness 0.001000 m
# Exposure_time 0.0964000 s
# Count_cutoff 388705 counts
# Beam_xy (730.00, 865.00) pixels
# Start_angle -145.0000 deg.
"""
    )
    assert h.model == "PILATUS 2M"
    assert h.serial == "24-0107 Diamond"
    assert h.date == "2016-03-05T09:01:45.613"
    assert h.pixel_size_m == (172e-6, 172e-6)
    assert h.sensor_material == "CdTe"
    assert h.sensor_thickness_m == pytest.approx(0.001)
    assert h.count_cutoff == 388705
    assert h.beam_xy_px == (730.0, 865.0)
    assert h.angles_deg["Start_angle"] == -145.0


def test_dtype_choice():
    assert choose_dtype("auto", 65534) is np.uint16
    assert choose_dtype("auto", 65535) is np.uint32
    assert choose_dtype("auto", None) is np.uint32
    assert choose_dtype("uint16", 10**6) is np.uint16


def test_convert_markers_and_overflow():
    a = np.array([[0, 5, -1, -2, 65533, -7]], dtype=np.int32)
    c = convert(a, Policy(np.uint16, "error"))
    assert c.data.tolist() == [[0, 5, 65535, 65534, 65533, 65534]]
    assert c.n_other_negative == 1
    with pytest.raises(OverflowError_):
        convert(np.array([[65534]], dtype=np.int32), Policy(np.uint16, "error"))
    c = convert(np.array([[70000, 1]], dtype=np.int32), Policy(np.uint16, "clip"))
    assert c.data.tolist() == [[65533, 1]]
    p = Policy(np.uint16, "clip")
    assert p.saturation_value(91799) == 65532
    p = Policy(np.uint32, "error")
    assert p.saturation_value(91799) == 91798
    c = convert(np.array([[-1, -2, 2**31 - 1]], dtype=np.int32), p)
    assert c.data.tolist() == [[2**32 - 1, 2**32 - 2, 2**31 - 1]]


def test_frames():
    assert np.allclose(to_mcstas([1, 2, 3]), [-1, 2, -3])
    R = rotation([0, 0, 1], 90)
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0])
    R, t = compose([("translation", [0, 0, 1], [1, 0, 0], 5.0), ("rotation", [0, 0, 1], [0, 0, 0], 90.0)])
    assert np.allclose(t, [0, 1, 5])


def test_names():
    assert nx_name("GON_OMEGA") == "omega"
    assert nx_name("DET_2THETA") == "two_theta"
    assert nx_name("DET_Z") == "det_z"
