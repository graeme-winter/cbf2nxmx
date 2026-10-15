# cbf2nxmx

Convert a sweep of CBF images with full imgCIF headers to one NXmx HDF5 file.
The output reads in DIALS (`dials.import`) and in `mxi_import` with the same
geometry as `dials.import` gives for the CBFs.

```
pip install .            # numpy, h5py, hdf5plugin
cbf2nxmx CIX1_1_00001.cbf -o CIX1_1.nxs                 # the first image: the rest follow
cbf2nxmx 'l-cyst_01_#####.cbf' -o l-cyst_01.nxs --dtype uint16 --overflow clip
cbf2nxmx 'x_*.cbf' -o x_master.h5 --frames-per-file 1000   # data files + VDS
```

## Options

| option | default | |
|---|---|---|
| `--dtype` | `auto` | `uint16`, `uint32`. `auto` is uint16 when count cutoff - 1 <= 65533, else uint32 |
| `--overflow` | `error` | `clip`: counts above 2^n-3 become 2^n-3, and saturation_value drops below them |
| `--compression` | `bslz4` | `lz4`, `none`: the three mxi reads; one frame per chunk |
| `--frames-per-file` | 0 | >0: `<stem>_data_NNNNNN.h5` files, a VDS `/entry/data/data` and `data_NNNNNN` links |
| `--image-range a,b` | all | 1-based, inclusive |
| `--nproc` | all cores | decoding processes |
| `--polarization` | off | write the CBF's polarisation ratio (dxtbx ignores it in CBF, so it changes DIALS' answer) |
| `--no-md5`, `--no-consistency-check` | | skip the checksum / header-agreement checks |

## Pixels

PILATUS CBF use -1 for module gaps and -2 for bad pixels. In the output these
become 2^n-1 (gap) and 2^n-2 (bad), the DECTRIS/mxi markers, and are also
recorded in `pixel_mask` (bit 0 gap, bit 1 bad, the union over all frames).
Other negative values are marked bad, and counted in the provenance note.
`saturation_value` is the CBF's `_array_intensities.overload` - 1, which is
dxtbx's trusted maximum for the CBF, capped at 2^n-3.

## Geometry

The imgCIF axis tables are authoritative. The PILATUS header in the CBF can
contradict them; on I24, for example, the header says `Oscillation_axis X.CW`
while the vertical omega is (0,1,0). The header supplies only the sensor
material and thickness, the exposure and the detector's name.

* Every imgCIF vector v is written as C v, with C = diag(-1, 1, -1), because
  dxtbx and mxi go from McStas back to imgCIF with that half turn.
* Vectors are written as given, not normalised. Only the pixel directions
  are unit vectors.
* The goniometer chain is the CBF's: `/entry/sample/depends_on` points to
  the innermost axis. Every axis has one value per image, and the scan axis
  also has `_increment_set` and `_end`.
* The detector chain is the CBF's (e.g. two_theta, det_z, det_y, det_x). The
  two element axes fold into `module_offset` (their offsets plus
  displacements), and `fast_` / `slow_pixel_direction` both depend on it.
* Axis names lose `GON_` and are lower-cased; `DET_2THETA` becomes `two_theta`.

What is checked: consecutive images; every header identical apart from
angles, dates and binary details; only the scan axis moves; even steps (to
1e-3 deg); a single goniometer chain; translation element axes; a source
along -z; MD5 of every binary.

## Checking a conversion

```
cbf2nxmx.verify out.nxs 'x_#####.cbf'                    # pixel round trip, no DIALS
dials.import out.nxs output.experiments=nxs.expt
cbf2nxmx.compare cbf.expt nxs.expt                        # geometry, from DIALS
mxi_import out.nxs -o mxi.expt && cbf2nxmx.compare cbf.expt mxi.expt
dials.python -m cbf2nxmx.verify out.nxs cbf.expt --dxtbx --every 10   # DIALS' masks and counts
```

`cbf2nxmx.compare` also reads an NXmx file directly (`cbf2nxmx.readback`,
dxtbx's reading rules in numpy). It reports these as expected differences:

* exposure time and epochs: dxtbx takes the integration time and frame dates
  from CBF, and frame_time from NXmx;
* the panel's mask list: PILATUS module gaps, which are in `pixel_mask` here;
* identifiers and axis names.

## Not yet

* True mini-CBF (no imgCIF axis tables): a header-only geometry path with
  beamline profiles.
* Several arrays or detector elements per image; moving detectors;
  `decreasing` array directions or non-zero element displacements (handled,
  but untested against dxtbx).
* Time zones: CBF dates carry none and are written as given.
