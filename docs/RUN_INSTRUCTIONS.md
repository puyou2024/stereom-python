# Running `stereom` — the Python port

This is a from-scratch Python port of the MATLAB toolbox
[Ziling-Wu/Quantitative-3D-structural-analysis-of-the-cellular-microstructures](https://github.com/Ziling-Wu/Quantitative-3D-structural-analysis-of-the-cellular-microstructures),
written against the analysis in `docs/stereom_implementation_spec.md` (that
document is the authoritative record of *why* each stage below works the
way it does, which MATLAB defect it fixes, and what — if anything — was
deliberately simplified relative to the original; this file is just the
"how do I run it" companion).

Paths below are relative to the repository root. The MATLAB plotting
scripts are in `matlab_viz/`. No MATLAB code from the original repository
is included here.

## 1. What changed, in one paragraph

Every voxel-level scalar loop in the original (node identification,
branch registration, node merging, and above all thickness — a full-volume
temporary allocated *per centerline voxel*, spec's own "performance
catastrophe") is replaced by vectorized NumPy/SciPy operations, a KD-tree
+ union-find for node merging, and a single Euclidean distance transform
for thickness. Skeletonization is `skimage.morphology.skeletonize(...,
method='lee')` — the same published algorithm the MATLAB `Skeleton3D.m`
implements, not a reimplementation. Ring search and profile fitting are
parallelized across CPU cores via `joblib`. Eight defects from the spec's
register (D4, D6, D8, D9, D10, D11, D13, D14, D17, D18 — ring-count
undercounting, N-4 angle undercounting, wrong orientation for
axis-aligned branches, wrong node-merge survivor, missed merge
neighbours, unclamped `acos`, thickness bounds errors, chord-vs-tangent
axis, the `open` flag silently overriding `merge_thres`, and the
full-volume-temporary bottleneck) are fixed, not reproduced. See
`docs/stereom_implementation_spec.md` Sec 5 and each module's docstring for
specifics.

## 2. Install

Requires Python >= 3.10.

```powershell
cd stereom-python
python -m venv .venv
.venv\Scripts\Activate.ps1        # or .venv/bin/activate on Linux/macOS
pip install -e ".[viz,yaml,parquet,dev]"
```

Extras:
- `viz` — `matplotlib`, for the optional plotting helpers in `stereom/viz.py`.
- `yaml` — lets `StereomConfig` read/write YAML in addition to JSON.
- `parquet` — lets table export use Parquet in addition to CSV.
- `hpc` — `mpi4py`, only needed if you wire in your own MPI-based batch driver.
- `dev` — `pytest` etc., for running the test suite.

Core dependencies (always installed): `numpy`, `scipy`, `scikit-image`,
`networkx`, `tifffile`, `joblib`, `pandas`.

### Run the tests

```powershell
pytest -q                    # full suite, ~50 fast tests + 2 slower end-to-end lattice tests
pytest -q -m "not slow"      # skip the two end-to-end lattice tests (rasterize+skeletonize), ~5s
```

The suite includes the phantom tests from spec Sec 9.1 — synthetic
straight/bent struts, an hourglass profile with a known analytic waist, a
simple cubic lattice, a cube-graph ring-dedup stress test, and (the two
the spec calls highest-yield) a 2x uniformly-scaled lattice and a
45-degree-rotated lattice, both checked for statistically unchanged
results. `tests/phantoms.py` has the generators if you want to build your
own synthetic volumes; `stereom/invariants.py` has the runtime-checkable
invariants from spec Sec 9.2 (`invariants.run_all(graph, ...)`).

## 3. Quick start — command line

```powershell
stereom run --input path\to\tiff_stack_dir --output path\to\results --voxel-size 0.65
```

or, if you didn't `pip install` the console script:

```powershell
python -m stereom.cli run --input path\to\tiff_stack_dir --output path\to\results --voxel-size 0.65
```

Write a config file with all defaults first if you want to edit
parameters before running:

```powershell
stereom init-config --output my_config.json
# ... edit my_config.json ...
stereom run --input path\to\tiff_stack_dir --output path\to\results --config my_config.json
```

Full CLI options:

```
stereom run
  --input PATH          TIFF-stack directory, or a .npz/.npy volume file (required)
  --output PATH          Output directory (required)
  --config PATH          StereomConfig JSON/YAML file (default: built-in defaults)
  --voxel-size FLOAT      Override voxel_size_um
  --n-workers INT         Override config.n_workers (parallelism; -1 = all cores)
  --start-page INT        First TIFF slice index (dir input only; default 0)
  --end-page INT          Last TIFF slice index (dir input only; default: autodetect)
  --sig-digits INT        Zero-padding width of slice filenames (default 4)
  --image-name STR        Filename prefix before the zero-padded index (default "")
  --style STR             File extension (default ".tif")
  --table-format {csv,parquet}
  --skip-rings / --skip-chains / --skip-fft
```

## 4. Quick start — Python API

```python
from stereom import io, pipeline
from stereom.config import StereomConfig

volume = io.load_volume(
    "path/to/tiff_stack_dir",
    voxel_size_um=0.65,
    axes=("L", "C", "R"),   # see Sec 7 below -- get this right
)

config = StereomConfig(voxel_size_um=0.65)   # or StereomConfig.load("my_config.json")
result = pipeline.run_full_pipeline(volume, config)

result.nodes_df      # pandas DataFrame: id, i, j, k, degree, voxel_degree
result.branches_df    # id, node_a, node_b, length_um, distance_um, tortuosity,
                      #   t_min/t_cen/t_mean_radius_um (+ _diameter_um), theta_deg, omega_deg
result.rings_df       # id, size, nodes, branch_ids, centroid, normal, area_um2, d_ring_um, flatness_r2
result.chains_df      # id, nodes, branch_ids, n_branches
result.profile_fits   # list[ProfileFitResult]: quadratic (a0,b,r2) and Paper-II quartic (quartic_a) fits
result.graph          # the underlying networkx.MultiGraph -- full topology + branch paths
```

Save everything:

```python
from stereom import tables
tables.export_tables(result.nodes_df, result.branches_df, "out_dir", fmt="csv")
tables.save_graph_npz(result.graph, "out_dir/graph.npz")
```

Reload a saved graph later without rerunning the pipeline:

```python
G = tables.load_graph_npz("out_dir/graph.npz")
```

## 5. Input formats

| Format | How | Notes |
|---|---|---|
| Numbered TIFF stack | `io.load_volume(folder, start_page=, end_page=, sig_digits=, image_name=, style=)` | Same naming convention as the original (`imageName` + zero-padded index + extension). Reads the whole stack in one vectorized call; stores `bool`, not `double` (spec Sec 3.1 fix: 8x smaller than uint8, 64x smaller than double). |
| Single multi-page TIFF / `.npy` array | `io.load_volume_array(array, voxel_size_um=, axes=, threshold=)` | `threshold=None` auto-picks the midpoint between min and max (correct for a genuinely binary 0/1 or 0/255 input). |
| `.npz` (this package's own format) | `io.load_volume_npz(path)` | Written by `io.save_volume_npz`. |

The CLI's `--input` accepts a directory (TIFF stack) or a `.npz`/`.npy`
file directly.

**Very large volumes** (too big for one machine's RAM): `tifffile` can
open TIFF stacks lazily via Zarr (`tifffile.imread(files, aszarr=True)`)
for chunked/lazy loading with `dask`; wire that into
`io.load_volume_array` if you need it — it isn't turned on by default
because the demo-scale volumes (10^7-10^8 voxels) fit comfortably in
memory as `bool` arrays.

## 6. Analyzing a region of an STL mesh, and visualizing results in MATLAB

Two extra pieces, for going from a surface mesh (e.g. `Raw_STL_models/*.stl`)
to a specific cropped region's analysis, then viewing the result in MATLAB
instead of Python.

### 6.1 STL -> cropped, voxelized region

Install the extra dependencies (`trimesh`, `shapely`, `pillow` -- only
needed for this step, not for the core pipeline):

```powershell
pip install -e ".[mesh]"
```

**Choosing `--pitch` is the single most consequence-laden setting in this
whole workflow, and it is easy to get badly wrong.** It must resolve
your structure's smallest real feature (e.g. the thinnest ligament/strut
diameter), not just fit the overall region size. Use
`pitch_from_feature_size`:

```python
from stereom.mesh_regions import pitch_from_feature_size
pitch = pitch_from_feature_size(min_feature_size=0.1, voxels_across=8)  # -> 0.0125 (mm), for a 100 um ligament
```

in the SAME units as your bounding box (mm, if that's what the STL uses).
Getting this wrong is not a minor accuracy issue: an under-resolved
pitch distorts the measured relative density (a real check on this
package's own test data found an under-resolved setup understating solid
volume fraction by roughly 3x before it was diagnosed and fixed -- see
`mesh_regions.py`'s module docstring) and can push skeletonization into
producing a medial SURFACE instead of a medial LINE (spec Sec 3.2's
documented failure mode), which then shows up downstream as a chaotic or
empty node/branch graph. As a floor: 8 voxels across the smallest
feature is a reasonable minimum for a clean skeleton; use more (12-20) if
thickness/density accuracy matters more than runtime to you.

Each Z-slice of the requested region is voxelized via an EXACT
cross-section (triangle-mesh/plane intersection, not a 3D flood-fill --
see the module docstring for why: on real, non-watertight STL exports, a
3D flood-fill can leak through small gaps and silently under-fill).
This needs no mesh repair and no mesh cropping/capping step, and
parallelizes across Z-slices with `--n-jobs`:

```powershell
stereom extract-regions --stl "Raw_STL_models\sample.stl" `
  --bbox 19.5 24.5 4 9 0 7.5 --name my_region --pitch 0.02 `
  --stl-units-to-um 1000 --n-jobs -1 --output-dir regions_out
```

- `--bbox XMIN XMAX YMIN YMAX ZMIN ZMAX` is in the STL FILE's own
  coordinate units (whatever you'd read off in a mesh viewer) -- NOT
  micrometres.
- `--pitch` is the voxel edge length, ALSO in STL file units (e.g. `0.02`
  if the STL is in mm and you want 20 um voxels -- see the resolution
  guidance above). Decoupled on purpose from `--stl-units-to-um` (which
  only tags the output volume's `voxel_size_um` metadata) so you can
  reason about resolution directly against what you see in the mesh.
- `--stl-units-to-um` defaults to 1000 (STL in millimetres, the common
  CAD/STL-export convention). **Check this against the physical sample
  size you expect** -- get it wrong and every length/thickness number
  downstream is silently wrong. `extract-regions` prints the mesh's
  `extents` before voxelizing specifically so you can sanity-check this.
- `--n-jobs` parallelizes across Z-slices (independent per-slice work --
  the natural place to use extra cores here). `-1` = all cores.

Many regions at once (recommended when you have several regions across
several files -- each source STL is loaded only once no matter how many
regions are cut from it):

```json
[
  {"name": "region_A", "stl": "Raw_STL_models/sample1.stl", "bbox": [19.5,24.5,4,9,0,7.5], "pitch_stl_units": 0.02},
  {"name": "region_B", "stl": "Raw_STL_models/sample2.stl", "bbox": [0,10,0,10,0,10], "pitch_stl_units": 0.015, "stl_units_to_um": 1000}
]
```

```powershell
stereom extract-regions --spec regions.json --output-dir regions_out
```

This writes, per region: `regions_out/{name}.npz` (feed straight into
`stereom run --input regions_out/{name}.npz ...`), `regions_out/{name}.stl`
(a crop of the source mesh, for the MATLAB overlay in Sec 6.3),
`regions_out/{name}_stl_alignment.json` (coordinate-alignment metadata
tying the two together -- `stereom run --export-mat` picks this up
automatically, see Sec 6.2), and `regions_out/{name}_slice_profile.csv`
(solid fraction per Z-slice -- see the automatic sanity check below).

`extract-regions` prints the resulting solid fraction, the number of
Z-slices that failed to rasterize outright (a slice can fail if a mesh
defect sits exactly on that slice's plane; it's retried at a slightly
offset height and, failing that, left void -- a warning fires if more
than 5% of slices fail), **and the number of slices flagged/healed/left
unresolved by an automatic outlier check.** This check exists because a
slice can "succeed" (no exception) and still be silently wrong: on real
data, `mesh.section()`'s 2D polygon assembly was found to occasionally
misassemble the cross-section at specific exact cutting heights into one
giant spurious polygon spanning nearly the whole part -- symptom: an
isolated 1-2 slice solid-fraction spike (and, downstream, an implausible
localized cluster of extra nodes/branches at that Z in the MATLAB
visualization) with no corresponding change in the underlying mesh's
local triangle density. Every extraction now compares each slice's solid
fraction against a robust local median of its neighbors and, for
outliers, retries with a small nudge to the cutting height, keeping the
result only if it now agrees with the local trend (see `mesh_regions.py`
module docstring for the full incident and the detection/healing logic).
Slices that stay anomalous after every nudge are left untouched and
reported as **unresolved** -- that means the check could not rule out
real, unusually dense local geometry, not that nothing is wrong; open
`{name}_slice_profile.csv` and look at the flagged Z-height before
trusting node/branch counts near it.

**Still sanity check the overall solid fraction against what you
independently expect for your material** -- if you have an expected
relative density from your design process and the measured value is off
by a large factor, don't just adjust the pitch until the numbers happen
to agree; that is curve fitting, not a fix. Re-read the module
docstring's account of how the original flood-fill under-counting bug
was diagnosed (an exact, voxelization-free cross-section area check) for
the kind of independent verification worth doing if you hit this.

### 6.2 Exporting results for MATLAB

```powershell
stereom run --input regions_out\my_region.npz --output results\my_region --export-mat
```

writes `results/my_region/results.mat` alongside the usual CSVs (or, to
regenerate a `.mat` from a results directory you already have:
`stereom export-mat --results-dir results\my_region`, adding
`--stl-alignment regions_out\my_region_stl_alignment.json` if you also
want the overlay data). It bundles nodes, branches (including every
branch's full ordered voxel path, needed for plotting -- not present in
the CSVs, which are summary tables only), rings, chains, inter-branch
angles and profile fits into one file MATLAB can `load()` directly, with
1-based row indices already resolved for you
(`branches.node_a_row`/`node_b_row` index straight into
`nodes.i`/`nodes.j`/`nodes.k` -- see `matlab_export.py`'s module
docstring for why graph node ids themselves aren't dense/contiguous and
can't be used as array indices directly). If the input volume came from
`stereom extract-regions`, its companion `.stl` and alignment metadata
are found automatically (via the `{name}_stl_alignment.json` sidecar next
to the input `.npz`) and copied alongside `results.mat` as a
`stl_overlay` struct, for Sec 6.3's transparency overlay.

### 6.3 Visualizing in MATLAB

`matlab_viz/visualize_stereom_results.m`
loads a `results.mat` and produces: the 3D network colored by thickness/
length/tortuosity, node-type and length/thickness histograms, the
orientation polar plot (paper's theta/omega convention, not the original
code's swapped naming), inter-branch angle histograms by node degree,
ring size histogram + 3D ring polygons, a per-chain colored plot,
profile-fit summaries, and -- if `results.mat` has a `stl_overlay` struct
(Sec 6.2) -- **the skeleton network overlaid on a semi-transparent render
of the source STL region**, specifically for visually verifying that the
extracted lines actually run through the real ligaments rather than
floating outside them or missing thin features. This needs MATLAB's
built-in `stlread` (R2018b+, returns a `triangulation` object); if yours
is older and only has a file-exchange `stlread` with a different return
signature, adjust the one `tri = stlread(...)` line in
`plotNetworkWithSTLOverlay` at the bottom of the script.

It's a single cell-mode (`%%`) script in the same style as the original
toolbox's `main_*.m` scripts -- set `resultsFile` at the top and run
section by section. If the original toolbox is on your MATLAB path it
reuses `FigureFormat.m` for consistent styling; otherwise a small
built-in equivalent is used automatically. Nothing in this script
recomputes any analysis result -- it only reads what `stereom` already
computed (same separation-of-concerns as `viz.py` on the Python side).

## 7. The one setting you must get right: `axes`

`StereomConfig.axes` (default `("L", "C", "R")`) declares which ARRAY
DIMENSION is which PHYSICAL axis:

- **L** — longitudinal / polar axis. `theta` (misorientation) is measured from it.
- **C** — circumferential.
- **R** — radial / the graded direction in a specimen. `omega` (in-plane
  azimuth) is measured from it.

Every orientation statistic (`theta_deg`, `omega_deg` on every branch, the
N-3 node planarity normal, the FFT peak interpretation) is computed
relative to this mapping. Get it wrong and every orientation number is
silently wrong — there is no way to detect this from the output alone.
This mirrors the original code's hardcoded "dim 1 = L, dim 3 = R"
assumption (spec Sec 2.3), except here it is an explicit, checked field
instead of an implicit convention buried in `myangle.m`.

If you don't care about orientation statistics for your use case, the
default is harmless either way.

## 8. Output formats

`stereom run` writes to `--output`:

| File | Contents |
|---|---|
| `nodes.csv` / `nodes.parquet` | `id, i, j, k, degree, voxel_degree` |
| `branches.csv` / `branches.parquet` | `id, node_a, node_b, n_path_voxels, length_um, distance_um, tortuosity, t_min/t_cen/t_mean_radius_um, t_min/t_mean_diameter_um, theta_deg, omega_deg` |
| `graph.npz` | Full graph: topology, node positions, every branch's ordered voxel path, all computed edge attributes. Reload with `tables.load_graph_npz`. |
| `rings.csv` | `id, size, nodes, branch_ids, centroid_{x,y,z}_um, normal_{x,y,z}, area_um2, d_ring_um, flatness_r2, max_plane_residual_um` |
| `chains.csv` | `id, nodes, branch_ids, n_branches` |
| `inter_branch_angles.csv` | `node, neighbor_a, neighbor_b, branch_a, branch_b, angle_deg` — one row per node per pair of incident branches (ALL pairs, fixing D6) |
| `node_planarity.csv` | `node, normal_{x,y,z}, out_angle_deg, planarity_index` for every degree-3 node |
| `profile_fits.csv` | Per-branch quadratic fit (`a0, b, r2`) + Paper II quartic fit (`quartic_a`), with `accepted`/`reject_reason` |
| `fft_spectrum.npz` | Normalized magnitude spectrum of the node-indicator volume (array key `spectrum`) |
| `run_summary.json` | Node/branch/drop counts from graph construction, elapsed time, the resolved `merge_d_m_um` |
| `config_used.json` | The exact config (including any auto-derived values) this run used — save this alongside results for reproducibility |
| `results.mat` (only with `--export-mat`) | Everything above, bundled for MATLAB — see Sec 6.2-6.3 |

All lengths, distances and thickness values are already in micrometres
(`voxel_size_um` is applied once, at the boundary — spec Sec 10.2
requirement 3). **Thickness is always a radius, explicitly named**
(`t_min_radius_um` etc.), with a diameter column alongside
(`t_min_diameter_um` = `2 * t_min_radius_um`) — never a bare `thickness`
(spec Sec 10.2 requirement 8; the original's `branthickness` is a radius
despite being called "thickness", a real ambiguity documented in spec
Sec 3.11).

## 9. Parameters

Everything tunable is a field on `StereomConfig`
(`stereom/config.py`) — no parameter is a bare literal buried in analysis
code. Defaults reproduce the shipped MATLAB pipeline's *effective*
behaviour (not always what a literal reading of its call sites suggests —
see the field docstrings for the `open`-flag / D17 case specifically).

| Field | Default | Meaning |
|---|---|---|
| `voxel_size_um` | 0.65 | Isotropic voxel size |
| `axes` | `('L','C','R')` | Array-dimension -> physical-axis mapping (Sec 7 above) |
| `node_degree_threshold` | 4 | Node-candidate voxel-neighbour threshold (leave alone) |
| `degree2_guard_voxels` | 5.0 | First-pass guard: don't dissolve a degree-2 node if its shorter incident branch exceeds this many voxels. `None` = unguarded (used for the pipeline's second pass automatically) |
| `merge_d_m_um` | `None` (auto) | The real node-clustering distance. `None` derives it as the mean strut radius from an EDT of the volume (spec Sec 4 rule for parameter #7) |
| `merge_touch_um` | `None` (auto = `voxel_size_um * sqrt(3)`) | "Directly touching" merge pass distance |
| `boundary_limit_um` | `None` (auto = mean branch length) | For analysis-time boundary exclusion (`stereom/boundary.py`) — NOT applied automatically inside `run_full_pipeline`; call `boundary.filter_interior_tables` yourself when you want it (spec Sec 3.7: this is a per-analysis decision, and the saved graph should keep every node/branch) |
| `thickness_min_path_voxels` | 10 | Branches shorter than this get no thickness profile |
| `fit_max_abs_z_um`, `fit_len_min_um`, `fit_len_max_um` | 25, 2, 35 | Profile-fit acceptance window. Use `profile_fit.derive_fit_limits(lengths)` for the mean+/-3SD auto-derivation rule instead, if your dataset's length scale differs a lot from the demo volume |
| `fit_include_linear_term` | `False` | `True` reproduces published Eq. 3's linear term; default matches the shipped code's actual (linear-term-free) fit |
| `ring_max_size` | 7 | Max ring size (branches) to search for |
| `ring_flatness_r2_min` | 0.5 | Minimum planarity R² for a ring to be kept (not implemented in the shipped MATLAB release — see Sec 10 below) |
| `chain_cos_threshold` | 0.5 | Minimum `|cos(angle)|` between successive branch chords for greedy chain growth to continue |
| `fft_window` | `True` | Apply a Tukey window before the node-array FFT |
| `n_workers` | -1 (all cores) | Parallelism for ring search (`joblib`) and `pipeline.run_batch` |

## 10. What's fixed vs. the original, and what's deliberately different

Full detail is in `docs/stereom_implementation_spec.md` Sec 5 (defect
register) and each module's docstring; the short version:

**Fixed defects** (see spec IDs): ring-count undercounting from an
overflowing numeric hash (D4) → exact sorted-tuple dedup; N-4+ node
inter-branch angles only measuring 4 of 6 pairs (D6) → all
`C(degree,2)` pairs; wrong orientation for axis-aligned branches (D8) →
`atan2`-based, no degenerate special-casing; node-merge survivor placed
at an arbitrary max-index voxel instead of the cluster centroid (D9);
fixed-window node-merge missing neighbours separated along an axis (D10)
→ KD-tree + union-find, exact and order-independent; unclamped `acos`
producing silently-wrong complex-to-real-mapped angles (D11); no bounds
checking in the thickness search (D13); chord-based instead of local-
tangent branch axis for thickness (D14); the `open` flag silently
overriding the passed merge distance (D17); and the thickness stage's
full-volume-temporary-per-sample-point performance catastrophe (D18) →
one Euclidean distance transform.

**Two things NOT implemented**, both genuinely absent from the papers'
released code and not fabricated here: a beam-model / finite-element
exporter (Paper II's FE pipeline was never released — spec Sec 1.3), and
a chunk-and-stitch driver for volumes too large for one machine (a
correct version needs to handle branches truncated at a chunk boundary
and avoid double-counting rings/chains crossing one — real additional
design work; `pipeline.run_batch` is the complete, correct tool for the
more common case of many independent volumes, and
`graph_clean.merge_clusters` is the building block a future stitcher
would reuse). See the `pipeline.py` module docstring for the full
rationale.

**One deliberate topology-construction simplification** worth knowing
about: the original iteratively re-runs node identification and branch
trimming on the shrinking VOLUME to erase dangling whisker branches
before finalizing node clusters. This port instead registers the graph
once and lets the "does this branch's far end touch a registered node"
test in `graph_build.py` exclude whiskers directly (which it does,
completely, on its own — see that module's docstring for why), skipping
the expensive iterative volume relabeling. This is explicitly the
refactor the spec recommends (Sec 3.4) and only affects the exact voxel
shape of node clusters in edge cases, which the subsequent KD-tree
cluster merge absorbs regardless.

## 11. Performance & HPC usage

### What got faster, and why

The dominant cost in the original was `main_thickness.m`: a full-volume
boolean multiply *per interior centerline voxel of every branch*
(~4.6x10^7 elements x ~30,000 sample points on the demo volume — spec's
own estimate is ~10^12 operations, "grab a cup of coffee"). That entire
stage is now one `scipy.ndimage.distance_transform_edt` call, O(N) once
for the whole volume. Node identification is one 3D convolution instead
of a scalar loop rebuilding a `meshgrid` per voxel. Node merging is a
KD-tree range query + union-find instead of a windowed linear scan.
Ring search is the same asymptotic complexity as the original's unrolled
BFS (bounded by `ring_max_size`) but exactly deduplicated and
parallelized across CPU cores.

On the shipped demo volume (386x386x308), the full pipeline (load,
skeletonize, build, clean, all descriptors, rings, chains, FFT) runs in
well under a minute on a laptop-class multi-core CPU — see the timing
line in `run_summary.json` / the CLI's stderr output for the actual
number on your machine and data.

### Multi-core parallelism (single machine / single SLURM node)

Ring search parallelizes automatically via `config.n_workers` (joblib,
default: all cores). This is the one stage with meaningfully
super-linear-in-node-count cost, and it is embarrassingly parallel across
seed nodes.

### Many independent volumes (the common HPC case)

If you have many specimens/replicates/imaging sessions rather than one
enormous volume, use `pipeline.run_batch`:

```python
from stereom.pipeline import run_batch
from stereom.io import load_volume

volumes = {name: load_volume(path, voxel_size_um=0.65) for name, path in specimen_paths.items()}
results = run_batch(volumes, config, n_jobs=-1)   # one process per volume, in parallel
```

`run_batch` sets `n_workers=1` inside each worker automatically, so you
don't oversubscribe CPUs by parallelizing both across volumes and within
ring search at once.

### Example SLURM job (single large volume, multi-core)

```bash
#!/bin/bash
#SBATCH --job-name=stereom
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00

module load python/3.11
source /path/to/venv/bin/activate

stereom run \
  --input /scratch/$USER/specimen_01/Binary \
  --output /scratch/$USER/specimen_01/results \
  --voxel-size 0.65 \
  --n-workers "$SLURM_CPUS_PER_TASK"
```

### Example SLURM array job (many independent volumes)

```bash
#!/bin/bash
#SBATCH --job-name=stereom_batch
#SBATCH --array=0-19
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module load python/3.11
source /path/to/venv/bin/activate

SPECIMEN=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" specimen_list.txt)
stereom run --input "$SPECIMEN" --output "results/$SLURM_ARRAY_TASK_ID" \
  --voxel-size 0.65 --n-workers "$SLURM_CPUS_PER_TASK"
```

(One `stereom run` per array task keeps each job's memory footprint
bounded and lets SLURM schedule them independently — usually simpler to
operate than a single `run_batch` call spanning many nodes.)

### Memory

Volumes are stored as `bool` (1 byte/voxel as a NumPy array, i.e. 8x
smaller than `uint8` and 64x smaller than `double`). The EDT and node-
identification convolution each need one same-size scratch array in
addition to the input; budget ~3-4x the raw volume size in RAM as a rule
of thumb (e.g. a 500^3 volume: 125M voxels ~ 125MB as bool, ~500MB peak
working set). For volumes that don't fit in memory at all, see the
"Very large volumes" note in Sec 5 and the chunking caveat in Sec 10.

## 12. Validated against the real demo volume

The pipeline was run end-to-end on the actual shipped demo data
(`Code/Data/Binary` in the original MATLAB repository, 386x386x308, 0.65 um/voxel) as a
smoke test, not just the synthetic phantoms:

```
[stereom] loaded volume (386, 386, 308), solid fraction 0.2849
[stereom] pipeline done in 28.0s: 1461 nodes, 2430 branches, merge_d_m=4.85 um
```

**28 seconds, full pipeline, laptop-class CPU** (4 worker processes) —
for comparison, `main_thickness.m` alone was described by the original
authors as taking long enough to "grab a cup of coffee" on this exact
volume.

Against spec Sec 9.3's regression targets, most statistics land close:

| Statistic | Spec's reported value | This port |
|---|---|---|
| Mean physical branch length | 21.7 +/- 7.9 um | 20.4 um (std 9.6) |
| Mean branch length/distance ratio (tortuosity) | 1.2 +/- 0.1 | 1.19 (std 0.11) |
| N-3 mean inter-branch angle | 118.1 +/- 3.5 deg | 116.1 deg (per-node-mean std 7.3) |
| Mean thickness (radius) | 6.2 +/- 1.5 um | 4.92 um (std 0.85) via EDT; see note below |

Two things differ more noticeably, both **expected** consequences of
real, documented fixes rather than bugs in the port:

- **Node-degree distribution** is more concentrated at degree 3 here
  (~70% vs. the paper's reported ~50%) than in the original release.
  This follows directly from fixing D9 (cluster survivor placed at the
  true centroid, not an arbitrary max-index voxel) and D17 (the merge
  distance is no longer silently overridden by a hardcoded ladder) —
  both change exactly which nodes get consolidated into which, and the
  paper's own reported merge distance was, per spec Sec 3.6.3, never
  actually the value the shipped code used. `merge_d_m_um` is worth
  tuning explicitly for your own data (Sec 9 above) rather than trusting
  the auto-derivation to reproduce a specific paper's numbers exactly.
- **Ring counts** (78/214/348/484 for 4/5/6/7-edge rings) are lower
  across the board than even the paper's own (under-counted, per D4)
  numbers. Two corrections pull in opposite directions here: fixing the
  hash-collision undercount (D4) pushes counts up, while adding the
  flatness filter (`ring_flatness_r2_min=0.5`) — absent from the shipped
  release entirely, present only as commented-out code — rejects
  non-planar candidates the original never filtered at all, pushing
  counts down. The net effect on raw counts isn't something the spec
  predicts numerically; set `ring_flatness_r2_min=0.0` if you want
  counts closer to the (uncorrected-for-planarity) original definition.
- **Mean thickness** is a radius from an inscribed-sphere Euclidean
  distance transform (the recommended replacement for the original's
  mean-surface-normal-distance method, spec Sec 3.11), not the same
  geometric quantity as the paper's number; a moderate, expected offset
  rather than a bug. Use `thickness.legacy_mean_radius_profile` (see its
  own docstring) if you need the original's exact geometric definition instead.

## 13. Visualization (optional, separate from analysis)

`stereom/viz.py` (needs the `viz` extra) has a few matplotlib helpers —
`plot_graph_3d`, `plot_histogram`, `plot_orientation_polar`,
`plot_fft_peaks`. None of them are called anywhere in `pipeline.py`;
every analysis function returns data, never a plot (spec Sec 10.2
requirement 1). Call these yourself on the `PipelineResult` you already
have:

```python
from stereom import viz
fig = viz.plot_graph_3d(result.graph)
fig.savefig("network.png", dpi=200)
```
