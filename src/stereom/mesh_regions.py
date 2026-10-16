"""STL -> region volume extraction.

Turns an STL mesh (e.g. a spinodoid/lattice sample, or any other
cellular-microstructure surface model) into a cropped, binary, voxelized
sub-volume in exactly the format ``pipeline.run_full_pipeline`` / the
``stereom run`` CLI expect as input.

Voxelization method -- and why it looks the way it does
---------------------------------------------------------
Voxelization proceeds one Z-slice at a time: ``mesh.section()`` computes
the EXACT plane-mesh intersection as closed 2D polygons at a given
height, which are then rasterized directly (PIL polygon fill, with holes
subtracted) onto the requested x/y grid.

This replaces an earlier design that voxelized the mesh's SURFACE (via
trimesh's ``ray`` or ``subdivide`` methods) and then flood-filled the
interior (either trimesh's own ``.fill()`` or a from-scratch
``scipy.ndimage.binary_fill_holes``). On this package's real test data
(a non-watertight, ~1M-triangle spinodoid STL export -- ``is_watertight``
is False and cannot be made True by ``fill_holes()``; the primary
connected component alone has 950k+ faces and evidently some genuine
small gaps/non-manifold edges scattered through it), BOTH flood-fill
variants were found to systematically UNDER-COUNT solid volume by
roughly 3x, verified against an exact, voxelization-independent
cross-section polygon-area measurement (``mesh.section`` + shapely area,
computed directly, no rasterization at all) at several heights within a
test region: ~35% by the exact area check vs. ~6-11% from either
flood-fill variant. The per-slice rasterization approach here matches
the exact area check (~35.6% on the same region) because it never
depends on 3D watertightness at all -- each Z-slice only needs its OWN
2D cross-section to consist of well-formed closed loops, which
``mesh.section`` provides even when the full 3D mesh has scattered local
defects elsewhere. A small number of slices (typically <2%) can still
fail if a defect sits exactly on that slice's plane; these are retried
at a slightly offset height and, failing that, left as void with the
failure count reported in the returned diagnostics -- check this number
after any extraction.

**Pitch selection matters more than it looks like it should.** Choosing
too coarse a pitch relative to the smallest real feature (e.g. the
thinnest ligament diameter in a strut lattice) doesn't just lose detail
at the edges -- for a thin-walled/thin-strut structure it can also
noticeably distort apparent solid fraction and, further down the
pipeline, push skeletonization into producing a medial SURFACE instead
of a medial LINE (spec Sec 3.2's documented failure mode), which then
manifests as a chaotic or empty node/branch graph. Use
``pitch_from_feature_size`` to derive a sensible pitch from your
structure's known minimum feature size rather than guessing.

Per-slice outlier detection and self-healing (why THIS exists)
----------------------------------------------------------------
Even when every slice "succeeds" (``_rasterize_z_slice`` returns a
result, no exception), a small number of EXACT cutting heights can
still be wrong: ``mesh.section()``'s 2D polygon assembly is a numerical
edge-finding/graph-traversal algorithm, and at specific z-heights where
the cutting plane is close to some coincidence in the mesh's vertex
data, it can misassemble the intersection segments into one spuriously
huge merged polygon instead of many small correct ones. Found on real
data: a handful of isolated single/double-slice heights where the
measured solid fraction jumped to ~65% against a ~35% baseline in
every neighboring slice, with the "solid" polygon's bounding box
spanning nearly the whole part -- while an independent check (local
triangle density in that z-band) showed the underlying mesh geometry
there is exactly as sparse as everywhere else. This is NOT a rare
mesh-specific fluke to special-case once: it's an inherent numerical
failure mode of exact-plane cross-sectioning, so it's handled generically
here rather than patched around for one file. Confirmed fix: nudging the
cutting height by a small fraction of one voxel pitch reliably escapes
the degenerate plane and reproduces the correct (neighbor-consistent)
result.

``voxelize_region`` therefore, after rasterizing all slices, computes a
robust local-trend profile of solid fraction vs. Z (median over a
rolling window), flags any slice that deviates abruptly from that trend
(not merely "different" -- real geometry varies smoothly slice-to-slice
at a properly chosen pitch, see above), and re-rasterizes flagged slices
at several small Z-nudges, keeping the first nudge whose result agrees
with the local trend.

**Misassembly can cluster, not just occur as isolated single slices.**
Also found on the SSCF dataset: a run of 7 anomalous slices within a
12-slice span (indices 510-521 of 938), all from the same degenerate-
plane mechanism. With the original 11-slice rolling window, more than
half the window at the cluster's center was itself anomalous, which
breaks the median (robust only up to ~50% contamination) -- several
members of the cluster masked each other from detection entirely, at
ANY reasonable `k` threshold, because the "local trend" was partly built
from the anomalies it was supposed to catch. Confirmed directly: with a
51-slice window on freshly-rasterized (unhealed) data, all 7 cluster
members are correctly flagged; with the original 11-slice window, only
1 of 7 was. This is why ``window`` defaults to 51, not 11.

**Revised healing fallback** (previously: leave an unresolved slice
untouched, on the theory it might be genuine dense geometry). On the
SSCF spinodoid dataset, every slice that nudging failed to resolve was
independently confirmed -- via overlaying the resulting skeleton on the
source STL, not via this module -- to produce a spurious, near-zero-
thickness cluster of extra skeleton branches, never legitimate dense
geometry. A flagged slice where no nudge agrees is now healed by
interpolating (unioning) the nearest non-outlier slice on each side,
which is still reported separately from nudge-healed slices in the
diagnostics (``n_outlier_slices_healed_by_nudge`` vs
``..._by_interpolation``) so the distinction remains auditable. Only a
slice with no non-outlier neighbor on either side (i.e. outliers
spanning the whole volume, a sign something is more seriously wrong)
is left truly unresolved. Check
``n_outlier_slices_detected/healed/unresolved`` in the returned
diagnostics after every extraction; ``extract_regions`` also writes a
per-slice solid-fraction CSV so this can be checked programmatically
without opening MATLAB -- but note that CSV alone was NOT enough to
catch the seed14 case on this dataset (an ~11-point spike under the
original k=6.0 detector threshold), which is what motivated lowering it
to k=3.0. **The most reliable check remains a direct look at the
resulting skeleton** (e.g. a histogram of node count vs. each axis, or
the MATLAB STL-overlay visualization) rather than trusting the
per-slice solid-fraction CSV alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from stereom.volume import Volume


@dataclass
class RegionSpec:
    name: str
    stl_path: str
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    pitch_stl_units: float
    """Voxel edge length, in the STL FILE's own coordinate units (not
    micrometres) -- e.g. 0.02 if the STL is in mm and you want 20 um
    voxels. Decoupled from ``stl_units_to_um`` on purpose: this controls
    how finely you sample the geometry, that controls what physical unit
    a "1" in the STL file means. See ``pitch_from_feature_size`` for how
    to derive this from your structure's known minimum feature size --
    do not guess, an under-resolved pitch silently distorts both
    relative density and skeleton quality (see module docstring)."""
    stl_units_to_um: float = 1000.0
    """Scale factor from STL file units to micrometres, used only to tag
    the output Volume's ``voxel_size_um`` correctly. Default 1000 assumes
    the STL is in millimetres (the common CAD/STL-export convention) --
    check ``mesh.extents`` against the physical sample size you expect
    BEFORE trusting this; get it wrong and every length/thickness number
    downstream is silently wrong."""

    @property
    def voxel_size_um(self) -> float:
        return self.pitch_stl_units * self.stl_units_to_um


def pitch_from_feature_size(min_feature_size: float, voxels_across: float = 8.0) -> float:
    """Derivation rule for ``RegionSpec.pitch_stl_units``: resolve the
    smallest feature of interest (e.g. the thinnest expected ligament
    diameter, IN STL FILE UNITS) with at least ``voxels_across`` voxels.
    8 is a reasonable minimum for getting both a clean 1-voxel-wide
    skeleton and a reasonably accurate thickness/relative-density
    measurement; use more (12-20) if you specifically care about
    thickness accuracy.
    """
    return min_feature_size / voxels_across


def load_regions_json(path: str | Path) -> list[RegionSpec]:
    """Load a list of regions from a JSON file. Each entry:
    ``{"name": ..., "stl": ..., "bbox": [xmin,xmax,ymin,ymax,zmin,zmax],
    "pitch_stl_units": ..., "stl_units_to_um": ...}`` (the last field
    optional, default 1000). ``stl`` paths are resolved relative to the
    JSON file's own directory if not absolute.
    """
    path = Path(path)
    entries = json.loads(path.read_text())
    specs = []
    for e in entries:
        stl = Path(e["stl"])
        if not stl.is_absolute():
            stl = path.parent / stl
        bbox = e["bbox"]
        specs.append(
            RegionSpec(
                name=e["name"],
                stl_path=str(stl),
                bbox_min=(bbox[0], bbox[2], bbox[4]),
                bbox_max=(bbox[1], bbox[3], bbox[5]),
                pitch_stl_units=float(e["pitch_stl_units"]),
                stl_units_to_um=float(e.get("stl_units_to_um", 1000.0)),
            )
        )
    return specs


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    mesh.merge_vertices()
    mesh.fix_normals()
    return mesh


def _section_to_2d(mesh: trimesh.Trimesh, z: float):
    sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if sec is None:
        return None, None
    to_2d = sec.to_2D if hasattr(sec, "to_2D") else sec.to_planar
    return to_2d()


def _rasterize_z_slice(
    mesh: trimesh.Trimesh, z: float, origin_xy: np.ndarray, nx: int, ny: int, pitch: float
) -> np.ndarray | None:
    try:
        planar, transform = _section_to_2d(mesh, z)
    except Exception:
        return None
    if planar is None:
        return np.zeros((nx, ny), dtype=bool)
    try:
        polys = planar.polygons_full
    except Exception:
        return None

    tx, ty = transform[0, 3], transform[1, 3]
    img = Image.new("L", (nx, ny), 0)
    draw = ImageDraw.Draw(img)
    for poly in polys:
        if poly is None or not poly.is_valid or poly.is_empty:
            continue
        ext = [((x + tx - origin_xy[0]) / pitch, (y + ty - origin_xy[1]) / pitch) for x, y in poly.exterior.coords]
        draw.polygon(ext, fill=1)
        for interior in poly.interiors:
            hole = [((x + tx - origin_xy[0]) / pitch, (y + ty - origin_xy[1]) / pitch) for x, y in interior.coords]
            draw.polygon(hole, fill=0)
    # PIL image array is (row=y, col=x); transpose to this package's (x, y, z) convention.
    return np.asarray(img, dtype=bool).T


def _local_median_profile(x: np.ndarray, window: int) -> np.ndarray:
    n = len(x)
    if n == 0:
        return x.astype(float).copy()
    window = min(window, n if n % 2 == 1 else n - 1)
    window = max(window, 1)
    half = window // 2
    padded = np.pad(x.astype(float), (half, half), mode="edge")
    return np.array([np.median(padded[i : i + window]) for i in range(n)])


def _detect_slice_outliers(
    frac: np.ndarray, window: int = 51, k: float = 3.0, min_scale: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """Flag Z-slices whose solid fraction deviates abruptly from the
    local trend of their neighbors -- see module docstring. Returns
    ``(outlier_mask, local_median)``. ``min_scale`` floors the noise
    scale at 1 percentage point of solid fraction so a locally
    near-noiseless profile (typical real slice-to-slice variation is
    ~1pt, per the real incident this defends against) doesn't make the
    detector hypersensitive to negligible residuals.

    ``k=3.0`` (was 6.0): real-world testing on the SSCF spinodoid dataset
    found the original k=6.0 threshold (with the global-median-residual
    scale estimate, itself typically pinned at the 1pt floor) missed a
    genuine ~11-point single-slice spike entirely -- confirmed downstream
    as a real, visible cluster of spurious near-zero-thickness skeleton
    branches at that exact Z-height (found by overlaying the skeleton on
    the source STL in MATLAB, not by this detector). k=3.0 catches that
    case and similar moderate spikes.

    ``window=51`` (was 11): a single isolated bad slice isn't the only
    real failure mode -- misassembled cross-sections can cluster in a
    tight run of consecutive Z-heights (confirmed on this dataset: 7 of
    12 consecutive slices anomalous in one run). With an 11-wide window,
    a cluster that dense makes MORE than half the window anomalous,
    which breaks the median itself (it's only robust up to ~50%
    contamination) -- the "local trend" becomes partly built from the
    very anomalies it's supposed to detect, masking members of the
    cluster from each other. A cluster of that size is a small minority
    of a 51-wide window, restoring the median's robustness, at the cost
    of being somewhat less locally-adaptive to genuine fast density
    gradients (fine for this use case; reduce if yours has sharp real
    Z-gradients over <25 slices).
    """
    local_median = _local_median_profile(frac, window)
    resid = frac - local_median
    scale = max(float(np.median(np.abs(resid))), min_scale)
    inside_mesh = local_median > 0.02  # don't flag legitimate outside-mesh boundary slices
    outlier = inside_mesh & (np.abs(resid) > k * scale)
    return outlier, local_median


def _heal_outlier_slice(
    mesh: trimesh.Trimesh,
    z: float,
    origin_xy: np.ndarray,
    nx: int,
    ny: int,
    pitch: float,
    expected: float,
) -> np.ndarray | None:
    """Try small Z-nudges to escape a cross-section polygon-assembly
    degeneracy (module docstring). Returns a replacement slice if some
    nudge's solid fraction agrees with ``expected`` (the local trend),
    else ``None`` (caller falls back to neighbor interpolation -- see
    ``voxelize_region``).

    Widened from the original 6 offsets (0.13/0.29/0.41, +-) to 12,
    filling in 0.07/0.19/0.35 as well: the degenerate cutting heights
    this escapes are a numerical coincidence in the mesh's own vertex
    data, not a fixed fraction of the pitch, so a finer sweep raises the
    odds of landing away from the degeneracy on the first attempt.
    """
    tol = max(0.05, 0.25 * expected)
    offsets = (0.07, -0.07, 0.13, -0.13, 0.19, -0.19, 0.29, -0.29, 0.35, -0.35, 0.41, -0.41)
    for frac_offset in offsets:
        candidate = _rasterize_z_slice(mesh, z + frac_offset * pitch, origin_xy, nx, ny, pitch)
        if candidate is None:
            continue
        if abs(candidate.mean() - expected) < tol:
            return candidate
    return None


def voxelize_region(
    mesh: trimesh.Trimesh,
    bbox_min,
    bbox_max,
    pitch: float,
    *,
    n_jobs: int = 1,
    max_retries: int = 2,
    retry_offset_frac: float = 0.13,
) -> tuple[np.ndarray, dict]:
    """Voxelize exactly ``[bbox_min, bbox_max]`` (STL file units) via
    exact per-Z-slice cross-section rasterization -- see module
    docstring for why this replaced a 3D-flood-fill approach. No mesh
    cropping or repair is needed beforehand.

    Returns ``(dense_bool_array, diagnostics)``, where diagnostics has
    ``n_failed_slices`` (slices where even the retries failed -- these
    are left as void; check this is small relative to ``n_total_slices``)
    and ``n_total_slices``.
    """
    bbox_min = np.asarray(bbox_min, dtype=np.float64)
    bbox_max = np.asarray(bbox_max, dtype=np.float64)
    nx = max(1, int(round((bbox_max[0] - bbox_min[0]) / pitch)))
    ny = max(1, int(round((bbox_max[1] - bbox_min[1]) / pitch)))
    nz = max(1, int(round((bbox_max[2] - bbox_min[2]) / pitch)))

    n_voxels_estimate = nx * ny * nz
    if n_voxels_estimate > 2e9:
        raise ValueError(
            f"pitch={pitch} over bbox {bbox_min}..{bbox_max} would produce "
            f"~{n_voxels_estimate:.2e} voxels -- almost certainly not what you "
            "want (and won't fit in RAM). Use a coarser pitch or a smaller region."
        )

    z_centers = bbox_min[2] + (np.arange(nz) + 0.5) * pitch
    origin_xy = bbox_min[:2]

    def _one(z: float) -> np.ndarray | None:
        result = _rasterize_z_slice(mesh, z, origin_xy, nx, ny, pitch)
        attempt = 1
        while result is None and attempt <= max_retries:
            offset = pitch * retry_offset_frac * attempt
            result = _rasterize_z_slice(mesh, z + offset, origin_xy, nx, ny, pitch)
            attempt += 1
        return result

    if n_jobs == 1 or nz < 8:
        slices = [_one(z) for z in z_centers]
    else:
        from joblib import Parallel, delayed

        slices = Parallel(n_jobs=n_jobs)(delayed(_one)(z) for z in z_centers)

    dense = np.zeros((nx, ny, nz), dtype=bool)
    n_failed = 0
    for k, s in enumerate(slices):
        if s is None:
            n_failed += 1
        else:
            dense[:, :, k] = s

    frac = dense.mean(axis=(0, 1))
    outlier_mask, local_median = _detect_slice_outliers(frac)
    outlier_indices = np.flatnonzero(outlier_mask)
    n_healed_by_nudge = 0
    nudge_failed_indices: list[int] = []
    for idx in outlier_indices:
        healed = _heal_outlier_slice(mesh, z_centers[idx], origin_xy, nx, ny, pitch, local_median[idx])
        if healed is not None:
            dense[:, :, idx] = healed
            n_healed_by_nudge += 1
        else:
            nudge_failed_indices.append(int(idx))

    # Any outlier no nudge could escape is, by construction, a confirmed
    # numerical misassembly (not "possibly real, leave it" -- module
    # docstring's original conservative stance was revised after finding
    # these consistently correspond to spurious dense skeleton clusters,
    # never legitimate dense geometry, on real data from this dataset).
    # Fall back to interpolating from the nearest NON-outlier slices on
    # each side, which is guaranteed to exist as long as outliers don't
    # span the entire volume.
    n_healed_by_interpolation = 0
    still_unresolved: list[int] = []
    good_mask = ~outlier_mask
    good_indices = np.flatnonzero(good_mask)
    for idx in nudge_failed_indices:
        before = good_indices[good_indices < idx]
        after = good_indices[good_indices > idx]
        if before.size == 0 and after.size == 0:
            still_unresolved.append(idx)
            continue
        lo = dense[:, :, before[-1]] if before.size else None
        hi = dense[:, :, after[0]] if after.size else None
        if lo is not None and hi is not None:
            interpolated = lo | hi  # union: conservative, avoids losing real thin features either side
        else:
            interpolated = lo if lo is not None else hi
        dense[:, :, idx] = interpolated
        n_healed_by_interpolation += 1

    return dense, {
        "n_failed_slices": n_failed,
        "n_total_slices": nz,
        "n_outlier_slices_detected": int(outlier_mask.sum()),
        "n_outlier_slices_healed": n_healed_by_nudge + n_healed_by_interpolation,
        "n_outlier_slices_healed_by_nudge": n_healed_by_nudge,
        "n_outlier_slices_healed_by_interpolation": n_healed_by_interpolation,
        "n_outlier_slices_unresolved": len(still_unresolved),
        "unresolved_outlier_slice_indices": still_unresolved,
    }


def crop_mesh_for_visualization(
    mesh: trimesh.Trimesh, bbox_min, bbox_max, pad: float = 0.0
) -> trimesh.Trimesh:
    """Crop the mesh to (a padded) ``[bbox_min, bbox_max]`` via six plane
    cuts, uncapped, for exporting a small companion STL to overlay in the
    MATLAB visualization -- NOT used for the analysis volume itself
    (``voxelize_region`` needs no cropping at all, see module docstring).
    Uncapped cuts are fine here: an open boundary where the crop plane
    sliced through the structure doesn't matter for a semi-transparent
    visual overlay, and avoids needing shapely/rtree/a triangulation
    engine just to re-cap the cut faces.
    """
    bbox_min = np.asarray(bbox_min, dtype=np.float64)
    bbox_max = np.asarray(bbox_max, dtype=np.float64)
    cropped = mesh.copy()
    planes = [
        (bbox_min - pad, [1, 0, 0]),
        (bbox_max + pad, [-1, 0, 0]),
        (bbox_min - pad, [0, 1, 0]),
        (bbox_max + pad, [0, -1, 0]),
        (bbox_min - pad, [0, 0, 1]),
        (bbox_max + pad, [0, 0, -1]),
    ]
    for origin, normal in planes:
        cropped = trimesh.intersections.slice_mesh_plane(
            cropped, plane_normal=normal, plane_origin=origin, cap=False
        )
        if cropped is None or len(cropped.faces) == 0:
            return cropped
    return cropped


def extract_regions(
    specs: list[RegionSpec],
    output_dir: str | Path,
    *,
    axes: tuple[str, str, str] = ("L", "C", "R"),
    n_jobs: int = 1,
    export_stl: bool = True,
    verbose: bool = True,
) -> dict[str, Volume]:
    """Extract every region in ``specs``, grouping by source STL file so
    each mesh is loaded only once no matter how many regions are cut from
    it. Writes ``{output_dir}/{name}.npz`` (loadable directly by
    ``stereom run --input {name}.npz`` or ``io.load_volume_npz``) for
    each region, and -- if ``export_stl`` -- a companion
    ``{output_dir}/{name}.stl`` (a crop of the source mesh, for the
    MATLAB visualization's transparent overlay). Returns the in-memory
    ``Volume`` objects too.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_file: dict[str, list[RegionSpec]] = {}
    for s in specs:
        by_file.setdefault(s.stl_path, []).append(s)

    results: dict[str, Volume] = {}
    for stl_path, group in by_file.items():
        if verbose:
            print(f"[mesh_regions] loading {stl_path} ...")
        mesh = load_mesh(stl_path)
        if verbose:
            print(
                f"[mesh_regions]   {len(mesh.faces)} faces, extents {mesh.extents}, "
                f"watertight={mesh.is_watertight}"
            )

        for spec in group:
            if verbose:
                print(f"[mesh_regions]   voxelizing '{spec.name}' at pitch={spec.pitch_stl_units} ...")
            region, diag = voxelize_region(mesh, spec.bbox_min, spec.bbox_max, spec.pitch_stl_units, n_jobs=n_jobs)
            volume = Volume(data=region, voxel_size=spec.voxel_size_um, axes=axes)
            out_path = output_dir / f"{spec.name}.npz"
            from stereom.io import save_volume_npz

            save_volume_npz(volume, out_path)
            results[spec.name] = volume

            # Per-slice solid-fraction profile: lets correctness be checked
            # programmatically (or by re-plotting) without opening MATLAB --
            # see module docstring re: the outlier-detection/healing step.
            import csv

            frac_by_z = region.mean(axis=(0, 1))
            unresolved = set(diag.get("unresolved_outlier_slice_indices", []))
            with (output_dir / f"{spec.name}_slice_profile.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["z_index", "z_stl_units", "solid_fraction", "unresolved_outlier"])
                for k, val in enumerate(frac_by_z):
                    z_k = spec.bbox_min[2] + (k + 0.5) * spec.pitch_stl_units
                    writer.writerow([k, f"{z_k:.6f}", f"{val:.6f}", k in unresolved])

            if export_stl:
                pad = spec.pitch_stl_units * 4
                cropped = crop_mesh_for_visualization(mesh, spec.bbox_min, spec.bbox_max, pad=pad)
                if cropped is not None and len(cropped.faces) > 0:
                    cropped.export(output_dir / f"{spec.name}.stl")
                    # Sidecar alignment metadata: lets a downstream consumer (the
                    # MATLAB overlay, or anything else) convert this companion
                    # STL's vertex coordinates (STL file units) into this
                    # volume's voxel-index coordinates: voxel = (stl_point -
                    # bbox_min) / pitch_stl_units. See matlab_export.py /
                    # visualize_stereom_results.m Section 11.
                    alignment = {
                        "stl_file": f"{spec.name}.stl",
                        "bbox_min_stl_units": list(spec.bbox_min),
                        "pitch_stl_units": spec.pitch_stl_units,
                        "stl_units_to_um": spec.stl_units_to_um,
                    }
                    (output_dir / f"{spec.name}_stl_alignment.json").write_text(json.dumps(alignment, indent=2))

            if verbose:
                print(
                    f"[mesh_regions]   '{spec.name}': shape {volume.shape}, "
                    f"solid fraction {volume.solid_fraction:.4f}, "
                    f"voxel_size_um={volume.voxel_size:.4f}, "
                    f"failed slices {diag['n_failed_slices']}/{diag['n_total_slices']}, "
                    f"outlier slices {diag['n_outlier_slices_detected']} detected / "
                    f"{diag['n_outlier_slices_healed']} healed / "
                    f"{diag['n_outlier_slices_unresolved']} unresolved -> {out_path}"
                )
                if diag["n_failed_slices"] > 0.05 * diag["n_total_slices"]:
                    print(
                        f"[mesh_regions]   WARNING: {diag['n_failed_slices']} of "
                        f"{diag['n_total_slices']} slices failed to rasterize (>5%) -- "
                        "the mesh may have more extensive defects than usual in this "
                        "region; treat the result with more caution than usual."
                    )
                if diag["n_outlier_slices_unresolved"] > 0:
                    print(
                        f"[mesh_regions]   WARNING: {diag['n_outlier_slices_unresolved']} slice(s) "
                        f"at index {diag['unresolved_outlier_slice_indices']} deviate abruptly from "
                        "their neighbors' solid fraction and could NOT be resolved by nudging the "
                        "cutting plane -- this may be a real, unusually dense local feature rather "
                        "than the usual cross-section numerical artifact (see module docstring); "
                        f"check {spec.name}_slice_profile.csv and inspect that region before trusting "
                        "downstream node/branch counts near it."
                    )
                if diag["n_outlier_slices_detected"] > 0.10 * diag["n_total_slices"]:
                    print(
                        f"[mesh_regions]   WARNING: {diag['n_outlier_slices_detected']} of "
                        f"{diag['n_total_slices']} slices ({diag['n_outlier_slices_detected'] / diag['n_total_slices']:.0%}) "
                        "were flagged as outliers -- unusually many; the outlier detector's "
                        "assumptions (smooth slice-to-slice variation) may not hold for this "
                        "mesh/region, so treat individual heals with extra caution."
                    )
                if volume.solid_fraction < 1e-4 or volume.solid_fraction > 0.999:
                    print(
                        f"[mesh_regions]   WARNING: solid fraction {volume.solid_fraction:.4f} "
                        "looks degenerate (empty or fully solid) -- check the bounding box "
                        "actually intersects the mesh, and that pitch_stl_units/stl_units_to_um "
                        "are set to sensible values for this file's coordinate scale."
                    )
    return results
