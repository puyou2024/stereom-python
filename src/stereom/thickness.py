"""Branch thickness. Replaces ``main_thickness.m`` Stages A-D (spec
Sec 3.11) -- by far the most expensive stage in the original pipeline (D18:
it allocates and multiplies a full-volume temporary for every interior
voxel of every branch, ~10^12 operations on the demo volume) and the
"single highest-value change" the spec identifies (Sec 7).

Default method: a single Euclidean distance transform of the whole volume,
sampled along each branch path. This is the standard "local thickness"
definition (directly comparable to ``porespy.filters.local_thickness`` and
BoneJ), is O(N_volume) once for the entire volume, and needs no magic
constants (sphere radius, angular tolerance, outlier guard -- all gone).
It reports the *inscribed* radius rather than the original's *mean
surface-normal* radius; for a circular cross-section the two agree, and
the difference is small and well-defined for the mildly non-circular
sections stereom struts actually have (spec Sec 3.11 refactor note).

A slower, opt-in ``legacy_mean_radius_profile`` reproduces the original's
geometric definition (mean distance to surface voxels near the
perpendicular plane) for users who need closer parity with the published
numbers, but fixes its three defects: D18 (KD-tree over surface voxels
instead of a full-volume temporary per sample point), D13 (no bounds
check -> KD-tree needs none), D14 (local tangent instead of the
endpoint-to-endpoint chord).

Per spec Sec 10.2 requirement 8: thickness is always reported as an
explicitly-named radius, with a diameter column also available -- never a
bare "thickness".
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from stereom.volume import Volume


def compute_edt(volume: Volume, *, max_chunk_voxels: int = 80_000_000) -> np.ndarray:
    """Euclidean distance transform of the solid phase, in physical units
    (micrometres), computed once for the whole volume.

    scipy's exact EDT allocates a transient ``int32`` indices array of
    shape ``(3, *volume.shape)`` internally regardless of whether indices
    are requested (an implementation detail of its separable algorithm,
    not something callable-away via kwargs) -- for a ~366M-voxel volume
    that alone is ~4GB and can fail to allocate even with more than that
    much free physical memory, depending on what else is running. Above
    ``max_chunk_voxels`` this transparently chunks along the volume's
    largest axis and stitches the results, which is mathematically exact
    (not an approximation): EDT(x) depends only on data within distance
    EDT(x) of x, so as long as each chunk is padded by more than any
    physically plausible feature radius, the padded chunk contains
    everything needed to compute the exact global EDT for its core
    (non-padding) region. Only internal chunk boundaries get padding --
    the volume's own true edges are left exactly as scipy would treat
    them (nothing padded there, same as the unchunked call).
    """
    if volume.data.size <= max_chunk_voxels:
        return ndimage.distance_transform_edt(
            volume.data, sampling=(volume.voxel_size,) * 3
        ).astype(np.float32)
    return _compute_edt_chunked(volume, max_chunk_voxels=max_chunk_voxels)


def _compute_edt_chunked(
    volume: Volume, *, max_chunk_voxels: int, pad_um: float = 400.0
) -> np.ndarray:
    data = volume.data
    vs = volume.voxel_size
    axis = int(np.argmax(data.shape))
    n = data.shape[axis]
    pad_vox = int(np.ceil(pad_um / vs))

    other_size = 1
    for i, s in enumerate(data.shape):
        if i != axis:
            other_size *= s
    max_padded_len = max(2 * pad_vox + 1, int(max_chunk_voxels // max(other_size, 1)))
    core_len = max(1, max_padded_len - 2 * pad_vox)
    n_chunks = max(1, int(np.ceil(n / core_len)))

    out = np.empty(data.shape, dtype=np.float32)
    base_slicer = [slice(None)] * data.ndim
    for i in range(n_chunks):
        core_lo = i * core_len
        core_hi = min(n, core_lo + core_len)
        pad_lo = max(0, core_lo - pad_vox)
        pad_hi = min(n, core_hi + pad_vox)

        chunk_slicer = list(base_slicer)
        chunk_slicer[axis] = slice(pad_lo, pad_hi)
        chunk = data[tuple(chunk_slicer)]

        chunk_edt = ndimage.distance_transform_edt(chunk, sampling=(vs,) * 3).astype(
            np.float32
        )

        rel_lo = core_lo - pad_lo
        rel_hi = rel_lo + (core_hi - core_lo)
        extract_slicer = list(base_slicer)
        extract_slicer[axis] = slice(rel_lo, rel_hi)
        write_slicer = list(base_slicer)
        write_slicer[axis] = slice(core_lo, core_hi)
        out[tuple(write_slicer)] = chunk_edt[tuple(extract_slicer)]
        del chunk_edt, chunk

    return out


def sample_profiles_edt(
    G: nx.MultiGraph, edt_um: np.ndarray, min_path_voxels: int = 10
) -> nx.MultiGraph:
    """Sample the EDT along every branch path. Sets ``profile_radius_um``
    (ndarray), ``t_min_radius_um``, ``t_cen_radius_um``, ``t_mean_radius_um``
    edge attributes, all radii in micrometres. Branches shorter than
    ``min_path_voxels`` get ``profile_radius_um = None`` and NaN summaries
    (matches the original's exclusion, spec Sec 3.11 failure mode 5)."""
    G = G.copy()
    for u, v, k, data in G.edges(keys=True, data=True):
        path = data["path"]
        if path.shape[0] < min_path_voxels:
            data["profile_radius_um"] = None
            data["t_min_radius_um"] = float("nan")
            data["t_cen_radius_um"] = float("nan")
            data["t_mean_radius_um"] = float("nan")
            continue
        profile = edt_um[path[:, 0], path[:, 1], path[:, 2]]
        data["profile_radius_um"] = profile
        data["t_min_radius_um"] = float(profile.min())
        data["t_cen_radius_um"] = float(profile[len(profile) // 2])
        data["t_mean_radius_um"] = float(profile.mean())
    return G


def surface_mask(volume: Volume) -> np.ndarray:
    """Boundary voxels of the solid phase: solid voxels with at least one
    background 6-neighbour. Replaces ``edge3(orig,'approxcanny',0.6)``
    (Canny edge detection, designed for grayscale imagery, applied to an
    already-binary segmentation) with the natural, parameter-free
    definition of "surface voxel" for a binary volume."""
    eroded = ndimage.binary_erosion(volume.data, structure=ndimage.generate_binary_structure(3, 1))
    return volume.data & ~eroded


def legacy_mean_radius_profile(
    G: nx.MultiGraph,
    volume: Volume,
    *,
    search_radius_um: float | None = None,
    angular_tolerance_deg: float = 2.0,
    tangent_window: int = 3,
    min_path_voxels: int = 10,
) -> nx.MultiGraph:
    """Reproduce the original's geometric definition -- mean distance from
    each centerline point to surface voxels lying within
    ``angular_tolerance_deg`` of the plane perpendicular to the branch's
    LOCAL tangent -- but via a KD-tree over surface voxels (built once)
    instead of a full-volume temporary per sample point (fixes D18), and a
    local tangent instead of the endpoint chord (fixes D14). No bounds
    checking is needed: a KD-tree query near a volume edge simply returns
    whatever surface voxels exist (fixes D13).

    Sets ``profile_radius_legacy_um`` (values) and
    ``profile_index_legacy`` (the corresponding index into ``path``, so
    callers can still compute correctly-aligned arclength) edge
    attributes. Points where no surface voxel satisfies the angular
    tolerance are OMITTED, not NaN-filled -- this matches what the
    original actually does (``thick`` is built by appending only accepted
    samples; skipped points are never inserted as a placeholder), which
    is why ``main_profile.m``'s "reject the branch if any NaN in t" guard
    is, in practice, close to a no-op for well-formed data.
    """
    G = G.copy()
    surf = surface_mask(volume)
    surf_coords_vox = np.argwhere(surf)
    if surf_coords_vox.shape[0] == 0:
        raise ValueError("Volume has no detectable surface voxels")
    surf_coords_um = surf_coords_vox * volume.voxel_size
    tree = cKDTree(surf_coords_um)

    if search_radius_um is None:
        # Must exceed the largest expected strut radius (spec Sec 4 rule #14).
        edt = compute_edt(volume)
        search_radius_um = 3.0 * float(edt.max())

    cos_tol = np.cos(np.deg2rad(90.0 - angular_tolerance_deg))

    for u, v, k, data in G.edges(keys=True, data=True):
        path = data["path"]
        n = path.shape[0]
        if n < min_path_voxels:
            data["profile_radius_legacy_um"] = None
            data["profile_index_legacy"] = None
            data["t_min_radius_legacy_um"] = float("nan")
            continue

        path_um = path.astype(np.float64) * volume.voxel_size
        values: list[float] = []
        indices: list[int] = []
        half = tangent_window // 2
        for j in range(1, n - 1):
            lo, hi = max(0, j - half), min(n, j + half + 1)
            tangent = path_um[hi - 1] - path_um[lo]
            tnorm = np.linalg.norm(tangent)
            if tnorm == 0:
                continue
            tangent /= tnorm

            p = path_um[j]
            idx = tree.query_ball_point(p, r=search_radius_um)
            if not idx:
                continue
            c = surf_coords_um[idx] - p
            cnorm = np.linalg.norm(c, axis=1)
            valid = cnorm > 0
            if not np.any(valid):
                continue
            cos_angle = np.abs((c[valid] @ tangent) / cnorm[valid])
            near_perp = cos_angle < cos_tol
            if not np.any(near_perp):
                continue
            values.append(float(cnorm[valid][near_perp].mean()))
            indices.append(j)

        profile = np.array(values, dtype=np.float64)
        data["profile_radius_legacy_um"] = profile
        data["profile_index_legacy"] = np.array(indices, dtype=np.int64)
        data["t_min_radius_legacy_um"] = float(profile.min()) if profile.size else float("nan")

    return G


def add_diameter_columns(nodes_or_branches_df, radius_cols: list[str]):
    """Add ``*_diameter_um`` columns alongside each named radius column
    (spec Sec 10.2 requirement 8: report both, explicitly named)."""
    df = nodes_or_branches_df.copy()
    for col in radius_cols:
        if col in df.columns:
            new_col = col.replace("radius", "diameter")
            df[new_col] = df[col] * 2.0
    return df
