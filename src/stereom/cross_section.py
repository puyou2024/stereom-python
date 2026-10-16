"""True branch cross-sections and an exactly-conserving volume partition.

Why this exists
---------------
``thickness.sample_profiles_edt`` reports the *inscribed* radius -- the
radius of the largest ball that fits inside the solid at that centerline
point. For a circular cross-section that is the whole story, and
``thickness``'s own module docstring says so: the inscribed-vs-mean
difference is "small and well-defined for the mildly non-circular sections
stereom struts actually have".

That premise is load-bearing, and it fails for structures whose sections
are elongated. The inscribed radius is the section's MINOR half-axis, so
modelling a branch as a circular cylinder of that radius recovers only
about ``b/a`` of its true area. Every ``pi * r^2`` quantity -- relative
density, load-bearing area, the SSCF/min-cut strength inputs -- inherits
that factor.

Two measurements, and the difference between them matters
--------------------------------------------------------
1. ``assign_solid_to_skeleton`` partitions EVERY solid voxel to its
   nearest skeleton voxel, hence to one branch or one junction cluster.
   This is the trustworthy route to area and density: the parts are
   disjoint and sum to the whole by construction, so a branch CANNOT be
   credited with material that belongs to its neighbour. Effective area
   is then ``assigned_volume / branch_length``.

2. ``sample_cross_sections`` cuts the plane perpendicular to the local
   tangent for section SHAPE (aspect ratio, principal half-widths), but
   keeps only the pixels the partition already assigned to this branch.

The naive version of (2) -- "take the connected component containing the
centre point" -- is wrong on a bicontinuous solid and was measured to be
wrong here: neighbouring ligaments touch within the cutting plane, the
component spans several of them, and the implied relative density came out
at 0.78 against a true solid fraction of 0.351. The ownership mask is what
makes the section measurement safe.

Sanity check to run on any new structure: ``sum(assigned volumes)`` must
equal the solid voxel count exactly, and ``sum(L * A_eff) / V`` must land
near the measured solid fraction. ``partition_report`` returns both.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy import ndimage
from scipy.ndimage import map_coordinates

from stereom.volume import Volume

NODE_OWNER = -1


def _perpendicular_basis(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors spanning the plane perpendicular to ``t``."""
    a = np.array([1.0, 0.0, 0.0])
    if abs(float(t @ a)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(t, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(t, e1)
    return e1, e2


def edge_order(G: nx.MultiGraph) -> list[tuple[int, int, int]]:
    """Stable (u, v, key) ordering used to index branches 0..nB-1."""
    return [(u, v, k) for u, v, k in G.edges(keys=True)]


def assign_solid_to_skeleton(
    G: nx.MultiGraph, volume: Volume, skel: np.ndarray | None = None,
    *, spur_policy: str = "nearest_branch",
) -> tuple[np.ndarray, list[tuple[int, int, int]], dict]:
    """Partition the solid phase among branches by nearest skeleton voxel.

    Returns ``(owner, order, stats)`` where ``owner`` is an int32 array the
    shape of the volume holding, for each solid voxel, the index into
    ``order`` of the branch that owns it, ``NODE_OWNER`` (-1) if the nearest
    skeleton voxel belongs to a junction cluster rather than any branch
    path, and -2 for background.

    Junction clusters are kept as a separate bucket on purpose: that
    material is real, but it belongs to no single branch, so folding it
    into one would double-count at every node. It is reported separately.

    ``spur_policy`` decides what happens to skeleton voxels that survive
    thinning but sit on no kept branch path -- the short medial spurs that
    ``graph_build``/``graph_clean`` prune as dangling. On this Spinodoid
    crop there are 1059 of them (median 4 voxels long), and they are not
    noise: seeding them takes 25.5% of the solid out of the branch network.
    They are the medial signature of a lobed, non-circular cross-section --
    the medial axis of an elongated section is a tree, not a point, so the
    extra structure shows up as spurs off the main curve. Pruning them is
    right for a clean topology and wrong for a volume budget.

      ``"nearest_branch"`` (default) does not seed them, so their material
      flows to the nearest kept branch or junction. Branch areas then
      account for the whole solid and need no correction factor.
      ``"separate"`` seeds them as their own bucket, which is what you want
      when auditing how much material the pruning moves.
    """
    data = volume.data.astype(bool)
    order = edge_order(G)
    index_of = {key: i for i, key in enumerate(order)}

    # Seed image: branch paths carry their branch index + 1; every other
    # skeleton voxel is a junction-cluster seed.
    seed = np.zeros(data.shape, dtype=np.int32)
    for u, v, k, ed in G.edges(keys=True, data=True):
        path = ed.get("path")
        if path is None or path.shape[0] == 0:
            continue
        i = index_of[(u, v, k)]
        seed[path[:, 0], path[:, 1], path[:, 2]] = i + 1

    if skel is not None and spur_policy == "separate":
        seed[skel & (seed == 0)] = -1
    for n in G.nodes():
        p = G.nodes[n]["pos"]
        if seed[int(p[0]), int(p[1]), int(p[2])] == 0:
            seed[int(p[0]), int(p[1]), int(p[2])] = -1

    has_seed = seed != 0
    if not has_seed.any():
        raise ValueError("No skeleton seeds found -- cannot partition.")

    _dist, inds = ndimage.distance_transform_edt(~has_seed, return_indices=True)
    nearest = seed[inds[0], inds[1], inds[2]]
    del inds, _dist

    owner = np.full(data.shape, -2, dtype=np.int32)
    owner[data] = np.where(nearest[data] > 0, nearest[data] - 1, NODE_OWNER)
    del nearest

    n_solid = int(data.sum())
    n_node_owned = int((owner == NODE_OWNER).sum())
    stats = dict(
        spur_policy=spur_policy,
        n_solid_voxels=n_solid,
        n_branch_owned=int(n_solid - n_node_owned),
        n_junction_owned=n_node_owned,
        junction_owned_fraction=float(n_node_owned / max(n_solid, 1)),
        n_branches=len(order),
    )
    return owner, order, stats


def branch_volumes(
    G: nx.MultiGraph, owner: np.ndarray, order: list[tuple[int, int, int]],
    volume: Volume
) -> nx.MultiGraph:
    """Attach the partition's per-branch volume and effective area.

    Sets ``v_assigned_um3``, ``t_area_eff_um2`` (= volume / length) and
    ``t_equiv_radius_um`` (= sqrt(area/pi)) on every edge. ``t_equiv_radius_um``
    is the radius to use wherever a branch is modelled as a circular
    cylinder -- it preserves AREA, which is what density and load-bearing
    section care about, unlike the inscribed radius which preserves the
    minor half-axis.
    """
    G = G.copy()
    vox3 = volume.voxel_size ** 3
    counts = np.bincount(owner[owner >= 0].ravel(), minlength=len(order))
    for i, (u, v, k) in enumerate(order):
        ed = G[u][v][k]
        vol_um3 = float(counts[i]) * vox3
        length = float(ed.get("length_um", np.nan))
        ed["v_assigned_um3"] = vol_um3
        if length and np.isfinite(length) and length > 0:
            area = vol_um3 / length
            ed["t_area_eff_um2"] = area
            ed["t_equiv_radius_um"] = float(np.sqrt(area / np.pi))
        else:
            ed["t_area_eff_um2"] = float("nan")
            ed["t_equiv_radius_um"] = float("nan")
    return G


def partition_report(G: nx.MultiGraph, volume: Volume, stats: dict) -> dict:
    """The two conservation checks that catch an over-counting section
    measurement before its numbers get used for anything."""
    V = float(np.prod(np.array(volume.data.shape, dtype=np.float64))) * volume.voxel_size ** 3
    L = np.array([d.get("length_um", np.nan) for _, _, d in G.edges(data=True)], float)
    A = np.array([d.get("t_area_eff_um2", np.nan) for _, _, d in G.edges(data=True)], float)
    Vb = np.array([d.get("v_assigned_um3", np.nan) for _, _, d in G.edges(data=True)], float)
    ok = np.isfinite(L) & np.isfinite(A)
    rho_from_branches = float(np.nansum(L[ok] * A[ok]) / V)
    return dict(
        solid_fraction_measured=float(volume.data.mean()),
        rho_from_branch_length_times_area=rho_from_branches,
        branch_assigned_volume_fraction=float(np.nansum(Vb) / V),
        junction_assigned_volume_fraction=float(
            stats["n_junction_owned"] * volume.voxel_size ** 3 / V),
        conservation_residual=float(
            volume.data.mean()
            - (np.nansum(Vb) + stats["n_junction_owned"] * volume.voxel_size ** 3) / V),
        **stats,
    )


def _measure_section(mask2d: np.ndarray, voxel_size_um: float) -> dict | None:
    """Section shape from the component of ``mask2d`` touching its centre."""
    c = mask2d.shape[0] // 2
    if not mask2d[c, c]:
        return None
    lab, _n = ndimage.label(mask2d, structure=np.ones((3, 3), dtype=int))
    comp = lab == lab[c, c]
    n_pix = int(comp.sum())
    if n_pix < 3:
        return None
    area = n_pix * voxel_size_um ** 2

    ys, xs = np.nonzero(comp)
    pts = np.column_stack([ys, xs]).astype(np.float64)
    pts -= pts.mean(axis=0)
    ev = np.linalg.eigvalsh(pts.T @ pts / n_pix)[::-1]
    hw_max = 2.0 * np.sqrt(max(float(ev[0]), 0.0)) * voxel_size_um
    hw_min = 2.0 * np.sqrt(max(float(ev[1]), 0.0)) * voxel_size_um
    dt = ndimage.distance_transform_edt(comp)
    return dict(
        area_um2=area,
        r_equiv_um=float(np.sqrt(area / np.pi)),
        halfwidth_max_um=hw_max,
        halfwidth_min_um=hw_min,
        aspect_ratio=(float(hw_max / hw_min) if hw_min > 1e-9 else float("nan")),
        r_inscribed_um=float(dt[c, c]) * voxel_size_um,
        touches_border=bool(comp[0].any() or comp[-1].any()
                            or comp[:, 0].any() or comp[:, -1].any()),
    )


def sample_cross_sections(
    G: nx.MultiGraph,
    volume: Volume,
    owner: np.ndarray,
    order: list[tuple[int, int, int]],
    *,
    edt_um: np.ndarray | None = None,
    tangent_window: int = 5,
    stride: int = 1,
    min_path_voxels: int = 10,
    extent_factor: float = 3.0,
    include_junction_voxels: bool = True,
) -> nx.MultiGraph:
    """Section SHAPE along each branch, restricted to voxels the partition
    assigned to that branch (plus, optionally, junction-owned voxels, which
    are genuinely part of the local section near a node).

    Sets the per-edge arrays ``section_index``, ``section_area_um2``,
    ``section_r_equiv_um``, ``section_halfwidth_max_um``,
    ``section_halfwidth_min_um``, ``section_aspect_ratio``,
    ``section_r_inscribed_um``, and the scalar ``t_aspect_ratio``.
    """
    from stereom.thickness import compute_edt

    G = G.copy()
    vox = volume.voxel_size
    if edt_um is None:
        edt_um = compute_edt(volume)
    index_of = {key: i for i, key in enumerate(order)}

    half_win = max(1, tangent_window // 2)
    n_measured = n_truncated = n_off = 0

    for u, v, k, ed in G.edges(keys=True, data=True):
        path = ed.get("path")
        bidx = index_of[(u, v, k)]
        if path is None or path.shape[0] < min_path_voxels:
            _blank_section_attrs(ed)
            continue

        p = path.astype(np.float64)
        n = p.shape[0]
        idxs, recs = [], []
        for j in range(half_win, n - half_win, max(1, stride)):
            tang = p[j + half_win] - p[j - half_win]
            nt = np.linalg.norm(tang)
            if nt == 0:
                continue
            tang = tang / nt
            e1, e2 = _perpendicular_basis(tang)
            r_here = float(edt_um[int(p[j][0]), int(p[j][1]), int(p[j][2])])
            h = max(4, int(np.ceil(extent_factor * max(r_here, vox) / vox)))

            g = np.arange(-h, h + 1, dtype=np.float64)
            uu, vv = np.meshgrid(g, g, indexing="ij")
            pts = (p[j][None, None, :]
                   + uu[..., None] * e1[None, None, :]
                   + vv[..., None] * e2[None, None, :])
            own = map_coordinates(owner, pts.reshape(-1, 3).T, order=0,
                                  mode="constant", cval=-2).reshape(uu.shape)
            mask = own == bidx
            if include_junction_voxels:
                mask |= own == NODE_OWNER

            rec = _measure_section(mask, vox)
            if rec is None:
                n_off += 1
                continue
            if rec["touches_border"]:
                n_truncated += 1
                continue
            idxs.append(j)
            recs.append(rec)
            n_measured += 1

        if not recs:
            _blank_section_attrs(ed)
            continue

        ed["section_index"] = np.asarray(idxs, dtype=np.int64)
        for key, attr in (("area_um2", "section_area_um2"),
                          ("r_equiv_um", "section_r_equiv_um"),
                          ("halfwidth_max_um", "section_halfwidth_max_um"),
                          ("halfwidth_min_um", "section_halfwidth_min_um"),
                          ("aspect_ratio", "section_aspect_ratio"),
                          ("r_inscribed_um", "section_r_inscribed_um")):
            ed[attr] = np.array([r[key] for r in recs], dtype=np.float64)
        ed["t_aspect_ratio"] = float(np.nanmean(ed["section_aspect_ratio"]))

    G.graph["cross_section_diagnostics"] = dict(
        n_sections_measured=n_measured,
        n_sections_dropped_truncated=n_truncated,
        n_sections_centre_not_owned=n_off,
        stride=stride, tangent_window=tangent_window,
        extent_factor=extent_factor,
        include_junction_voxels=include_junction_voxels,
    )
    return G


def _blank_section_attrs(ed: dict) -> None:
    for a in ("section_index", "section_area_um2", "section_r_equiv_um",
              "section_halfwidth_max_um", "section_halfwidth_min_um",
              "section_aspect_ratio", "section_r_inscribed_um"):
        ed[a] = None
    ed["t_aspect_ratio"] = float("nan")
