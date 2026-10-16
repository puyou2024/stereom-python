"""Ring (loop) detection and geometry. Replaces ``findloop.m`` (309 lines)
and ``planeFitting_v2.m`` / ``ringSize_{4,5,6,7}E.m`` (spec Sec 3.16).

Fixes, all per spec's own recommended remedies:

* **D4 (the serious one)**: the original deduplicates rings by hashing the
  PRODUCT of node linear indices, scaled and rounded -- for the demo
  volume this overflows IEEE double's exact-integer range by ~30 orders of
  magnitude, so distinct rings collide and are silently dropped, with the
  undercount growing with volume size. Here, a ring is canonicalized as
  ``tuple(sorted(nodes))`` and deduplicated on that exact tuple: O(1) per
  ring, exact, no numeric overflow possible.

* Ancestor exclusion during the DFS is full (excludes every node already
  on the current path), not just the immediate parent, so no post-hoc
  "does this candidate repeat a node" filter is needed at all (the
  original's unrolled BFS only excludes the parent at each level and
  relies on a separate filter pass to catch the rest).

* Ring area is computed from the ring's node vertices projected onto the
  fitted plane (Newell's formula on the projected polygon), instead of fan
  triangulation from an arbitrary first vertex on a possibly non-planar
  point set (spec Sec 3.16 caveat i) using an unstable Heron's-formula
  variant (caveat ii). This single function replaces the four
  near-duplicate ``ringSize_{4,5,6,7}E.m`` files (fixes D24) since it does
  not care how many vertices the polygon has.

* The flatness filter (R^2 >= threshold, PCA-based) and the redundancy
  filter (via ``networkx.minimum_cycle_basis``, giving a minimal
  independent cycle set "for free" -- spec Sec 8 row 14) are both
  implemented; neither exists in the shipped MATLAB release (present only
  as commented-out code, spec Sec 1.3 / 3.16).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from stereom.geometry import oriented_path


def _find_cycles_from_seed(
    G: nx.MultiGraph, seed: int, min_size: int, max_size: int
) -> list[tuple[int, ...]]:
    cycles: list[tuple[int, ...]] = []
    path: list[int] = [seed]
    path_set: set[int] = {seed}

    def dfs() -> None:
        current = path[-1]
        for neighbor in G.neighbors(current):
            if neighbor == seed:
                if len(path) >= min_size:
                    cycles.append(tuple(path))
                continue
            if neighbor in path_set or len(path) >= max_size:
                continue
            path.append(neighbor)
            path_set.add(neighbor)
            dfs()
            path.pop()
            path_set.discard(neighbor)

    dfs()
    return cycles


def _find_cycles_for_seeds(
    G: nx.MultiGraph, seeds: list[int], min_size: int, max_size: int
) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for s in seeds:
        out.extend(_find_cycles_from_seed(G, s, min_size, max_size))
    return out


def find_rings(
    G: nx.MultiGraph,
    min_size: int = 4,
    max_size: int = 7,
    seed_min_degree: int = 3,
    n_jobs: int = 1,
) -> list[tuple[int, ...]]:
    """Bounded enumeration of simple cycles of size in
    ``[min_size, max_size]``, deduplicated exactly (fixes D4). Seeds are
    restricted to nodes with degree >= ``seed_min_degree``, matching the
    original's ``node(:,2)>2`` ring-search-seed filter (spec Sec 3.16).

    ``n_jobs`` parallelizes the (embarrassingly parallel) per-seed search
    across processes via joblib -- the natural place to use extra cores on
    an HPC node for this stage.
    """
    seeds = [n for n in G.nodes() if G.degree(n) >= seed_min_degree]
    if not seeds:
        return []

    if n_jobs == 1 or len(seeds) < 64:
        all_cycles = _find_cycles_for_seeds(G, seeds, min_size, max_size)
    else:
        from joblib import Parallel, delayed

        workers = n_jobs if n_jobs > 0 else (os.cpu_count() or 1)
        chunks = [c.tolist() for c in np.array_split(np.array(seeds), workers) if len(c) > 0]
        results = Parallel(n_jobs=n_jobs)(
            delayed(_find_cycles_for_seeds)(G, chunk, min_size, max_size) for chunk in chunks
        )
        all_cycles = [c for r in results for c in r]

    seen: dict[tuple[int, ...], tuple[int, ...]] = {}
    for cyc in all_cycles:
        key = tuple(sorted(cyc))
        if key not in seen:
            seen[key] = cyc
    return list(seen.values())


def _polygon_area_newell(points_um: np.ndarray) -> float:
    if points_um.shape[0] < 3:
        return 0.0
    nxt = np.roll(points_um, -1, axis=0)
    cross_sum = np.sum(np.cross(points_um, nxt), axis=0)
    return 0.5 * float(np.linalg.norm(cross_sum))


def _ring_boundary_paths(
    G: nx.MultiGraph, cycle: tuple[int, ...]
) -> tuple[list[np.ndarray], list[int]] | None:
    n = len(cycle)
    paths, branch_ids = [], []
    for i in range(n):
        a, b = cycle[i], cycle[(i + 1) % n]
        candidates = [
            (k, data) for u, v, k, data in G.edges(a, keys=True, data=True) if v == b or u == b
        ]
        if not candidates:
            return None
        k, data = min(candidates, key=lambda kd: kd[1].get("tortuosity", float("inf")))
        paths.append(oriented_path(data, a))
        branch_ids.append(data.get("id", k))
    return paths, branch_ids


@dataclass
class Ring:
    id: int
    size: int
    nodes: tuple[int, ...]
    branch_ids: tuple[int, ...]
    centroid_um: np.ndarray
    normal: np.ndarray
    area_um2: float
    d_ring_um: float
    flatness_r2: float
    max_plane_residual_um: float


def ring_geometry(
    G: nx.MultiGraph,
    cycle: tuple[int, ...],
    voxel_size_um: float,
    *,
    flatness_r2_min: float = 0.5,
    plane_residual_max_um: float | None = None,
    ring_id: int = -1,
) -> Ring | None:
    """Fit a total-least-squares plane through the ring's full branch
    boundary (all voxels, matching the original's plane-fit scope), then
    compute area from the ring's NODE VERTICES projected onto that plane,
    via Newell's formula -- matching ``main_ring.m``, which fits the plane
    to the dense voxel boundary (for robustness) but computes
    ``ringSize_NE`` from only the N node positions (``lp4``/etc, not
    ``bran{}``). Newell's formula replaces the original's fan-triangulation
    from an arbitrary first vertex plus an unstable Heron's-formula
    variant, and is a single implementation for any polygon size, fixing
    D24 (four near-identical ``ringSize_{4,5,6,7}E.m`` files).

    Returns ``None`` if the ring fails the flatness filter (spec Sec 3.16
    "must add it") or the optional absolute-residual filter (mirrors
    ``main_ring.m``'s ``max(errors) > 5`` voxel guard)."""
    result = _ring_boundary_paths(G, cycle)
    if result is None:
        return None
    paths, branch_ids = result
    boundary_voxels = np.vstack(paths) if any(p.shape[0] for p in paths) else np.zeros((0, 3))
    node_positions = np.array([G.nodes[n]["pos"] for n in cycle], dtype=np.float64)

    plane_fit_points = boundary_voxels.astype(np.float64) if boundary_voxels.shape[0] >= 3 else node_positions
    plane_fit_um = plane_fit_points * voxel_size_um
    centroid = plane_fit_um.mean(axis=0)
    centered = plane_fit_um - centroid
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]

    total_var = float(np.sum(s**2))
    # NOTE: s[-1] (smallest singular value) can never exceed 1/3 of the
    # total by construction (it is the smallest of exactly three
    # non-negative values), so a naive `1 - s3^2/total` is bounded below
    # by 2/3 for ANY point set -- a `flatness_r2_min` threshold below
    # that could never reject anything. Rescaling against the isotropic
    # (fully 3D-scattered) baseline of s3^2/total = 1/3 gives a measure
    # that spans the full [0, 1] range: 0 for isotropic/non-planar data,
    # 1 for a perfect plane.
    flatness_r2 = max(0.0, min(1.0, 1.0 - 3.0 * (s[-1] ** 2) / total_var)) if total_var > 0 else 1.0
    residuals = centered @ normal
    max_residual = float(np.max(np.abs(residuals)))

    if flatness_r2 < flatness_r2_min:
        return None
    if plane_residual_max_um is not None and max_residual > plane_residual_max_um:
        return None

    node_um = node_positions * voxel_size_um
    node_residuals = (node_um - centroid) @ normal
    projected_nodes = node_um - np.outer(node_residuals, normal)
    area = _polygon_area_newell(projected_nodes)
    d_ring = 2.0 * float(np.sqrt(area / np.pi)) if area > 0 else 0.0

    return Ring(
        id=ring_id,
        size=len(cycle),
        nodes=cycle,
        branch_ids=tuple(branch_ids),
        centroid_um=centroid,
        normal=normal,
        area_um2=area,
        d_ring_um=d_ring,
        flatness_r2=flatness_r2,
        max_plane_residual_um=max_residual,
    )


def analyze_rings(
    G: nx.MultiGraph,
    voxel_size_um: float,
    *,
    min_size: int = 4,
    max_size: int = 7,
    seed_min_degree: int = 3,
    flatness_r2_min: float = 0.5,
    plane_residual_max_um: float | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Full ring pipeline: find candidate cycles, fit + filter their
    geometry, return one row per surviving ring."""
    cycles = find_rings(G, min_size, max_size, seed_min_degree, n_jobs)
    rows = []
    for i, cyc in enumerate(cycles):
        ring = ring_geometry(
            G,
            cyc,
            voxel_size_um,
            flatness_r2_min=flatness_r2_min,
            plane_residual_max_um=plane_residual_max_um,
            ring_id=i,
        )
        if ring is None:
            continue
        rows.append(
            {
                "id": ring.id,
                "size": ring.size,
                "nodes": ring.nodes,
                "branch_ids": ring.branch_ids,
                "centroid_x_um": ring.centroid_um[0],
                "centroid_y_um": ring.centroid_um[1],
                "centroid_z_um": ring.centroid_um[2],
                "normal_x": ring.normal[0],
                "normal_y": ring.normal[1],
                "normal_z": ring.normal[2],
                "area_um2": ring.area_um2,
                "d_ring_um": ring.d_ring_um,
                "flatness_r2": ring.flatness_r2,
                "max_plane_residual_um": ring.max_plane_residual_um,
            }
        )
    return pd.DataFrame(rows)


def _reorder_as_cycle(G: nx.MultiGraph, nodes: list[int]) -> tuple[int, ...] | None:
    """Reconstruct a definitive cyclic traversal order for a set of nodes
    known to form a simple cycle in ``G`` (used because
    ``networkx.minimum_cycle_basis``'s per-cycle node order is an
    implementation detail we should not rely on)."""
    node_set = set(nodes)
    start = nodes[0]
    order = [start]
    visited = {start}
    current = start
    for _ in range(len(nodes) - 1):
        candidates = [n for n in G.neighbors(current) if n in node_set and n not in visited]
        if not candidates:
            return None
        current = candidates[0]
        order.append(current)
        visited.add(current)
    if start not in G.neighbors(current):
        return None
    return tuple(order)


def minimum_cycle_basis_rings(
    G: nx.MultiGraph, max_size: int | None = None
) -> list[tuple[int, ...]]:
    """The Paper II "redundancy filter" (discard higher-order rings
    composable from lower-order ones) obtained for free via
    ``networkx.minimum_cycle_basis`` on the simplified (parallel-edge-
    collapsed) topology -- spec Sec 3.16 / Sec 8 row 14."""
    simple = nx.Graph()
    simple.add_nodes_from(G.nodes())
    simple.add_edges_from((u, v) for u, v, _ in G.edges(keys=True))
    basis = nx.minimum_cycle_basis(simple)
    cycles = []
    for nodes in basis:
        if max_size is not None and len(nodes) > max_size:
            continue
        ordered = _reorder_as_cycle(simple, list(nodes))
        if ordered is not None:
            cycles.append(ordered)
    return cycles
