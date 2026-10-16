"""Branch path ordering and length/distance/tortuosity. Replaces
``branch_order.m`` and ``branch_length.m`` (spec Sec 3.8-3.9).

``order_path`` must run before ``compute_lengths`` -- the raw path stored
by ``graph_build.build_graph`` (and re-spliced by
``graph_clean.dissolve_degree2``) is an unordered voxel set; a
consecutive-distance sum over an unordered set is meaningless (spec
Sec 3.9 "hard dependency on Sec 3.8").

One simplification relative to ``branch_order.m``, documented here: the
original picks a starting voxel as "whichever immediate neighbour of
``connect(i,1)`` is in the branch", and if *more than one* neighbour
qualifies (a rare degenerate case at a fat junction boundary), it seeds the
walk with all of them simultaneously. We pick a single such neighbour
(first found). This cannot change the final *set* of voxels visited --
the "jump to nearest unvisited voxel" fallback still guarantees every
voxel is eventually included -- it only affects which of several
topologically-equivalent starting voxels is used, which does not affect
length, since length is a sum over the whole path regardless of starting
point.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def _order_single_path(
    voxels: np.ndarray, node_a_pos: np.ndarray, gap_repair_voxels: float
) -> np.ndarray:
    voxels = np.unique(voxels, axis=0)
    n = voxels.shape[0]
    if n <= 1:
        return voxels

    cheb_to_a = np.max(np.abs(voxels - node_a_pos), axis=1)
    adjacent = np.where(cheb_to_a <= 1)[0]
    if adjacent.size > 0:
        start = int(adjacent[0])
    else:
        start = int(np.argmin(np.linalg.norm(voxels - node_a_pos, axis=1)))

    visited = np.zeros(n, dtype=bool)
    order = np.empty(n, dtype=np.int64)
    order[0] = start
    visited[start] = True
    for step in range(1, n):
        last = voxels[order[step - 1]]
        cheb = np.max(np.abs(voxels - last), axis=1)
        candidates = np.where((cheb <= 1) & ~visited)[0]
        if candidates.size > 0:
            nxt = int(candidates[0])
        else:
            unvisited = np.where(~visited)[0]
            d = np.linalg.norm(voxels[unvisited] - last, axis=1)
            nxt = int(unvisited[int(np.argmin(d))])
        order[step] = nxt
        visited[nxt] = True

    path = voxels[order]
    if n > 2:
        if np.linalg.norm(path[-1].astype(np.float64) - path[-2].astype(np.float64)) > gap_repair_voxels:
            if np.linalg.norm(path[-1].astype(np.float64) - path[0].astype(np.float64)) <= gap_repair_voxels:
                path = np.vstack([path[-1:], path[:-1]])
            else:
                path = path[:-1]
    return path


def oriented_path(data: dict, start_node: int) -> np.ndarray:
    """Return ``data['path']`` reoriented (reversed if necessary) so it
    runs from ``start_node`` towards the other endpoint, using the
    ``node_a``/``node_b`` attributes set at build time. Needed anywhere a
    branch's path must be walked in a specific direction (splicing in
    ``graph_clean.dissolve_degree2``, ring-boundary reconstruction in
    ``rings.py``) -- MultiGraph edge iteration order does not reliably
    preserve which endpoint was originally "a"."""
    path = data["path"]
    if data.get("node_a") == start_node:
        return path
    if data.get("node_b") == start_node:
        return path[::-1]
    return path


def order_paths(G: nx.MultiGraph, gap_repair_voxels: float = 10.0) -> nx.MultiGraph:
    """Order every branch's interior voxels into a walk from its ``node_a``
    endpoint to its ``node_b`` endpoint. Equivalent to ``branch_order.m``."""
    G = G.copy()
    for u, v, k, data in G.edges(keys=True, data=True):
        path = data["path"]
        if path.shape[0] == 0:
            continue
        node_a = data.get("node_a", u)
        pos_a = np.asarray(G.nodes[node_a]["pos"], dtype=np.float64)
        data["path"] = _order_single_path(path, pos_a, gap_repair_voxels)
    return G


def compute_lengths(G: nx.MultiGraph, voxel_size_um: float) -> nx.MultiGraph:
    """Compute curved length, node-to-node (Euclidean) distance and
    tortuosity for every branch. Equivalent to ``branch_length.m``, but
    stores results directly in physical units (spec Sec 10.2 requirement
    3: convert to um once, at the boundary).

    Requires ``order_paths`` to have been run first.
    """
    G = G.copy()
    for u, v, k, data in G.edges(keys=True, data=True):
        path = data["path"]
        pos_a = np.asarray(G.nodes[u]["pos"], dtype=np.float64)
        pos_b = np.asarray(G.nodes[v]["pos"], dtype=np.float64)

        if path.shape[0] == 0:
            curved = 0.0
        else:
            path_f = path.astype(np.float64)
            seg_sum = (
                float(np.linalg.norm(np.diff(path_f, axis=0), axis=1).sum())
                if path_f.shape[0] > 1
                else 0.0
            )
            stub_a = min(
                float(np.linalg.norm(path_f[0] - pos_a)), float(np.linalg.norm(path_f[-1] - pos_a))
            )
            stub_b = min(
                float(np.linalg.norm(path_f[0] - pos_b)), float(np.linalg.norm(path_f[-1] - pos_b))
            )
            curved = seg_sum + stub_a + stub_b

        distance = float(np.linalg.norm(pos_a - pos_b))
        tortuosity = curved / distance if distance > 0 else float("nan")

        data["length_um"] = curved * voxel_size_um
        data["distance_um"] = distance * voxel_size_um
        data["tortuosity"] = tortuosity

    return G
