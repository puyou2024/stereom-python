"""Boundary exclusion. Replaces ``isboundary.m`` (spec Sec 3.7).

Applied at analysis time, on the summary tables -- NOT during graph
registration/cleaning, so the saved graph always contains every node and
branch. Spec Sec 3.7 explicitly calls this "the right design"; it is kept
that way here.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def node_is_boundary(pos: tuple[int, int, int], shape: tuple[int, int, int], limit_voxels: float) -> bool:
    return any(pos[d] < limit_voxels or (shape[d] - 1 - pos[d]) < limit_voxels for d in range(3))


def branch_is_boundary(
    path: np.ndarray,
    pos_a: tuple[int, int, int],
    pos_b: tuple[int, int, int],
    shape: tuple[int, int, int],
    limit_voxels: float,
) -> bool:
    """True if ANY voxel of the branch (path interior or either endpoint
    node) lies within ``limit_voxels`` of a volume face -- matches the
    original, which tests the whole branch's voxel set at once."""
    pts = [pos_a, pos_b]
    if path.shape[0] > 0:
        pts.extend(map(tuple, path.tolist()))
    return any(node_is_boundary(p, shape, limit_voxels) for p in pts)


def interior_node_ids(G: nx.MultiGraph, shape: tuple[int, int, int], limit_voxels: float) -> set[int]:
    return {n for n in G.nodes() if not node_is_boundary(G.nodes[n]["pos"], shape, limit_voxels)}


def filter_interior_tables(
    nodes_df: pd.DataFrame,
    branches_df: pd.DataFrame,
    G: nx.MultiGraph,
    shape: tuple[int, int, int],
    limit_voxels: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return copies of ``nodes_df``/``branches_df`` restricted to nodes
    and branches with no voxel within ``limit_voxels`` of a face."""
    interior_nodes = interior_node_ids(G, shape, limit_voxels)
    nodes_out = nodes_df[nodes_df["id"].isin(interior_nodes)].reset_index(drop=True)

    path_by_id = {data.get("id", k): data["path"] for u, v, k, data in G.edges(keys=True, data=True)}
    keep_branch = []
    for _, row in branches_df.iterrows():
        u, v, bid = int(row["node_a"]), int(row["node_b"]), int(row["id"])
        path = path_by_id.get(bid, np.zeros((0, 3), dtype=np.int64))
        keep_branch.append(
            not branch_is_boundary(path, G.nodes[u]["pos"], G.nodes[v]["pos"], shape, limit_voxels)
        )
    branches_out = branches_df[np.array(keep_branch)].reset_index(drop=True)
    return nodes_out, branches_out
