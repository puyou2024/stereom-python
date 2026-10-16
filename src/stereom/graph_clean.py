"""Graph cleaning: node merging, degree-2 dissolution, parallel-edge
removal. Replaces ``merge_connectednodes.m``, ``merge_double.m``,
``merge_nearby.m`` and ``DeleteRepeatedBran.m`` (spec Sec 3.6, 3.10).

Design notes / departures from the MATLAB code, all deliberate and all
documented in the spec as the recommended fix:

* ``merge_clusters`` replaces BOTH ``merge_connectednodes`` (touching-scale
  merge) and ``merge_nearby`` (the real ``d_m`` cluster merge) with a single
  correct implementation: KD-tree neighbour queries + union-find connected
  components, always placing the survivor at the cluster centroid. This
  fixes D9 (max-index survivor instead of centroid), D10 (windowed scans
  miss neighbours separated along an axis by more than the window size) and
  is order-independent (fixes the greedy, order-dependent behaviour noted
  in spec Sec 3.6.3). It is called twice in the reference pipeline order
  (touching-scale, then real ``d_m``) purely to mirror that pipeline;
  mechanically both calls are the same function at different radii.

* ``dissolve_degree2`` does not replicate ``merge_double``'s endpoint-
  distance comparison for choosing concatenation order between the two
  spliced path halves. That comparison is immaterial to any final result:
  ``geometry.order_path`` (the mandatory next stage, matching
  ``branch_order.m``) re-walks and re-orients every path from scratch
  regardless of how its voxels were concatenated. Skipping it is a
  simplification with no behavioural effect, not an approximation.

* We do NOT implement a "remove all degree-1 nodes" trim pass. A degree-1
  node in this graph is a genuine network dead end (both its position and
  its neighbour's are valid registered junctions -- see
  ``graph_build``'s docstring for why true one-ended whiskers never make it
  into the graph at all). The reference pipeline keeps such dead-end
  branches in ``bran_final``/``connect_final``; it only excludes degree<=2
  nodes from the *node-type statistics table*, which is a per-analysis
  filter (see ``rings.py``, ``angles.py``), not a graph mutation. Deleting
  them here would silently drop real branches from every downstream
  descriptor.
"""

from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from stereom.geometry import oriented_path as _oriented_path


def _drop_self_loops_and_isolated(G: nx.MultiGraph) -> nx.MultiGraph:
    G.remove_edges_from(list(nx.selfloop_edges(G, keys=True)))
    G.remove_nodes_from([n for n in list(G.nodes()) if G.degree(n) == 0])
    return G


def dissolve_degree2(G: nx.MultiGraph, guard_voxels: float | None = None) -> nx.MultiGraph:
    """Splice away degree-2 nodes, replacing their two incident branches
    with one. Equivalent to ``merge_double.m`` (spec Sec 3.6.2).

    Parameters
    ----------
    guard_voxels:
        If set, a degree-2 node is only dissolved when the SHORTER of its
        two incident branch paths has <= this many voxels (prevents
        splicing two long branches into one very long one). ``None``
        dissolves unconditionally -- matches the reference pipeline's
        second, unguarded call.
    """
    G = G.copy()
    worklist = list(G.nodes())
    while worklist:
        v = worklist.pop()
        if v not in G:
            continue
        if G.degree(v) != 2:
            continue
        incident = list(G.edges(v, keys=True, data=True))
        if len(incident) != 2:
            # a degree-2 node with a single multi-edge to the same
            # neighbour (parallel pair) -- leave for drop_parallel_edges.
            continue
        (_, u1, k1, d1), (_, u2, k2, d2) = incident
        path1, path2 = d1["path"], d2["path"]
        if guard_voxels is not None and min(len(path1), len(path2)) > guard_voxels:
            continue

        pos_v = np.asarray(G.nodes[v]["pos"], dtype=np.int64)
        G.remove_edge(v, u1, k1)
        G.remove_edge(v, u2, k2)
        G.remove_node(v)

        if u1 == u2:
            # v was the apex of a 2-branch loop back to the same node;
            # dissolving it removes that small loop entirely (matches the
            # self-loop cleanup the original applies after every merge
            # stage).
            continue

        # Orient each half so the spliced result reads u1 -> v -> u2
        # (node_a=u1, node_b=u2). The exact concatenation order has no
        # effect on any downstream result -- see module docstring -- but
        # getting node_a/node_b right matters for ring reconstruction,
        # which walks a cycle in a specific direction and must know which
        # end of each branch's path is which.
        path1_oriented = _oriented_path(d1, u1)
        path2_oriented = _oriented_path(d2, v)
        combined_path = np.vstack([path1_oriented, pos_v[None, :], path2_oriented])
        G.add_edge(u1, u2, path=combined_path, node_a=u1, node_b=u2)
        worklist.append(u1)
        worklist.append(u2)

    return _drop_self_loops_and_isolated(G)


def merge_clusters(G: nx.MultiGraph, d_m_voxels: float) -> nx.MultiGraph:
    """Merge every connected component of the "within `d_m_voxels`"
    proximity graph on node positions into a single node at the cluster
    centroid. Replaces ``merge_connectednodes.m`` and ``merge_nearby.m``
    (spec Sec 3.6.1, 3.6.3) with one correct, order-independent
    implementation (fixes D9, D10, D17).
    """
    G = G.copy()
    node_ids = list(G.nodes())
    if len(node_ids) < 2 or d_m_voxels <= 0:
        return G

    positions = np.array([G.nodes[n]["pos"] for n in node_ids], dtype=np.float64)
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=d_m_voxels, output_type="ndarray")
    if pairs.shape[0] == 0:
        return G

    parent = {n: n for n in node_ids}

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j in pairs:
        union(node_ids[i], node_ids[j])

    clusters: dict[int, list[int]] = defaultdict(list)
    for n in node_ids:
        clusters[find(n)].append(n)

    for members in clusters.values():
        if len(members) < 2:
            continue
        member_set = set(members)

        intra_edges = {
            (min(u, v), max(u, v), k)
            for u, v, k in G.edges(members, keys=True)
            if u in member_set and v in member_set
        }
        for u, v, k in intra_edges:
            if G.has_edge(u, v, k):
                G.remove_edge(u, v, k)

        pos_arr = np.array([G.nodes[m]["pos"] for m in members], dtype=np.float64)
        centroid = tuple(int(x) for x in np.round(pos_arr.mean(axis=0)))

        survivor, rest = members[0], members[1:]
        for m in rest:
            if m not in G:
                continue
            for u, v, k, data in list(G.edges(m, keys=True, data=True)):
                other = v if u == m else u
                G.remove_edge(u, v, k)
                if other == survivor or other not in G:
                    continue  # would become a self-loop on the survivor; drop
                data = dict(data)
                if data.get("node_a") == m:
                    data["node_a"] = survivor
                if data.get("node_b") == m:
                    data["node_b"] = survivor
                G.add_edge(survivor, other, **data)
            G.remove_node(m)

        if survivor in G:
            G.nodes[survivor]["pos"] = centroid
            G.nodes[survivor]["voxel_degree"] = -1  # synthesized centroid, no voxel-level count

    return _drop_self_loops_and_isolated(G)


def drop_parallel_edges(G: nx.MultiGraph, tortuosity_attr: str = "tortuosity") -> nx.MultiGraph:
    """For every node pair joined by more than one branch, keep only the
    one with the smallest tortuosity. Replaces ``DeleteRepeatedBran.m``
    (spec Sec 3.10) -- generalized to correctly handle 3+ parallel edges
    between the same pair (the original's pairwise comparison logic can
    misbehave in that rare case; keeping the global minimum is the clearly
    intended behaviour: "keep the straighter one").

    Requires ``tortuosity_attr`` to already be set on every edge (i.e. run
    after a first ``geometry.compute_lengths`` pass; recompute lengths
    again afterwards -- spec Sec 3.10 "note the ordering dependency").
    """
    G = G.copy()
    groups: dict[tuple[int, int], list[tuple[int, int, int, float]]] = defaultdict(list)
    for u, v, k, data in G.edges(keys=True, data=True):
        key = (min(u, v), max(u, v))
        groups[key].append((u, v, k, float(data.get(tortuosity_attr, np.inf))))

    for items in groups.values():
        if len(items) <= 1:
            continue
        items_sorted = sorted(items, key=lambda x: x[3])
        for u, v, k, _ in items_sorted[1:]:
            if G.has_edge(u, v, k):
                G.remove_edge(u, v, k)

    return _drop_self_loops_and_isolated(G)
