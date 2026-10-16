"""Skeleton -> graph. Replaces ``node_identification.m``, ``branch_trim.m``
and ``branch_sort2.m`` (spec Sec 3.3-3.5).

Two deliberate departures from the MATLAB pipeline, both explicitly
sanctioned by the spec's own refactor notes:

1. Node identification is one convolution instead of a scalar loop calling
   ``neighbor.m`` (rebuilding a meshgrid) once per skeleton voxel -- spec
   Sec 3.3 refactor note, ~1000x claimed speedup.

2. We do NOT run the MATLAB driver's iterative
   ``node_identification -> branch_trim -> relabel`` fixed point on the
   *volume*. Instead we register the graph once (this module) and prune
   dangling degree-1 dead ends at the *graph* level afterwards
   (``graph_clean.trim_dangling``) -- spec Sec 3.4 refactor note ("operate
   on the graph, not the volume ... O(V+E) instead of O(passes x volume)").
   A single ``branch_sort2``-equivalent pass already drops true one-ended
   whisker segments on its own (an endpoint touching zero node voxels is
   simply never registered, so the branch fails the "exactly 2 registered
   ends" test) -- the MATLAB pre-trimming loop's only *additional* effect
   is to shrink node-candidate clusters that bordered a whisker before it
   was physically erased from the volume. That is a second-order shape
   correction to node clusters, not a topology change, and is subsumed by
   the explicit KD-tree cluster merge in ``graph_clean.merge_clusters``.
   This is documented here because it is the single largest intentional
   behavioural difference from the reference implementation.

Every function below takes and returns plain arrays / a
``networkx.MultiGraph`` -- no plotting, no global state (spec Sec 10.2
requirements 1-2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from scipy import ndimage

_OFFSETS_26 = np.array(
    [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ],
    dtype=np.int64,
)
_FULL_CONNECTIVITY = np.ones((3, 3, 3), dtype=np.uint8)


def identify_nodes(skel: np.ndarray, degree_threshold: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized replacement for ``node_identification.m`` (spec Sec 3.3).

    A skeleton voxel is a node candidate if its 27-neighbourhood (including
    itself) contains ``>= degree_threshold`` skeleton voxels -- one
    convolution instead of a scalar loop.

    Returns
    -------
    node_mask:
        bool array, same shape as ``skel``. True at candidate node voxels.
    voxel_degree:
        int16 array. At candidate node voxels, ``count - 1`` (the "voxel
        degree" diagnostic of spec Sec 2.2 -- NOT true graph degree).
        Zero elsewhere.
    """
    counts = ndimage.convolve(
        skel.astype(np.uint8), np.ones((3, 3, 3), dtype=np.uint8), mode="constant", cval=0
    )
    node_mask = skel & (counts >= degree_threshold)
    voxel_degree = np.zeros(skel.shape, dtype=np.int16)
    voxel_degree[node_mask] = (counts[node_mask] - 1).astype(np.int16)
    return node_mask, voxel_degree


def _same_label_neighbor_count(labeled: np.ndarray) -> np.ndarray:
    """For every voxel, count how many of its 26 neighbours share its
    (nonzero) label. Zero for background voxels. Fully vectorized -- no
    per-voxel or per-label Python loop, unlike the scalar ``neighbor.m``
    calls the original performs for every voxel of every branch."""
    padded = np.pad(labeled, 1, mode="constant", constant_values=0)
    count = np.zeros(labeled.shape, dtype=np.int8)
    nx_, ny_, nz_ = labeled.shape
    for dx, dy, dz in _OFFSETS_26:
        shifted = padded[1 + dx : 1 + dx + nx_, 1 + dy : 1 + dy + ny_, 1 + dz : 1 + dz + nz_]
        count += (shifted == labeled) & (labeled != 0)
    return count


def _batch_node_neighbors(
    coords: np.ndarray, node_mask: np.ndarray
) -> dict[tuple[int, int, int], list[tuple[int, int, int]]]:
    """For every voxel in ``coords`` (M,3), find which of its 26 neighbours
    are True in ``node_mask``. Vectorized across all M voxels at once
    (this is the operation that dominated cost in the original -- spec
    Sec 3.3/3.5 -- reduced here to array broadcasting plus one small
    Python loop over M, which is O(branch count), not O(volume))."""
    if coords.shape[0] == 0:
        return {}
    shape = np.array(node_mask.shape)
    neighbor_coords = coords[:, None, :] + _OFFSETS_26[None, :, :]  # (M,26,3)
    in_bounds = np.all((neighbor_coords >= 0) & (neighbor_coords < shape), axis=-1)  # (M,26)
    clipped = np.clip(neighbor_coords, 0, shape - 1)
    is_node = np.zeros(in_bounds.shape, dtype=bool)
    is_node[in_bounds] = node_mask[
        clipped[in_bounds][:, 0], clipped[in_bounds][:, 1], clipped[in_bounds][:, 2]
    ]

    result: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    for m in range(coords.shape[0]):
        hits = neighbor_coords[m][is_node[m]]
        result[tuple(int(v) for v in coords[m])] = [tuple(int(v) for v in h) for h in hits]
    return result


@dataclass
class GraphBuildStats:
    n_candidate_nodes: int = 0
    n_segments: int = 0
    n_branches_kept: int = 0
    n_dropped_dangling: int = 0
    n_dropped_ambiguous: int = 0
    n_self_loops_dropped: int = 0


def build_graph(
    skel: np.ndarray, degree_threshold: int = 4
) -> tuple[nx.MultiGraph, GraphBuildStats]:
    """Build the branch/node graph from a skeleton, in one pass.

    Replaces ``branch_sort2.m`` (and, per the module docstring, the
    ``branch_trim`` pre-loop). Returns a ``networkx.MultiGraph`` because
    parallel edges (two branches joining the same node pair) are a real,
    expected structure at this stage -- they are only resolved later, by
    ``graph_clean.drop_parallel_edges``, exactly mirroring the original's
    ordering (spec Sec 3.5, 3.10).

    Node attributes: ``pos`` (tuple[int,int,int] voxel coords),
    ``voxel_degree`` (int, -1 if this position is a synthesized centroid
    rather than an original node-mask voxel).
    Edge attributes: ``path`` (ndarray (n,3) int, UNORDERED interior
    voxels, node voxels excluded -- matches ``bran{i}``).
    """
    node_mask, voxel_degree_field = identify_nodes(skel, degree_threshold)
    return build_graph_from_node_mask(skel, node_mask, voxel_degree_field)


def build_graph_from_node_mask(
    skel: np.ndarray, node_mask: np.ndarray, voxel_degree_field: np.ndarray | None = None
) -> tuple[nx.MultiGraph, GraphBuildStats]:
    """Same as ``build_graph``, but takes an already-computed
    ``node_mask`` instead of deriving one via ``identify_nodes``. Split out
    from ``build_graph`` so graph assembly can be unit-tested against a
    hand-specified node set, independent of ``identify_nodes``'s emergent
    "fat junction cluster" behaviour (multiple mutually-adjacent voxels
    around a real >=3-arm junction all independently satisfying
    ``k >= degree_threshold`` -- a real, documented property of the
    algorithm, not a defect, but one that makes hand-predicting exact
    registered voxel positions from a rasterized volume impractical for
    some targeted tests)."""
    stats = GraphBuildStats()
    if voxel_degree_field is None:
        voxel_degree_field = np.zeros(skel.shape, dtype=np.int16)
    stats.n_candidate_nodes = int(node_mask.sum())

    segments = skel & ~node_mask
    labeled, n_segments = ndimage.label(segments, structure=_FULL_CONNECTIVITY)
    stats.n_segments = int(n_segments)

    if n_segments == 0:
        return nx.MultiGraph(), stats

    nbcount = _same_label_neighbor_count(labeled)

    fg_coords = np.argwhere(labeled > 0)
    fg_labels = labeled[fg_coords[:, 0], fg_coords[:, 1], fg_coords[:, 2]]
    order = np.argsort(fg_labels, kind="stable")
    fg_coords = fg_coords[order]
    fg_labels = fg_labels[order]
    uniq_labels, start_idx = np.unique(fg_labels, return_index=True)
    voxel_groups = dict(zip(uniq_labels.tolist(), np.split(fg_coords, start_idx[1:])))

    fg_nbcount = nbcount[fg_coords[:, 0], fg_coords[:, 1], fg_coords[:, 2]]
    is_candidate_endpoint = (fg_nbcount == 1) | (fg_nbcount == 0)
    endpoint_candidates = fg_coords[is_candidate_endpoint]
    endpoint_to_node_neighbors = _batch_node_neighbors(endpoint_candidates, node_mask)

    def _register(ep: np.ndarray) -> tuple[int, int, int] | None:
        neighbors = endpoint_to_node_neighbors.get(tuple(int(v) for v in ep), [])
        if len(neighbors) == 0:
            return None
        if len(neighbors) == 1:
            return neighbors[0]
        arr = np.asarray(neighbors, dtype=np.float64)
        centroid = tuple(int(v) for v in np.round(arr.mean(axis=0)))
        return centroid

    voxel_degree_lookup: dict[tuple[int, int, int], int] = {}
    node_voxel_coords = np.argwhere(node_mask)
    for c in node_voxel_coords:
        voxel_degree_lookup[tuple(int(v) for v in c)] = int(voxel_degree_field[tuple(c)])

    branches: list[dict] = []
    for label_id, voxels in voxel_groups.items():
        if voxels.shape[0] == 1:
            v = voxels[0]
            neighbors = endpoint_to_node_neighbors.get(tuple(int(x) for x in v), [])
            uniq = sorted(set(neighbors))
            if len(uniq) == 2:
                branches.append({"a": uniq[0], "b": uniq[1], "path": voxels})
            else:
                stats.n_dropped_dangling += 1
            continue

        nb_here = nbcount[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
        endpoint_voxels = voxels[nb_here == 1]
        registered = [r for ep in endpoint_voxels if (r := _register(ep)) is not None]
        if len(registered) == 2:
            branches.append({"a": registered[0], "b": registered[1], "path": voxels})
        elif len(registered) < 2:
            stats.n_dropped_dangling += 1
        else:
            stats.n_dropped_ambiguous += 1

    branches = [b for b in branches if b["a"] != b["b"]]
    stats.n_self_loops_dropped = (stats.n_segments - stats.n_dropped_dangling - stats.n_dropped_ambiguous) - len(branches)
    stats.n_branches_kept = len(branches)

    graph = nx.MultiGraph()
    position_to_id: dict[tuple[int, int, int], int] = {}

    def _node_id(pos: tuple[int, int, int]) -> int:
        if pos not in position_to_id:
            nid = len(position_to_id)
            position_to_id[pos] = nid
            graph.add_node(nid, pos=pos, voxel_degree=voxel_degree_lookup.get(pos, -1))
        return position_to_id[pos]

    for branch_id, b in enumerate(branches):
        ida, idb = _node_id(b["a"]), _node_id(b["b"])
        graph.add_edge(
            ida, idb, key=branch_id, id=branch_id, path=b["path"], node_a=ida, node_b=idb
        )

    return graph, stats
