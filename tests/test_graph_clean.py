import networkx as nx
import numpy as np

from stereom.graph_clean import dissolve_degree2, drop_parallel_edges, merge_clusters


def _line_graph(positions: list[tuple[int, int, int]]) -> nx.MultiGraph:
    G = nx.MultiGraph()
    for i, p in enumerate(positions):
        G.add_node(i, pos=p, voxel_degree=-1)
    for i in range(len(positions) - 1):
        G.add_edge(
            i, i + 1, key=i, id=i, path=np.zeros((0, 3), dtype=np.int64), node_a=i, node_b=i + 1
        )
    return G


def test_merge_clusters_uses_centroid_not_max_index_fixes_D9():
    G = nx.MultiGraph()
    G.add_node(5, pos=(0, 0, 0), voxel_degree=-1)
    G.add_node(2, pos=(2, 0, 0), voxel_degree=-1)
    G.add_node(9, pos=(1, 0, 0), voxel_degree=-1)  # highest id, NOT the centroid
    G.add_node(20, pos=(100, 0, 0), voxel_degree=-1)
    G.add_edge(20, 5, key=0, id=0, path=np.zeros((0, 3), dtype=np.int64), node_a=20, node_b=5)
    G.add_edge(20, 2, key=1, id=1, path=np.zeros((0, 3), dtype=np.int64), node_a=20, node_b=2)
    G.add_edge(20, 9, key=2, id=2, path=np.zeros((0, 3), dtype=np.int64), node_a=20, node_b=9)

    merged = merge_clusters(G, d_m_voxels=3.0)
    survivors = [n for n in merged.nodes() if n != 20]
    assert len(survivors) == 1
    pos = merged.nodes[survivors[0]]["pos"]
    assert pos == (1, 0, 0)  # centroid of (0,0,0),(2,0,0),(1,0,0), NOT max-id node 9's position


def test_merge_clusters_finds_axis_separated_neighbours_fixes_D10():
    # These two nodes are far apart in id-insertion order but close in
    # space -- a fixed lookahead window (as in the original) could miss
    # them; a KD-tree cannot.
    G = nx.MultiGraph()
    G.add_node(0, pos=(0, 0, 0), voxel_degree=-1)
    G.add_node(1, pos=(1, 0, 0), voxel_degree=-1)
    G.add_edge(0, 1000, key=0, id=0, path=np.zeros((0, 3), dtype=np.int64), node_a=0, node_b=1000)
    G.add_edge(1, 1000, key=1, id=1, path=np.zeros((0, 3), dtype=np.int64), node_a=1, node_b=1000)
    G.add_node(1000, pos=(0, 500, 0), voxel_degree=-1)  # keeps 0 and 1 non-isolated
    eid = 2
    for i in range(2, 50):
        # Widely spaced (100 apart) so they never cluster with each other
        # -- only nodes 0 and 1 (1 voxel apart) should ever merge. Each is
        # given a trivial self-contained edge so isolated-node cleanup
        # (a real, separate behaviour) doesn't remove it regardless of
        # whether the merge logic under test works.
        anchor = 2000 + i
        G.add_node(i, pos=(1000 + i * 100, 0, 0), voxel_degree=-1)
        G.add_node(anchor, pos=(1000 + i * 100, 500, 0), voxel_degree=-1)
        G.add_edge(i, anchor, key=eid, id=eid, path=np.zeros((0, 3), dtype=np.int64), node_a=i, node_b=anchor)
        eid += 1

    n_before = G.number_of_nodes()
    merged = merge_clusters(G, d_m_voxels=2.0)
    assert merged.number_of_nodes() == n_before - 1


def test_merge_clusters_deletes_intracluster_branches_not_selfloops():
    G = nx.MultiGraph()
    G.add_node(0, pos=(0, 0, 0), voxel_degree=-1)
    G.add_node(1, pos=(1, 0, 0), voxel_degree=-1)
    G.add_node(2, pos=(100, 0, 0), voxel_degree=-1)
    G.add_edge(0, 1, key=0, id=0, path=np.zeros((0, 3), dtype=np.int64), node_a=0, node_b=1)
    G.add_edge(0, 2, key=1, id=1, path=np.zeros((0, 3), dtype=np.int64), node_a=0, node_b=2)

    merged = merge_clusters(G, d_m_voxels=3.0)
    assert not any(u == v for u, v, k in merged.edges(keys=True))
    assert merged.number_of_edges() == 1  # the 0-1 intra-cluster edge is gone, 0/1-2 survives once


def test_dissolve_degree2_splices_and_sums_lengths():
    from stereom import geometry

    G = _line_graph([(0, 0, 0), (10, 0, 0), (25, 0, 0)])
    G = geometry.order_paths(G)
    G = geometry.compute_lengths(G, voxel_size_um=1.0)
    dissolved = dissolve_degree2(G, guard_voxels=None)
    assert dissolved.number_of_nodes() == 2
    assert dissolved.number_of_edges() == 1
    (u, v, k, data), = dissolved.edges(keys=True, data=True)
    dissolved2 = geometry.compute_lengths(dissolved, voxel_size_um=1.0)
    (_, _, _, data2), = dissolved2.edges(keys=True, data=True)
    assert np.isclose(data2["distance_um"], 25.0)


def test_dissolve_degree2_guard_prevents_long_merge():
    G = nx.MultiGraph()
    G.add_node(0, pos=(0, 0, 0), voxel_degree=-1)
    G.add_node(1, pos=(100, 0, 0), voxel_degree=-1)
    G.add_node(2, pos=(200, 0, 0), voxel_degree=-1)
    long_path = np.array([[i, 0, 0] for i in range(1, 100)], dtype=np.int64)
    G.add_edge(0, 1, key=0, id=0, path=long_path, node_a=0, node_b=1)
    G.add_edge(1, 2, key=1, id=1, path=long_path + 100, node_a=1, node_b=2)
    dissolved = dissolve_degree2(G, guard_voxels=5.0)
    assert dissolved.number_of_nodes() == 3  # both branches too long -> node 1 kept


def test_dissolve_degree2_apex_of_two_branch_loop_vanishes():
    G = nx.MultiGraph()
    G.add_node(0, pos=(0, 0, 0), voxel_degree=-1)
    G.add_node(1, pos=(10, 0, 0), voxel_degree=-1)
    G.add_edge(0, 1, key=0, id=0, path=np.zeros((0, 3), dtype=np.int64), node_a=0, node_b=1)
    G.add_edge(0, 1, key=1, id=1, path=np.zeros((0, 3), dtype=np.int64), node_a=0, node_b=1)
    # give node 0 a third neighbour with degree 2, both of whose edges go to node 1
    G.add_node(2, pos=(5, 5, 0), voxel_degree=-1)
    G.remove_edge(0, 1, 1)
    G.add_edge(0, 2, key=2, id=2, path=np.zeros((0, 3), dtype=np.int64), node_a=0, node_b=2)
    G.add_edge(2, 1, key=3, id=3, path=np.zeros((0, 3), dtype=np.int64), node_a=2, node_b=1)
    dissolved = dissolve_degree2(G, guard_voxels=None)
    assert 2 not in dissolved.nodes()
    assert not any(u == v for u, v, k in dissolved.edges(keys=True))


def test_drop_parallel_edges_keeps_min_tortuosity():
    G = nx.MultiGraph()
    G.add_node(0, pos=(0, 0, 0), voxel_degree=-1)
    G.add_node(1, pos=(10, 0, 0), voxel_degree=-1)
    G.add_edge(0, 1, key=0, id=0, path=np.zeros((0, 3), dtype=np.int64), node_a=0, node_b=1, tortuosity=1.5)
    G.add_edge(0, 1, key=1, id=1, path=np.zeros((0, 3), dtype=np.int64), node_a=0, node_b=1, tortuosity=1.05)
    G.add_edge(0, 1, key=2, id=2, path=np.zeros((0, 3), dtype=np.int64), node_a=0, node_b=1, tortuosity=1.2)

    deduped = drop_parallel_edges(G)
    remaining = list(deduped.edges(keys=True, data=True))
    assert len(remaining) == 1
    assert remaining[0][3]["id"] == 1
