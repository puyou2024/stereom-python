import networkx as nx
import numpy as np

from stereom.planarity import node_planarity


def _make_n3_node(center0, p1, p2, p3):
    G = nx.MultiGraph()
    G.add_node(0, pos=center0, voxel_degree=-1)
    for i, p in enumerate((p1, p2, p3), start=1):
        G.add_node(i, pos=p, voxel_degree=-1)
        G.add_edge(0, i, key=i, id=i, path=np.zeros((0, 3), dtype=np.int64), node_a=0, node_b=i)
    return G


def test_planar_n3_node_gives_zero_planarity_index():
    p1 = (100, 0, 0)
    p2 = (int(100 * np.cos(np.deg2rad(120))), int(100 * np.sin(np.deg2rad(120))), 0)
    p3 = (int(100 * np.cos(np.deg2rad(240))), int(100 * np.sin(np.deg2rad(240))), 0)
    G = _make_n3_node((0, 0, 0), p1, p2, p3)
    normal, out_angle, p = node_planarity(G, 0)
    assert np.isclose(p, 0.0, atol=0.05)
    assert 0.0 <= p <= 1.0


def test_nonplanar_n3_node_gives_positive_planarity_index():
    p1 = (100, 0, 0)
    p2 = (int(100 * np.cos(np.deg2rad(120))), int(100 * np.sin(np.deg2rad(120))), 0)
    p3 = (int(100 * np.cos(np.deg2rad(240))), int(100 * np.sin(np.deg2rad(240))), 0)
    G = _make_n3_node((0, 0, 60), p1, p2, p3)  # centre node lifted out of the neighbour plane
    normal, out_angle, p = node_planarity(G, 0)
    assert p > 0.05
    assert 0.0 <= p <= 1.0


def test_non_degree_3_returns_none():
    G = nx.MultiGraph()
    G.add_node(0, pos=(0, 0, 0), voxel_degree=-1)
    G.add_node(1, pos=(10, 0, 0), voxel_degree=-1)
    G.add_edge(0, 1, key=0, id=0, path=np.zeros((0, 3), dtype=np.int64), node_a=0, node_b=1)
    assert node_planarity(G, 0) is None
