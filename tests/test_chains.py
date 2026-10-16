import networkx as nx
import numpy as np

from stereom.chains import greedy_chains


def _straight_line_graph(n_segments: int, step: float = 20.0) -> nx.MultiGraph:
    G = nx.MultiGraph()
    for i in range(n_segments + 1):
        G.add_node(i, pos=(int(i * step), 0, 0), voxel_degree=-1)
    for i in range(n_segments):
        G.add_edge(i, i + 1, key=i, id=i, path=np.zeros((0, 3), dtype=np.int64), node_a=i, node_b=i + 1)
    return G


def test_collinear_chain_is_a_single_chain():
    G = _straight_line_graph(5)
    chains_df = greedy_chains(G, cos_threshold=0.5)
    assert len(chains_df) == 1
    assert chains_df.iloc[0]["n_branches"] == 5


def test_sharp_turn_breaks_the_chain():
    G = nx.MultiGraph()
    G.add_node(0, pos=(0, 0, 0), voxel_degree=-1)
    G.add_node(1, pos=(20, 0, 0), voxel_degree=-1)
    G.add_node(2, pos=(20, 20, 0), voxel_degree=-1)  # 90 degree turn
    G.add_edge(0, 1, key=0, id=0, path=np.zeros((0, 3), dtype=np.int64), node_a=0, node_b=1)
    G.add_edge(1, 2, key=1, id=1, path=np.zeros((0, 3), dtype=np.int64), node_a=1, node_b=2)
    chains_df = greedy_chains(G, cos_threshold=0.5)
    assert len(chains_df) == 2
    assert set(chains_df["n_branches"]) == {1}


def test_every_branch_appears_in_exactly_one_chain():
    G = _straight_line_graph(3)
    G.add_node(10, pos=(0, 50, 0), voxel_degree=-1)
    G.add_edge(0, 10, key=10, id=10, path=np.zeros((0, 3), dtype=np.int64), node_a=0, node_b=10)
    chains_df = greedy_chains(G, cos_threshold=0.5)
    all_branch_ids = [b for row in chains_df["branch_ids"] for b in row]
    assert sorted(all_branch_ids) == sorted([0, 1, 2, 10])
