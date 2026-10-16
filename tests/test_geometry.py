import numpy as np

from stereom import geometry
from tests.phantoms import single_branch_graph


def test_straight_branch_length_and_tortuosity():
    # A straight path from (0,0,0) to (0,0,100), nodes exactly at the ends.
    path = np.array([[0, 0, i] for i in range(1, 100)], dtype=np.int64)
    G = single_branch_graph((0, 0, 0), (0, 0, 100), path)
    G = geometry.order_paths(G)
    G = geometry.compute_lengths(G, voxel_size_um=0.5)

    data = G.edges[0, 1, 0]
    assert np.isclose(data["distance_um"], 50.0)
    assert np.isclose(data["length_um"], 50.0, atol=1e-6)
    assert np.isclose(data["tortuosity"], 1.0, atol=1e-6)


def test_bent_branch_tortuosity_matches_arc_over_chord():
    # Quarter-circle arc of radius R, densely sampled -> curved length ~ arc,
    # chord = R*sqrt(2).
    R = 50.0
    n = 500
    thetas = np.linspace(0, np.pi / 2, n)
    path = np.stack([R * np.sin(thetas), R * (1 - np.cos(thetas)), np.zeros(n)], axis=1)
    pos_a = tuple(path[0].tolist())
    pos_b = tuple(path[-1].tolist())
    G = single_branch_graph(pos_a, pos_b, path[1:-1])
    G = geometry.compute_lengths(G, voxel_size_um=1.0)  # path already dense/ordered; skip order_paths

    data = G.edges[0, 1, 0]
    expected_arc = R * (np.pi / 2)
    expected_chord = R * np.sqrt(2)
    assert np.isclose(data["length_um"], expected_arc, rtol=1e-3)
    assert np.isclose(data["distance_um"], expected_chord, rtol=1e-6)
    assert np.isclose(data["tortuosity"], expected_arc / expected_chord, rtol=1e-3)
    assert data["tortuosity"] >= 1.0


def test_order_paths_repairs_disordered_input():
    # Shuffle a simple straight-line voxel path; order_paths must recover
    # a contiguous walk from node_a to node_b.
    rng = np.random.default_rng(0)
    ordered = np.array([[0, 0, i] for i in range(1, 30)], dtype=np.int64)
    shuffled = ordered.copy()
    rng.shuffle(shuffled)
    G = single_branch_graph((0, 0, 0), (0, 0, 30), shuffled)
    G = geometry.order_paths(G)
    path = G.edges[0, 1, 0]["path"]
    steps = np.linalg.norm(np.diff(path.astype(float), axis=0), axis=1)
    assert np.all(steps <= np.sqrt(3) + 1e-9)
    assert set(map(tuple, path.tolist())) == set(map(tuple, ordered.tolist()))
