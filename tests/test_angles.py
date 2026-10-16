import numpy as np

from stereom.angles import inter_branch_angles
from tests.phantoms import star_graph


def test_n3_node_120_degree_angles():
    # Three neighbours at 120 degrees apart in a plane.
    offsets = [
        (100 * np.cos(np.deg2rad(a)), 100 * np.sin(np.deg2rad(a)), 0)
        for a in (0, 120, 240)
    ]
    G = star_graph((0, 0, 0), offsets)
    df = inter_branch_angles(G)
    assert len(df) == 3  # C(3,2)
    assert np.allclose(sorted(df["angle_deg"]), [120.0, 120.0, 120.0], atol=0.5)


def test_n4_node_tetrahedral_angles_all_six_pairs_fixes_D6():
    # Regular tetrahedron directions from the origin.
    offsets = [
        (1, 1, 1),
        (1, -1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
    ]
    offsets = [tuple(50 * np.array(o)) for o in offsets]
    G = star_graph((0, 0, 0), offsets)
    df = inter_branch_angles(G)
    assert len(df) == 6  # C(4,2) -- catches D6 (original only found 4)
    tetra_angle = np.degrees(np.arccos(-1 / 3))
    assert np.allclose(sorted(df["angle_deg"]), [tetra_angle] * 6, atol=0.5)


def test_angle_count_matches_degree_choose_2():
    import itertools

    n_neighbors = 5
    offsets = []
    for i in range(n_neighbors):
        a = 2 * np.pi * i / n_neighbors
        offsets.append((200 * np.cos(a), 200 * np.sin(a), (i % 2) * 30))
    G = star_graph((0, 0, 0), offsets)
    df = inter_branch_angles(G)
    expected = len(list(itertools.combinations(range(n_neighbors), 2)))
    assert len(df) == expected


def test_angle_clamped_for_near_parallel_vectors_fixes_D11():
    # Two neighbours in almost exactly the same direction -- floating point
    # error could push the cosine argument slightly past 1.0. Must not
    # raise or return NaN/complex.
    G = star_graph((0, 0, 0), [(1000, 0, 0), (1000, 1e-9, 0)])
    df = inter_branch_angles(G)
    assert len(df) == 1
    assert not np.isnan(df["angle_deg"].iloc[0])
    assert 0.0 <= df["angle_deg"].iloc[0] <= 180.0
