import networkx as nx
import numpy as np

from stereom.config import StereomConfig
from stereom.profile_fit import fit_profiles


def _straight_path_graph(n: int, values: np.ndarray, length_um: float) -> nx.MultiGraph:
    """A straight path of n voxels along dim 0 (arclength = 0..n-1 at
    voxel_size_um=1.0, matching what profile_fit will independently
    recompute), carrying a precomputed radius profile."""
    path = np.stack([np.arange(n), np.zeros(n), np.zeros(n)], axis=1).astype(np.int64)
    G = nx.MultiGraph()
    G.add_node(0, pos=(0, 0, 0), voxel_degree=-1)
    G.add_node(1, pos=(n - 1, 0, 0), voxel_degree=-1)
    G.add_edge(
        0, 1, key=0, id=0, path=path, node_a=0, node_b=1, profile_radius_um=values, length_um=length_um
    )
    return G


def test_quartic_fit_recovers_input_parameter():
    n = 61
    waist_idx = n // 2
    arclen = np.arange(n, dtype=float)  # matches profile_fit's own arclength computation exactly
    z_true = arclen - arclen[waist_idx]
    length_um = 30.0
    t_min = 5.0
    a_true = 3.0
    t = t_min * (a_true * (z_true / length_um) ** 2 + 1.0) ** 2

    G = _straight_path_graph(n, t, length_um)
    config = StereomConfig(fit_len_min_um=1.0, fit_len_max_um=100.0, fit_max_abs_z_um=100.0)
    results = fit_profiles(G, voxel_size_um=1.0, config=config)
    assert len(results) == 1
    r = results[0]
    assert r.accepted
    assert r.quartic_a is not None
    assert np.isclose(r.quartic_a, a_true, rtol=0.02)


def test_quadratic_fit_recovers_curvature():
    n = 61
    waist_idx = n // 2
    arclen = np.arange(n, dtype=float)
    z_true = arclen - arclen[waist_idx]
    b_true = 0.02
    t = 5.0 + b_true * z_true**2

    G = _straight_path_graph(n, t, length_um=30.0)
    config = StereomConfig(fit_len_min_um=1.0, fit_len_max_um=100.0, fit_max_abs_z_um=100.0)
    results = fit_profiles(G, voxel_size_um=1.0, config=config)
    r = results[0]
    assert r.accepted
    assert np.isclose(r.b, b_true, rtol=0.02)
    assert r.a1 is None  # no linear term by default


def test_branch_outside_length_window_is_rejected():
    n = 61
    waist_idx = n // 2
    arclen = np.arange(n, dtype=float)
    z_true = arclen - arclen[waist_idx]
    t = 5.0 * (1.0 * (z_true / 1000.0) ** 2 + 1.0) ** 2
    G = _straight_path_graph(n, t, length_um=1000.0)
    config = StereomConfig()  # default fit_len_max_um=35
    results = fit_profiles(G, voxel_size_um=1.0, config=config)
    assert not results[0].accepted
    assert results[0].reject_reason == "out_of_length_window"
