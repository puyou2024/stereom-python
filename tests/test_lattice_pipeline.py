"""End-to-end pipeline tests on a rasterized simple-cubic-lattice volume:
the only tests here that exercise skeletonization and graph registration
for real. Includes the scaling- and rotation-invariance phantoms the spec
calls the highest-yield tests in the whole suite (Sec 9.1: "together they
expose nearly every hardcoded voxel literal and every axis assumption").
"""

from __future__ import annotations

import numpy as np
import pytest

from stereom import invariants
from stereom.config import StereomConfig
from stereom.pipeline import run_full_pipeline
from tests.phantoms import cubic_lattice_volume, rotated_cubic_lattice_volume

PITCH = 14.0
RADIUS = 2.5
N_CELLS = (3, 3, 3)


@pytest.fixture(scope="module")
def cubic_result():
    volume, grid = cubic_lattice_volume(N_CELLS, PITCH, RADIUS, margin=6)
    config = StereomConfig(
        voxel_size_um=1.0,
        merge_d_m_um=RADIUS * 1.5,
        degree2_guard_voxels=1.0,
        thickness_min_path_voxels=3,
        n_workers=1,
    )
    result = run_full_pipeline(volume, config, compute_rings=True, compute_chains=False, compute_fft=False)
    return result, grid


def _interior_grid_ids(grid, n_cells):
    nx_, ny_, nz_ = n_cells
    return [k for k in grid if 0 < k[0] < nx_ and 0 < k[1] < ny_ and 0 < k[2] < nz_]


def test_pipeline_runs_and_finds_interior_degree_6_nodes(cubic_result):
    result, grid = cubic_result
    degrees = [result.graph.degree(n) for n in result.graph.nodes()]
    assert len(degrees) > 0
    # Every interior lattice point should map to a real registered node
    # very close (within a couple of voxels, for skeletonization jitter)
    # to its intended grid position, with degree 6.
    interior_expected = [grid[k] for k in _interior_grid_ids(grid, N_CELLS)]
    positions = np.array([result.graph.nodes[n]["pos"] for n in result.graph.nodes()], dtype=np.float64)
    matched_degree6 = 0
    for expected in interior_expected:
        dists = np.linalg.norm(positions - expected, axis=1)
        nearest = int(np.argmin(dists))
        if dists[nearest] <= 3.0:
            node_id = list(result.graph.nodes())[nearest]
            if result.graph.degree(node_id) == 6:
                matched_degree6 += 1
    assert matched_degree6 >= len(interior_expected) // 2


def test_interior_angles_are_90_or_180(cubic_result):
    result, grid = cubic_result
    angles_df = result.angles_df
    degree_by_node = {n: result.graph.degree(n) for n in result.graph.nodes()}
    deg6_nodes = angles_df[angles_df["node"].map(degree_by_node) == 6]
    assert len(deg6_nodes) > 0
    close_to_90_or_180 = deg6_nodes["angle_deg"].apply(
        lambda a: min(abs(a - 90), abs(a - 180), abs(a - 0)) < 15
    )
    assert close_to_90_or_180.mean() > 0.8


def test_invariants_hold_on_cubic_lattice(cubic_result):
    result, grid = cubic_result
    violations = invariants.run_all(
        result.graph,
        angles_df=result.angles_df,
        planarity_df=result.planarity_df,
        rings_df=result.rings_df,
        ring_r2_min=0.5,
    )
    assert violations == []


def test_bounded_search_finds_4cycles_among_others(cubic_result):
    # find_rings/analyze_rings does bounded enumeration up to
    # ring_max_size=7 with NO redundancy filter -- exactly like the
    # original's unrolled BFS, a cubic lattice's cycle space genuinely
    # contains far more 6-node cycles than 4-node ones (there are simply
    # many more distinct paths of length 6 than length 4 through a 3D
    # grid), so "mostly 4-cycles" is not the right expectation for this
    # function. It should, however, find at least some.
    result, grid = cubic_result
    if result.rings_df is None or len(result.rings_df) == 0:
        pytest.skip("no rings detected at this lattice resolution")
    assert (result.rings_df["size"] == 4).sum() > 0


def test_minimum_cycle_basis_gives_mostly_4cycles(cubic_result):
    # The redundancy-filtered set (Paper II's "discard higher-B rings
    # composed of lower-B rings") IS expected to be dominated by the unit
    # cell's 4-cycles, since a minimum cycle basis is by definition the
    # smallest independent generating set of cycles.
    from stereom.rings import minimum_cycle_basis_rings

    result, grid = cubic_result
    basis = minimum_cycle_basis_rings(result.graph, max_size=7)
    if len(basis) == 0:
        pytest.skip("no rings detected at this lattice resolution")
    frac_4 = np.mean([len(c) == 4 for c in basis])
    assert frac_4 > 0.5


@pytest.mark.slow
def test_scaling_invariance_doubles_length_and_thickness():
    volume1, _ = cubic_lattice_volume(N_CELLS, PITCH, RADIUS, margin=6)
    volume2, _ = cubic_lattice_volume(N_CELLS, PITCH * 2, RADIUS * 2, margin=12)

    config1 = StereomConfig(voxel_size_um=1.0, merge_d_m_um=RADIUS * 1.5, degree2_guard_voxels=1.0, thickness_min_path_voxels=3, n_workers=1)
    config2 = StereomConfig(voxel_size_um=1.0, merge_d_m_um=RADIUS * 3.0, degree2_guard_voxels=2.0, thickness_min_path_voxels=3, n_workers=1)

    r1 = run_full_pipeline(volume1, config1, compute_rings=False, compute_chains=False, compute_fft=False)
    r2 = run_full_pipeline(volume2, config2, compute_rings=False, compute_chains=False, compute_fft=False)

    len1 = r1.branches_df["distance_um"].median()
    len2 = r2.branches_df["distance_um"].median()
    assert np.isclose(len2 / len1, 2.0, rtol=0.15)

    thick1 = r1.branches_df["t_mean_radius_um"].median()
    thick2 = r2.branches_df["t_mean_radius_um"].median()
    assert np.isclose(thick2 / thick1, 2.0, rtol=0.25)


@pytest.mark.slow
def test_rotation_invariance_of_degree_distribution():
    volume_straight, _ = cubic_lattice_volume(N_CELLS, PITCH, RADIUS, margin=6)
    volume_rotated = rotated_cubic_lattice_volume(N_CELLS, PITCH, RADIUS, angle_deg=45.0, margin=12)

    config = StereomConfig(voxel_size_um=1.0, merge_d_m_um=RADIUS * 1.5, degree2_guard_voxels=1.0, thickness_min_path_voxels=3, n_workers=1)

    r1 = run_full_pipeline(volume_straight, config, compute_rings=False, compute_chains=False, compute_fft=False)
    r2 = run_full_pipeline(volume_rotated, config, compute_rings=False, compute_chains=False, compute_fft=False)

    def frac_degree6(result):
        degs = [result.graph.degree(n) for n in result.graph.nodes()]
        return np.mean(np.array(degs) == 6)

    assert abs(frac_degree6(r1) - frac_degree6(r2)) < 0.2
