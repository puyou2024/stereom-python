import numpy as np

from stereom import thickness
from tests.phantoms import single_branch_graph, straight_cylinder_volume


def test_edt_thickness_on_straight_cylinder_recovers_radius():
    radius = 6.0
    length = 80.0
    volume, p0, p1 = straight_cylinder_volume(length, radius, voxel_size=1.0)
    edt = thickness.compute_edt(volume)

    # Sample along the exact centreline, well clear of the capsule end caps
    # (the profile near the ends is contaminated by the hemispherical cap,
    # which is a real geometric effect, not a bug -- so we only assert
    # on interior samples).
    i0 = int(p0[0]) + int(radius) + 5
    i1 = int(p1[0]) - int(radius) - 5
    path = np.array([[i, int(round(p0[1])), int(round(p0[2]))] for i in range(i0, i1)], dtype=np.int64)
    G = single_branch_graph(tuple(p0.astype(int)), tuple(p1.astype(int)), path)
    G = thickness.sample_profiles_edt(G, edt, min_path_voxels=5)

    data = G.edges[0, 1, 0]
    profile = data["profile_radius_um"]
    assert profile is not None
    # EDT gives the INSCRIBED radius; on a voxelized cylinder of radius r
    # this should be within about 1 voxel of the analytic radius.
    assert np.all(np.abs(profile - radius) <= 1.5)
    assert abs(data["t_min_radius_um"] - radius) <= 1.5
    assert abs(data["t_mean_radius_um"] - radius) <= 1.5


def test_short_branch_is_excluded_from_profile():
    radius = 4.0
    volume, p0, p1 = straight_cylinder_volume(40.0, radius, voxel_size=1.0)
    edt = thickness.compute_edt(volume)
    path = np.array([[int(p0[0]) + 1, int(p0[1]), int(p0[2])]], dtype=np.int64)  # 1 voxel
    G = single_branch_graph(tuple(p0.astype(int)), tuple(p1.astype(int)), path)
    G = thickness.sample_profiles_edt(G, edt, min_path_voxels=10)
    data = G.edges[0, 1, 0]
    assert data["profile_radius_um"] is None
    assert np.isnan(data["t_min_radius_um"])
