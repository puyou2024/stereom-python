"""Tests for STL region extraction, in particular the slice-rasterization
voxelization method (mesh_regions.voxelize_region), which replaced an
earlier 3D-flood-fill approach found to under-count solid volume by
roughly 3x on real, non-watertight lattice STL exports (see
mesh_regions.py's module docstring). These tests use synthetic meshes
with exactly known volume/geometry so the voxelization can be checked
against an analytic answer, not just "looks plausible".
"""

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from stereom.mesh_regions import pitch_from_feature_size, voxelize_region  # noqa: E402
import stereom.mesh_regions as mesh_regions  # noqa: E402


def test_solid_cube_gives_expected_density():
    # A 10x10x10 unit box, fully solid; voxelizing a 4x4x4 sub-region
    # entirely inside it should come out ~100% solid.
    mesh = trimesh.creation.box(extents=[10, 10, 10])
    mesh.apply_translation([5, 5, 5])  # bounds become [0,10]^3
    region, diag = voxelize_region(mesh, (3, 3, 3), (7, 7, 7), pitch=0.5)
    assert diag["n_failed_slices"] == 0
    assert region.mean() > 0.98


def test_region_outside_mesh_is_empty():
    mesh = trimesh.creation.box(extents=[10, 10, 10])
    mesh.apply_translation([5, 5, 5])
    region, diag = voxelize_region(mesh, (20, 20, 20), (24, 24, 24), pitch=0.5)
    assert region.mean() == 0.0


def test_sphere_volume_fraction_matches_analytic():
    # A sphere of radius 5 centred at the origin, cropped to a bounding
    # cube of half-extent 5 (i.e. exactly the sphere's bounding box):
    # analytic volume fraction of a sphere inscribed in its bounding cube
    # is (4/3 pi r^3) / (2r)^3 = pi/6 ~= 0.5236.
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=5.0)
    region, diag = voxelize_region(mesh, (-5, -5, -5), (5, 5, 5), pitch=0.1)
    assert diag["n_failed_slices"] == 0
    assert np.isclose(region.mean(), np.pi / 6, atol=0.02)


def test_hollow_shell_has_lower_density_than_solid():
    # A thin hollow shell (difference of two concentric spheres, built by
    # inverting the inner sphere's normals and unioning the two meshes as
    # a single non-watertight-looking surface, mimicking how a real
    # thin-walled lattice STL is represented) should rasterize to a small
    # solid fraction, not something close to a filled sphere -- this is
    # exactly the "thin wall vs. filled interior" distinction the
    # flood-fill approach got wrong on real data.
    outer = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
    inner = trimesh.creation.icosphere(subdivisions=3, radius=4.5)
    inner.invert()
    shell = trimesh.util.concatenate([outer, inner])
    region, diag = voxelize_region(shell, (-5, -5, -5), (5, 5, 5), pitch=0.1)
    solid_fraction = region.mean()
    outer_volume_fraction = np.pi / 6
    assert solid_fraction < outer_volume_fraction * 0.5


def test_pitch_from_feature_size():
    assert pitch_from_feature_size(100.0, voxels_across=8.0) == pytest.approx(12.5)
    assert pitch_from_feature_size(320.0, voxels_across=10.0) == pytest.approx(32.0)


def test_voxelize_region_rejects_absurdly_fine_pitch():
    mesh = trimesh.creation.box(extents=[100, 100, 100])
    with pytest.raises(ValueError):
        voxelize_region(mesh, (0, 0, 0), (100, 100, 100), pitch=0.0001)


# --- Outlier-detection / self-healing regression tests ------------------
#
# These target a real bug found on production data: mesh.section()'s 2D
# polygon assembly can misassemble the intersection at specific EXACT
# cutting heights into one spuriously huge merged polygon, even though
# every individual slice call "succeeds" (no exception, so the existing
# failed-slice retry never triggers). Symptom: an isolated 1-2 slice
# solid-fraction spike (e.g. ~65% vs a ~35% neighborhood baseline) with
# no corresponding change in local mesh triangle density. Confirmed fix:
# nudging the cutting height by a small fraction of one pitch. See
# mesh_regions.py's module docstring for the full incident writeup.


def test_detect_slice_outliers_flags_isolated_spike_not_boundary():
    n = 40
    baseline = 0.35 + 0.02 * np.sin(np.linspace(0, 3, n))
    frac = baseline.copy()
    frac[0:3] = 0.0  # legitimate "outside the mesh" boundary run
    frac[20] = 0.9  # injected artifact spike, matching the real incident's shape
    outlier, _ = mesh_regions._detect_slice_outliers(frac)
    assert outlier[20]
    assert not outlier[:3].any()
    assert outlier.sum() == 1


def test_voxelize_region_self_heals_isolated_bad_slice(monkeypatch):
    # A slice rasterized at exactly z=10.5 is wrong (bogus, ~2.5x too
    # solid); any nudged z near it rasterizes correctly -- exactly the
    # observed real-world failure mode.
    normal = np.zeros((10, 10), dtype=bool)
    normal[:4, :] = True  # 40% solid
    bogus = np.ones((10, 10), dtype=bool)  # 100% solid
    bad_z = 10.5

    def fake_rasterize(mesh, z, origin_xy, nx, ny, pitch):
        return bogus.copy() if abs(z - bad_z) < 1e-9 else normal.copy()

    monkeypatch.setattr(mesh_regions, "_rasterize_z_slice", fake_rasterize)

    dense, diag = voxelize_region(None, (0, 0, 0), (10, 10, 20), pitch=1.0)
    assert diag["n_outlier_slices_detected"] == 1
    assert diag["n_outlier_slices_healed"] == 1
    assert diag["n_outlier_slices_unresolved"] == 0
    assert dense[:, :, 10].mean() == pytest.approx(0.4)


def test_voxelize_region_heals_persistent_anomaly_by_interpolation(monkeypatch):
    # Same setup, but the "bad" slice is bad across its whole neighborhood
    # (every nudge attempted still lands inside it). Revised behavior
    # (see module docstring): every such case observed on real SSCF data
    # turned out to be a numerical artifact, never genuine dense geometry,
    # so this is now healed by interpolating from the nearest non-outlier
    # neighbor slices rather than left untouched.
    normal = np.zeros((10, 10), dtype=bool)
    normal[:4, :] = True
    bogus = np.ones((10, 10), dtype=bool)
    bad_idx = 10

    def fake_rasterize(mesh, z, origin_xy, nx, ny, pitch):
        return bogus.copy() if int(z // 1.0) == bad_idx else normal.copy()

    monkeypatch.setattr(mesh_regions, "_rasterize_z_slice", fake_rasterize)

    dense, diag = voxelize_region(None, (0, 0, 0), (10, 10, 20), pitch=1.0)
    assert diag["n_outlier_slices_detected"] == 1
    assert diag["n_outlier_slices_healed"] == 1
    assert diag["n_outlier_slices_healed_by_nudge"] == 0
    assert diag["n_outlier_slices_healed_by_interpolation"] == 1
    assert diag["n_outlier_slices_unresolved"] == 0
    # neighbors on both sides are "normal" (40% solid); interpolation
    # unions them, which for identical neighbor patterns reproduces "normal"
    assert dense[:, :, bad_idx].mean() == pytest.approx(0.4)


def test_voxelize_region_truly_unresolved_only_without_any_good_neighbor(monkeypatch):
    # If outliers span the ENTIRE volume (no non-outlier slice on either
    # side to interpolate from), healing genuinely cannot proceed --
    # exercise that fallback path directly.
    bogus = np.ones((10, 10), dtype=bool)

    def fake_rasterize(mesh, z, origin_xy, nx, ny, pitch):
        return bogus.copy()

    def fake_detect(frac, *a, **k):
        return np.ones_like(frac, dtype=bool), np.zeros_like(frac)

    monkeypatch.setattr(mesh_regions, "_rasterize_z_slice", fake_rasterize)
    monkeypatch.setattr(mesh_regions, "_detect_slice_outliers", fake_detect)

    dense, diag = voxelize_region(None, (0, 0, 0), (10, 10, 5), pitch=1.0)
    assert diag["n_outlier_slices_detected"] == 5
    assert diag["n_outlier_slices_healed"] == 0
    assert diag["n_outlier_slices_unresolved"] == 5
