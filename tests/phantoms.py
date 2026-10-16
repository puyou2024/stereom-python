"""Synthetic volume and graph builders for tests, per spec Sec 9.1.

Two flavours, deliberately:

* Hand-built ``networkx.MultiGraph`` objects with exact, known node
  positions and paths -- used to test the MATH of an individual stage
  (length, thickness, orientation, angles, planarity, ring geometry,
  profile fitting) against an analytic expectation with no voxelization
  noise in the way.

* Rasterized binary VOLUMES (a capsule -- cylinder with hemispherical caps
  -- drawn per strut) -- used for end-to-end pipeline tests that must
  exercise skeletonization and graph registration for real, in particular
  the rotation- and scaling-invariance tests the spec calls out as
  highest-yield (Sec 9.1: "together they expose nearly every hardcoded
  voxel literal and every axis assumption").
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from stereom.volume import Volume


class CapsuleRasterizer:
    """Accumulates a union of capsules (cylinder + hemispherical caps) into
    a boolean volume. Precomputes the voxel coordinate grid once so many
    struts can be drawn efficiently."""

    def __init__(self, shape: tuple[int, int, int]):
        self.shape = shape
        ii, jj, kk = np.indices(shape)
        self.coords = np.stack([ii, jj, kk], axis=-1).astype(np.float64)
        self.volume = np.zeros(shape, dtype=bool)

    def add_capsule(self, p0, p1, radius: float) -> None:
        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)
        d = p1 - p0
        l2 = float(d @ d)
        if l2 == 0:
            dist = np.linalg.norm(self.coords - p0, axis=-1)
        else:
            t = np.clip(((self.coords - p0) @ d) / l2, 0.0, 1.0)
            proj = p0 + t[..., None] * d
            dist = np.linalg.norm(self.coords - proj, axis=-1)
        self.volume |= dist <= radius


def straight_cylinder_volume(
    length: float, radius: float, *, margin: int = 6, voxel_size: float = 1.0
) -> tuple[Volume, np.ndarray, np.ndarray]:
    """A single straight capsule along dim 0. Returns (volume, p0, p1)."""
    n0 = int(np.ceil(length + 2 * margin))
    shape = (n0, int(np.ceil(2 * radius + 2 * margin)), int(np.ceil(2 * radius + 2 * margin)))
    r = CapsuleRasterizer(shape)
    p0 = np.array([margin, shape[1] / 2, shape[2] / 2])
    p1 = np.array([margin + length, shape[1] / 2, shape[2] / 2])
    r.add_capsule(p0, p1, radius)
    return Volume(data=r.volume, voxel_size=voxel_size), p0, p1


def bent_cylinder_volume(
    arc_radius: float, arc_angle_deg: float, strut_radius: float, *, margin: int = 8, n_segments: int = 40
) -> tuple[Volume, np.ndarray]:
    """A circular-arc strut approximated by short capsule segments (dense
    enough that the polyline length converges to the analytic arc length).
    Returns (volume, ordered centreline points used to build it)."""
    angle = np.deg2rad(arc_angle_deg)
    thetas = np.linspace(0, angle, n_segments + 1)
    pts_local = np.stack([arc_radius * np.sin(t) * 0 + arc_radius * (1 - np.cos(t)), arc_radius * np.sin(t), np.zeros_like(t)], axis=1)
    extent = arc_radius * 1.2 + margin
    shape = (int(np.ceil(extent + margin)), int(np.ceil(extent + margin)), int(np.ceil(2 * strut_radius + 2 * margin)))
    offset = np.array([margin, margin, shape[2] / 2])
    pts = pts_local + offset
    r = CapsuleRasterizer(shape)
    for k in range(n_segments):
        r.add_capsule(pts[k], pts[k + 1], strut_radius)
    return Volume(data=r.volume, voxel_size=1.0), pts


def cubic_lattice_volume(
    n_cells: tuple[int, int, int], pitch: float, radius: float, *, margin: int = 6
) -> tuple[Volume, dict[tuple[int, int, int], np.ndarray]]:
    """Simple cubic lattice: grid points at every ``(i,j,k)`` for
    ``i in [0,nx], j in [0,ny], k in [0,nz]``, struts along +x/+y/+z from
    each point. Interior points have degree 6, all angles 90/180 deg,
    every ring is a 4-cycle of side ``pitch``."""
    nx_, ny_, nz_ = n_cells
    shape = (
        int(nx_ * pitch + 2 * margin),
        int(ny_ * pitch + 2 * margin),
        int(nz_ * pitch + 2 * margin),
    )
    grid: dict[tuple[int, int, int], np.ndarray] = {}
    for i in range(nx_ + 1):
        for j in range(ny_ + 1):
            for k in range(nz_ + 1):
                grid[(i, j, k)] = np.array(
                    [margin + i * pitch, margin + j * pitch, margin + k * pitch], dtype=np.float64
                )

    r = CapsuleRasterizer(shape)
    for (i, j, k), p0 in grid.items():
        for di, dj, dk in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            nb = (i + di, j + dj, k + dk)
            if nb in grid:
                r.add_capsule(p0, grid[nb], radius)

    return Volume(data=r.volume, voxel_size=1.0), grid


def transform_volume_grid_points(
    grid: dict[tuple[int, int, int], np.ndarray], rotation: np.ndarray, translation: np.ndarray
) -> dict[tuple[int, int, int], np.ndarray]:
    return {key: rotation @ p + translation for key, p in grid.items()}


def rotated_cubic_lattice_volume(
    n_cells: tuple[int, int, int], pitch: float, radius: float, angle_deg: float, *, margin: int = 12
) -> Volume:
    """Same lattice as ``cubic_lattice_volume`` but rotated ``angle_deg``
    about the dim-2 axis before rasterizing, in a larger volume so the
    rotated structure still fits -- the rotation/axis-convention-leakage
    phantom (spec Sec 9.1)."""
    nx_, ny_, nz_ = n_cells
    theta = np.deg2rad(angle_deg)
    rot = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    raw_grid: dict[tuple[int, int, int], np.ndarray] = {}
    for i in range(nx_ + 1):
        for j in range(ny_ + 1):
            for k in range(nz_ + 1):
                raw_grid[(i, j, k)] = np.array([i * pitch, j * pitch, k * pitch], dtype=np.float64)

    centre = np.array([nx_ * pitch, ny_ * pitch, nz_ * pitch]) / 2.0
    rotated = {key: rot @ (p - centre) + centre for key, p in raw_grid.items()}

    extent = max(nx_, ny_) * pitch * 1.5 + 2 * margin
    zextent = nz_ * pitch + 2 * margin
    shape = (int(extent), int(extent), int(zextent))
    shift = np.array([extent / 2 - centre[0], extent / 2 - centre[1], margin])
    grid = {key: p + shift for key, p in rotated.items()}

    r = CapsuleRasterizer(shape)
    for (i, j, k), p0 in raw_grid.items():
        for di, dj, dk in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            nb = (i + di, j + dj, k + dk)
            if nb in raw_grid:
                r.add_capsule(grid[(i, j, k)], grid[nb], radius)

    return Volume(data=r.volume, voxel_size=1.0)


# ---------------------------------------------------------------------
# Synthetic graphs (exact, no voxelization noise)
# ---------------------------------------------------------------------


def single_branch_graph(pos_a: tuple[int, int, int], pos_b: tuple[int, int, int], path: np.ndarray) -> nx.MultiGraph:
    G = nx.MultiGraph()
    G.add_node(0, pos=pos_a, voxel_degree=-1)
    G.add_node(1, pos=pos_b, voxel_degree=-1)
    G.add_edge(0, 1, key=0, id=0, path=path, node_a=0, node_b=1)
    return G


def star_graph(center: tuple[float, float, float], neighbor_offsets: list[tuple[float, float, float]]) -> nx.MultiGraph:
    """One central node, N neighbours at given offsets (float coords are
    rounded to the nearest int voxel position -- callers wanting exact
    real-valued angle tests should pass offsets whose rounded voxel
    positions still realize the intended geometry, or bypass this by
    building the graph directly)."""
    G = nx.MultiGraph()
    c = tuple(int(round(x)) for x in center)
    G.add_node(0, pos=c, voxel_degree=-1)
    for idx, off in enumerate(neighbor_offsets, start=1):
        p = tuple(int(round(center[d] + off[d])) for d in range(3))
        G.add_node(idx, pos=p, voxel_degree=-1)
        G.add_edge(0, idx, key=idx, id=idx, path=np.zeros((0, 3), dtype=np.int64), node_a=0, node_b=idx)
    return G
