"""Runtime-checkable invariants from spec Sec 9.2, as functions returning a
list of violation strings (empty = pass) rather than raising immediately,
so a test can assert on the list and see every failure at once.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd


def check_no_self_loops(G: nx.MultiGraph) -> list[str]:
    return [f"self-loop at node {u}" for u, v, k in G.edges(keys=True) if u == v]


def check_no_parallel_edges(G: nx.MultiGraph) -> list[str]:
    seen: dict[tuple[int, int], int] = {}
    for u, v, k in G.edges(keys=True):
        key = (min(u, v), max(u, v))
        seen[key] = seen.get(key, 0) + 1
    return [f"{c} parallel edges between {k}" for k, c in seen.items() if c > 1]


def check_tortuosity(G: nx.MultiGraph, tol: float = 1e-6) -> list[str]:
    out = []
    for u, v, k, data in G.edges(keys=True, data=True):
        t = data.get("tortuosity")
        if t is None or (isinstance(t, float) and math.isnan(t)):
            continue
        if t < 1.0 - tol:
            out.append(f"branch {data.get('id', k)}: tortuosity {t} < 1")
    return out


def check_thickness_ordering(G: nx.MultiGraph, tol: float = 1e-6) -> list[str]:
    out = []
    for u, v, k, data in G.edges(keys=True, data=True):
        t_min, t_mean = data.get("t_min_radius_um"), data.get("t_mean_radius_um")
        if t_min is None or t_mean is None or math.isnan(t_min) or math.isnan(t_mean):
            continue
        if t_min > t_mean + tol:
            out.append(f"branch {data.get('id', k)}: t_min {t_min} > t_mean {t_mean}")
    return out


def check_orientation_ranges(G: nx.MultiGraph, tol: float = 1e-6) -> list[str]:
    out = []
    for u, v, k, data in G.edges(keys=True, data=True):
        theta, omega = data.get("theta_deg"), data.get("omega_deg")
        if theta is not None and not (-tol <= theta <= 90 + tol):
            out.append(f"branch {data.get('id', k)}: theta {theta} out of [0,90]")
        if omega is not None and not (-tol <= omega < 360 + tol):
            out.append(f"branch {data.get('id', k)}: omega {omega} out of [0,360)")
    return out


def check_path_contiguity(G: nx.MultiGraph, max_step: float = math.sqrt(3) + 1e-6) -> list[str]:
    """Every consecutive pair of voxels in an ordered branch path (spec's
    `branch_order` output) must be 26-adjacent -- catches unrepaired path
    jumps."""
    out = []
    for u, v, k, data in G.edges(keys=True, data=True):
        path = data.get("path")
        if path is None or path.shape[0] < 2:
            continue
        steps = np.linalg.norm(np.diff(path.astype(np.float64), axis=0), axis=1)
        if np.any(steps > max_step):
            out.append(f"branch {data.get('id', k)}: path has a step of {steps.max():.3f} voxels")
    return out


def check_angle_pair_count(G: nx.MultiGraph, angles_df: pd.DataFrame) -> list[str]:
    """n_angles(node) == degree*(degree-1)/2 -- catches D6 (only measuring
    cyclically-adjacent pairs instead of all pairs)."""
    out = []
    counts = angles_df.groupby("node").size().to_dict() if len(angles_df) else {}
    for n in G.nodes():
        deg = G.degree(n)
        expected = deg * (deg - 1) // 2
        actual = counts.get(n, 0)
        if deg >= 2 and actual != expected:
            out.append(f"node {n} (degree {deg}): expected {expected} angle pairs, got {actual}")
    return out


def check_planarity_range(planarity_df: pd.DataFrame) -> list[str]:
    bad = planarity_df[(planarity_df["planarity_index"] < -1e-9) | (planarity_df["planarity_index"] > 1 + 1e-9)]
    return [f"node {row.node}: planarity_index {row.planarity_index} out of [0,1]" for row in bad.itertuples()]


def check_ring_flatness(rings_df: pd.DataFrame, r2_min: float) -> list[str]:
    bad = rings_df[rings_df["flatness_r2"] < r2_min - 1e-9]
    return [f"ring {row.id}: flatness_r2 {row.flatness_r2} < {r2_min}" for row in bad.itertuples()]


def run_all(
    G: nx.MultiGraph,
    *,
    angles_df: pd.DataFrame | None = None,
    planarity_df: pd.DataFrame | None = None,
    rings_df: pd.DataFrame | None = None,
    ring_r2_min: float = 0.5,
    expect_no_parallel_edges: bool = True,
) -> list[str]:
    violations = []
    violations += check_no_self_loops(G)
    if expect_no_parallel_edges:
        violations += check_no_parallel_edges(G)
    violations += check_tortuosity(G)
    violations += check_thickness_ordering(G)
    violations += check_orientation_ranges(G)
    violations += check_path_contiguity(G)
    if angles_df is not None:
        violations += check_angle_pair_count(G, angles_df)
    if planarity_df is not None:
        violations += check_planarity_range(planarity_df)
    if rings_df is not None and len(rings_df):
        violations += check_ring_flatness(rings_df, ring_r2_min)
    return violations
