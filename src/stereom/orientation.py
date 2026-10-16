"""Branch orientation. Replaces ``myangle.m`` / ``vectorAngle.m`` (spec
Sec 3.13), fixing D8 (degenerate cases silently return 0 instead of the
correct 90/atan2 value for axis-aligned branches -- "a substantive defect"
per the spec, since axis-aligned branches are exactly the population of
interest in a textured material) by using ``atan2`` throughout, and using
the Volume's explicit axis mapping instead of hardcoded array dimensions
(spec Sec 2.3, 10.2 requirement 4).

Naming follows the PAPERS, not the code: ``theta`` is misorientation from
the L axis (0-90 deg), ``omega`` is the in-plane azimuth in the C-R plane
measured from R (0-360 deg). The original swaps these names (spec
Sec 2.4) -- do not port that swap.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from stereom.volume import AxisTriple


def branch_orientation(
    pos_a: tuple[int, int, int], pos_b: tuple[int, int, int], axes: AxisTriple
) -> tuple[float, float]:
    """theta, omega (degrees) for the chord from ``pos_a`` to ``pos_b``.

    ``omega`` requires an orientation convention for an otherwise
    undirected chord (swapping ``pos_a``/``pos_b`` would flip the raw
    azimuth by 180 deg). We canonicalize by requiring the L-component be
    non-negative, matching the original's intent (spec Sec 3.13: "the
    vector is first given a canonical sign"). For a branch exactly
    perpendicular to L (``dl == 0``), no canonical direction exists --
    ``omega`` is then genuinely ambiguous up to a 180 deg flip depending on
    which endpoint was labelled ``a``; this is an intrinsic property of an
    undirected transverse chord, not a bug (contrast with D8, where the
    original returned a plain wrong value for this case instead of an
    inherently ambiguous one).
    """
    d = np.asarray(pos_b, dtype=np.float64) - np.asarray(pos_a, dtype=np.float64)
    li, ci, ri = axes.index("L"), axes.index("C"), axes.index("R")
    dl, dc, dr = d[li], d[ci], d[ri]

    if dl < 0:
        dl, dc, dr = -dl, -dc, -dr

    theta = float(np.degrees(np.arctan2(np.hypot(dc, dr), dl)))
    omega = float(np.degrees(np.arctan2(dc, dr)) % 360.0)
    return theta, omega


def compute_orientations(G: nx.MultiGraph, axes: AxisTriple) -> nx.MultiGraph:
    """Set ``theta_deg`` and ``omega_deg`` edge attributes for every
    branch, computed from the node-to-node chord (not the local tangent --
    spec Sec 3.13 notes this is the branch's net direction, a defensible
    but distinct choice from tangent-based orientation; stated explicitly
    here per spec Sec 10.2 requirement 7)."""
    G = G.copy()
    for u, v, k, data in G.edges(keys=True, data=True):
        theta, omega = branch_orientation(G.nodes[u]["pos"], G.nodes[v]["pos"], axes)
        data["theta_deg"] = theta
        data["omega_deg"] = omega
    return G


def orientation_table(G: nx.MultiGraph) -> pd.DataFrame:
    rows = [
        {"branch_id": data.get("id", k), "theta_deg": data["theta_deg"], "omega_deg": data["omega_deg"]}
        for u, v, k, data in G.edges(keys=True, data=True)
        if "theta_deg" in data
    ]
    return pd.DataFrame(rows)
