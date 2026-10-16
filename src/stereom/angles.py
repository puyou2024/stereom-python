"""Inter-branch angles. Replaces ``interangle.m`` / ``main_interBranchAngle.m``
(spec Sec 3.14), fixing D6 (only cyclically-adjacent neighbour pairs were
measured, giving 4 angles instead of the 6 = C(4,2) distinct pairs at an
N-4 node, in an order that depended on ``find``'s arbitrary return order)
and D11 (``acos`` domain overflow from floating-point error silently
mapped to 0/180 deg via ``real()``/``abs()``, instead of being clamped).

Angles are computed between node-to-node CHORD vectors at the shared
node, not local tangents -- the papers do not state which convention is
used; this matches the original's choice and is stated explicitly (spec
Sec 3.14 "document your choice").
"""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pandas as pd


def inter_branch_angles(G: nx.MultiGraph) -> pd.DataFrame:
    """One row per (node, unordered pair of incident branches), with the
    angle in degrees between the two node-to-neighbour chord vectors.
    ``n_angles(node) == degree*(degree-1)/2`` for every node (the
    invariant that catches D6, spec Sec 9.2)."""
    rows = []
    for n in G.nodes():
        pos_n = np.asarray(G.nodes[n]["pos"], dtype=np.float64)
        incident = []
        for u, v, k, data in G.edges(n, keys=True, data=True):
            other = v if u == n else u
            incident.append((other, data.get("id", k)))
        if len(incident) < 2:
            continue
        for (n1, b1), (n2, b2) in itertools.combinations(incident, 2):
            v1 = np.asarray(G.nodes[n1]["pos"], dtype=np.float64) - pos_n
            v2 = np.asarray(G.nodes[n2]["pos"], dtype=np.float64) - pos_n
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            if denom == 0:
                continue
            cos_angle = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
            angle_deg = float(np.degrees(np.arccos(cos_angle)))
            rows.append(
                {
                    "node": n,
                    "neighbor_a": n1,
                    "neighbor_b": n2,
                    "branch_a": b1,
                    "branch_b": b2,
                    "angle_deg": angle_deg,
                }
            )
    return pd.DataFrame(rows, columns=["node", "neighbor_a", "neighbor_b", "branch_a", "branch_b", "angle_deg"])


def node_degree_table(G: nx.MultiGraph) -> pd.DataFrame:
    return pd.DataFrame(
        [{"node": n, "degree": G.degree(n), "pos": G.nodes[n]["pos"]} for n in G.nodes()]
    )
