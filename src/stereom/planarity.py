"""N-3 node orientation and planarity. Replaces ``ori3.m`` (spec Sec 3.15).

The plane is fitted through the THREE NEIGHBOUR nodes (not the equiangular
central-axis construction the companion document derives from the papers'
text, ``v ~ (u1-u2) x (u2-u3)``) -- these are different estimators that
coincide only when the three incident branches have equal length. This
port keeps ``ori3``'s version (cheaper, more stable) and states that
choice explicitly, per spec Sec 3.15 / 10.2 requirement 7. Do not port
``offangle.m`` (D12: crosses two unit vectors and treats the result as a
unit vector; the acos argument it produces is not actually a cosine).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def node_planarity(G: nx.MultiGraph, node_id: int) -> tuple[np.ndarray, float, float] | None:
    """Returns ``(normal, out_angle_deg, planarity_index)`` for a
    degree-3 node, or ``None`` if the node's degree is not 3 or the
    neighbours are degenerate (collinear / coincident).

    ``planarity_index = |cos(out_angle_deg)|`` in ``[0, 1]``: 0 for a
    perfectly planar node (each branch at 90 deg to the fitted normal),
    matching the sense-check in spec Sec 3.15.
    """
    neighbors = [v if u == node_id else u for u, v, k in G.edges(node_id, keys=True)]
    if len(neighbors) != 3:
        return None

    pos0 = np.asarray(G.nodes[node_id]["pos"], dtype=np.float64)
    p1, p2, p3 = (np.asarray(G.nodes[n]["pos"], dtype=np.float64) for n in neighbors)

    v1 = p2 - p1
    v2 = p3 - p1
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None
    v1, v2 = v1 / n1, v2 / n2
    normal = np.cross(v2, v1)
    nn = np.linalg.norm(normal)
    if nn == 0:  # collinear neighbours -- plane undefined
        return None
    normal = normal / nn

    us = []
    for p in (p1, p2, p3):
        u = p - pos0
        un = np.linalg.norm(u)
        if un == 0:
            return None
        us.append(u / un)

    thetas = np.array(
        [np.degrees(np.arccos(np.clip(np.dot(u, normal), -1.0, 1.0))) for u in us]
    )
    if thetas.mean() > 90:
        normal = -normal
        out_angle = 180.0 - float(thetas.max())
    else:
        out_angle = float(thetas.max())

    planarity_index = float(abs(np.cos(np.radians(out_angle))))
    return normal, out_angle, planarity_index


def planarity_table(G: nx.MultiGraph) -> pd.DataFrame:
    rows = []
    for n in G.nodes():
        if G.degree(n) != 3:
            continue
        result = node_planarity(G, n)
        if result is None:
            continue
        normal, out_angle, p = result
        rows.append(
            {
                "node": n,
                "normal_x": normal[0],
                "normal_y": normal[1],
                "normal_z": normal[2],
                "out_angle_deg": out_angle,
                "planarity_index": p,
            }
        )
    return pd.DataFrame(rows)
