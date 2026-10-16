"""Chain detection. Replaces ``chain.m`` (spec Sec 3.17), which as shipped
cannot run at all (D1: calls ``connected_node()``, not present anywhere in
the repository) and, even fixed, reads the wrong orientation column (D7:
linear-indexes an M x 2 ``branOrien`` array, reading omega where theta was
clearly intended) and hardcodes both the growth axis and a fixed
monotonicity direction.

The primary implementation here is the direction-agnostic greedy alignment
growth the spec recommends instead of trying to repair the threshold rule
(Sec 3.17 refactor note): from each unclaimed branch, extend outward in
both directions to whichever neighbouring, not-yet-claimed branch
maximizes ``|cos(angle between successive chord vectors)|``, stopping once
that falls below ``cos_threshold``. Every branch ends up in exactly one
chain. This needs no axis assumption and no arbitrary monotonicity rule.

``legacy_threshold_chains`` is also provided, implementing (for the first
time in a runnable form) the original's actual rule -- monotone advance
along a configurable axis, gated by misorientation from that axis -- for
users who specifically want that behaviour reproduced rather than the
recommended replacement.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def _ekey(u: int, v: int, k: int) -> tuple[int, int, int]:
    return (min(u, v), max(u, v), k)


def _grow_one_side(
    G: nx.MultiGraph, pivot: int, prev: int, cos_threshold: float, used: set
) -> tuple[list[int], list[int]]:
    ref_chord = np.asarray(G.nodes[pivot]["pos"], dtype=np.float64) - np.asarray(
        G.nodes[prev]["pos"], dtype=np.float64
    )
    nodes_out: list[int] = []
    branches_out: list[int] = []
    current = pivot
    while True:
        best = None
        for u, v, k, data in G.edges(current, keys=True, data=True):
            ek = _ekey(u, v, k)
            if ek in used:
                continue
            other = v if u == current else u
            chord = np.asarray(G.nodes[other]["pos"], dtype=np.float64) - np.asarray(
                G.nodes[current]["pos"], dtype=np.float64
            )
            denom = np.linalg.norm(ref_chord) * np.linalg.norm(chord)
            if denom == 0:
                continue
            cos_val = float(np.dot(ref_chord, chord) / denom)
            if best is None or abs(cos_val) > abs(best[0]):
                best = (cos_val, ek, other, chord, data.get("id", k))
        if best is None or abs(best[0]) < cos_threshold:
            break
        cos_val, ek, other, chord, branch_id = best
        used.add(ek)
        nodes_out.append(other)
        branches_out.append(branch_id)
        ref_chord = chord if cos_val >= 0 else -chord
        current = other
    return nodes_out, branches_out


def greedy_chains(G: nx.MultiGraph, cos_threshold: float = 0.5) -> pd.DataFrame:
    """Partition every branch into a maximal chain of geometrically
    aligned neighbours. Returns one row per chain: ``nodes`` (ordered node
    id sequence), ``branch_ids``, ``n_branches``."""
    used: set = set()
    rows = []
    chain_id = 0
    for u0, v0, k0, data0 in list(G.edges(keys=True, data=True)):
        ek0 = _ekey(u0, v0, k0)
        if ek0 in used:
            continue
        used.add(ek0)
        fwd_nodes, fwd_branches = _grow_one_side(G, v0, u0, cos_threshold, used)
        bwd_nodes, bwd_branches = _grow_one_side(G, u0, v0, cos_threshold, used)
        nodes = list(reversed(bwd_nodes)) + [u0, v0] + fwd_nodes
        branches = list(reversed(bwd_branches)) + [data0.get("id", k0)] + fwd_branches
        rows.append({"id": chain_id, "nodes": tuple(nodes), "branch_ids": tuple(branches), "n_branches": len(branches)})
        chain_id += 1
    return pd.DataFrame(rows)


def legacy_threshold_chains(
    G: nx.MultiGraph, axis_index: int, theta_max_deg: float = 60.0
) -> list[list[int | None]]:
    """Reproduce the original's rule: from each node, follow every incident
    branch whose far node advances monotonically along array dimension
    ``axis_index`` AND whose misorientation from that axis
    (``theta_deg`` edge attribute -- requires ``orientation.compute_orientations``
    to have run first, with ``axes`` set so that dimension is 'L') is
    ``<= theta_max_deg``, recursing; chains are delimited by ``None``.
    Fixes D1 (the recursion is fully self-contained -- no missing
    ``connected_node``) and D7 (uses ``theta_deg``, the misorientation,
    not the azimuth the original actually indexed)."""
    chains: list[list[int | None]] = []

    def grow(n: int, acc: list[int | None]) -> None:
        pos_n = G.nodes[n]["pos"][axis_index]
        survivors = []
        for u, v, k, data in G.edges(n, keys=True, data=True):
            other = v if u == n else u
            if G.nodes[other]["pos"][axis_index] <= pos_n:
                continue
            theta = data.get("theta_deg")
            if theta is None or theta > theta_max_deg:
                continue
            survivors.append(other)
        if survivors:
            for s in survivors:
                acc.append(n)
                grow(s, acc)
        else:
            acc.append(n)
            acc.append(None)

    for n in G.nodes():
        acc: list[int | None] = []
        grow(n, acc)
        chains.append(acc)
    return chains
