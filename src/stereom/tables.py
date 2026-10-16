"""Conversion between the working ``networkx.MultiGraph`` representation
and flat tables / a portable on-disk format.

Implements the "stable integer node IDs, decoupled from voxel positions"
schema of spec Sec 2.5, and makes the graph serializable to any standard
tabular format (here: ``.npz`` for the full graph incl. ragged branch
paths, plus plain CSV/Parquet export for the summary tables) -- spec
Sec 2.5 "removes the rewrite-on-merge fragility entirely".
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

_EDGE_SCALAR_ATTRS = [
    "length_um",
    "distance_um",
    "tortuosity",
    "t_min_radius_um",
    "t_cen_radius_um",
    "t_mean_radius_um",
    "theta_deg",
    "omega_deg",
    # Non-circular section descriptors (stereom.cross_section). Optional:
    # every consumer reads them via .get(attr, nan) and load_graph_npz skips
    # members absent from an older .npz, so adding them here does not break
    # graphs saved before they existed.
    "v_assigned_um3",
    "t_area_eff_um2",
    "t_equiv_radius_um",
    "t_aspect_ratio",
]


def nodes_to_dataframe(G: nx.MultiGraph) -> pd.DataFrame:
    rows = []
    for n in G.nodes():
        data = G.nodes[n]
        pos = data["pos"]
        rows.append(
            {
                "id": n,
                "i": pos[0],
                "j": pos[1],
                "k": pos[2],
                "degree": G.degree(n),
                "voxel_degree": data.get("voxel_degree", -1),
            }
        )
    return pd.DataFrame(rows, columns=["id", "i", "j", "k", "degree", "voxel_degree"])


def branches_to_dataframe(G: nx.MultiGraph) -> pd.DataFrame:
    rows = []
    for u, v, k, data in G.edges(keys=True, data=True):
        row = {
            "id": data.get("id", k),
            "node_a": data.get("node_a", u),
            "node_b": data.get("node_b", v),
            "n_path_voxels": int(data["path"].shape[0]) if "path" in data else 0,
        }
        for attr in _EDGE_SCALAR_ATTRS:
            row[attr] = data.get(attr, float("nan"))
        row["t_min_diameter_um"] = row["t_min_radius_um"] * 2.0
        row["t_mean_diameter_um"] = row["t_mean_radius_um"] * 2.0
        rows.append(row)
    cols = ["id", "node_a", "node_b", "n_path_voxels"] + _EDGE_SCALAR_ATTRS + [
        "t_min_diameter_um",
        "t_mean_diameter_um",
    ]
    return pd.DataFrame(rows, columns=cols)


def save_graph_npz(G: nx.MultiGraph, path: str | Path) -> None:
    """Save the full graph (topology, positions, branch paths, all
    computed edge attributes) to a single portable ``.npz`` file."""
    nodes = list(G.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}
    positions = np.array([G.nodes[n]["pos"] for n in nodes], dtype=np.int64)
    voxel_degrees = np.array([G.nodes[n].get("voxel_degree", -1) for n in nodes], dtype=np.int64)

    edge_u, edge_v, edge_ids, path_arrays = [], [], [], []
    attr_values: dict[str, list[float]] = {a: [] for a in _EDGE_SCALAR_ATTRS}
    for u, v, k, data in G.edges(keys=True, data=True):
        edge_u.append(node_index[u])
        edge_v.append(node_index[v])
        edge_ids.append(data.get("id", k))
        path_arrays.append(data.get("path", np.zeros((0, 3), dtype=np.int64)))
        for a in _EDGE_SCALAR_ATTRS:
            attr_values[a].append(data.get(a, np.nan))

    path_concat = np.vstack(path_arrays) if path_arrays else np.zeros((0, 3), dtype=np.int64)
    path_lengths = np.array([p.shape[0] for p in path_arrays], dtype=np.int64)

    save_dict = dict(
        node_ids=np.array(nodes, dtype=np.int64),
        positions=positions,
        voxel_degrees=voxel_degrees,
        edge_u=np.array(edge_u, dtype=np.int64),
        edge_v=np.array(edge_v, dtype=np.int64),
        edge_ids=np.array(edge_ids, dtype=np.int64),
        path_concat=path_concat,
        path_lengths=path_lengths,
    )
    for a in _EDGE_SCALAR_ATTRS:
        save_dict[a] = np.array(attr_values[a], dtype=np.float64)

    np.savez_compressed(path, **save_dict)


def load_graph_npz(path: str | Path) -> nx.MultiGraph:
    # Read every member array into an in-memory ndarray ONCE up front. An
    # NpzFile (what `np.load` returns for a .npz) is a lazy zip reader --
    # `z["key"]` re-reads and re-decompresses the WHOLE array from the
    # archive on every single access, not just the first. Indexing it
    # per-node/per-edge inside a loop (as this function used to) is O(N *
    # array_size) instead of O(array_size): fine for a few thousand edges,
    # catastrophic (multi-minute stalls, spurious MemoryError from repeated
    # large transient allocations) for a graph with 10k+ edges -- exactly
    # what Gyroid's finer lattice produces vs the Spinodoid graphs this was
    # first validated against.
    with np.load(path) as z:
        node_ids = z["node_ids"]
        positions = z["positions"]
        voxel_degrees = z["voxel_degrees"]
        edge_u_arr = z["edge_u"]
        edge_v_arr = z["edge_v"]
        edge_ids_arr = z["edge_ids"]
        path_concat = z["path_concat"]
        path_lengths = z["path_lengths"]
        edge_attrs = {a: z[a] for a in _EDGE_SCALAR_ATTRS if a in z}

    G = nx.MultiGraph()
    for i, nid in enumerate(node_ids):
        G.add_node(
            int(nid),
            pos=tuple(int(x) for x in positions[i]),
            voxel_degree=int(voxel_degrees[i]),
        )

    offsets = np.concatenate([[0], np.cumsum(path_lengths)])
    for e in range(len(edge_u_arr)):
        u = int(node_ids[edge_u_arr[e]])
        v = int(node_ids[edge_v_arr[e]])
        edge_id = int(edge_ids_arr[e])
        path = path_concat[offsets[e] : offsets[e + 1]]
        data = {"id": edge_id, "path": path, "node_a": u, "node_b": v}
        for a, arr in edge_attrs.items():
            data[a] = float(arr[e])
        G.add_edge(u, v, key=edge_id, **data)
    return G


def export_tables(nodes_df: pd.DataFrame, branches_df: pd.DataFrame, out_dir: str | Path, fmt: str = "csv") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        nodes_df.to_csv(out_dir / "nodes.csv", index=False)
        branches_df.to_csv(out_dir / "branches.csv", index=False)
    elif fmt == "parquet":
        nodes_df.to_parquet(out_dir / "nodes.parquet", index=False)
        branches_df.to_parquet(out_dir / "branches.parquet", index=False)
    else:
        raise ValueError(f"Unknown export format: {fmt!r} (use 'csv' or 'parquet')")
