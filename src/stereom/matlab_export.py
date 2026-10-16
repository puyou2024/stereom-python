"""Export analysis results to a single MATLAB ``.mat`` file, for the
visualization scripts in ``matlab_viz/`` (see RUN_INSTRUCTIONS.md).

Every other output of this package (CSV/Parquet tables, ``graph.npz``) is
Python-native. This module exists purely so MATLAB -- which cannot read
``.npz`` or a ``networkx`` graph, and where ragged per-branch voxel paths
don't fit naturally into a CSV -- has a single, direct, no-glue-code file
to load. Nothing here is used by ``pipeline.py`` or any other analysis
code (spec Sec 10.2 requirement 1: computation stays separate from any
particular presentation format, MATLAB visualization included).

Node identity note: graph node ids are stable but NOT contiguous once
nodes have been removed by cleaning/merging (spec Sec 2.5's "stable
integer IDs, decoupled from voxel positions" -- ids are stable, not
necessarily dense). The exported ``nodes`` struct keeps the original
``id`` for cross-referencing against the CSV outputs, and ALSO assigns a
dense 1-based ``row`` field; ``branches.node_a_row``/``node_b_row`` index
directly into that row numbering, which is what MATLAB plotting code
should actually use (``nodes.i(branches.node_a_row(k))`` etc.) since
MATLAB has no direct equivalent of a hash-map id lookup as convenient as
a plain array index.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.io import savemat


def _cell(items: list) -> np.ndarray:
    """A Python list -> a MATLAB cell array (via scipy.io.savemat's
    convention: an object-dtype ndarray, one element per cell)."""
    cell = np.empty((len(items), 1), dtype=object)
    for i, item in enumerate(items):
        cell[i, 0] = item
    return cell


def _col(values, dtype=np.float64) -> np.ndarray:
    return np.asarray(values, dtype=dtype).reshape(-1, 1)


def _parse_tuple_column(series: pd.Series) -> list[tuple]:
    """rings.csv/chains.csv round-trip tuple-valued columns (``nodes``,
    ``branch_ids``) as their Python repr string (e.g. ``"(0, 1, 2, 3)"``)
    -- reparse them back into tuples when reloading from CSV."""
    out = []
    for v in series:
        if isinstance(v, str):
            out.append(ast.literal_eval(v))
        else:
            out.append(tuple(v))
    return out


def _is_stringlike_column(series: pd.Series) -> bool:
    """True if this column holds strings that need re-parsing (via
    ``_parse_tuple_column``) rather than already-tuple values -- e.g. after
    a CSV round-trip. Checking ``series.dtype == object`` alone is not
    reliable: pandas >= 2.x can infer a dedicated string dtype (its own
    ``StringDtype``, not ``object``) for CSV string columns depending on
    the ``future.infer_string`` option (the default in pandas 3.x), which
    silently skipped re-parsing here and let raw ``"(24, 631, ...)"``
    strings leak into ``np.asarray(..., dtype=np.float64)`` downstream."""
    if series.dtype == object:
        return True
    first = next((v for v in series if pd.notna(v)), None)
    return isinstance(first, str)


def _nodes_struct(graph: nx.MultiGraph) -> tuple[dict, dict[int, int]]:
    nodes = list(graph.nodes())
    node_row = {n: i for i, n in enumerate(nodes)}  # 0-based
    if nodes:
        pos = np.array([graph.nodes[n]["pos"] for n in nodes], dtype=np.float64)
    else:
        pos = np.zeros((0, 3), dtype=np.float64)  # np.array([]) would be 1-D, breaking pos[:, 0:1] below
    struct = {
        "id": _col(nodes),
        "row": _col(np.arange(1, len(nodes) + 1)),  # 1-based, for direct MATLAB indexing
        "i": pos[:, 0:1],
        "j": pos[:, 1:2],
        "k": pos[:, 2:3],
        "degree": _col([graph.degree(n) for n in nodes]),
        "voxel_degree": _col([graph.nodes[n].get("voxel_degree", -1) for n in nodes]),
    }
    return struct, node_row


def _branches_struct(graph: nx.MultiGraph, node_row: dict[int, int]) -> dict:
    ids, node_a_row, node_b_row = [], [], []
    length_um, distance_um, tortuosity = [], [], []
    t_min, t_cen, t_mean = [], [], []
    theta, omega = [], []
    paths = []
    for u, v, k, data in graph.edges(keys=True, data=True):
        ids.append(data.get("id", k))
        node_a_row.append(node_row[data.get("node_a", u)] + 1)
        node_b_row.append(node_row[data.get("node_b", v)] + 1)
        length_um.append(data.get("length_um", np.nan))
        distance_um.append(data.get("distance_um", np.nan))
        tortuosity.append(data.get("tortuosity", np.nan))
        t_min.append(data.get("t_min_radius_um", np.nan))
        t_cen.append(data.get("t_cen_radius_um", np.nan))
        t_mean.append(data.get("t_mean_radius_um", np.nan))
        theta.append(data.get("theta_deg", np.nan))
        omega.append(data.get("omega_deg", np.nan))
        path = data.get("path")
        paths.append(path.astype(np.float64) if path is not None else np.zeros((0, 3)))

    return {
        "id": _col(ids),
        "node_a_row": _col(node_a_row),
        "node_b_row": _col(node_b_row),
        "length_um": _col(length_um),
        "distance_um": _col(distance_um),
        "tortuosity": _col(tortuosity),
        "t_min_radius_um": _col(t_min),
        "t_cen_radius_um": _col(t_cen),
        "t_mean_radius_um": _col(t_mean),
        "t_min_diameter_um": _col(t_min) * 2.0,
        "t_mean_diameter_um": _col(t_mean) * 2.0,
        "theta_deg": _col(theta),
        "omega_deg": _col(omega),
        "path": _cell(paths),
    }


def _rings_struct(rings_df: pd.DataFrame, node_row: dict[int, int]) -> dict | None:
    if rings_df is None or len(rings_df) == 0:
        return None
    node_lists = _parse_tuple_column(rings_df["nodes"]) if _is_stringlike_column(rings_df["nodes"]) else list(rings_df["nodes"])
    branch_lists = (
        _parse_tuple_column(rings_df["branch_ids"]) if _is_stringlike_column(rings_df["branch_ids"]) else list(rings_df["branch_ids"])
    )
    node_rows_1based = [
        _col([node_row[n] + 1 for n in nl if n in node_row], dtype=np.float64).ravel() for nl in node_lists
    ]
    return {
        "id": _col(rings_df["id"]),
        "size": _col(rings_df["size"]),
        "node_row": _cell(node_rows_1based),
        "branch_id": _cell([np.asarray(b, dtype=np.float64) for b in branch_lists]),
        "centroid_x_um": _col(rings_df["centroid_x_um"]),
        "centroid_y_um": _col(rings_df["centroid_y_um"]),
        "centroid_z_um": _col(rings_df["centroid_z_um"]),
        "normal_x": _col(rings_df["normal_x"]),
        "normal_y": _col(rings_df["normal_y"]),
        "normal_z": _col(rings_df["normal_z"]),
        "area_um2": _col(rings_df["area_um2"]),
        "d_ring_um": _col(rings_df["d_ring_um"]),
        "flatness_r2": _col(rings_df["flatness_r2"]),
    }


def _chains_struct(chains_df: pd.DataFrame, node_row: dict[int, int]) -> dict | None:
    if chains_df is None or len(chains_df) == 0:
        return None
    node_lists = _parse_tuple_column(chains_df["nodes"]) if _is_stringlike_column(chains_df["nodes"]) else list(chains_df["nodes"])
    branch_lists = (
        _parse_tuple_column(chains_df["branch_ids"]) if _is_stringlike_column(chains_df["branch_ids"]) else list(chains_df["branch_ids"])
    )
    node_rows_1based = [
        _col([node_row[n] + 1 for n in nl if n in node_row], dtype=np.float64).ravel() for nl in node_lists
    ]
    return {
        "id": _col(chains_df["id"]),
        "n_branches": _col(chains_df["n_branches"]),
        "node_row": _cell(node_rows_1based),
        "branch_id": _cell([np.asarray(b, dtype=np.float64) for b in branch_lists]),
    }


def _angles_struct(angles_df: pd.DataFrame, node_row: dict[int, int]) -> dict | None:
    if angles_df is None or len(angles_df) == 0:
        return None
    return {
        "node_row": _col([node_row[n] + 1 for n in angles_df["node"]]),
        "branch_a": _col(angles_df["branch_a"]),
        "branch_b": _col(angles_df["branch_b"]),
        "angle_deg": _col(angles_df["angle_deg"]),
    }


def _profile_fits_struct(fits: Any) -> dict | None:
    """``fits`` is either a list[ProfileFitResult] (from an in-process
    PipelineResult) or a pandas DataFrame (reloaded from
    ``profile_fits.csv``)."""
    if fits is None:
        return None
    if isinstance(fits, pd.DataFrame):
        if len(fits) == 0:
            return None
        df = fits
        return {
            "branch_id": _col(df["branch_id"]),
            "accepted": _col(df["accepted"].astype(float)),
            "a0": _col(df["a0"]),
            "b": _col(df["b"]),
            "r2": _col(df["r2"]),
            "t_min_um": _col(df["t_min_um"]),
            "length_um": _col(df["length_um"]),
            "quartic_a": _col(df["quartic_a"]),
        }
    if len(fits) == 0:
        return None
    return {
        "branch_id": _col([f.branch_id for f in fits]),
        "accepted": _col([float(f.accepted) for f in fits]),
        "a0": _col([f.a0 for f in fits]),
        "b": _col([f.b for f in fits]),
        "r2": _col([f.r2 for f in fits]),
        "t_min_um": _col([f.t_min_um for f in fits]),
        "length_um": _col([f.length_um for f in fits]),
        "quartic_a": _col([f.quartic_a if f.quartic_a is not None else np.nan for f in fits]),
    }


def export_matlab(
    graph: nx.MultiGraph,
    path: str | Path,
    *,
    pipeline_result: Any = None,
    extra_csv_dir: str | Path | None = None,
    stl_alignment_path: str | Path | None = None,
) -> None:
    """Write ``path`` (a ``.mat`` file) with everything the MATLAB
    visualization scripts need.

    Pass ONE of:

    - ``pipeline_result``: the ``PipelineResult`` from
      ``pipeline.run_full_pipeline`` (same process, right after a run --
      has rings/chains/angles/profile-fits directly, no reparsing needed).
    - ``extra_csv_dir``: a results directory previously written by
      ``stereom run`` (``rings.csv``, ``chains.csv``,
      ``inter_branch_angles.csv``, ``profile_fits.csv``) to reload from
      disk instead.

    Either way, ``graph`` supplies node/branch topology, positions and
    voxel paths (from ``graph.npz`` if not already in memory).

    ``stl_alignment_path``: the ``{name}_stl_alignment.json`` sidecar
    written by ``mesh_regions.extract_regions`` next to a region's
    ``.npz``/``.stl``, if this volume came from an STL region extraction.
    If given, embeds a ``stl_overlay`` struct (``stl_file``,
    ``bbox_min_stl_units``, ``pitch_stl_units``) so the MATLAB script can
    convert the companion STL's vertex coordinates into this graph's
    voxel-index coordinate system for the transparent overlay.
    """
    node_struct, node_row = _nodes_struct(graph)
    branch_struct = _branches_struct(graph, node_row)

    rings_df = chains_df = angles_df = None
    fits: Any = None

    if pipeline_result is not None:
        rings_df = pipeline_result.rings_df
        chains_df = pipeline_result.chains_df
        angles_df = pipeline_result.angles_df
        fits = pipeline_result.profile_fits
    elif extra_csv_dir is not None:
        extra_csv_dir = Path(extra_csv_dir)
        if (extra_csv_dir / "rings.csv").exists():
            rings_df = pd.read_csv(extra_csv_dir / "rings.csv")
        if (extra_csv_dir / "chains.csv").exists():
            chains_df = pd.read_csv(extra_csv_dir / "chains.csv")
        if (extra_csv_dir / "inter_branch_angles.csv").exists():
            angles_df = pd.read_csv(extra_csv_dir / "inter_branch_angles.csv")
        if (extra_csv_dir / "profile_fits.csv").exists():
            fits = pd.read_csv(extra_csv_dir / "profile_fits.csv")

    mat_dict: dict[str, Any] = {"nodes": node_struct, "branches": branch_struct}

    rings_struct = _rings_struct(rings_df, node_row)
    if rings_struct is not None:
        mat_dict["rings"] = rings_struct

    chains_struct = _chains_struct(chains_df, node_row)
    if chains_struct is not None:
        mat_dict["chains"] = chains_struct

    angles_struct = _angles_struct(angles_df, node_row)
    if angles_struct is not None:
        mat_dict["angles"] = angles_struct

    fits_struct = _profile_fits_struct(fits)
    if fits_struct is not None:
        mat_dict["profile_fits"] = fits_struct

    if stl_alignment_path is not None and Path(stl_alignment_path).exists():
        import json

        alignment = json.loads(Path(stl_alignment_path).read_text())
        mat_dict["stl_overlay"] = {
            "stl_file": alignment["stl_file"],
            "bbox_min_stl_units": np.asarray(alignment["bbox_min_stl_units"], dtype=np.float64).reshape(1, 3),
            "pitch_stl_units": float(alignment["pitch_stl_units"]),
        }

    savemat(str(path), mat_dict, long_field_names=True)
