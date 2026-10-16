"""Optional visualization helpers. Deliberately separate from every
analysis function in the rest of the package (spec Sec 10.2 requirement 1:
"No plotting inside any analysis function. Analysis functions return data
structures.") -- everything here takes already-computed results (arrays /
DataFrames / a graph) and returns a matplotlib ``Figure``; nothing here is
called by ``pipeline.py``.

Requires the ``viz`` extra (``pip install stereom[viz]``).
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def plot_graph_3d(G: nx.MultiGraph, *, node_size: float = 10.0, elev: float = 20.0, azim: float = 120.0):
    import matplotlib.pyplot as plt  # noqa: local import, optional dependency

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for u, v, k, data in G.edges(keys=True, data=True):
        path = data.get("path")
        pos_a = np.asarray(G.nodes[u]["pos"], dtype=np.float64)
        pos_b = np.asarray(G.nodes[v]["pos"], dtype=np.float64)
        pts = [pos_a]
        if path is not None and path.shape[0] > 0:
            pts.extend(path.astype(np.float64))
        pts.append(pos_b)
        pts_arr = np.array(pts)
        ax.plot(pts_arr[:, 0], pts_arr[:, 1], pts_arr[:, 2], color="tab:green", linewidth=0.8)

    positions = np.array([G.nodes[n]["pos"] for n in G.nodes()], dtype=np.float64)
    if positions.size:
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], color="tab:red", s=node_size)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)
    return fig


def plot_histogram(values: np.ndarray, *, bins: int | np.ndarray = 30, xlabel: str = "", title: str = ""):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.hist(np.asarray(values), bins=bins, color="tab:blue", edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    return fig


def plot_orientation_polar(theta_deg: np.ndarray, omega_deg: np.ndarray):
    """theta = radial coordinate (misorientation), omega = angular
    coordinate (azimuth) -- paper convention, not swapped (spec Sec 2.4)."""
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")
    ax.plot(np.deg2rad(omega_deg), theta_deg, "o", markersize=3, alpha=0.6)
    return fig


def plot_fft_peaks(spectrum: np.ndarray, threshold: float = 0.5):
    import matplotlib.pyplot as plt

    coords = np.argwhere(spectrum > threshold)
    values = spectrum[spectrum > threshold]
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    if coords.size:
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=values, cmap="gray")
    return fig
