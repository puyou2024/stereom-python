"""3D-FFT of the node array. Replaces ``main_fft.m`` (spec Sec 3.18).

Fixes: the unconventional shift order (`fftshift(fftn(ifftshift(x)))` is
the standard; the original visibly does the opposite nesting, which is
invisible in a magnitude spectrum but wrong for clarity/consistency); the
`elsess` syntax error that prevents the original script from parsing at
all (not applicable here); and the complete absence of windowing, which
lets the rectangular volume boundary produce sinc sidelobes that can be
mistaken for structural peaks (spec Sec 3.18) -- a Tukey window is applied
by default.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.signal.windows import tukey


def node_indicator_volume(G: nx.MultiGraph, shape: tuple[int, int, int]) -> np.ndarray:
    """Rasterize node positions as a sparse binary indicator volume --
    matches the original: the transform is taken over node positions at
    the original voxel grid, not over the segmented structure."""
    vol = np.zeros(shape, dtype=np.float64)
    for n in G.nodes():
        pos = G.nodes[n]["pos"]
        vol[pos] = 1.0
    return vol


def node_fft(
    G: nx.MultiGraph, shape: tuple[int, int, int], *, window: bool = True, window_alpha: float = 0.5
) -> np.ndarray:
    """Magnitude spectrum of the node-indicator volume, normalized to
    ``[0, 1]``."""
    vol = node_indicator_volume(G, shape)
    if window:
        wins = [tukey(n, alpha=window_alpha) for n in shape]
        vol = vol * wins[0][:, None, None] * wins[1][None, :, None] * wins[2][None, None, :]

    spectrum = np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(vol))))
    lo, hi = float(spectrum.min()), float(spectrum.max())
    if hi > lo:
        spectrum = (spectrum - lo) / (hi - lo)
    return spectrum


def find_peaks(spectrum: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Indices (into the shifted spectrum array) where the normalized
    magnitude exceeds ``threshold``."""
    return np.argwhere(spectrum > threshold)


def peak_index_to_spacing_um(
    peak_index: tuple[int, int, int], shape: tuple[int, int, int], voxel_size_um: float
) -> float:
    """Convert a peak's array index (in the fftshift'd spectrum, DC at
    ``shape[d]//2``) to a real-space spacing in micrometres, via the
    standard DFT relationship: a peak at index offset ``m`` from centre
    along an axis of length ``N`` corresponds to spatial frequency
    ``f = m / (N * voxel_size_um)`` [1/um], i.e. a real-space period of
    ``1/f``. Returns the spacing corresponding to the peak's full 3D
    reciprocal-space offset (its Euclidean norm in frequency space)."""
    offsets = np.array(
        [peak_index[d] - shape[d] // 2 for d in range(3)], dtype=np.float64
    )
    freq_per_voxel = offsets / np.array(shape, dtype=np.float64)  # cycles / voxel, per axis
    freq_per_um = freq_per_voxel / voxel_size_um  # cycles / um, per axis
    freq_mag = float(np.linalg.norm(freq_per_um))
    return float("inf") if freq_mag == 0 else 1.0 / freq_mag
