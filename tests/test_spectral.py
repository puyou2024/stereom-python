import networkx as nx
import numpy as np

from stereom.spectral import node_fft, peak_index_to_spacing_um


def _periodic_node_graph(shape, spacing: int) -> nx.MultiGraph:
    G = nx.MultiGraph()
    nid = 0
    for i in range(spacing // 2, shape[0], spacing):
        for j in range(spacing // 2, shape[1], spacing):
            for k in range(spacing // 2, shape[2], spacing):
                G.add_node(nid, pos=(i, j, k), voxel_degree=-1)
                nid += 1
    return G


def test_fft_spectrum_is_normalized_and_finite():
    shape = (32, 32, 32)
    G = _periodic_node_graph(shape, spacing=8)
    spectrum = node_fft(G, shape, window=True)
    assert spectrum.shape == shape
    assert np.all(np.isfinite(spectrum))
    assert spectrum.min() >= 0.0
    assert np.isclose(spectrum.max(), 1.0)


def test_peak_index_to_spacing_matches_known_periodicity():
    # A perfectly periodic lattice's FFT is itself a Dirac comb: EVERY
    # harmonic (all integer multiples of the fundamental frequency) is an
    # equally strong peak, not just one. The fundamental (smallest
    # nonzero offset from DC) is the one that corresponds to the real-
    # space spacing; higher harmonics tie at the same magnitude and must
    # not be picked instead.
    shape = (64, 64, 64)
    spacing_voxels = 8
    G = _periodic_node_graph(shape, spacing=spacing_voxels)
    spectrum = node_fft(G, shape, window=False)

    axis_profile = spectrum[:, shape[1] // 2, shape[2] // 2]
    center = shape[0] // 2
    candidates = np.where(axis_profile > 0.9)[0]
    candidates = candidates[candidates != center]
    fundamental_idx = int(candidates[np.argmin(np.abs(candidates - center))])

    spacing_um = peak_index_to_spacing_um(
        (fundamental_idx, shape[1] // 2, shape[2] // 2), shape, voxel_size_um=1.0
    )
    assert np.isclose(spacing_um, spacing_voxels, rtol=0.2)
