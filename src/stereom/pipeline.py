"""End-to-end orchestration, matching the reference pipeline's stage order
(spec Sec 11 "quick-reference call graph") while running each stage
through the fixed/vectorized implementation in its own module. No
plotting happens here or anywhere in this module (spec Sec 10.2
requirement 1); see ``viz.py`` for optional, separate visualization.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from stereom import angles, chains, geometry, graph_build, graph_clean, orientation, planarity
from stereom import profile_fit, rings, skeleton, spectral, tables, thickness
from stereom.config import StereomConfig, d_m_from_radius
from stereom.graph_build import GraphBuildStats
from stereom.volume import Volume


@dataclass
class PipelineResult:
    volume: Volume
    skeleton: np.ndarray
    graph: nx.MultiGraph
    build_stats: GraphBuildStats
    merge_d_m_um: float
    nodes_df: pd.DataFrame
    branches_df: pd.DataFrame
    profile_fits: list[profile_fit.ProfileFitResult]
    angles_df: pd.DataFrame
    planarity_df: pd.DataFrame
    rings_df: pd.DataFrame | None = None
    chains_df: pd.DataFrame | None = None
    fft_spectrum: np.ndarray | None = None


def clean_graph(
    G: nx.MultiGraph, config: StereomConfig, merge_d_m_um: float
) -> nx.MultiGraph:
    """Node merging + degree-2 dissolution, in the reference pipeline's
    order (spec Sec 3.6 / Sec 11): touching-scale merge, guarded
    dissolution, real-d_m merge, unguarded dissolution."""
    touch_um = config.merge_touch_um or (config.voxel_size_um * np.sqrt(3))
    G = graph_clean.merge_clusters(G, touch_um / config.voxel_size_um)
    G = graph_clean.dissolve_degree2(G, config.degree2_guard_voxels)
    G = graph_clean.merge_clusters(G, merge_d_m_um / config.voxel_size_um)
    G = graph_clean.dissolve_degree2(G, None)
    return G


def compute_descriptors(
    G: nx.MultiGraph, volume: Volume, edt_um: np.ndarray, config: StereomConfig
) -> nx.MultiGraph:
    """Path ordering -> lengths -> parallel-edge dedup -> lengths again
    (spec Sec 3.10 "note the ordering dependency") -> thickness ->
    orientation."""
    G = geometry.order_paths(G)
    G = geometry.compute_lengths(G, config.voxel_size_um)
    G = graph_clean.drop_parallel_edges(G)
    G = geometry.compute_lengths(G, config.voxel_size_um)
    G = thickness.sample_profiles_edt(G, edt_um, config.thickness_min_path_voxels)
    G = orientation.compute_orientations(G, config.axes)
    return G


def run_full_pipeline(
    volume: Volume,
    config: StereomConfig,
    *,
    compute_rings: bool = True,
    compute_chains: bool = True,
    compute_fft: bool = True,
) -> PipelineResult:
    """Run the complete pipeline on a single (already in-memory) volume.
    For volumes too large to fit in memory, or too large to process
    single-threaded in reasonable time, see ``pipeline.run_chunked``."""
    if not np.isclose(config.voxel_size_um, volume.voxel_size, rtol=1e-9):
        import warnings

        warnings.warn(
            f"config.voxel_size_um ({config.voxel_size_um}) does not match "
            f"volume.voxel_size ({volume.voxel_size}) -- using the volume's own "
            "value for every physical-unit conversion in this run (a Volume "
            "loaded from a file carries its own authoritative voxel size; "
            "StereomConfig.voxel_size_um is only a convenience default for "
            "io.load_volume on a raw TIFF stack that has none). Pass "
            "voxel_size_um=volume.voxel_size explicitly when constructing "
            "StereomConfig if you want this to be silent.",
            stacklevel=2,
        )
    config = dataclasses.replace(config, voxel_size_um=volume.voxel_size)

    skel = skeleton.skeletonize_volume(volume)
    G, build_stats = graph_build.build_graph(skel, config.node_degree_threshold)

    edt_um = thickness.compute_edt(volume)

    if config.merge_d_m_um is None:
        # Sample the EDT AT SKELETON (medial-axis) voxels, not over the
        # whole solid phase. The two differ by roughly 2x in practice: most
        # solid voxels sit near a surface (small EDT), while skeleton
        # voxels sit at local EDT ridges close to the true local strut
        # radius -- which is what "mean strut radius" (spec Sec 4's
        # derivation rule for this parameter) actually means. Using the
        # whole-solid-phase mean systematically undershoots d_m.
        skeleton_radii = edt_um[skel]
        mean_radius_um = (
            float(skeleton_radii.mean()) if skeleton_radii.size else config.voxel_size_um
        )
        merge_d_m_um = d_m_from_radius(mean_radius_um)
    else:
        merge_d_m_um = config.merge_d_m_um

    G = clean_graph(G, config, merge_d_m_um)
    G = compute_descriptors(G, volume, edt_um, config)

    profile_fits = profile_fit.fit_profiles(G, config.voxel_size_um, config)
    angles_df = angles.inter_branch_angles(G)
    planarity_df = planarity.planarity_table(G)

    rings_df = None
    if compute_rings:
        rings_df = rings.analyze_rings(
            G,
            config.voxel_size_um,
            max_size=config.ring_max_size,
            flatness_r2_min=config.ring_flatness_r2_min,
            plane_residual_max_um=config.ring_plane_residual_max_um,
            n_jobs=config.n_workers,
        )

    chains_df = None
    if compute_chains:
        chains_df = chains.greedy_chains(G, config.chain_cos_threshold)

    fft_spectrum = None
    if compute_fft:
        fft_spectrum = spectral.node_fft(G, volume.shape, window=config.fft_window)

    nodes_df = tables.nodes_to_dataframe(G)
    branches_df = tables.branches_to_dataframe(G)

    return PipelineResult(
        volume=volume,
        skeleton=skel,
        graph=G,
        build_stats=build_stats,
        merge_d_m_um=merge_d_m_um,
        nodes_df=nodes_df,
        branches_df=branches_df,
        profile_fits=profile_fits,
        angles_df=angles_df,
        planarity_df=planarity_df,
        rings_df=rings_df,
        chains_df=chains_df,
        fft_spectrum=fft_spectrum,
    )


def run_batch(
    volumes: dict[str, Volume], config: StereomConfig, *, n_jobs: int = -1, **kwargs
) -> dict[str, PipelineResult]:
    """Run ``run_full_pipeline`` over several independent volumes (e.g. one
    per specimen / replicate / imaging session) in parallel across
    processes. This is the natural HPC entry point for the common case of
    "many separate volumes to analyse", as opposed to one volume too large
    to fit in memory -- see the module note below for that case.

    Ring detection's own internal parallelism (``config.n_workers``) is
    disabled inside each worker (set to 1) to avoid oversubscribing CPUs
    when multiple volumes are already running in parallel; the outer
    ``n_jobs`` is where parallelism should be spent in that case.
    """
    from joblib import Parallel, delayed

    inner_config = StereomConfig(**{**config.to_dict(), "n_workers": 1})

    def _one(name: str, vol: Volume) -> tuple[str, PipelineResult]:
        return name, run_full_pipeline(vol, inner_config, **kwargs)

    results = Parallel(n_jobs=n_jobs)(delayed(_one)(name, vol) for name, vol in volumes.items())
    return dict(results)


# NOTE on volumes too large for a single machine / single pass (spec Sec 7
# "Scaling to Paper II volumes"): this module does not implement the
# overlapping-block chunk-and-stitch driver the spec sketches (split into
# overlapping sub-volumes, build a graph per chunk, match nodes in the
# overlap region). A correct stitcher has to handle branches truncated at
# a chunk boundary and avoid double-counting rings/chains that cross one,
# which is real additional design work, not a thin wrapper over what is
# here -- shipping a version that silently mishandles those cases would be
# worse than not shipping it. ``graph_clean.merge_clusters`` (KD-tree +
# union-find over node positions) is the building block a stitcher would
# reuse to merge nodes across chunk overlaps; ``run_batch`` above is the
# correct, complete tool for the *different* (and, for most users, more
# common) case of many independent volumes.


def derive_boundary_limit_um(branches_df: pd.DataFrame) -> float:
    """Derivation rule for parameter #10 (spec Sec 4): one mean branch
    length, from a first descriptor pass."""
    lengths = branches_df["length_um"].dropna()
    return float(lengths.mean()) if len(lengths) else 0.0
