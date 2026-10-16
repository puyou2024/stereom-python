"""All tunable parameters for the pipeline, in one place.

This is the direct answer to spec Sec 10.2 requirement 6 ("all parameters in
one config object, defaulting to the Sec 4 table, with derivation helpers").
Every "magic constant" flagged in the spec's defect register and parameter
registry (Sec 4-5) is a field here, not a literal buried in analysis code.

Values default to whatever the shipped MATLAB code's *effective* behaviour
was (documented per-field), not necessarily what a literal reading of its
call sites suggests -- e.g. the ``open`` flag bug (D17) meant the passed
``merge_thres=10`` was never actually used; the effective default merge
distance for the demo volume was 12 voxels, so that is what is reproduced
here, with a clear note to override it via ``merge_d_m`` for real work.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np

DistanceOrFn = Union[float, Callable[[np.ndarray], np.ndarray]]


@dataclass
class StereomConfig:
    # ------------------------------------------------------------------
    # Volume / units (spec Sec 10.2 req. 3-4)
    # ------------------------------------------------------------------
    voxel_size_um: float = 0.65
    """Physical size of one (isotropic) voxel, in micrometres. Param #1."""

    axes: tuple[str, str, str] = ("L", "C", "R")
    """Which array dimension is Longitudinal / Circumferential / Radial.
    MUST be set correctly for your volume's orientation (spec Sec 2.3) --
    the original code hardcoded dim 1 = L, dim 3 = R with no way to change
    it; getting this wrong silently corrupts every orientation statistic."""

    # ------------------------------------------------------------------
    # Node identification (spec Sec 3.3)
    # ------------------------------------------------------------------
    node_degree_threshold: int = 4
    """Param #3. A skeleton voxel is a node candidate if its 27-neighbourhood
    (including itself) contains >= this many skeleton voxels. Matches the
    paper's "more than two neighbouring voxels" criterion. Leave alone
    unless you have a specific reason to change it."""

    # ------------------------------------------------------------------
    # Graph cleaning (spec Sec 3.6)
    # ------------------------------------------------------------------
    degree2_guard_voxels: Optional[float] = 5.0
    """Param #5, first pass. When dissolving a degree-2 node (splicing its
    two incident branches into one), only do so if the SHORTER of the two
    incident branches has <= this many voxels. ``None`` disables the guard
    (dissolve unconditionally) -- used for the second, unguarded pass in
    the reference pipeline (see ``pipeline.clean_graph``)."""

    merge_d_m_um: Optional[float] = None
    """Param #6/#7, the real node-clustering distance ``d_m``, in
    micrometres. ``None`` => auto-derive as the mean strut radius from an
    EDT of the volume (spec Sec 4 derivation rule for #7). The MATLAB
    code's ``open=1`` branch silently overrode the passed value with a
    position-dependent ladder (12/10/8 voxels, defect D17); that ladder is
    NOT reproduced here -- pass ``merge_d_m_um`` as a callable of a node's
    physical position (array of shape (N,3) in voxel coords -> array of N
    distances in voxels) if you need a graded-material threshold."""

    merge_touch_um: Optional[float] = None
    """Distance for the first ("directly touching") merge pass, replacing
    ``merge_connectednodes``. ``None`` => derived as
    ``voxel_size_um * sqrt(3)`` (26-neighbour touching distance)."""

    # ------------------------------------------------------------------
    # Boundary exclusion (spec Sec 3.7)
    # ------------------------------------------------------------------
    boundary_limit_um: Optional[float] = None
    """Param #10. Nodes/branches with any voxel within this distance of a
    volume face are excluded from *statistics* (not from the saved graph --
    keep that separation, spec Sec 3.7 says it is the right design).
    ``None`` => derived as one mean branch length after a first pass."""

    # ------------------------------------------------------------------
    # Thickness (spec Sec 3.11)
    # ------------------------------------------------------------------
    thickness_min_path_voxels: int = 10
    """Param #17. Branches with fewer path voxels than this get no
    thickness profile (matches original; EDT method makes this less
    necessary but it is kept for statistical comparability)."""

    # ------------------------------------------------------------------
    # Profile fitting (spec Sec 3.12)
    # ------------------------------------------------------------------
    fit_max_abs_z_um: float = 25.0
    fit_len_min_um: float = 2.0
    fit_len_max_um: float = 35.0
    """Param #18. Branch acceptance window for profile fitting. ``None``
    on any of these three enables auto-derivation from the branch length
    distribution (mean +/- 3 SD) instead -- see
    ``pipeline.derive_fit_limits``."""

    fit_include_linear_term: bool = False
    """If True, fit t(z) = a0 + a1*z + b*z^2 (Paper I Eq. 3 as published).
    Default False reproduces the shipped code's actual fit (pure quadratic,
    no linear term) -- see spec Sec 3.12."""

    # ------------------------------------------------------------------
    # Inter-branch angles / rings (spec Sec 3.14, 3.16)
    # ------------------------------------------------------------------
    ring_max_size: int = 7
    """Param #21. Maximum ring size (number of branches) to search for.
    Structural, not a tunable in the original (unrolled to depth 7)."""

    ring_flatness_r2_min: float = 0.5
    """Param #24. Minimum PCA-based planarity R^2 for a ring to be kept.
    Not implemented in the shipped MATLAB release (present only as
    commented-out code); implemented here per spec Sec 3.16."""

    ring_plane_residual_max_um: Optional[float] = None
    """Optional additional filter on max out-of-plane residual (um) of a
    ring's branch voxels, mirroring the ``max(errors) > 5`` (voxel) guard
    in ``main_ring.m``. ``None`` disables it (the R^2 filter above is the
    principled replacement)."""

    # ------------------------------------------------------------------
    # Chains (spec Sec 3.17)
    # ------------------------------------------------------------------
    chain_cos_threshold: float = 0.5
    """Minimum |cos(angle)| between successive branch chord vectors for
    greedy chain growth to continue (replaces the broken monotone +
    fixed-axis threshold rule, spec Sec 3.17 refactor note). 0.5 ~= 60 deg,
    matching the original gate's angular scale."""

    # ------------------------------------------------------------------
    # FFT (spec Sec 3.18)
    # ------------------------------------------------------------------
    fft_window: bool = True
    """Apply a Tukey window before the node-array FFT (fixes the "no
    windowing -> sinc sidelobes" issue noted in spec Sec 3.18)."""

    fft_display_threshold: float = 0.5

    # ------------------------------------------------------------------
    # Performance / HPC
    # ------------------------------------------------------------------
    n_workers: int = -1
    """Worker processes for embarrassingly-parallel stages (ring search
    across seed nodes, batch processing of independent volumes). -1 =
    all available cores (joblib convention)."""

    random_seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        return d

    def save(self, path: str | Path) -> None:
        path = Path(path)
        d = self.to_dict()
        if path.suffix in (".yaml", ".yml"):
            import yaml  # optional dependency

            path.write_text(yaml.safe_dump(d, sort_keys=False))
        else:
            path.write_text(json.dumps(d, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "StereomConfig":
        path = Path(path)
        text = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            import yaml

            d = yaml.safe_load(text)
        else:
            d = json.loads(text)
        d["axes"] = tuple(d["axes"])
        return cls(**d)


def d_m_from_radius(mean_radius_um: float, factor: float = 1.0) -> float:
    """Derivation rule for parameter #7 (spec Sec 4): set the node-merge
    distance to (a multiple of) the mean strut radius, measured from an EDT
    of the segmented volume."""
    return mean_radius_um * factor


def boundary_from_length(mean_branch_length_um: float, factor: float = 1.0) -> float:
    """Derivation rule for parameter #10 (spec Sec 4)."""
    return mean_branch_length_um * factor
