"""The Volume data structure: the single explicit carrier of shape, physical
scale and axis convention. Every downstream stage takes a Volume (or the
graph built from one) rather than a bare ndarray, so voxel size and axis
identity can never be silently lost -- see spec Sec 2.5 / 10.2 (requirements
3 and 4: explicit units, explicit axis mapping).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Physical axis names in the paper's convention: L = longitudinal (polar
#: axis, theta measured from it), C = circumferential, R = radial (omega
#: measured from it). See spec Sec 2.3.
AxisTriple = tuple[str, str, str]


@dataclass
class Volume:
    """A binary (or skeleton) 3D volume with explicit physical metadata.

    Attributes
    ----------
    data:
        ``bool`` array of shape ``(nx, ny, nz)``. Solid phase is ``True``.
    voxel_size:
        Isotropic physical size of one voxel, in micrometres.
    axes:
        Which physical axis (``'L'``, ``'C'``, ``'R'``) each array dimension
        represents. ``axes[0]`` is dim 0 (rows), etc. Orientation code reads
        this instead of hardcoding "dim 1 is longitudinal" as the MATLAB
        code does (spec defect class D8 / Sec 2.3).
    """

    data: np.ndarray
    voxel_size: float = 1.0
    axes: AxisTriple = field(default_factory=lambda: ("L", "C", "R"))

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError(f"Volume.data must be 3D, got shape {self.data.shape}")
        if self.data.dtype != np.bool_:
            self.data = self.data.astype(np.bool_)
        if self.voxel_size <= 0:
            raise ValueError(f"voxel_size must be positive, got {self.voxel_size}")
        if set(self.axes) != {"L", "C", "R"}:
            raise ValueError(f"axes must be a permutation of ('L','C','R'), got {self.axes}")

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def n_voxels(self) -> int:
        return int(self.data.size)

    @property
    def solid_fraction(self) -> float:
        return float(self.data.mean())

    def axis_index(self, physical_axis: str) -> int:
        """Return which array dimension (0, 1 or 2) carries `physical_axis`."""
        return self.axes.index(physical_axis)

    def to_um(self, voxel_length: float) -> float:
        return voxel_length * self.voxel_size

    def with_data(self, new_data: np.ndarray) -> "Volume":
        """Return a copy of this Volume with different voxel data but the
        same physical metadata (used after skeletonization, padding, etc.)."""
        return Volume(data=new_data, voxel_size=self.voxel_size, axes=self.axes)
