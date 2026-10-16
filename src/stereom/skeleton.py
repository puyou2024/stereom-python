"""Skeletonization. Replaces the entire ``toolbox/skeleton/`` directory
(``Skeleton3D.m`` + 5 LUT/simple-point helpers, ~875 lines, 20% of the
original repo) with a single call into scikit-image's implementation of the
*same published algorithm* (Lee, Kashyap & Chu 1994) -- spec Sec 1.2/3.2/8
row 2. This is a like-for-like substitution, not a reimplementation.

The only behaviour explicitly worth preserving from the original is the
zero-padding before thinning, to avoid edge-of-volume artefacts (spec
Sec 1.2, 3.2 refactor note). We do that here explicitly.
"""

from __future__ import annotations

import numpy as np
from skimage.morphology import skeletonize

from stereom.volume import Volume


def skeletonize_volume(volume: Volume, *, pad: int = 1) -> np.ndarray:
    """Return a boolean skeleton array, same shape as ``volume.data``.

    Parameters
    ----------
    pad:
        Voxels of zero-padding applied on every side before thinning, then
        cropped back off. Matches ``Skeleton3D.m``'s behaviour of padding
        by 1 voxel to avoid boundary distortion of the thinning result.
    """
    if pad > 0:
        padded = np.pad(volume.data, pad, mode="constant", constant_values=False)
    else:
        padded = volume.data

    skel = skeletonize(padded, method="lee")

    if pad > 0:
        sl = tuple(slice(pad, -pad) for _ in range(3))
        skel = skel[sl]

    return skel.astype(np.bool_)
