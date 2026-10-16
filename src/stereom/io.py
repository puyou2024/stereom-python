"""Volume I/O. Replaces Sec 3.1 of the spec (``01_main_dataProcessing.m``
Sec 1): reads a numbered TIFF stack, but preallocated, stored as ``bool``,
and without the divide-by-zero trap of min-max rescaling a binary stack
(spec Sec 3.1 failure modes, param table #2).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import tifffile

from stereom.volume import AxisTriple, Volume


def load_volume(
    folder: str | Path,
    *,
    start_page: int = 0,
    end_page: int | None = None,
    sig_digits: int = 4,
    image_name: str = "",
    style: str = ".tif",
    voxel_size_um: float = 0.65,
    axes: AxisTriple = ("L", "C", "R"),
    threshold: float | None = None,
) -> Volume:
    """Load a numbered TIFF slice stack into a :class:`Volume`.

    Mirrors the naming convention of the original ``01_main_dataProcessing.m``
    (``imageName`` + zero-padded index + ``style``), but reads the whole
    stack in one vectorized call, preallocates, and stores ``bool`` instead
    of ``double`` (8x smaller than uint8, 64x smaller than double -- spec
    Sec 3.1). No min-max rescale is performed: since the input is expected
    to already be a binary segmentation, we threshold instead, which cannot
    divide by zero on a constant slice the way the original min-max step
    could.

    Parameters
    ----------
    threshold:
        Pixel value above which a voxel counts as solid. ``None`` (default)
        auto-selects the midpoint between the stack's min and max value,
        which reproduces the original code's behaviour for a genuinely
        binary (0/255 or 0/1) input.
    """
    folder = Path(folder)
    if end_page is None:
        pattern = re.compile(rf"^{re.escape(image_name)}(\d{{{sig_digits}}}){re.escape(style)}$")
        indices = sorted(
            int(m.group(1)) for p in folder.iterdir() if (m := pattern.match(p.name))
        )
        if not indices:
            raise FileNotFoundError(f"No slices matching pattern found in {folder}")
        start_page, end_page = indices[0], indices[-1]

    paths = [
        folder / f"{image_name}{page:0{sig_digits}d}{style}"
        for page in range(start_page, end_page + 1)
    ]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} slice(s) missing, e.g. {missing[0]}")

    first = tifffile.imread(paths[0])
    stack = np.empty((first.shape[0], first.shape[1], len(paths)), dtype=first.dtype)
    stack[:, :, 0] = first
    for k, p in enumerate(paths[1:], start=1):
        stack[:, :, k] = tifffile.imread(p)

    if threshold is None:
        lo, hi = float(stack.min()), float(stack.max())
        threshold = (lo + hi) / 2.0 if hi > lo else lo
    data = stack > threshold

    return Volume(data=data, voxel_size=voxel_size_um, axes=axes)


def load_volume_array(
    array: np.ndarray,
    *,
    voxel_size_um: float = 0.65,
    axes: AxisTriple = ("L", "C", "R"),
    threshold: float | None = None,
) -> Volume:
    """Build a :class:`Volume` directly from an in-memory 3D array (e.g. a
    single multi-page TIFF, a `.npy` array, or a zarr/dask chunk already
    materialised to memory)."""
    if array.dtype == np.bool_:
        data = array
    else:
        if threshold is None:
            lo, hi = float(array.min()), float(array.max())
            threshold = (lo + hi) / 2.0 if hi > lo else lo
        data = array > threshold
    return Volume(data=data, voxel_size=voxel_size_um, axes=axes)


def save_volume_tiff(volume: Volume, folder: str | Path, *, sig_digits: int = 4, style: str = ".tif") -> None:
    """Write a Volume back out as a numbered TIFF stack (dim 2 = slice)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    data_u8 = (volume.data.astype(np.uint8)) * 255
    for k in range(data_u8.shape[2]):
        tifffile.imwrite(folder / f"{k:0{sig_digits}d}{style}", data_u8[:, :, k])


def save_volume_npz(volume: Volume, path: str | Path) -> None:
    path = Path(path)
    np.savez_compressed(
        path, data=volume.data, voxel_size=volume.voxel_size, axes=np.array(volume.axes)
    )


def load_volume_npz(path: str | Path) -> Volume:
    with np.load(path) as z:
        return Volume(
            data=z["data"].astype(np.bool_),
            voxel_size=float(z["voxel_size"]),
            axes=tuple(str(a) for a in z["axes"]),
        )
