"""Branch profile fitting. Replaces ``main_branchfitting.m`` (buggy -- calls
``vecdist`` with the wrong arity, D5) and its corrected duplicate
``General/main_profile.m`` (spec Sec 3.12).

Three things the spec settles about the published papers, all reproduced
here deliberately:

1. The coordinate ``z`` is anchored at the thickness MINIMUM (the "waist"),
   not the branch midpoint -- Paper II's convention, already present in
   the Paper I code.
2. Normalization is by ``t_min``, not midpoint thickness -- also Paper II's
   convention.
3. The fitted quadratic has NO linear term by default
   (``StereomConfig.fit_include_linear_term=False``), matching the shipped
   code's actual fit rather than the published Eq. (3). Set that flag to
   True to reproduce Eq. (3) instead.

The Paper II quartic model (``t/t_min = (a*(z/L)^2 + 1)^2``) is provided
alongside the quadratic since it is not present anywhere in this
repository and must be written from the paper description (spec Sec 1.3,
8 row 10).
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.optimize import curve_fit

from stereom.config import StereomConfig


@dataclass
class ProfileFitResult:
    branch_id: int
    accepted: bool
    reject_reason: str | None
    a0: float = float("nan")
    a1: float | None = None
    b: float = float("nan")
    r2: float = float("nan")
    t_min_um: float = float("nan")
    length_um: float = float("nan")
    quartic_a: float | None = None
    z_um: np.ndarray | None = None
    t_um: np.ndarray | None = None


def _arclength_um(path: np.ndarray, voxel_size_um: float) -> np.ndarray:
    path_f = path.astype(np.float64)
    if path_f.shape[0] < 2:
        return np.zeros(path_f.shape[0])
    d = np.linalg.norm(np.diff(path_f, axis=0), axis=1) * voxel_size_um
    return np.concatenate([[0.0], np.cumsum(d)])


def _fit_quadratic(
    z: np.ndarray, t: np.ndarray, include_linear: bool
) -> tuple[float, float | None, float, float]:
    if include_linear:
        A = np.column_stack([np.ones_like(z), z, z**2])
    else:
        A = np.column_stack([np.ones_like(z), z**2])
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((t - pred) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if include_linear:
        a0, a1, b = (float(c) for c in coef)
    else:
        a0, b = (float(c) for c in coef)
        a1 = None
    return a0, a1, b, r2


def _fit_quartic(z_over_l: np.ndarray, t_over_tmin: np.ndarray) -> float | None:
    def model(x: np.ndarray, a: float) -> np.ndarray:
        return (a * x**2 + 1.0) ** 2

    try:
        popt, _ = curve_fit(model, z_over_l, t_over_tmin, p0=[1.0], maxfev=2000)
        return float(popt[0])
    except Exception:
        return None


def fit_profiles(
    G: nx.MultiGraph,
    voxel_size_um: float,
    config: StereomConfig,
    *,
    profile_attr: str = "profile_radius_um",
    index_attr: str | None = None,
) -> list[ProfileFitResult]:
    """Fit every branch's radius profile to a waist-anchored quadratic
    (and, where possible, the Paper II quartic), applying the acceptance
    filters of spec Sec 3.12 / parameter #18.

    Parameters
    ----------
    profile_attr:
        Edge attribute holding the (already-um) radius profile. Use
        ``"profile_radius_um"`` for the default EDT method or
        ``"profile_radius_legacy_um"`` for the legacy surface method.
    index_attr:
        Edge attribute holding the path-index of each profile sample
        (needed for the legacy method, whose samples are a subset of the
        path with omitted points -- see ``thickness.legacy_mean_radius_profile``).
        ``None`` (default) assumes a dense profile aligned 1:1 with ``path``.
    """
    results: list[ProfileFitResult] = []
    for u, v, k, data in G.edges(keys=True, data=True):
        branch_id = data.get("id", k)
        profile = data.get(profile_attr)
        length_um = data.get("length_um", float("nan"))

        if profile is None or len(profile) == 0:
            results.append(ProfileFitResult(branch_id, False, "no_profile"))
            continue

        path = data["path"]
        arclen = _arclength_um(path, voxel_size_um)
        if index_attr is not None:
            idx = data.get(index_attr)
            if idx is None or len(idx) == 0:
                results.append(ProfileFitResult(branch_id, False, "no_profile"))
                continue
            sample_arclen = arclen[idx]
        else:
            sample_arclen = arclen[: len(profile)]

        t = np.asarray(profile, dtype=np.float64)
        valid = ~np.isnan(t)
        t, sample_arclen = t[valid], sample_arclen[valid]

        if t.size < 2:
            results.append(ProfileFitResult(branch_id, False, "too_few_samples"))
            continue

        a_idx = int(np.argmin(t))
        z = sample_arclen - sample_arclen[a_idx]
        max_abs_z = float(np.max(np.abs(z)))

        if (
            max_abs_z > config.fit_max_abs_z_um
            or length_um > config.fit_len_max_um
            or length_um < config.fit_len_min_um
        ):
            results.append(ProfileFitResult(branch_id, False, "out_of_length_window"))
            continue

        a0, a1, b, r2 = _fit_quadratic(z, t, config.fit_include_linear_term)
        t_min = float(t.min())
        quartic_a = None
        if length_um > 0 and t_min > 0:
            quartic_a = _fit_quartic(z / length_um, t / t_min)

        results.append(
            ProfileFitResult(
                branch_id=branch_id,
                accepted=True,
                reject_reason=None,
                a0=a0,
                a1=a1,
                b=b,
                r2=r2,
                t_min_um=t_min,
                length_um=length_um,
                quartic_a=quartic_a,
                z_um=z,
                t_um=t,
            )
        )
    return results


def pooled_normalized_arrays(
    results: list[ProfileFitResult],
) -> tuple[np.ndarray, np.ndarray]:
    """Pool accepted branches' (z/L, t/t_min) points across the whole
    dataset -- the normalized abscissa spans roughly [-1, 1] because the
    waist can sit anywhere along the branch (spec Sec 3.12 point 3,
    resolving the factor-of-two ambiguity in Paper I's Eq. 3: it is the
    full length, not the half length)."""
    zs, ts = [], []
    for r in results:
        if not r.accepted or r.length_um <= 0 or r.t_min_um <= 0:
            continue
        zs.append(r.z_um / r.length_um)
        ts.append(r.t_um / r.t_min_um)
    if not zs:
        return np.array([]), np.array([])
    return np.concatenate(zs), np.concatenate(ts)


def derive_fit_limits(
    length_um_values: np.ndarray, n_sd: float = 3.0
) -> tuple[float, float]:
    """Derivation rule for parameter #18 (spec Sec 4): reject beyond
    mean +/- n_sd standard deviations of the branch-length distribution,
    instead of the fixed 2/35 um literals, when ``config.fit_len_min_um``/
    ``fit_len_max_um`` are left at their defaults for a dataset with a very
    different length scale."""
    mean = float(np.mean(length_um_values))
    sd = float(np.std(length_um_values))
    return max(0.0, mean - n_sd * sd), mean + n_sd * sd
