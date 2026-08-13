"""Per-connected-component probability statistics and confidence-aware gating.

This is a probability-aware companion to :mod:`nnunet_isles.inference.cc_postproc`
that keeps or drops connected components of a binary mask based on the
foreground probabilities that produced it. It powers the "drop low
confidence CCs" and adaptive-size-gate post-processors.

The module deliberately avoids importing from :mod:`cc_postproc` to keep the
import graph acyclic; the structuring-element helper is rebuilt locally with
identical semantics to ``apply_cc_filter``.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import find_objects, label
from scipy.ndimage import maximum as ndi_maximum


def _build_structure(ndim: int, connectivity: int) -> np.ndarray:
    """Structuring element matching cc_postproc.apply_cc_filter semantics."""
    if ndim == 3:
        if connectivity == 26:
            return np.ones((3, 3, 3), dtype=bool)
        if connectivity == 18:
            struct = np.ones((3, 3, 3), dtype=bool)
            for corner in [
                (0, 0, 0),
                (0, 0, 2),
                (0, 2, 0),
                (2, 0, 0),
                (0, 2, 2),
                (2, 0, 2),
                (2, 2, 0),
                (2, 2, 2),
            ]:
                struct[corner] = False
            return struct
        if connectivity == 6:
            struct = np.zeros((3, 3, 3), dtype=bool)
            struct[1, 1, :] = True
            struct[1, :, 1] = True
            struct[:, 1, 1] = True
            return struct
        raise ValueError(f"connectivity must be 6, 18 or 26 for 3D; got {connectivity}")
    # 2D (or other) fallback - full neighbourhood (8-conn in 2D).
    return np.ones((3,) * ndim, dtype=bool)


def _label(mask: np.ndarray, connectivity: int) -> tuple[np.ndarray, int]:
    """Label the CCs of a boolean mask with the requested connectivity."""
    struct = _build_structure(mask.ndim, connectivity)
    labelled, n = label(mask, structure=struct)
    return labelled, int(n)


def cc_probability_stats(
    prob_fg: np.ndarray,
    mask: np.ndarray,
    *,
    connectivity: int = 26,
) -> list[dict]:
    """Return per-connected-component statistics.

    Each dict has:
      - ``label``: int (1..N), the CC id in the labelled array
      - ``voxels``: int, voxel count of the CC
      - ``max_prob``: float, max ``prob_fg`` over the CC's voxels
      - ``mean_prob``: float, mean ``prob_fg`` over the CC's voxels
      - ``prob_mass``: float, sum of ``prob_fg`` over the CC's voxels
      - ``bbox``: tuple[tuple[int, int], ...], per-axis ``(lo, hi_exclusive)``
    Returns ``[]`` when the mask has no positive voxels.
    """
    if prob_fg.shape != mask.shape:
        raise ValueError(f"prob_fg shape {prob_fg.shape} != mask shape {mask.shape}")
    mask_bool = mask.astype(bool) if mask.dtype != bool else mask
    if not mask_bool.any():
        return []

    labelled, n = _label(mask_bool, connectivity)
    if n == 0:
        return []

    prob_flat = prob_fg.astype(np.float64, copy=False).ravel()
    lab_flat = labelled.ravel()

    counts = np.bincount(lab_flat, minlength=n + 1)
    sums = np.bincount(lab_flat, weights=prob_flat, minlength=n + 1)
    # Per-CC maximum via scipy.ndimage.maximum (returns list aligned with index).
    max_vals = ndi_maximum(prob_fg, labels=labelled, index=np.arange(1, n + 1))
    max_arr = np.atleast_1d(np.asarray(max_vals, dtype=np.float64))

    slices = find_objects(labelled)
    stats: list[dict] = []
    for i in range(1, n + 1):
        vox = int(counts[i])
        if vox == 0:
            # Background label 0 is skipped; positive labels are always
            # non-empty from scipy.ndimage.label, so this branch is defensive.
            continue
        sl = slices[i - 1]
        bbox = tuple((int(s.start), int(s.stop)) for s in sl)
        mass = float(sums[i])
        stats.append(
            {
                "label": i,
                "voxels": vox,
                "max_prob": float(max_arr[i - 1]),
                "mean_prob": mass / vox,
                "prob_mass": mass,
                "bbox": bbox,
            }
        )
    return stats


def drop_low_confidence_ccs(
    prob_fg: np.ndarray,
    mask: np.ndarray,
    *,
    min_max_prob: float = 0.0,
    min_mean_prob: float = 0.0,
    min_prob_mass: float = 0.0,
    min_voxels: int = 0,
    connectivity: int = 26,
) -> np.ndarray:
    """Drop CCs failing any of the ``min_*`` thresholds; return uint8 mask.

    - The four ``min_*`` are ANDed: a CC survives iff ALL its stats meet or
      exceed the corresponding ``min_*``.
    - Defaults are all 0 -> identity (mask returned unchanged, as uint8).
    - Empty input mask returns an all-zero uint8 mask (same shape).
    """
    if prob_fg.shape != mask.shape:
        raise ValueError(f"prob_fg shape {prob_fg.shape} != mask shape {mask.shape}")

    mask_bool = mask.astype(bool) if mask.dtype != bool else mask
    if not mask_bool.any():
        return np.zeros(mask.shape, dtype=np.uint8)

    identity = min_max_prob <= 0.0 and min_mean_prob <= 0.0 and min_prob_mass <= 0.0 and min_voxels <= 0
    if identity:
        return mask_bool.astype(np.uint8)

    labelled, n = _label(mask_bool, connectivity)
    if n == 0:
        return np.zeros(mask.shape, dtype=np.uint8)

    prob_flat = prob_fg.astype(np.float64, copy=False).ravel()
    lab_flat = labelled.ravel()

    counts = np.bincount(lab_flat, minlength=n + 1)
    sums = np.bincount(lab_flat, weights=prob_flat, minlength=n + 1)
    max_vals = ndi_maximum(prob_fg, labels=labelled, index=np.arange(1, n + 1))
    max_arr = np.atleast_1d(np.asarray(max_vals, dtype=np.float64))

    keep = np.zeros(n + 1, dtype=bool)
    for i in range(1, n + 1):
        vox = int(counts[i])
        if vox == 0:
            continue
        mass = float(sums[i])
        mean_p = mass / vox
        max_p = float(max_arr[i - 1])
        if vox >= min_voxels and max_p >= min_max_prob and mean_p >= min_mean_prob and mass >= min_prob_mass:
            keep[i] = True

    survivors = keep[labelled]
    return survivors.astype(np.uint8)


def adaptive_size_gate(
    prob_fg: np.ndarray,
    mask: np.ndarray,
    size_curve: list[tuple[float, int]],
    *,
    connectivity: int = 26,
) -> np.ndarray:
    """Piecewise size floor conditioned on per-CC ``max_prob``.

    ``size_curve`` is a list of ``(min_max_prob_threshold, min_voxels_floor)``
    tuples interpreted as breakpoints sorted by ascending threshold. For each
    CC, take the bracket with the highest ``min_max_prob_threshold`` that is
    still ``<= max_prob`` and use its floor. If the CC has fewer voxels than
    that floor, drop it.

    Rationale: high-confidence CCs get a low size floor (kept even if tiny),
    while low-confidence CCs must be larger to survive.

    Empty ``size_curve`` or empty mask -> return input mask cast to uint8.
    """
    if prob_fg.shape != mask.shape:
        raise ValueError(f"prob_fg shape {prob_fg.shape} != mask shape {mask.shape}")

    mask_bool = mask.astype(bool) if mask.dtype != bool else mask
    if not mask_bool.any() or len(size_curve) == 0:
        return mask_bool.astype(np.uint8)

    # Sort breakpoints by ascending probability threshold.
    curve_sorted = sorted(size_curve, key=lambda t: t[0])
    thresholds = np.asarray([t[0] for t in curve_sorted], dtype=np.float64)
    floors = np.asarray([t[1] for t in curve_sorted], dtype=np.int64)

    labelled, n = _label(mask_bool, connectivity)
    if n == 0:
        return np.zeros(mask.shape, dtype=np.uint8)

    lab_flat = labelled.ravel()
    counts = np.bincount(lab_flat, minlength=n + 1)
    max_vals = ndi_maximum(prob_fg, labels=labelled, index=np.arange(1, n + 1))
    max_arr = np.atleast_1d(np.asarray(max_vals, dtype=np.float64))

    keep = np.zeros(n + 1, dtype=bool)
    for i in range(1, n + 1):
        vox = int(counts[i])
        if vox == 0:
            continue
        max_p = float(max_arr[i - 1])
        # Find the highest bracket the CC qualifies for.
        # searchsorted with side='right' returns insertion index; the last
        # bracket with threshold <= max_p is at (idx - 1).
        idx = int(np.searchsorted(thresholds, max_p, side="right")) - 1
        if idx < 0:
            # max_prob is below every threshold -> no bracket applies.
            # Conservative choice: drop the CC (no floor guarantees keeping it).
            continue
        floor = int(floors[idx])
        if vox >= floor:
            keep[i] = True

    survivors = keep[labelled]
    return survivors.astype(np.uint8)


__all__ = [
    "cc_probability_stats",
    "drop_low_confidence_ccs",
    "adaptive_size_gate",
]
