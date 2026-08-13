"""Connected-component post-processing for the Pillar-1 inference stack.

Reference: Liu et al., arXiv:2408.02929 ("Segmenting Small Stroke Lesions"),
which reports +1.3% Dice and +2.4% Lesion-F1 on ATLAS R2.0 from CC-based
post-processing.

Two operations:
  * `apply_cc_filter` - drops CCs smaller than `min_voxels`.
  * `apply_lesion_size_suppression` - scales each CC's mask probability by
    a per-bucket suppress factor (Liu MAPPING-style learnt curve).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label


def apply_cc_filter(pred: np.ndarray, min_voxels: int, *, connectivity: int = 26) -> np.ndarray:
    """Drop connected components smaller than `min_voxels`.

    Args:
        pred: integer/bool ndarray (any shape) - binary mask.
        min_voxels: lower threshold; CCs strictly smaller than this are removed.
        connectivity: 6 / 18 / 26 for 3D. Default 26 (face+edge+corner).

    Returns:
        ndarray of same shape and dtype as `pred`.
    """
    pred_bool = pred.astype(bool) if pred.dtype != bool else pred
    if pred_bool.sum() == 0 or min_voxels <= 1:
        return pred.copy()

    # Build scipy structuring element for the requested connectivity.
    if pred_bool.ndim == 3:
        if connectivity == 26:
            struct = np.ones((3, 3, 3), dtype=bool)
        elif connectivity == 18:
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
        elif connectivity == 6:
            struct = np.zeros((3, 3, 3), dtype=bool)
            struct[1, 1, :] = True
            struct[1, :, 1] = True
            struct[:, 1, 1] = True
        else:
            raise ValueError(f"connectivity must be 6, 18 or 26; got {connectivity}")
    else:
        # 2D fallback - 8-connectivity for any non-3D input.
        struct = np.ones((3,) * pred_bool.ndim, dtype=bool)

    labelled, n = label(pred_bool, structure=struct)
    if n == 0:
        return pred.copy()
    # Vectorised: compute size of each CC, mask out small ones.
    sizes = np.bincount(labelled.ravel())
    keep = sizes >= min_voxels
    keep[0] = False  # background
    valid_mask = keep[labelled]
    out = pred.copy()
    out[~valid_mask] = 0
    return out


def apply_lesion_size_suppression(
    pred: np.ndarray,
    voxel_volume_mm3: float,
    suppress_curve: dict[str, float],
    *,
    connectivity: int = 26,
) -> np.ndarray:
    """Per-CC probabilistic suppression based on the CC's volume bucket.

    The curve maps the bucket name (e.g. "<0.5mL") to a multiplier in [0, 1]
    applied to that CC's voxels in the output mask. With a hard {0,1} mask
    input, multiplier 0 fully drops the CC, multiplier 1 keeps it.

    Args:
        pred: binary uint8/bool mask.
        voxel_volume_mm3: physical voxel volume (read from plans / nifti header).
        suppress_curve: e.g. `{"<0.5mL": 0.3, "0.5-5mL": 1.0, "5-50mL": 1.0, ">=50mL": 1.0}`.

    Returns:
        Mask with each CC scaled by its suppress factor. With a {0,1} input,
        the output is also {0,1} after rounding (CC kept iff suppress >= 0.5).
    """
    from nnunet_isles.losses._volume_weights import bucket_for_volume

    pred_bool = pred.astype(bool)
    if pred_bool.sum() == 0:
        return pred.copy()

    struct = (np.ones((3, 3, 3), dtype=bool) if connectivity == 26 else None) if pred_bool.ndim == 3 else None
    labelled, n = label(pred_bool, structure=struct)
    if n == 0:
        return pred.copy()
    out = pred.copy()
    for i in range(1, n + 1):
        cc = labelled == i
        vol_ml = cc.sum() * voxel_volume_mm3 / 1000.0
        bucket = bucket_for_volume(vol_ml)
        suppress = float(suppress_curve.get(bucket, 1.0))
        if suppress < 0.5:
            out[cc] = 0
    return out


__all__ = ["apply_cc_filter", "apply_lesion_size_suppression"]
