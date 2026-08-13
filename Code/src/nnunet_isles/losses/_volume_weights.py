"""Per-sample lesion-volume weight utilities for the bucket-weighted loss.

The weight function up-weights the 0.5-5 mL bucket (the V2 regression
locus) and down-weights very-large lesions where Dice is already
saturated. Designed to be called from `train_step` after the highest-
resolution segmentation target is materialised.

Usage in a trainer:

    from nnunet_isles.losses._volume_weights import compute_per_sample_volume_weight

    voxel_spacing = tuple(float(s) for s in self.configuration_manager.spacing)  # (sz, sy, sx)
    sample_weights = compute_per_sample_volume_weight(target[0], voxel_spacing)
    loss_val = self.loss(seg_output, target, sample_weights=sample_weights)
"""

from __future__ import annotations

import math

import torch


def compute_per_sample_volume_ml(
    seg_highres: torch.Tensor, voxel_spacing_mm: tuple[float, float, float]
) -> torch.Tensor:
    """Return per-sample lesion volume in millilitres.

    seg_highres: (B, 1, *spatial) int/bool/float - the highest-resolution
        ground-truth mask (deep-supervision level 0).
    voxel_spacing_mm: (sz, sy, sx) plans-space spacing - constant per
        experiment, read from `configuration_manager.spacing`.

    Returns: (B,) float tensor on the same device as `seg_highres`.
    """
    if seg_highres.ndim < 4:
        raise ValueError(f"seg_highres must be (B,1,*spatial); got shape {tuple(seg_highres.shape)}")
    voxel_volume_mm3 = float(voxel_spacing_mm[0]) * float(voxel_spacing_mm[1]) * float(voxel_spacing_mm[2])
    # Robust to int / bool / float labels.
    fg = (seg_highres > 0).to(torch.float32)
    # Sum over channel + spatial → per-sample voxel count.
    voxel_counts = fg.flatten(start_dim=1).sum(dim=1)
    return voxel_counts * (voxel_volume_mm3 / 1000.0)


def compute_per_sample_volume_weight(
    seg_highres: torch.Tensor,
    voxel_spacing_mm: tuple[float, float, float],
    *,
    target_ml: float = 5.0,
    w_min: float = 0.5,
    w_max: float = 4.0,
    empty_mask_weight: float = 2.0,
) -> torch.Tensor:
    """Per-sample loss weight = `clip(target_ml / max(vol_ml, eps), w_min, w_max)`.

    Up-weights small lesions, down-weights very large ones. Empty masks
    get `empty_mask_weight` (cap-friendly default 2.0).

    Args:
        seg_highres: (B, 1, *spatial) GT mask.
        voxel_spacing_mm: plans-space spacing tuple.
        target_ml: the volume at which weight == 1.0. Default 5.0 mL -
            puts the centre of the 0.5-5 mL V2-regression bucket at unit weight.
        w_min, w_max: clip range.
        empty_mask_weight: fallback for cases with no foreground.

    Returns: (B,) float tensor.
    """
    vol_ml = compute_per_sample_volume_ml(seg_highres, voxel_spacing_mm)
    # Empty mask = literally zero voxels; tiny-but-nonzero lesions get the
    # clipped-up weight (sub-0.5 mL lesions are the V2 frontier - we want
    # them weighted high, not falling back to `empty_mask_weight`).
    empty = vol_ml == 0
    # Floor the divisor at 0.01 mL = 10 mm³ ≈ 1 voxel at 1mm iso - avoids
    # 0/0 while still letting w_max cap dominate for the smallest CCs.
    safe_vol = vol_ml.clamp(min=0.01)
    weights = (target_ml / safe_vol).clamp(min=w_min, max=w_max)
    weights[empty] = empty_mask_weight
    return weights


def log_volume_for_conditioning(
    seg_highres: torch.Tensor, voxel_spacing_mm: tuple[float, float, float]
) -> torch.Tensor:
    """Log(volume_ml + 1) for diffusion conditioning and similar contexts."""
    vol = compute_per_sample_volume_ml(seg_highres, voxel_spacing_mm)
    return torch.log(vol + 1.0)


def _bucket_volume(vol_ml: float) -> str:
    """Match the V1/V2 EDA's bucket scheme (lesion_buckets_ml=[0.5, 5.0, 50.0])."""
    if vol_ml < 0.5:
        return "<0.5mL"
    if vol_ml < 5.0:
        return "0.5-5mL"
    if vol_ml < 50.0:
        return "5-50mL"
    return ">=50mL"


def bucket_for_volume(vol_ml: float) -> str:
    """Public alias of `_bucket_volume` - used by per-bucket threshold tuning."""
    return _bucket_volume(vol_ml)


__all__ = [
    "compute_per_sample_volume_ml",
    "compute_per_sample_volume_weight",
    "log_volume_for_conditioning",
    "bucket_for_volume",
    "_bucket_volume",
]


# Hide the math import lint warning when only used implicitly in future helpers.
_ = math
