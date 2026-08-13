"""ISLES 2026 metric set.

Definitions follow ISLES'24 precedent (final ranking rules for ISLES'26 to be
locked from the official challenge page in M0). All functions take binary
uint8 numpy arrays (foreground=1, background=0) and a spacing tuple (mm).
"""

from __future__ import annotations

import numpy as np


def _binarize(arr: np.ndarray) -> np.ndarray:
    return (arr > 0).astype(np.uint8)


def dice_coefficient(pred: np.ndarray, gt: np.ndarray, smooth: float = 1.0e-7) -> float:
    """Case-level Dice = 2|A∩B| / (|A|+|B|). Empty/empty returns 1.0."""
    p, g = _binarize(pred), _binarize(gt)
    inter = float(np.logical_and(p, g).sum())
    denom = float(p.sum() + g.sum())
    if denom == 0:
        return 1.0
    return float((2.0 * inter + smooth) / (denom + smooth))


def absolute_volume_difference_ml(
    pred: np.ndarray, gt: np.ndarray, spacing_mm: tuple[float, float, float]
) -> float:
    """|V_pred - V_gt| in millilitres (1 mL = 1000 mm³)."""
    voxel_volume_ml = float(spacing_mm[0] * spacing_mm[1] * spacing_mm[2]) / 1000.0
    v_pred = float(_binarize(pred).sum()) * voxel_volume_ml
    v_gt = float(_binarize(gt).sum()) * voxel_volume_ml
    return abs(v_pred - v_gt)


def _connected_components(mask: np.ndarray):
    from scipy.ndimage import label

    return label(_binarize(mask))  # (labelled, n_components)


def lesion_count_f1(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Count-level F1 + absolute count difference."""
    _, n_pred = _connected_components(pred)
    _, n_gt = _connected_components(gt)
    abs_diff = abs(n_pred - n_gt)
    if n_pred + n_gt == 0:
        return {"count_f1": 1.0, "count_pred": 0, "count_gt": 0, "count_abs_diff": 0}
    f1 = 2 * min(n_pred, n_gt) / (n_pred + n_gt)
    return {
        "count_f1": float(f1),
        "count_pred": int(n_pred),
        "count_gt": int(n_gt),
        "count_abs_diff": int(abs_diff),
    }


def lesion_wise_f1(pred: np.ndarray, gt: np.ndarray, iou_threshold: float = 0.20) -> float:
    """Instance-level F1: TP if max per-component IoU vs any GT component >= threshold."""
    pred_lbl, n_pred = _connected_components(pred)
    gt_lbl, n_gt = _connected_components(gt)
    if n_pred == 0 and n_gt == 0:
        return 1.0
    if n_pred == 0 or n_gt == 0:
        return 0.0

    # Compute component IoU matrix (n_pred x n_gt).
    iou = np.zeros((n_pred, n_gt), dtype=np.float32)
    for i in range(1, n_pred + 1):
        p_mask = pred_lbl == i
        p_area = float(p_mask.sum())
        for j in range(1, n_gt + 1):
            g_mask = gt_lbl == j
            inter = float(np.logical_and(p_mask, g_mask).sum())
            if inter == 0:
                continue
            union = p_area + float(g_mask.sum()) - inter
            iou[i - 1, j - 1] = inter / union if union > 0 else 0.0

    tp = int((iou.max(axis=1) >= iou_threshold).sum()) if iou.size > 0 else 0
    fp = n_pred - tp
    fn = n_gt - int((iou.max(axis=0) >= iou_threshold).sum()) if iou.size > 0 else n_gt
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


def hausdorff_95(
    pred: np.ndarray, gt: np.ndarray, spacing_mm: tuple[float, float, float], percentile: float = 95.0
) -> float:
    """95th-percentile symmetric surface distance in mm.

    Returns float('inf') if either mask is empty (consumer should treat as worst).
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        return float("inf")

    p, g = _binarize(pred), _binarize(gt)
    if p.sum() == 0 or g.sum() == 0:
        return float("inf")

    spacing = np.asarray(spacing_mm, dtype=np.float64)
    # Surface = mask XOR eroded(mask)
    from scipy.ndimage import binary_erosion

    struct = np.ones((3, 3, 3), dtype=np.uint8)
    p_surf = p ^ binary_erosion(p, structure=struct, border_value=0)
    g_surf = g ^ binary_erosion(g, structure=struct, border_value=0)

    dt_g = distance_transform_edt(~g.astype(bool), sampling=spacing)
    dt_p = distance_transform_edt(~p.astype(bool), sampling=spacing)
    d_p_to_g = dt_g[p_surf.astype(bool)]
    d_g_to_p = dt_p[g_surf.astype(bool)]
    if d_p_to_g.size == 0 or d_g_to_p.size == 0:
        return float("inf")
    all_distances = np.concatenate([d_p_to_g, d_g_to_p])
    return float(np.percentile(all_distances, percentile))
