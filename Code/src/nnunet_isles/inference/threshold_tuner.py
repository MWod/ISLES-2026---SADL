"""Sigmoid threshold + temperature scaling for the Pillar-1 inference stack.

Operates on per-case NPZ softmax files saved by `finalize.py --save-softmax`.
The NPZ files are written by nnU-Net's `predict_from_files(save_probabilities=True)`
in plans space (1 mm iso) with key `probabilities` (shape `(2, *spatial)`).

Two modes:
  * Global: one threshold across all val cases.
  * Per-bucket: separate threshold per lesion-size bucket (uses the predicted
    CC volume at threshold 0.5 to assign each case to a bucket - bootstrap;
    not GT-leaking).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_softmax_npz(path: Path) -> np.ndarray:
    """Return foreground probability (single channel, shape `(*spatial,)`).

    nnU-Net's predict_from_files saves the full softmax under key
    `probabilities` shape `(C=num_classes, *spatial)`. For binary lesion
    segmentation, channel 1 is the foreground.
    """
    with np.load(str(path)) as f:
        prob = f["probabilities"]
    if prob.ndim < 3:
        raise ValueError(f"unexpected softmax shape {prob.shape} in {path}")
    if prob.shape[0] >= 2:
        return prob[1]
    return prob[0]


def apply_temperature(prob_fg: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to a foreground probability map.

    For 2-class softmax `p_fg`, the logit is `log(p_fg / (1 - p_fg))`.
    Scaling by 1/T and reapplying sigmoid gives the calibrated foreground prob.
    """
    if abs(temperature - 1.0) < 1e-6:
        return prob_fg
    eps = 1e-7
    logit = np.log(np.clip(prob_fg, eps, 1.0 - eps) / np.clip(1.0 - prob_fg, eps, 1.0 - eps))
    return 1.0 / (1.0 + np.exp(-logit / temperature))


def dice_at(prob_fg: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    pred = (prob_fg > threshold).astype(np.uint8)
    p = pred > 0
    g = gt > 0
    denom = p.sum() + g.sum()
    return float(2 * (p & g).sum() / denom) if denom > 0 else 1.0


def sweep_threshold(
    softmax_paths: list[Path],
    gt_paths: list[Path],
    *,
    candidates: tuple[float, ...] = tuple(np.arange(0.20, 0.65, 0.025)),
    temperature: float = 1.0,
) -> dict[float, float]:
    """Return {threshold: mean dice over the val set} for each candidate."""
    if len(softmax_paths) != len(gt_paths):
        raise ValueError("softmax_paths and gt_paths must be aligned 1:1")

    # Load all softmax + GT once, in memory (small val sets).
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for sp, gp in zip(softmax_paths, gt_paths, strict=True):
        prob = load_softmax_npz(sp)
        gt = _load_gt(gp)
        if prob.shape != gt.shape:
            # nnU-Net's softmax is in plans space; GT may have been resampled
            # already to match. Skip mismatches with a warning.
            continue
        if temperature != 1.0:
            prob = apply_temperature(prob, temperature)
        pairs.append((prob, gt))

    out = {}
    for t in candidates:
        scores = [dice_at(p, g, float(t)) for p, g in pairs]
        out[float(t)] = float(np.mean(scores)) if scores else 0.0
    return out


def _load_gt(path: Path) -> np.ndarray:
    import SimpleITK as sitk

    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.uint8)


def fit_per_bucket_threshold(
    softmax_paths: list[Path],
    gt_paths: list[Path],
    case_buckets: dict[str, str],
    *,
    candidates: tuple[float, ...] = tuple(np.arange(0.20, 0.65, 0.025)),
) -> dict[str, float]:
    """Fit a separate threshold per lesion-size bucket.

    `case_buckets`: {session_id: bucket_name}.
    """
    from collections import defaultdict

    per_bucket_pairs: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for sp, gp in zip(softmax_paths, gt_paths, strict=True):
        sid = gp.name.replace(".nii.gz", "")
        bucket = case_buckets.get(sid, "unknown")
        prob = load_softmax_npz(sp)
        gt = _load_gt(gp)
        if prob.shape != gt.shape:
            continue
        per_bucket_pairs[bucket].append((prob, gt))

    out: dict[str, float] = {}
    for bucket, pairs in per_bucket_pairs.items():
        best_t, best_dice = 0.5, -1.0
        for t in candidates:
            scores = [dice_at(p, g, float(t)) for p, g in pairs]
            mean = float(np.mean(scores))
            if mean > best_dice:
                best_dice = mean
                best_t = float(t)
        out[bucket] = best_t
    return out


def save_sweep_result(
    out_path: Path,
    *,
    candidates: list[float],
    scores: dict[float, float],
    best_threshold: float,
    temperature: float,
    per_bucket: dict[str, float] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "candidates": candidates,
        "scores": {f"{k:.3f}": v for k, v in scores.items()},
        "best_threshold": best_threshold,
        "temperature": temperature,
    }
    if per_bucket is not None:
        payload["per_bucket"] = per_bucket
    out_path.write_text(json.dumps(payload, indent=2))


__all__ = [
    "load_softmax_npz",
    "apply_temperature",
    "dice_at",
    "sweep_threshold",
    "fit_per_bucket_threshold",
    "save_sweep_result",
]
