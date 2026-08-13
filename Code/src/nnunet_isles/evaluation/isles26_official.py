"""Official ISLES 2026 evaluator (wrapper).

Verbatim re-implementation of ``utils/eval_utils.py`` from the challenge
repository at https://github.com/ezequieldlrosa/isles26 (commit fetched
2026-07-28), packaged as a Python module so we can call it from CLIs and
tests without a git submodule.

The metric definitions are:

* **Dice** - ``panoptica.result.global_bin_dsc`` (semantic Dice over the
  full binary volume).
* **Lesion-wise F1 (RQ)** - ``panoptica.result.rq`` under
  :class:`~panoptica.NaiveThresholdMatching` with ``matching_threshold=0.25``.
* **Absolute Lesion Count Difference** -
  ``abs(num_ref_instances - num_pred_instances)`` with connected components
  from :class:`~panoptica.ConnectedComponentsInstanceApproximator`.
* **Absolute Volume Difference (mL)** - ``|V_gt - V_pred|`` with the
  voxel volume in millilitres (i.e. ``prod(spacing_mm) / 1000``).
* **PR-AUC** - voxel-wise precision-recall AUC from
  :func:`sklearn.metrics.precision_recall_curve`, taking the raw soft map
  as input (no thresholding).

We wrap them in :func:`evaluate_case_official` which returns a plain dict
with a stable column order, plus a Pillar-1 mL bucket label so we can
break the leaderboard down the same way we already do internally.

DO NOT copy this module into the submission Docker - Panoptica is a
scoring-side dependency only.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


_EMPTY_VALUE = 1.0
_MATCHING_IOU = 0.25


# ------------------------------------------------------------------ Panoptica evaluator singleton
# Panoptica prints a banner on first import; wrapping the imports here
# keeps callers clean and lets us build the evaluator lazily so tests that
# never touch panoptica don't pay the import cost.

_EVALUATOR = None


def _get_evaluator():
    global _EVALUATOR
    if _EVALUATOR is None:
        from panoptica import (
            ConnectedComponentsInstanceApproximator,
            InputType,
            NaiveThresholdMatching,
            Panoptica_Evaluator,
        )

        _EVALUATOR = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(matching_threshold=_MATCHING_IOU),
        )
    return _EVALUATOR


# ------------------------------------------------------------------ individual metrics


def compute_pr_auc(
    ground_truth: ArrayLike,
    prediction_map: ArrayLike,
    empty_value: float = _EMPTY_VALUE,
) -> float:
    """PR-AUC from a soft prediction map against a binary GT.

    Verbatim from the challenge's :file:`utils/eval_utils.py`. The soft map
    need not be bounded in ``[0, 1]``.
    """
    from sklearn.metrics import auc, precision_recall_curve

    gt_flat = np.asarray(ground_truth).astype(np.bool_).ravel()
    pred_flat = np.asarray(prediction_map).astype(np.float32).ravel()
    if gt_flat.shape != pred_flat.shape:
        raise ValueError("Shape mismatch: ground_truth and prediction_map must match.")

    if int(gt_flat.sum()) == 0:
        # All-negative GT: reward flat / trivial predictions with `empty_value`,
        # otherwise precision is 0 by definition (all positive predictions
        # are false).
        if np.all(pred_flat == pred_flat[0]):
            return float(empty_value)
        return 0.0

    try:
        precision, recall, _ = precision_recall_curve(gt_flat, pred_flat)
        return float(auc(recall, precision))
    except ValueError:
        return float("nan")


def compute_absolute_volume_difference(
    im1: ArrayLike,
    im2: ArrayLike,
    voxel_size: ArrayLike,
) -> float:
    """|V_1 - V_2| in millilitres. ``voxel_size`` is the per-voxel mL scalar."""
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)
    voxel_size = np.asarray(voxel_size).astype(float)

    if im1.shape != im2.shape:
        warnings.warn(
            "Shape mismatch: ground_truth and prediction have difference shapes."
            " The absolute volume difference is computed with mismatching shape masks",
            stacklevel=2,
        )
    return float(abs(np.sum(im1) * voxel_size - np.sum(im2) * voxel_size))


def compute_dice_f1_instance_difference(
    ground_truth: ArrayLike,
    prediction: ArrayLike,
    empty_value: float = _EMPTY_VALUE,
) -> tuple[float, int, float]:
    """Panoptica-based Dice + lesion-wise F1 + absolute lesion count diff.

    Returns ``(f1_score, instance_count_difference, dice_score)`` - same
    order as the challenge's :func:`compute_dice_f1_instance_difference`.
    """
    gt_int = np.asarray(ground_truth).astype(int)
    pr_int = np.asarray(prediction).astype(int)
    result, _ = _get_evaluator().evaluate(pr_int, gt_int, verbose=False)["ungrouped"]

    lcd = int(abs(result.num_ref_instances - result.num_pred_instances))
    if result.num_ref_instances == 0 and result.num_pred_instances == 0:
        return float(empty_value), lcd, float(empty_value)
    return float(result.rq), lcd, float(result.global_bin_dsc)


# ------------------------------------------------------------------ per-case bundle


@dataclass
class OfficialCaseMetrics:
    """Structured result of a single-case scoring pass."""

    subject_id: str
    dice: float
    lesion_f1: float
    abs_lesion_count_diff: int
    abs_volume_diff_ml: float
    pr_auc: float
    lesion_volume_ml_gt: float
    lesion_volume_ml_pred: float
    bucket_ml: str

    def as_dict(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "dice": self.dice,
            "lesion_f1": self.lesion_f1,
            "abs_lesion_count_diff": self.abs_lesion_count_diff,
            "abs_volume_diff_ml": self.abs_volume_diff_ml,
            "pr_auc": self.pr_auc,
            "lesion_volume_ml_gt": self.lesion_volume_ml_gt,
            "lesion_volume_ml_pred": self.lesion_volume_ml_pred,
            "bucket_ml": self.bucket_ml,
        }


def _bucket_ml(vol_ml: float) -> str:
    """Return the Pillar-1 lesion-size bucket string for ``vol_ml``."""
    if vol_ml < 0.5:
        return "<0.5ml"
    if vol_ml < 5.0:
        return "0.5-5ml"
    if vol_ml < 50.0:
        return "5-50ml"
    return ">=50ml"


def evaluate_case_official(
    subject_id: str,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    soft_map: np.ndarray | None,
    voxel_size_ml: float,
) -> OfficialCaseMetrics:
    """Run all five official metrics on a single case.

    Parameters
    ----------
    subject_id
        Session identifier (e.g. ``sub-r018s001_ses-1``); pass-through label
        for the returned record.
    gt_mask, pred_mask
        Same-shape ``uint8`` / bool arrays. Voxel-aligned; no resampling here.
    soft_map
        Foreground probability map matching ``gt_mask.shape``. If ``None`` we
        report ``pr_auc = NaN`` (the case can still contribute to Dice /
        Lesion-F1 / volume metrics).
    voxel_size_ml
        Physical volume of one voxel in millilitres.
    """
    lesion_f1, lcd, dice = compute_dice_f1_instance_difference(gt_mask, pred_mask)
    abs_vdiff_ml = compute_absolute_volume_difference(gt_mask, pred_mask, np.asarray(voxel_size_ml))
    pr_auc = compute_pr_auc(gt_mask, soft_map) if soft_map is not None else float("nan")

    vol_gt_ml = float(np.sum(gt_mask > 0)) * voxel_size_ml
    vol_pr_ml = float(np.sum(pred_mask > 0)) * voxel_size_ml
    bucket = _bucket_ml(vol_gt_ml)

    return OfficialCaseMetrics(
        subject_id=subject_id,
        dice=dice,
        lesion_f1=lesion_f1,
        abs_lesion_count_diff=lcd,
        abs_volume_diff_ml=abs_vdiff_ml,
        pr_auc=pr_auc,
        lesion_volume_ml_gt=vol_gt_ml,
        lesion_volume_ml_pred=vol_pr_ml,
        bucket_ml=bucket,
    )


__all__ = [
    "OfficialCaseMetrics",
    "compute_absolute_volume_difference",
    "compute_dice_f1_instance_difference",
    "compute_pr_auc",
    "evaluate_case_official",
]
