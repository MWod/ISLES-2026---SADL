"""Per-case metric computation runner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nnunet_isles.evaluation.metrics import (
    absolute_volume_difference_ml,
    dice_coefficient,
    hausdorff_95,
    lesion_count_f1,
    lesion_wise_f1,
)


@dataclass
class CaseMetrics:
    session_id: str
    site: str
    lesion_bucket: str | None
    dice: float
    hd95: float
    avd_ml: float
    lesion_f1: float
    count_f1: float
    count_pred: int
    count_gt: int
    count_abs_diff: int

    def to_row(self) -> dict[str, float | str | int | None]:
        return {
            "session_id": self.session_id,
            "site": self.site,
            "lesion_bucket": self.lesion_bucket,
            "dice": self.dice,
            "hd95": self.hd95,
            "avd_ml": self.avd_ml,
            "lesion_f1": self.lesion_f1,
            "count_f1": self.count_f1,
            "count_pred": self.count_pred,
            "count_gt": self.count_gt,
            "count_abs_diff": self.count_abs_diff,
        }


def compute_case_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing_mm: tuple[float, float, float],
    session_id: str,
    site: str,
    lesion_bucket: str | None = None,
    iou_threshold: float = 0.20,
) -> CaseMetrics:
    count = lesion_count_f1(pred, gt)
    return CaseMetrics(
        session_id=session_id,
        site=site,
        lesion_bucket=lesion_bucket,
        dice=dice_coefficient(pred, gt),
        hd95=hausdorff_95(pred, gt, spacing_mm),
        avd_ml=absolute_volume_difference_ml(pred, gt, spacing_mm),
        lesion_f1=lesion_wise_f1(pred, gt, iou_threshold=iou_threshold),
        count_f1=count["count_f1"],
        count_pred=count["count_pred"],
        count_gt=count["count_gt"],
        count_abs_diff=count["count_abs_diff"],
    )
