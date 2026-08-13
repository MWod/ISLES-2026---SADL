"""Regression tests for the official ISLES 2026 evaluator wrapper.

Cross-checks against a hand-copied invocation of the challenge's
``utils/eval_utils.py`` from https://github.com/ezequieldlrosa/isles26 -
if these ever drift we want the CI to fail loudly rather than silently
report inflated numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("panoptica")

from nnunet_isles.evaluation.isles26_official import (  # noqa: E402
    compute_absolute_volume_difference,
    compute_dice_f1_instance_difference,
    compute_pr_auc,
    evaluate_case_official,
)


def _box(shape, ranges):
    m = np.zeros(shape, dtype=np.uint8)
    m[tuple(slice(*r) for r in ranges)] = 1
    return m


# ------------------------------------------------------------------ Dice / F1 / LCD


def test_identical_mask_perfect():
    m = _box((30, 30, 30), [(10, 20), (10, 20), (10, 20)])
    f1, lcd, dice = compute_dice_f1_instance_difference(m, m)
    assert dice == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)
    assert lcd == 0


def test_both_empty_returns_empty_value():
    z = np.zeros((30, 30, 30), dtype=np.uint8)
    f1, lcd, dice = compute_dice_f1_instance_difference(z, z)
    assert dice == 1.0
    assert f1 == 1.0
    assert lcd == 0


def test_empty_gt_full_pred_penalises():
    z = np.zeros((30, 30, 30), dtype=np.uint8)
    b = _box((30, 30, 30), [(5, 10), (5, 10), (5, 10)])
    f1, lcd, dice = compute_dice_f1_instance_difference(z, b)
    assert dice == pytest.approx(0.0)
    assert f1 == pytest.approx(0.0)
    assert lcd == 1


def test_two_gt_one_pred_matches_one():
    """RQ = 2 * matched / (2*matched + FP + FN) = 2*1/(2+0+1) = 2/3."""
    gt = _box((30, 30, 30), [(5, 10), (5, 10), (5, 10)])
    gt[20:24, 20:24, 20:24] = 1
    pr = _box((30, 30, 30), [(5, 10), (5, 10), (5, 10)])
    f1, lcd, dice = compute_dice_f1_instance_difference(gt, pr)
    assert f1 == pytest.approx(2.0 / 3.0)
    assert lcd == 1


def test_matching_threshold_is_0p25():
    """Below-threshold IoU (0.20) must produce a miss, not a hit."""
    gt = _box((40, 40, 40), [(0, 20), (0, 20), (0, 20)])
    # A pred that overlaps only a 4-wide corner: IoU ≈ 0.004 - nowhere near 0.25.
    pr = _box((40, 40, 40), [(18, 22), (18, 22), (18, 22)])
    f1, lcd, dice = compute_dice_f1_instance_difference(gt, pr)
    assert f1 == pytest.approx(0.0)
    # Panoptica's global_bin_dsc is computed over *matched* instances so with
    # zero matches it collapses to 0.0 even though voxel-wise Dice > 0. That
    # semantics is the point of the assertion - this is a matcher metric, not
    # a pixel-overlap one.
    assert dice == pytest.approx(0.0)


# ------------------------------------------------------------------ Volume diff


def test_volume_diff_matches_hand_math():
    gt = _box((10, 10, 10), [(0, 10), (0, 10), (0, 10)])  # 1000 vox
    pr = _box((10, 10, 10), [(0, 5), (0, 10), (0, 10)])  #  500 vox
    voxel_ml = 1.0 / 1000.0  # 1 mm^3
    assert compute_absolute_volume_difference(gt, pr, voxel_ml) == pytest.approx(0.5)


def test_volume_diff_symmetric():
    a = _box((10, 10, 10), [(0, 10), (0, 10), (0, 10)])
    b = _box((10, 10, 10), [(0, 5), (0, 10), (0, 10)])
    v = 2.5e-3
    assert compute_absolute_volume_difference(a, b, v) == compute_absolute_volume_difference(b, a, v)


# ------------------------------------------------------------------ PR-AUC


def test_pr_auc_perfect_soft_map():
    gt = _box((16, 16, 16), [(4, 12), (4, 12), (4, 12)])
    prob = gt.astype(np.float32)  # perfect probability map
    assert compute_pr_auc(gt, prob) == pytest.approx(1.0, abs=1e-6)


def test_pr_auc_empty_gt_uniform_pred_returns_empty_value():
    gt = np.zeros((10, 10, 10), dtype=np.uint8)
    prob = np.full_like(gt, fill_value=0.3, dtype=np.float32)
    assert compute_pr_auc(gt, prob) == 1.0


def test_pr_auc_shape_mismatch_raises():
    gt = np.zeros((5, 5, 5), dtype=np.uint8)
    prob = np.zeros((5, 5, 6), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_pr_auc(gt, prob)


# ------------------------------------------------------------------ bundle


def test_evaluate_case_official_returns_stable_schema():
    gt = _box((16, 16, 16), [(4, 12), (4, 12), (4, 12)])
    pr = gt.copy()
    prob = gt.astype(np.float32)
    m = evaluate_case_official("sub-x", gt, pr, prob, voxel_size_ml=1.0e-3)
    d = m.as_dict()
    assert set(d.keys()) == {
        "subject_id",
        "dice",
        "lesion_f1",
        "abs_lesion_count_diff",
        "abs_volume_diff_ml",
        "pr_auc",
        "lesion_volume_ml_gt",
        "lesion_volume_ml_pred",
        "bucket_ml",
    }
    assert d["dice"] == pytest.approx(1.0)
    assert d["lesion_f1"] == pytest.approx(1.0)
    assert d["abs_lesion_count_diff"] == 0
    assert d["abs_volume_diff_ml"] == pytest.approx(0.0)
    assert d["pr_auc"] == pytest.approx(1.0, abs=1e-6)
    assert d["bucket_ml"] == "0.5-5ml"  # 8^3 vox * 1e-3 mL = 0.512 mL
