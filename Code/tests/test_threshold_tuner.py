"""Tests for threshold + temperature tuning."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from nnunet_isles.inference.threshold_tuner import (
    apply_temperature,
    dice_at,
    fit_per_bucket_threshold,
    load_softmax_npz,
    sweep_threshold,
)


def _make_softmax_npz(path: Path, prob_fg: np.ndarray) -> None:
    """Save a 2-channel softmax NPZ (bg, fg) in nnU-Net's expected format."""
    full = np.stack([1.0 - prob_fg, prob_fg], axis=0).astype(np.float32)
    np.savez(str(path), probabilities=full)


def _make_gt(path: Path, mask: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), np.eye(4)), str(path))


def test_load_softmax_npz_returns_foreground(tmp_path: Path):
    prob = np.array([[[0.1, 0.7], [0.8, 0.3]]], dtype=np.float32)  # shape (1, 2, 2)
    _make_softmax_npz(tmp_path / "case.npz", prob)
    fg = load_softmax_npz(tmp_path / "case.npz")
    np.testing.assert_array_almost_equal(fg, prob)


def test_apply_temperature_identity_at_T1():
    prob = np.array([0.1, 0.5, 0.9])
    out = apply_temperature(prob, 1.0)
    np.testing.assert_array_almost_equal(out, prob)


def test_apply_temperature_smooths_extreme_probs():
    """T>1 makes probs less confident (closer to 0.5)."""
    prob = np.array([0.9, 0.99])
    out = apply_temperature(prob, 2.0)
    assert (out < prob).all()
    assert (out > 0.5).all()


def test_dice_at_perfect_match():
    fg = np.array([[1.0, 0.0], [0.0, 1.0]])
    gt = np.array([[1, 0], [0, 1]])
    assert dice_at(fg, gt, 0.5) == 1.0


def test_dice_at_no_overlap():
    fg = np.array([[1.0, 0.0], [0.0, 0.0]])
    gt = np.array([[0, 1], [1, 0]])
    assert dice_at(fg, gt, 0.5) == 0.0


def test_sweep_threshold_finds_optimum(tmp_path: Path):
    """Construct a softmax where the optimum threshold is 0.30 (foreground
    probability at lesion voxels is 0.35; threshold > 0.35 misses; lower
    keeps the lesion)."""
    prob = np.zeros((10, 10, 10), dtype=np.float32)
    prob[2:5, 2:5, 2:5] = 0.35
    gt = np.zeros((10, 10, 10), dtype=np.uint8)
    gt[2:5, 2:5, 2:5] = 1

    npz_path = tmp_path / "case.npz"
    gt_path = tmp_path / "case.nii.gz"
    _make_softmax_npz(npz_path, prob)
    _make_gt(gt_path, gt)

    scores = sweep_threshold(
        [npz_path],
        [gt_path],
        candidates=(0.20, 0.30, 0.40, 0.50),
    )
    # At t=0.20 or 0.30 we capture the lesion (Dice=1); at t=0.40 or 0.50 we miss it.
    assert scores[0.30] == 1.0
    assert scores[0.50] == 0.0


def test_per_bucket_threshold_can_differ(tmp_path: Path):
    """Per-bucket fitter produces DIFFERENT thresholds when the optimal
    threshold differs across buckets.

    Small-bucket case: GT has a 2-voxel lesion at prob=0.30; threshold MUST be
    < 0.30 to capture it (Dice 1.0). At threshold 0.50, the lesion is missed
    and Dice drops to 0.
    Large-bucket case: GT has a 4×4×4 lesion at prob=0.45 surrounded by
    spurious prob=0.25 noise; threshold MUST be > 0.25 to suppress noise."""
    # Small lesion: prob 0.30 in lesion, 0 elsewhere.
    prob_small = np.zeros((10, 10, 10), dtype=np.float32)
    prob_small[0, 0, 0:2] = 0.30
    gt_small = np.zeros((10, 10, 10), dtype=np.uint8)
    gt_small[0, 0, 0:2] = 1

    # Large lesion: prob 0.45 in lesion, 0.25 spurious noise outside.
    prob_large = np.zeros((10, 10, 10), dtype=np.float32)
    prob_large[:] = 0.25
    prob_large[1:5, 1:5, 1:5] = 0.45
    gt_large = np.zeros((10, 10, 10), dtype=np.uint8)
    gt_large[1:5, 1:5, 1:5] = 1

    n_s = tmp_path / "small.npz"
    n_l = tmp_path / "large.npz"
    g_s = tmp_path / "small.nii.gz"
    g_l = tmp_path / "large.nii.gz"
    _make_softmax_npz(n_s, prob_small)
    _make_softmax_npz(n_l, prob_large)
    _make_gt(g_s, gt_small)
    _make_gt(g_l, gt_large)

    case_buckets = {"small": "<0.5mL", "large": "5-50mL"}
    out = fit_per_bucket_threshold(
        [n_s, n_l],
        [g_s, g_l],
        case_buckets,
        candidates=(0.20, 0.275, 0.35, 0.50),
    )
    # Small: 0.20 or 0.275 capture the lesion at prob 0.30 → Dice 1.0
    # Large: 0.275 or 0.35 suppress the 0.25 noise → Dice 1.0; 0.50 misses the 0.45 lesion → Dice 0
    assert out["<0.5mL"] < 0.30
    assert out["5-50mL"] > 0.25
    # Both buckets got fitted thresholds.
    assert set(out) == {"<0.5mL", "5-50mL"}


def test_sweep_returns_zero_dice_for_empty_pair_lists():
    out = sweep_threshold([], [])
    # No cases → 0.0 mean for every threshold.
    assert all(v == 0.0 for v in out.values())
