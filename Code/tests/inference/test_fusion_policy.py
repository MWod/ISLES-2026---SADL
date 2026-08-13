"""Tests for `apply_policy` and `mean_prob_weighted` in `fusion.py`.

These wire the `DecisionPolicy` into the fusion + threshold + CC-cleanup
pipeline, and exercise the never-empty rescue branch. Regressions here mean
the applier no longer honours the on-disk policy contract.
"""

from __future__ import annotations

import numpy as np
import pytest
from nnunet_isles.inference import fusion
from nnunet_isles.inference.policy import DecisionPolicy

# ---------------------------------------------------------------------------
# mean_prob_weighted
# ---------------------------------------------------------------------------


def test_mean_prob_weighted_single_dominant_weight_returns_member_zero():
    """weights=[1, 0, 0] must return member 0 (clipped)."""
    rng = np.random.default_rng(0)
    arr = rng.random((3, 4, 4, 4)).astype(np.float32)  # already in [0, 1)
    out = fusion.mean_prob_weighted(arr, weights=[1.0, 0.0, 0.0])
    assert out.shape == (4, 4, 4)
    assert np.allclose(out, np.clip(arr[0], 0.0, 1.0))


def test_mean_prob_weighted_uniform_matches_clipped_mean():
    """weights=None must equal the arithmetic mean of the clipped members."""
    arr = np.array(
        [
            np.full((2, 2, 2), 0.2, dtype=np.float32),
            np.full((2, 2, 2), 0.5, dtype=np.float32),
            np.full((2, 2, 2), 0.8, dtype=np.float32),
        ]
    )
    out = fusion.mean_prob_weighted(arr, weights=None)
    expected = np.clip(arr, 0.0, 1.0).mean(axis=0)
    assert np.allclose(out, expected)
    assert np.allclose(out, 0.5)


def test_mean_prob_weighted_clips_out_of_range_members_before_averaging():
    """Values above 1 / below 0 must be clipped before the weighted sum."""
    arr = np.array(
        [
            np.full((2, 2, 2), 1.5, dtype=np.float32),  # will clip to 1.0
            np.full((2, 2, 2), -0.3, dtype=np.float32),  # will clip to 0.0
        ]
    )
    out = fusion.mean_prob_weighted(arr, weights=None)
    # (1.0 + 0.0) / 2 = 0.5 everywhere.
    assert np.allclose(out, 0.5)


def test_mean_prob_weighted_rejects_negative_weight():
    arr = np.zeros((2, 3, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="weights"):
        fusion.mean_prob_weighted(arr, weights=[1.0, -0.1])


def test_mean_prob_weighted_rejects_wrong_length_weight_vector():
    arr = np.zeros((3, 3, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="weights"):
        fusion.mean_prob_weighted(arr, weights=[1.0, 0.0])


# ---------------------------------------------------------------------------
# apply_policy
# ---------------------------------------------------------------------------


def _flat_bucket_policy(**overrides) -> DecisionPolicy:
    """Default policy with a single flat bucket (threshold 0.5) - a clean
    baseline that lets tests toggle just the fields they care about.
    """
    defaults = {
        "mode": "mean",
        "weights": [1.0, 0.0, 0.0],
        "bucket_edges_ml": [],
        "threshold_by_bucket": [0.5],
        "min_voxels_by_bucket": [0],
        "min_max_prob": 0.0,
        "min_mean_prob": 0.0,
        "min_prob_mass": 0.0,
        "never_empty": False,
        "rescue_min_prob": 0.10,
        "connectivity": 26,
        "voxel_volume_ml": 0.001,
    }
    defaults.update(overrides)
    return DecisionPolicy(**defaults)


def test_apply_policy_single_dominant_weight_matches_apply_decision_layer():
    """weights=[1, 0, 0] + min_max_prob=0 + never_empty=False must reduce to
    the raw thresholded output of member 0."""
    rng = np.random.default_rng(42)
    stack = rng.random((3, 16, 16, 16)).astype(np.float32)

    policy = _flat_bucket_policy(weights=[1.0, 0.0, 0.0], never_empty=False)
    got = fusion.apply_policy(stack, policy)
    expected = fusion.apply_decision_layer(stack[0], mode="mean", threshold=0.5)
    assert got.dtype == np.uint8
    assert got.shape == (16, 16, 16)
    np.testing.assert_array_equal(got, expected)


def test_apply_policy_all_zero_stack_never_empty_rescues():
    """An all-zero stack with never_empty=True must still return a non-empty mask."""
    stack = np.zeros((3, 8, 8, 8), dtype=np.float32)
    policy = _flat_bucket_policy(never_empty=True)
    mask = fusion.apply_policy(stack, policy)
    assert mask.dtype == np.uint8
    assert mask.sum() > 0  # rescue fired


def test_apply_policy_all_zero_stack_never_empty_off_returns_empty():
    """Sanity check: with never_empty=False, an all-zero stack stays empty."""
    stack = np.zeros((3, 8, 8, 8), dtype=np.float32)
    policy = _flat_bucket_policy(never_empty=False)
    mask = fusion.apply_policy(stack, policy)
    assert mask.sum() == 0


def test_apply_policy_min_max_prob_removes_low_confidence_cc_keeps_high():
    """CC-A (max 0.7) vs CC-B (max 0.95); min_max_prob=0.9 keeps only CC-B."""
    shape = (16, 16, 16)
    fg = np.zeros(shape, dtype=np.float32)

    # CC-A: small blob peaking at 0.70 (survives 0.5 threshold, fails 0.9 gate).
    fg[2:4, 2:4, 2:4] = 0.7
    # CC-B: separate blob peaking at 0.95 (well past both gates).
    fg[10:13, 10:13, 10:13] = 0.95

    # Single-map input path (ndim == 3), which exercises the same FP-cleanup branch.
    policy = _flat_bucket_policy(min_max_prob=0.9, never_empty=False)
    mask = fusion.apply_policy(fg, policy)

    assert mask[2:4, 2:4, 2:4].sum() == 0  # CC-A dropped
    assert int(mask[10:13, 10:13, 10:13].sum()) == 3 * 3 * 3  # CC-B kept in full


def test_apply_policy_adaptive_threshold_bucket_zero_drops_to_lower_threshold():
    """Adaptive threshold: pred_vol_ml=0.03 lands in bucket 0 (<0.5 mL),
    so the re-binarise pass uses threshold 0.30 instead of the nominal 0.5.
    """
    shape = (16, 16, 16)
    fg = np.zeros(shape, dtype=np.float32)

    # 30 voxels at 0.7: these count at *both* nominal 0.5 and adaptive 0.3.
    fg[0, 0, :30] = 0.7
    # 20 voxels at 0.4: only counted after the drop to threshold 0.3.
    fg[0, 1, :20] = 0.4

    # 30 voxels * 0.001 mL/vox = 0.03 mL < 0.5 mL -> bucket 0 -> threshold 0.30.
    policy = _flat_bucket_policy(
        bucket_edges_ml=[0.5, 5.0, 50.0],
        threshold_by_bucket=[0.30, 0.35, 0.40, 0.45],
        min_voxels_by_bucket=[0, 0, 0, 0],
        never_empty=False,
        voxel_volume_ml=0.001,
    )
    mask = fusion.apply_policy(fg, policy)

    # Manual application of the bucket-0 threshold (0.30) - with all FP gates
    # at zero this is exactly what apply_policy should produce.
    expected = fusion.threshold_mask(fg, 0.30)
    # Allow a tiny drift for future FP-cleanup churn (per contract).
    drift = int(np.abs(mask.astype(int) - expected.astype(int)).sum())
    assert drift <= 2, f"adaptive-threshold mask drifted by {drift} voxels from manual 0.30 threshold"
    # And qualitatively: the mask is bigger than the nominal-0.5 mask, because
    # the adaptive threshold lets the 0.4-valued voxels in.
    nominal = fusion.threshold_mask(fg, 0.5)
    assert int(mask.sum()) > int(nominal.sum())


def test_apply_policy_returns_uint8_mask():
    """The return type must be uint8, regardless of the input dtype."""
    stack = np.zeros((3, 8, 8, 8), dtype=np.float64)  # double, not float32
    policy = _flat_bucket_policy(never_empty=True)
    mask = fusion.apply_policy(stack, policy)
    assert mask.dtype == np.uint8


def test_apply_policy_noisy_or_stack_recovers_minority_signal():
    """noisy_or mode: one strong member at one voxel is preserved end-to-end."""
    stack = np.zeros((3, 8, 8, 8), dtype=np.float32)
    stack[0, 3, 3, 3] = 0.95  # single confident member at one voxel
    policy = _flat_bucket_policy(mode="noisy_or", weights=None, never_empty=False)
    mask = fusion.apply_policy(stack, policy)
    assert mask[3, 3, 3] == 1


def test_apply_policy_k_of_n_returns_mask_and_skips_cleanup():
    """k_of_n short-circuits: k=2 keeps a voxel with 2 votes, drops single-vote voxels."""
    stack = np.zeros((3, 8, 8, 8), dtype=np.float32)
    stack[0, 1, 1, 1] = 0.9
    stack[1, 1, 1, 1] = 0.9  # 2 votes at (1,1,1)
    stack[2, 4, 4, 4] = 0.9  # only 1 vote at (4,4,4)

    policy = _flat_bucket_policy(
        mode="k_of_n",
        k=2,
        member_threshold=0.5,
        never_empty=False,
    )
    mask = fusion.apply_policy(stack, policy)
    assert mask.dtype == np.uint8
    assert mask[1, 1, 1] == 1
    assert mask[4, 4, 4] == 0


def test_apply_policy_k_of_n_honours_fp_gates():
    """k_of_n must still route through the policy's FP gates.

    Build a small 3-voxel CC that fires above threshold in >= k of N members
    (so it survives the k_of_n vote), then set ``min_voxels_by_bucket`` high
    enough that the CC is dropped by :func:`drop_low_confidence_ccs`.

    * never_empty=False -> the surviving mask must be empty.
    * never_empty=True  -> the empty mask must be rescued from the weighted-mean
      fg field, so the return is non-empty.
    """
    stack = np.zeros((3, 8, 8, 8), dtype=np.float32)
    # 3-voxel connected component, all members firing at 0.9.
    for m in range(3):
        stack[m, 2, 2, 2] = 0.9
        stack[m, 2, 2, 3] = 0.9
        stack[m, 2, 2, 4] = 0.9
    # Sanity: k=2 vote must accept these three voxels pre-gating.
    pre_gate = fusion.k_of_n_mask(stack, k=2, member_threshold=0.5)
    assert int(pre_gate.sum()) == 3

    # min_voxels=100 is well above the CC's 3-voxel size -> gate must drop it.
    policy_no_rescue = _flat_bucket_policy(
        mode="k_of_n",
        k=2,
        member_threshold=0.5,
        weights=[1.0, 1.0, 1.0],
        min_voxels_by_bucket=[100],
        never_empty=False,
    )
    mask = fusion.apply_policy(stack, policy_no_rescue)
    assert mask.dtype == np.uint8
    assert int(mask.sum()) == 0, "min_voxels gate did not drop the small CC in k_of_n mode"

    # Same policy but never_empty=True -> rescue kicks in from the weighted mean.
    policy_rescue = _flat_bucket_policy(
        mode="k_of_n",
        k=2,
        member_threshold=0.5,
        weights=[1.0, 1.0, 1.0],
        min_voxels_by_bucket=[100],
        never_empty=True,
    )
    rescued = fusion.apply_policy(stack, policy_rescue)
    assert rescued.dtype == np.uint8
    assert int(rescued.sum()) > 0, "never_empty rescue must fire from the weighted-mean fg"


def test_largest_prob_component_connectivity():
    """Corner-diagonal voxel pair merges under connectivity=26 but not under 6."""
    prob = np.zeros((5, 5, 5), dtype=np.float32)
    # Two voxels sharing only a corner (all three indices differ by 1).
    prob[1, 1, 1] = 0.9
    prob[2, 2, 2] = 0.9

    mask_26 = fusion.largest_prob_component(prob, min_prob=0.1, connectivity=26)
    mask_6 = fusion.largest_prob_component(prob, min_prob=0.1, connectivity=6)

    # 26-connectivity merges the pair -> the single CC of 2 voxels wins.
    assert int(mask_26.sum()) == 2
    assert mask_26[1, 1, 1] == 1 and mask_26[2, 2, 2] == 1

    # 6-connectivity keeps them as two separate 1-voxel CCs of equal mass;
    # largest_prob_component returns exactly one of them.
    assert int(mask_6.sum()) == 1
    assert (mask_6[1, 1, 1] == 1) ^ (mask_6[2, 2, 2] == 1)
