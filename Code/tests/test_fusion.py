"""Unit tests for nnunet_isles.inference.fusion (Pillar 1 detection fusion)."""

from __future__ import annotations

import numpy as np
from nnunet_isles.evaluation.metrics import dice_coefficient
from nnunet_isles.inference import fusion


def test_mean_prob():
    a = np.zeros((4, 4, 4), np.float32)
    b = np.ones((4, 4, 4), np.float32)
    m = fusion.mean_prob([a, b])
    assert np.allclose(m, 0.5)


def test_noisy_or_raises_minority_signal():
    # one member fires 0.9 at a voxel, two members near 0 -> mean dilutes, noisy-OR keeps.
    p = np.zeros((3, 2, 2, 2), np.float32)  # (M=3, *spatial)
    p[0, 0, 0, 0] = 0.9
    mean = fusion.mean_prob(p)
    nor = fusion.noisy_or(p)
    assert mean[0, 0, 0] < 0.35  # 0.9/3
    assert nor[0, 0, 0] > 0.85  # ~0.9
    # noisy-OR is always >= max member prob
    assert np.all(nor + 1e-6 >= p.max(axis=0))


def test_k_of_n_mask():
    p = np.zeros((3, 2, 2, 2), np.float32)
    p[0, 0, 0, 0] = 0.9  # 1 vote
    p[1, 0, 0, 0] = 0.9  # -> 2 votes at (0,0,0)
    p[0, 1, 1, 1] = 0.9  # 1 vote at (1,1,1)
    m2 = fusion.k_of_n_mask(p, k=2)
    assert m2[0, 0, 0] == 1 and m2[1, 1, 1] == 0
    m1 = fusion.k_of_n_mask(p, k=1)
    assert m1[0, 0, 0] == 1 and m1[1, 1, 1] == 1


def test_never_empty_rescue_on_subthreshold_case():
    # A real lesion the model under-fires: max prob 0.35 (< 0.5 threshold).
    prob = np.zeros((6, 6, 6), np.float32)
    prob[2:4, 2:4, 2:4] = 0.35
    assert fusion.threshold_mask(prob, 0.5).sum() == 0  # baseline misses it
    rescued = fusion.never_empty_mask(prob, threshold=0.5, rescue_min_prob=0.10)
    assert rescued.sum() > 0  # never empty
    assert rescued[3, 3, 3] == 1  # recovers the blob


def test_never_empty_picks_highest_mass_component():
    prob = np.zeros((8, 8, 8), np.float32)
    prob[1, 1, 1] = 0.4  # tiny speck (1 voxel)
    prob[4:7, 4:7, 4:7] = 0.3  # bigger low-prob blob (27 voxels, more mass)
    rescued = fusion.never_empty_mask(prob, threshold=0.5, rescue_min_prob=0.10)
    assert rescued[5, 5, 5] == 1 and rescued[1, 1, 1] == 0


def test_never_empty_argmax_fallback_when_all_tiny():
    prob = np.full((4, 4, 4), 0.02, np.float32)
    prob[2, 2, 2] = 0.03
    rescued = fusion.never_empty_mask(prob, threshold=0.5, rescue_min_prob=0.10)
    assert rescued.sum() == 1 and rescued[2, 2, 2] == 1


def test_apply_decision_layer_single_map_never_empty():
    prob = np.zeros((6, 6, 6), np.float32)
    prob[2:4, 2:4, 2:4] = 0.35  # under-fired lesion
    m0 = fusion.apply_decision_layer(prob, mode="mean", threshold=0.5, never_empty=False)
    m1 = fusion.apply_decision_layer(prob, mode="mean", threshold=0.5, never_empty=True)
    assert m0.sum() == 0 and m1.sum() > 0  # never_empty recovers it


def test_apply_decision_layer_stack_modes():
    p = np.zeros((3, 4, 4, 4), np.float32)
    p[0, 1, 1, 1] = 0.9  # single confident member at one voxel
    mean = fusion.apply_decision_layer(p, mode="mean", threshold=0.5)  # 0.3 -> below thr
    nor = fusion.apply_decision_layer(p, mode="noisy_or", threshold=0.5)  # ~0.9 -> above
    kof = fusion.apply_decision_layer(p, mode="k_of_n", k=1)
    assert mean[1, 1, 1] == 0
    assert nor[1, 1, 1] == 1
    assert kof[1, 1, 1] == 1


def test_apply_decision_layer_min_voxels_filter():
    prob = np.zeros((10, 10, 10), np.float32)
    prob[0, 0, 0] = 0.9  # 1-voxel speck
    prob[5:8, 5:8, 5:8] = 0.9  # 27-voxel blob
    m = fusion.apply_decision_layer(prob, mode="mean", threshold=0.5, min_voxels=10)
    assert m[6, 6, 6] == 1 and m[0, 0, 0] == 0


def test_best_per_case_threshold_finds_optimum():
    gt = np.zeros((6, 6, 6), np.uint8)
    gt[2:4, 2:4, 2:4] = 1
    prob = np.zeros((6, 6, 6), np.float32)
    prob[2:4, 2:4, 2:4] = 0.35  # lesion at 0.35
    prob[0, 0, 0] = 0.6  # a spurious high voxel
    t, d = fusion.best_per_case_threshold(prob, gt)
    assert t <= 0.30  # any threshold below the lesion's 0.35 prob captures it
    # at the oracle threshold, dice beats the 0.5-threshold dice
    assert d > dice_coefficient(fusion.threshold_mask(prob, 0.5), gt)
