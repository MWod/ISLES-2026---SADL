"""Tests for BucketWeightedDiceTopK loss + IslesTrainerBucketWeighted."""

from __future__ import annotations

import pytest
import torch
from nnunet_isles.losses._volume_weights import compute_per_sample_volume_weight
from nnunet_isles.losses.bucket_weighted_dice_topk import BucketWeightedDiceTopK


def _make_logits(B: int, spatial: tuple[int, ...], fg_voxels: int) -> torch.Tensor:
    """Build 2-class logits where the first `fg_voxels` voxels predict fg=high."""
    logits = torch.full((B, 2) + spatial, -2.0, dtype=torch.float32)
    flat = logits.view(B, 2, -1)
    flat[:, 0, :fg_voxels] = -2.0
    flat[:, 1, :fg_voxels] = 2.0
    return logits


def _make_target(B: int, spatial: tuple[int, ...], fg_voxels: int) -> torch.Tensor:
    """Build (B, 1, *spatial) target with the first fg_voxels voxels set."""
    tgt = torch.zeros((B, 1) + spatial, dtype=torch.long)
    flat = tgt.view(B, 1, -1)
    flat[:, 0, :fg_voxels] = 1
    return tgt


def test_loss_matches_uniform_mean_when_no_weights():
    """sample_weights=None should equal mean over per-sample losses."""
    torch.manual_seed(0)
    logits = torch.randn(3, 2, 8, 8, 8, requires_grad=True)
    target = _make_target(3, (8, 8, 8), fg_voxels=20)
    loss = BucketWeightedDiceTopK()
    out_none = loss(logits, target, sample_weights=None)
    out_uniform = loss(logits, target, sample_weights=torch.ones(3))
    assert torch.allclose(out_none, out_uniform, atol=1e-6)


def test_per_sample_dice_increases_with_more_fp():
    """Per-sample dice loss responds to per-sample prediction quality."""
    spatial = (6, 6, 6)
    # Sample 0: FG predicted EVERYWHERE → high FP, low dice score.
    # Sample 1: FG predicted ONLY in the GT region → perfect dice.
    # Bad: predicts FG everywhere.
    logits_bad = torch.zeros((1, 2) + spatial)
    logits_bad[:, 0] = -4.0  # BG very low → softmax FG ≈ 1 everywhere
    logits_bad[:, 1] = 4.0
    # Good: predicts FG only at the target region (first slice); BG elsewhere.
    logits_good = torch.zeros((1, 2) + spatial)
    logits_good[:, 0] = 4.0  # BG high everywhere by default
    logits_good[:, 1] = -4.0  # FG low everywhere by default
    logits_good[:, 0, :, :, 0] = -4.0  # flip for target slice
    logits_good[:, 1, :, :, 0] = 4.0
    logits = torch.cat([logits_bad, logits_good], dim=0)
    # Target: fg voxels are exactly the first slice (6*6=36 voxels).
    target = torch.zeros((2, 1) + spatial, dtype=torch.long)
    target[:, 0, :, :, 0] = 1
    loss = BucketWeightedDiceTopK(weight_ce=0.0, weight_dice=1.0)
    per_b = loss._per_sample_dice_loss(logits, target)
    assert per_b.shape == (2,)
    # Sample 0 (BAD) should have HIGHER dice loss than sample 1 (GOOD).
    assert per_b[0] > per_b[1]
    # And the good one should be near zero.
    assert per_b[1].item() < 0.1


def test_weights_change_relative_contribution():
    """Up-weighting the harder sample should pull the total loss UP toward its per-sample loss."""
    spatial = (6, 6, 6)
    # Easy: predicts FG only in lesion slice; BG elsewhere (low FP).
    logits_easy = torch.zeros((1, 2) + spatial)
    logits_easy[:, 0] = 4.0
    logits_easy[:, 1] = -4.0
    logits_easy[:, 0, :, :, 0] = -4.0
    logits_easy[:, 1, :, :, 0] = 4.0
    # Hard: predicts BG everywhere (misses lesion → dice ≈ 1).
    logits_hard = torch.zeros((1, 2) + spatial)
    logits_hard[:, 0] = 4.0
    logits_hard[:, 1] = -4.0
    logits = torch.cat([logits_easy, logits_hard], dim=0)
    target = torch.zeros((2, 1) + spatial, dtype=torch.long)
    target[:, 0, :, :, 0] = 1
    loss = BucketWeightedDiceTopK()

    uniform = loss(logits, target, sample_weights=torch.tensor([1.0, 1.0]))
    weighted_to_hard = loss(logits, target, sample_weights=torch.tensor([0.1, 10.0]))
    weighted_to_easy = loss(logits, target, sample_weights=torch.tensor([10.0, 0.1]))
    # Up-weighting the hard sample pulls the loss toward L_hard; easy pulls down.
    assert weighted_to_hard > uniform > weighted_to_easy


def test_gradient_norm_scales_with_weight():
    """A larger sample weight on a given sample produces a larger gradient norm there."""
    torch.manual_seed(3)
    spatial = (4, 4, 4)
    logits = torch.randn((2, 2) + spatial, requires_grad=True)
    target = _make_target(2, spatial, fg_voxels=5)
    sw_a = torch.tensor([4.0, 1.0])
    sw_b = torch.tensor([1.0, 4.0])
    loss = BucketWeightedDiceTopK()

    # Run loss A: weight 4 on sample 0 → larger grads on sample 0's logits.
    logits_a = logits.detach().clone().requires_grad_(True)
    loss_a = loss(logits_a, target, sample_weights=sw_a)
    loss_a.backward()
    grad_norm_0_a = logits_a.grad[0].norm().item()
    grad_norm_1_a = logits_a.grad[1].norm().item()

    logits_b = logits.detach().clone().requires_grad_(True)
    loss_b = loss(logits_b, target, sample_weights=sw_b)
    loss_b.backward()
    grad_norm_0_b = logits_b.grad[0].norm().item()
    grad_norm_1_b = logits_b.grad[1].norm().item()

    # When sample 0 is up-weighted, its grad-norm is larger than when it's down-weighted.
    assert grad_norm_0_a > grad_norm_0_b
    assert grad_norm_1_b > grad_norm_1_a


def test_deep_supervision_wrapper_compatibility():
    """The loss must work inside a DeepSupervisionWrapper with 3-arg signature."""
    from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

    torch.manual_seed(4)
    spatial0 = (8, 8, 8)
    spatial1 = (4, 4, 4)  # half-resolution DS level
    logits_0 = torch.randn((2, 2) + spatial0, requires_grad=True)
    logits_1 = torch.randn((2, 2) + spatial1, requires_grad=True)
    tgt_0 = _make_target(2, spatial0, fg_voxels=15)
    tgt_1 = _make_target(2, spatial1, fg_voxels=2)
    sw_per_level = [torch.tensor([2.0, 0.5]), torch.tensor([2.0, 0.5])]

    inner = BucketWeightedDiceTopK()
    wrapped = DeepSupervisionWrapper(inner, weight_factors=[2.0 / 3.0, 1.0 / 3.0])
    out = wrapped([logits_0, logits_1], [tgt_0, tgt_1], sw_per_level)
    assert out.ndim == 0  # scalar
    out.backward()
    assert logits_0.grad is not None
    assert logits_1.grad is not None


def test_volume_weight_to_loss_pipeline():
    """End-to-end CC2 → loss: small mask gets w_max, large gets w_min, and the
    composite weighted loss reflects this with both samples included."""
    spatial = (10, 10, 10)
    # Sample 0: 10 voxels = 0.01 mL → w_max=4.0
    # Sample 1: 50000 voxels = 50 mL → w_min=0.5
    target_high = torch.zeros((2, 1) + spatial, dtype=torch.long)
    target_high[0, 0].view(-1)[:10] = 1
    target_high[1, 0].view(-1)[:50_000] = 1  # would be capped - but spatial only has 1000 voxels; OK
    # We have only 1000 voxels per sample (10*10*10=1000). So 50000 → all 1000 = 1 mL → still gets clipped above 1.0
    # Patch: enlarge spatial.
    spatial = (40, 40, 40)
    target_high = torch.zeros((2, 1) + spatial, dtype=torch.long)
    target_high[0, 0].view(-1)[:10] = 1
    target_high[1, 0].view(-1)[:50_000] = 1  # 50 mL at 1mm³

    weights = compute_per_sample_volume_weight(target_high, (1.0, 1.0, 1.0), target_ml=5.0)
    assert float(weights[0]) == pytest.approx(4.0)
    assert float(weights[1]) == pytest.approx(0.5)

    logits = torch.randn((2, 2) + spatial, requires_grad=True)
    loss = BucketWeightedDiceTopK()
    out = loss(logits, target_high, sample_weights=weights)
    assert out.ndim == 0
    assert torch.isfinite(out)


def test_loss_raises_on_shape_mismatch():
    logits = torch.randn(2, 2, 4, 4, 4)
    target = _make_target(2, (4, 4, 4), fg_voxels=3)
    loss = BucketWeightedDiceTopK()
    with pytest.raises(ValueError, match="sample_weights shape"):
        loss(logits, target, sample_weights=torch.tensor([1.0, 1.0, 1.0]))


def test_registry_constructor():
    """Ensure the LOSS_REGISTRY key resolves to a working instance."""
    from nnunet_isles.registry import LOSS_REGISTRY

    inst = LOSS_REGISTRY.build("bucket_weighted_dice_topk", weight_ce=2.0, weight_dice=0.5, k_pct=20.0)
    assert isinstance(inst, BucketWeightedDiceTopK)
    assert inst.weight_ce == 2.0
    assert inst.weight_dice == 0.5
    assert inst.k_pct == 20.0
