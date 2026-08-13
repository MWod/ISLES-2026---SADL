"""Per-sample volume-bucket-weighted DC + Top-K CE loss.

Unlike upstream `DC_and_topk_loss`, this loss is computed PER-SAMPLE and
then reduced with caller-supplied per-sample weights `sample_weights[B]`,
which lets the trainer up-weight small/mid-small-lesion samples (the V2
0.5-5mL regression bucket) without contaminating the gradients of
large-lesion samples that already train well.

Reduction:
    L = sum_b (w_b * L_b) / sum_b w_b                  (sample_weights given)
    L = mean_b (L_b)                                   (sample_weights is None - equivalent to uniform)

The per-sample dice + per-sample top-K CE are computed from primitives
rather than calling upstream `SoftDiceLoss` / `TopKLoss` (which both
collapse the batch axis internally). DeepSupervisionWrapper still
zips per-level args, so callers pass `sample_weights` as a list with
the same value at each DS level - see `IslesTrainerBucketWeighted`.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from nnunet_isles.registry import LOSS_REGISTRY


class BucketWeightedDiceTopK(nn.Module):
    """DC + Top-K CE with caller-supplied per-sample weights."""

    def __init__(
        self,
        *,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0,
        k_pct: float = 10.0,
        smooth: float = 1.0e-5,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if k_pct <= 0.0 or k_pct > 100.0:
            raise ValueError(f"k_pct must be in (0, 100]; got {k_pct}")
        self.weight_ce = float(weight_ce)
        self.weight_dice = float(weight_dice)
        self.k_pct = float(k_pct)
        self.smooth = float(smooth)
        self.label_smoothing = float(label_smoothing)

    def _per_sample_dice_loss(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-sample foreground soft-dice loss in [0, 1]. Shape (B,)."""
        # net_output: (B, C, *spatial) logits; we softmax over channels.
        probs = F.softmax(net_output, dim=1)
        # Take foreground channel only (binary task - class index 1).
        fg_prob = probs[:, 1]
        # Target is (B, 1, *spatial) int or (B, *spatial) - flatten to (B, 1+spatial).
        tgt = target[:, 0] if target.ndim == fg_prob.ndim + 1 else target
        tgt_fg = (tgt > 0).to(fg_prob.dtype)
        # Sum over spatial only, keeping batch.
        axes = tuple(range(1, fg_prob.ndim))
        tp = (fg_prob * tgt_fg).sum(dim=axes)
        fp = (fg_prob * (1.0 - tgt_fg)).sum(dim=axes)
        fn = ((1.0 - fg_prob) * tgt_fg).sum(dim=axes)
        dice_score = (2.0 * tp + self.smooth) / (2.0 * tp + fp + fn + self.smooth)
        return 1.0 - dice_score  # (B,) - loss form

    def _per_sample_topk_ce(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-sample top-k% cross-entropy. Shape (B,)."""
        tgt = target[:, 0].long() if target.ndim == net_output.ndim else target.long()
        # F.cross_entropy with reduction='none' returns (B, *spatial) per-voxel CE.
        ce = F.cross_entropy(net_output, tgt, reduction="none", label_smoothing=self.label_smoothing)
        # Flatten spatial -> (B, V), then per-sample top-k%.
        ce_b = ce.flatten(start_dim=1)
        num_voxels = ce_b.shape[1]
        k = max(1, int(num_voxels * self.k_pct / 100.0))
        topk_vals, _ = torch.topk(ce_b, k=k, dim=1, sorted=False)
        return topk_vals.mean(dim=1)  # (B,)

    def forward(
        self,
        net_output: torch.Tensor,
        target: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Weighted scalar loss.

        net_output: (B, C, *spatial) logits, C=2 for binary task.
        target:     (B, 1, *spatial) or (B, *spatial) labels.
        sample_weights: (B,) optional. None ⇒ uniform mean reduction.
        """
        dice_per_b = self._per_sample_dice_loss(net_output, target) if self.weight_dice != 0.0 else 0.0
        ce_per_b = self._per_sample_topk_ce(net_output, target) if self.weight_ce != 0.0 else 0.0
        per_sample = self.weight_dice * dice_per_b + self.weight_ce * ce_per_b
        if not torch.is_tensor(per_sample):
            # Degenerate config - both weights zero. Return scalar 0 for safety.
            return torch.zeros((), device=net_output.device, dtype=net_output.dtype)

        if sample_weights is None:
            return per_sample.mean()
        sw = sample_weights.to(per_sample.dtype).to(per_sample.device)
        if sw.shape != per_sample.shape:
            raise ValueError(
                f"sample_weights shape {tuple(sw.shape)} must match per-sample loss shape {tuple(per_sample.shape)}"
            )
        denom = sw.sum().clamp_min(1.0e-8)
        return (sw * per_sample).sum() / denom


@LOSS_REGISTRY.register("bucket_weighted_dice_topk")
def build_bucket_weighted_dice_topk(
    weight_ce: float = 1.0,
    weight_dice: float = 1.0,
    k_pct: float = 10.0,
    smooth: float = 1.0e-5,
    label_smoothing: float = 0.0,
    **_: Any,
) -> BucketWeightedDiceTopK:
    """Hydra/registry constructor."""
    return BucketWeightedDiceTopK(
        weight_ce=weight_ce,
        weight_dice=weight_dice,
        k_pct=k_pct,
        smooth=smooth,
        label_smoothing=label_smoothing,
    )


__all__ = ["BucketWeightedDiceTopK", "build_bucket_weighted_dice_topk"]
