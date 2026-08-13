"""Dice + Top-K CE loss - thin wrapper around nnU-Net's DC_and_topk_loss.

Top-K CE focuses the CE term on the K% hardest pixels per batch, which biases
training toward small or hard lesions where the bulk-mean CE loss is washed
out by easy background. Pair with soft-dice as the global term.

Default k_pct=10 follows Wu et al., "Top-k loss for boundary-aware segmentation."
"""

from __future__ import annotations

from typing import Any

from nnunet_isles.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("dice_topk")
def build_dice_topk(
    weight_ce: float = 1.0,
    weight_dice: float = 1.0,
    k_pct: float = 10.0,
    batch_dice: bool = True,
    smooth: float = 1.0e-5,
    ddp: bool = False,
    label_smoothing: float = 0.0,
    **_: Any,
) -> Any:
    """Return an nnU-Net DC_and_topk_loss configured for binary lesion segmentation."""
    from nnunetv2.training.loss.compound_losses import DC_and_topk_loss

    return DC_and_topk_loss(
        soft_dice_kwargs={"batch_dice": batch_dice, "smooth": smooth, "do_bg": False, "ddp": ddp},
        ce_kwargs={"k": float(k_pct), "label_smoothing": float(label_smoothing)},
        weight_ce=weight_ce,
        weight_dice=weight_dice,
        ignore_label=None,
    )
