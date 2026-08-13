"""IslesTrainerBucketWeighted - per-sample volume-weighted DC+TopK loss.

Each training step computes per-sample lesion volume from the
highest-resolution post-augmentation target, maps it to a weight via
`compute_per_sample_volume_weight` (target_ml=5.0, [0.5, 4.0] clip),
and passes the weight tensor through to `BucketWeightedDiceTopK`. The
loss is computed once per deep-supervision level with the SAME weights,
matching the DeepSupervisionWrapper's per-level call signature.

The pattern of computing per-batch metadata in `train_step` and threading
it through a custom loss mirrors `IslesTrainerSDT`. nnU-Net's data loader
strips arbitrary dict keys from the transform output, so per-sample state
cannot ride alongside the segmentation through the standard batch dict -
it must be derived inside the trainer from `batch["target"][0]`.
"""

from __future__ import annotations

from typing import Any

import torch

from nnunet_isles.losses._volume_weights import compute_per_sample_volume_weight
from nnunet_isles.registry import TRAINER_REGISTRY
from nnunet_isles.trainers.isles_trainer import IslesTrainer


@TRAINER_REGISTRY.register("IslesTrainerBucketWeighted")
class IslesTrainerBucketWeighted(IslesTrainer):
    """DC+TopK with per-sample weights derived from each sample's lesion volume."""

    # Volume-weight knobs (overridable per-experiment via class attr in train.py).
    isles_bucket_target_ml: float = 5.0
    isles_bucket_w_min: float = 0.5
    isles_bucket_w_max: float = 4.0
    isles_bucket_empty_weight: float = 2.0

    def _per_sample_weights(self, highres_target: torch.Tensor) -> torch.Tensor:
        cls = type(self)
        spacing = tuple(float(s) for s in self.configuration_manager.spacing)
        if len(spacing) != 3:
            raise ValueError(f"BucketWeighted trainer expects 3D spacing; got {spacing}")
        return compute_per_sample_volume_weight(
            highres_target,
            spacing,  # type: ignore[arg-type]
            target_ml=cls.isles_bucket_target_ml,
            w_min=cls.isles_bucket_w_min,
            w_max=cls.isles_bucket_w_max,
            empty_mask_weight=cls.isles_bucket_empty_weight,
        )

    def _ds_weights(self, weights: torch.Tensor, n_levels: int) -> list[torch.Tensor]:
        """Replicate per-sample weight tensor across deep-supervision levels."""
        return [weights] * n_levels

    def train_step(self, batch: dict) -> dict:  # type: ignore[override]
        from torch.amp.autocast_mode import autocast

        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
            highres = target[0]
            n_levels = len(target)
        else:
            target = target.to(self.device, non_blocking=True)
            highres = target
            n_levels = 1

        sample_weights = self._per_sample_weights(highres).detach()

        self.optimizer.zero_grad(set_to_none=True)
        amp_ctx = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else _NullCtx()
        with amp_ctx:
            output = self.network(data)
            loss_val = self._call_loss(output, target, sample_weights, n_levels)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss_val).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": loss_val.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:  # type: ignore[override]
        from torch.amp.autocast_mode import autocast

        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
            highres = target[0]
            n_levels = len(target)
        else:
            target = target.to(self.device, non_blocking=True)
            highres = target
            n_levels = 1

        sample_weights = self._per_sample_weights(highres).detach()

        amp_ctx = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else _NullCtx()
        with amp_ctx, torch.no_grad():
            output = self.network(data)
            loss_val = self._call_loss(output, target, sample_weights, n_levels)

        return self._collect_val_metrics(output, target, loss_val)

    def _call_loss(
        self,
        output: Any,
        target: Any,
        sample_weights: torch.Tensor,
        n_levels: int,
    ) -> torch.Tensor:
        """Invoke `self.loss(output, target, sample_weights_per_level)` correctly
        for both DeepSupervisionWrapper and the bare loss case."""
        from nnunet_isles.losses.bucket_weighted_dice_topk import BucketWeightedDiceTopK

        if isinstance(output, list) and self.enable_deep_supervision:
            sw_levels = self._ds_weights(sample_weights, n_levels)
            return self.loss(output, target, sw_levels)
        # Bare loss path.
        if isinstance(self.loss, BucketWeightedDiceTopK):
            return self.loss(output, target, sample_weights)
        # Fallback: e.g. DS wrapper but output is not a list (shouldn't happen).
        return self.loss(output, target)

    def _collect_val_metrics(self, seg_output: Any, target: Any, loss_val: torch.Tensor) -> dict:
        """Replicate upstream nnUNetTrainer.validation_step's TP/FP/FN accounting."""
        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

        if self.enable_deep_supervision:
            seg_output_for_metric = seg_output[0]
            target_for_metric = target[0]
        else:
            seg_output_for_metric = seg_output
            target_for_metric = target

        axes = [0] + list(range(2, seg_output_for_metric.ndim))
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(seg_output_for_metric) > 0.5).long()
        else:
            output_seg = seg_output_for_metric.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(
                seg_output_for_metric.shape, device=seg_output_for_metric.device, dtype=torch.float32
            )
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target_for_metric != self.label_manager.ignore_label).float()
                target_for_metric[target_for_metric == self.label_manager.ignore_label] = 0
            else:
                if target_for_metric.dtype == torch.bool:
                    mask = ~target_for_metric[:, -1:]
                else:
                    mask = (1 - target_for_metric[:, -1:]).float()
                target_for_metric = target_for_metric[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot, target_for_metric, axes=axes, mask=mask
        )
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            "loss": loss_val.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
