"""IslesTrainerCohortMoE - multi-expert seg head trainer.

Builds `PlainConvUNetWithCohortMoE` in place of the plans-described network
and adds a load-balancing auxiliary loss to the standard DC+TopK loss:

    total = main_loss + aux_weight * network.load_balancing_loss()

The gating is UNSUPERVISED (no cohort label needed in the batch) - the
network learns which expert head suits each sample. The aux loss keeps
gating from collapsing to one expert.
"""

from __future__ import annotations

from typing import Any

import torch

from nnunet_isles.registry import TRAINER_REGISTRY
from nnunet_isles.trainers.isles_trainer import IslesTrainer


@TRAINER_REGISTRY.register("IslesTrainerCohortMoE")
class IslesTrainerCohortMoE(IslesTrainer):
    """nnUNetTrainer subclass that swaps PlainConvUNet → PlainConvUNetWithCohortMoE."""

    isles_moe_num_experts: int = 3
    isles_moe_load_balance_weight: float = 0.01

    @classmethod
    def build_network_architecture(  # type: ignore[override]
        cls,
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        # MUST be @classmethod (or @staticmethod), NOT an instance method. nnU-Net's
        # `nnUNetPredictor` calls `trainer_class.build_network_architecture(...)`
        # via the CLASS (not a bound instance) - see predict_from_raw_data.py
        # `inspect.signature(trainer_class.build_network_architecture)`. With an
        # instance-method signature `self` is not auto-bound, so `plans_manager`
        # gets consumed as `self` and the call fails with "missing positional
        # argument: num_output_channels". Without @classmethod the predictor
        # fails at initialize_from_trained_model_folder.
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

        from nnunet_isles.networks.plainconv_with_cohort_moe import PlainConvUNetWithCohortMoE

        network = get_network_from_plans(
            "nnunet_isles.networks.plainconv_with_cohort_moe.PlainConvUNetWithCohortMoE",
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        if not isinstance(network, PlainConvUNetWithCohortMoE):
            raise RuntimeError("Expected PlainConvUNetWithCohortMoE; got " + type(network).__name__)
        # Wire knob from class attr - the get_network_from_plans path doesn't accept
        # extra kwargs cleanly, so we set num_experts after construction by rebuilding
        # the MoE head if it differs from the default.
        if int(cls.isles_moe_num_experts) != network.num_experts:
            # Rebuild with the right K. Cheap because the experts are 1x1x1 convs.
            from nnunet_isles.networks.plainconv_with_cohort_moe import _MoESegHead

            old = network._moe_head
            in_channels = old.experts[0].weight.shape[1]
            num_classes = old.experts[0].weight.shape[0]
            new_head = _MoESegHead(
                in_channels, num_classes, int(cls.isles_moe_num_experts), type(old.experts[0])
            )
            with torch.no_grad():
                src_w = old.experts[0].weight
                src_b = old.experts[0].bias
                for expert in new_head.experts:
                    expert.weight.copy_(src_w)
                    if src_b is not None and expert.bias is not None:
                        expert.bias.copy_(src_b)
            network.decoder.seg_layers[-1] = new_head
            network._moe_head = new_head
            network.num_experts = int(cls.isles_moe_num_experts)
            # Rebuild gating head with correct K output dim.
            import torch.nn as nn

            bottleneck_channels = network.encoder.output_channels[-1]
            conv_op = network.encoder.conv_op
            if conv_op is nn.Conv3d:
                pool = nn.AdaptiveAvgPool3d(1)
            elif conv_op is nn.Conv2d:
                pool = nn.AdaptiveAvgPool2d(1)
            else:
                raise ValueError(f"Unsupported conv_op: {conv_op}")
            network.gating_head = nn.Sequential(
                pool,
                nn.Flatten(),
                nn.Linear(bottleneck_channels, int(cls.isles_moe_num_experts)),
            )
        return network

    def train_step(self, batch: dict) -> dict:  # type: ignore[override]
        from torch.amp.autocast_mode import autocast

        cls = type(self)
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        amp_ctx = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else _NullCtx()
        with amp_ctx:
            output = self.network(data)
            main_loss = self.loss(output, target)
            aux = self.network.load_balancing_loss()
            loss_val = main_loss + float(cls.isles_moe_load_balance_weight) * aux

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
        else:
            target = target.to(self.device, non_blocking=True)

        amp_ctx = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else _NullCtx()
        with amp_ctx, torch.no_grad():
            output = self.network(data)
            loss_val = self.loss(output, target)

        return self._collect_val_metrics(output, target, loss_val)

    def _collect_val_metrics(self, seg_output: Any, target: Any, loss_val: torch.Tensor) -> dict:
        """Identical accounting to upstream nnUNetTrainer.validation_step."""
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
