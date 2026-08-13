"""PlainConvUNetWithCohortMoE.

Replaces ONLY the highest-resolution seg head of a `PlainConvUNet` with
K=3 parallel "expert" 1×1×1 conv heads. A small gating MLP fed by the
encoder bottleneck (AdaptiveAvgPool3d(1) → Linear(C, K) → softmax)
produces per-sample expert weights `w[B, K]`. The final highest-res
output is `sum_k w_{b,k} * expert_k(features)`.

Deep-supervision levels other than the highest-res are left untouched
(single seg head per level, as upstream). MoE only at the highest res
matches the "cohort-shift at lesion detail" motivation - cohort-shift
likely manifests in fine spatial pattern, not coarse anatomy.

Forward returns the same shape as upstream `PlainConvUNet.forward`:
  * a single tensor when `deep_supervision=False`
  * a list of tensors `[highres_moe_output, ds_level_1, ...]` when
    `deep_supervision=True`.

A load-balancing auxiliary loss is exposed via `self.last_gate_weights`
(shape `(B, K)`); the paired `IslesTrainerCohortMoE` reads this each
step and adds an aux term to keep gating from collapsing to one expert.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_network_architectures.architectures.unet import PlainConvUNet


class _MoESegHead(nn.Module):
    """K parallel 1×1×1 conv heads. The network's forward stashes
    `current_gate: (B, K)` here before invoking the decoder."""

    def __init__(self, in_channels: int, num_classes: int, num_experts: int, conv_op: type) -> None:
        super().__init__()
        if conv_op not in (nn.Conv3d, nn.Conv2d):
            raise ValueError(f"Unsupported conv_op for MoE head: {conv_op}")
        self.num_experts = int(num_experts)
        self.experts = nn.ModuleList(
            [
                conv_op(in_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
                for _ in range(num_experts)
            ]
        )
        # Default gate - overwritten per-forward by the parent network. Initialised to uniform
        # so the head can also be used as a vanilla averaged ensemble when called outside MoE flow.
        self.register_buffer("default_gate", torch.full((1, num_experts), 1.0 / num_experts))
        self.current_gate: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, *spatial). Compute K expert outputs → stack → weight by gate.
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # (B, K, num_classes, *spatial)
        gate = self.default_gate.expand(x.shape[0], -1) if self.current_gate is None else self.current_gate
        # Broadcast gate over spatial dims.
        spatial_ndim = expert_outs.ndim - 3  # excluding B, K, num_classes
        gate_b = gate.view(gate.shape[0], gate.shape[1], 1, *([1] * spatial_ndim))
        out = (expert_outs * gate_b).sum(dim=1)
        return out


class PlainConvUNetWithCohortMoE(PlainConvUNet):
    """PlainConvUNet with mixture-of-experts at the highest-res seg head."""

    def __init__(self, *args, num_experts: int = 3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_experts = int(num_experts)
        bottleneck_channels = self.encoder.output_channels[-1]
        # Build gating MLP from bottleneck features.
        conv_op = self.encoder.conv_op
        if conv_op is nn.Conv3d:
            pool = nn.AdaptiveAvgPool3d(1)
        elif conv_op is nn.Conv2d:
            pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError(f"Unsupported conv_op for MoE gating: {conv_op}")
        self.gating_head = nn.Sequential(
            pool,
            nn.Flatten(),
            nn.Linear(bottleneck_channels, self.num_experts),
        )

        # Replace highest-resolution seg head with an MoE head.
        old_head = self.decoder.seg_layers[-1]
        # Recover in_channels from the existing head's weight shape.
        in_channels = old_head.weight.shape[1]
        num_classes = old_head.weight.shape[0]
        moe_head = _MoESegHead(in_channels, num_classes, self.num_experts, conv_op)
        # Initialise each expert from the original head's weights (warm-start;
        # without this, the experts start identical zero-init and collapse).
        with torch.no_grad():
            for expert in moe_head.experts:
                expert.weight.copy_(old_head.weight)
                if old_head.bias is not None and expert.bias is not None:
                    expert.bias.copy_(old_head.bias)
        self.decoder.seg_layers[-1] = moe_head
        self._moe_head = moe_head  # convenience alias

        # Last gate weights for the trainer's load-balancing loss.
        self.last_gate_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        skips = self.encoder(x)
        # Compute gating from encoder bottleneck.
        gate_logits = self.gating_head(skips[-1])  # (B, K)
        gate = F.softmax(gate_logits, dim=1)
        # Stash for the MoE head + for the trainer to consume.
        self._moe_head.current_gate = gate
        self.last_gate_weights = gate
        seg_out = self.decoder(skips)
        # Clear to avoid stale gate in next call.
        self._moe_head.current_gate = None
        return seg_out

    def load_balancing_loss(self) -> torch.Tensor:
        """Standard MoE load-balancing aux loss: penalises deviation from uniform expert usage."""
        if self.last_gate_weights is None:
            raise RuntimeError("load_balancing_loss requires a forward pass first")
        gate = self.last_gate_weights  # (B, K)
        usage = gate.mean(dim=0)  # (K,) - fraction routed to expert k
        target = torch.full_like(usage, 1.0 / gate.shape[1])
        return ((usage - target) ** 2).sum()


__all__ = ["PlainConvUNetWithCohortMoE", "_MoESegHead"]
