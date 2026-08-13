"""Tests for PlainConvUNetWithCohortMoE network."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


def _build_tiny_moe_network(num_experts: int = 3, deep_supervision: bool = True):
    """Build a 2-stage 3D PlainConvUNetWithCohortMoE for fast unit tests."""
    from nnunet_isles.networks.plainconv_with_cohort_moe import PlainConvUNetWithCohortMoE

    net = PlainConvUNetWithCohortMoE(
        input_channels=1,
        n_stages=3,
        features_per_stage=(8, 16, 32),
        conv_op=nn.Conv3d,
        kernel_sizes=(3, 3, 3),
        strides=((1, 1, 1), (2, 2, 2), (2, 2, 2)),
        n_conv_per_stage=2,
        num_classes=2,
        n_conv_per_stage_decoder=(2, 2),
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=deep_supervision,
        num_experts=num_experts,
    )
    return net


def test_forward_shape_matches_vanilla():
    net = _build_tiny_moe_network(num_experts=3, deep_supervision=True)
    x = torch.randn(2, 1, 16, 16, 16)
    out = net(x)
    assert isinstance(out, list)
    # First entry is highest-resolution; should match input spatial.
    assert out[0].shape == (2, 2, 16, 16, 16)
    # Subsequent entries are lower resolution.
    for o in out[1:]:
        assert o.shape[0] == 2 and o.shape[1] == 2


def test_forward_shape_without_deep_supervision():
    net = _build_tiny_moe_network(num_experts=3, deep_supervision=False)
    x = torch.randn(2, 1, 16, 16, 16)
    out = net(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 2, 16, 16, 16)


def test_gating_weights_are_softmax_normalized():
    net = _build_tiny_moe_network(num_experts=3)
    x = torch.randn(2, 1, 16, 16, 16)
    _ = net(x)
    gate = net.last_gate_weights
    assert gate is not None
    assert gate.shape == (2, 3)
    sums = gate.sum(dim=1)
    assert torch.allclose(sums, torch.ones(2), atol=1e-5)
    assert (gate >= 0).all() and (gate <= 1).all()


def test_load_balancing_loss_is_zero_at_uniform_gating():
    """If we force the gate to be uniform, the load-balancing loss should be 0."""
    net = _build_tiny_moe_network(num_experts=3)
    x = torch.randn(2, 1, 16, 16, 16)
    _ = net(x)
    # Patch the gate to uniform.
    net.last_gate_weights = torch.full((2, 3), 1.0 / 3.0)
    aux = net.load_balancing_loss()
    assert float(aux) < 1e-6


def test_load_balancing_loss_is_positive_on_collapse():
    """If the gate collapses to one expert, the aux loss is positive."""
    net = _build_tiny_moe_network(num_experts=3)
    x = torch.randn(2, 1, 16, 16, 16)
    _ = net(x)
    collapsed = torch.zeros(2, 3)
    collapsed[:, 0] = 1.0
    net.last_gate_weights = collapsed
    aux = net.load_balancing_loss()
    assert float(aux) > 0.0


def test_backward_propagates_through_experts_and_gate():
    """Gradients should flow through every expert + the gating MLP."""
    net = _build_tiny_moe_network(num_experts=3)
    x = torch.randn(2, 1, 16, 16, 16, requires_grad=False)
    out = net(x)
    loss = out[0].sum() + 0.1 * net.load_balancing_loss()
    loss.backward()
    # Every expert head should have non-zero grad on at least one weight.
    for expert in net._moe_head.experts:
        assert expert.weight.grad is not None
        assert (expert.weight.grad.abs().sum().item()) > 0.0
    # Gating MLP linear weight should also have grad.
    linear = [m for m in net.gating_head.modules() if isinstance(m, nn.Linear)][0]
    assert linear.weight.grad is not None
    assert linear.weight.grad.abs().sum().item() > 0.0


def test_experts_are_warm_started_with_same_weights():
    """All K expert heads should initially share the same weights (warm-start
    from the original PlainConvUNet head) - otherwise they would diverge unstably."""
    net = _build_tiny_moe_network(num_experts=3)
    first = net._moe_head.experts[0].weight.data.clone()
    for e in net._moe_head.experts[1:]:
        assert torch.allclose(e.weight.data, first)


def test_num_experts_knob_changes_gate_dim():
    net = _build_tiny_moe_network(num_experts=5)
    x = torch.randn(2, 1, 16, 16, 16)
    _ = net(x)
    assert net.last_gate_weights.shape == (2, 5)


def test_gate_cleared_after_forward():
    """After forward returns, the head's `current_gate` should be cleared so a
    fresh forward computes a fresh gate."""
    net = _build_tiny_moe_network(num_experts=3)
    x = torch.randn(2, 1, 16, 16, 16)
    _ = net(x)
    assert net._moe_head.current_gate is None


def test_load_balancing_loss_raises_without_forward_pass():
    net = _build_tiny_moe_network(num_experts=3)
    with pytest.raises(RuntimeError):
        net.load_balancing_loss()


def test_build_network_architecture_callable_at_class_level():
    """nnU-Net's `nnUNetPredictor.initialize_from_trained_model_folder` calls
    `trainer_class.build_network_architecture(plans_manager, ...)` via the
    CLASS, not via a bound instance. The override must be a @classmethod (or
    @staticmethod) for `self` not to consume the first positional arg.

    Regression test: build_network_architecture must be a @classmethod so
    upstream's inspect.signature check treats it correctly at finalize time."""
    import inspect

    from nnunet_isles.trainers.isles_trainer_cohort_moe import IslesTrainerCohortMoE

    for trainer_cls in (IslesTrainerCohortMoE,):
        sig = inspect.signature(trainer_cls.build_network_architecture)
        params = list(sig.parameters.keys())
        assert "self" not in params, (
            f"{trainer_cls.__name__}.build_network_architecture must not list 'self' - "
            f"got params={params}. Decorate with @classmethod or @staticmethod."
        )
        # And `plans_manager` MUST be the first positional (matches upstream's call site).
        assert params[0] == "plans_manager", (
            f"{trainer_cls.__name__}.build_network_architecture must take plans_manager "
            f"as the first positional arg; got params={params}"
        )
