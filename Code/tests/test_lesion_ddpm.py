"""Tests for LesionDDPM + LesionDDPMSampler."""

from __future__ import annotations

import torch
from nnunet_isles.diffusion import LesionDDPM, LesionDDPMSampler
from nnunet_isles.diffusion.lesion_ddpm import linear_beta_schedule, sinusoidal_timestep_embedding


def test_sinusoidal_embedding_shape_and_range():
    t = torch.arange(0, 10, dtype=torch.long)
    emb = sinusoidal_timestep_embedding(t, dim=64)
    assert emb.shape == (10, 64)
    assert torch.isfinite(emb).all()
    assert emb.abs().max() <= 1.0001


def test_linear_beta_schedule_increases():
    betas = linear_beta_schedule(num_steps=10, beta_start=1e-4, beta_end=0.02)
    assert betas.shape == (10,)
    for i in range(1, 10):
        assert betas[i] > betas[i - 1]


def test_forward_shape_preserves_spatial():
    net = LesionDDPM(in_channels=1, base_channels=8, cond_dim=32)
    x = torch.randn(2, 1, 16, 16, 16)
    t = torch.tensor([100, 300], dtype=torch.long)
    log_vol = torch.tensor([0.5, 1.0])
    eps = net(x, t, log_vol)
    assert eps.shape == x.shape


def test_q_sample_interpolates_x0_and_noise():
    """At t=0, q_sample(x_0, 0, noise) ≈ x_0 (alpha_cumprod ≈ 1).
    At t=high, q_sample ≈ noise (alpha_cumprod ≈ 0)."""
    # 1000 steps with default schedule lets cumprod reach ~0 by the end.
    sampler = LesionDDPMSampler(num_steps=1000)
    torch.manual_seed(0)
    x_0 = torch.randn(1, 1, 8, 8, 8)
    noise = torch.randn_like(x_0)
    t_low = torch.tensor([0])
    t_high = torch.tensor([999])
    out_low = sampler.q_sample(x_0, t_low, noise)
    out_high = sampler.q_sample(x_0, t_high, noise)
    # Low t: closer to x_0 than to noise.
    assert (out_low - x_0).norm() < (out_low - noise).norm()
    # High t: closer to noise than to x_0.
    assert (out_high - noise).norm() < (out_high - x_0).norm()


def test_sample_reverse_returns_correct_shape():
    net = LesionDDPM(in_channels=1, base_channels=8, cond_dim=32)
    net.eval()
    sampler = LesionDDPMSampler(num_steps=5)
    log_vol = torch.tensor([0.3, 0.7])
    out = sampler.sample(net, shape=(2, 1, 8, 8, 8), log_vol=log_vol, device=torch.device("cpu"))
    assert out.shape == (2, 1, 8, 8, 8)
    assert torch.isfinite(out).all()


def test_backward_one_step():
    """The training step (predict noise → MSE) should produce finite gradients."""
    import torch.nn.functional as F

    net = LesionDDPM(in_channels=1, base_channels=8, cond_dim=32)
    sampler = LesionDDPMSampler(num_steps=10)
    x_0 = torch.randn(2, 1, 8, 8, 8)
    log_vol = torch.tensor([0.5, 1.0])
    t = torch.randint(0, 10, (2,))
    noise = torch.randn_like(x_0)
    x_t = sampler.q_sample(x_0, t, noise)
    eps_pred = net(x_t, t, log_vol)
    loss = F.mse_loss(eps_pred, noise)
    loss.backward()
    grad_count = sum(1 for p in net.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total = sum(1 for p in net.parameters() if p.requires_grad)
    assert grad_count > 0
    # Not necessarily all parameters get gradients (e.g., if some conditioning paths
    # are unused at this batch size), but most should.
    assert grad_count / total > 0.5


def test_volume_conditioning_changes_output():
    """Different log_vol values should produce different network outputs."""
    torch.manual_seed(0)
    net = LesionDDPM(in_channels=1, base_channels=8, cond_dim=32)
    net.eval()
    x = torch.randn(1, 1, 8, 8, 8)
    t = torch.tensor([100], dtype=torch.long)
    with torch.no_grad():
        out_small = net(x, t, torch.tensor([0.1]))
        out_large = net(x, t, torch.tensor([5.0]))
    # Outputs should differ - volume conditioning is active.
    assert not torch.allclose(out_small, out_large, atol=1e-4)
