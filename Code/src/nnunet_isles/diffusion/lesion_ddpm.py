"""C3a - minimal 3D conditional DDPM for synthetic small-lesion patch generation.

A self-contained DDPM (no MONAI dep) trained on 64^3 image patches centered
on real ATLAS V2 lesions, conditioned on the lesion log-volume scalar. At
inference we sample new patches conditioned on a target log-volume to
populate a synthetic-lesion bank for the C3b paste-augmentation transform.

The architecture is a small 3D U-Net (~3 stages, base 16 channels) with:
  * Sinusoidal timestep embedding fed into every conv block via FiLM.
  * Scalar volume conditioning (single float) projected to the same FiLM dim.

Training loss is the standard DDPM ε-prediction MSE.
Forward (noising) schedule is linear in β.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_beta_schedule(num_steps: int, beta_start: float = 1.0e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_steps)


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal embedding (Vaswani et al. / DDPM)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class _FiLMConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, cond_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.cond_proj = nn.Linear(cond_dim, 2 * out_c)
        self.skip = nn.Conv3d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm1(self.conv1(x))
        h = F.silu(h)
        h = self.norm2(self.conv2(h))
        # FiLM gating from (time + volume) embedding.
        scale, shift = self.cond_proj(cond).chunk(2, dim=1)
        h = h * (1.0 + scale[..., None, None, None]) + shift[..., None, None, None]
        h = F.silu(h)
        return h + self.skip(x)


class LesionDDPM(nn.Module):
    """Conditional ε-prediction 3D U-Net for DDPM training."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        cond_dim: int = 128,
    ) -> None:
        super().__init__()
        self.cond_dim = int(cond_dim)
        # Time + volume → joint condition vector.
        self.time_mlp = nn.Sequential(
            nn.Linear(self.cond_dim, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim),
        )
        self.volume_mlp = nn.Sequential(
            nn.Linear(1, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim),
        )

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.enc1 = _FiLMConvBlock(in_channels, c1, self.cond_dim)
        self.down1 = nn.Conv3d(c1, c1, 4, stride=2, padding=1)
        self.enc2 = _FiLMConvBlock(c1, c2, self.cond_dim)
        self.down2 = nn.Conv3d(c2, c2, 4, stride=2, padding=1)
        self.bottleneck = _FiLMConvBlock(c2, c3, self.cond_dim)
        self.up2 = nn.ConvTranspose3d(c3, c2, 4, stride=2, padding=1)
        self.dec2 = _FiLMConvBlock(c2 * 2, c2, self.cond_dim)
        self.up1 = nn.ConvTranspose3d(c2, c1, 4, stride=2, padding=1)
        self.dec1 = _FiLMConvBlock(c1 * 2, c1, self.cond_dim)
        self.out = nn.Conv3d(c1, in_channels, 1)

    def _condition(self, t: torch.Tensor, log_vol: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_timestep_embedding(t, self.cond_dim)
        t_emb = self.time_mlp(t_emb)
        v_emb = self.volume_mlp(log_vol.view(-1, 1).float())
        return t_emb + v_emb

    def forward(self, x: torch.Tensor, t: torch.Tensor, log_vol: torch.Tensor) -> torch.Tensor:
        """Predict noise ε given noisy x_t, timestep t, and scalar log-volume."""
        cond = self._condition(t, log_vol)
        e1 = self.enc1(x, cond)
        e2 = self.enc2(self.down1(e1), cond)
        b = self.bottleneck(self.down2(e2), cond)
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1), cond)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), cond)
        return self.out(d1)


class LesionDDPMSampler:
    """DDPM forward/reverse process - runs alongside `LesionDDPM`."""

    def __init__(self, num_steps: int = 1000, beta_start: float = 1.0e-4, beta_end: float = 0.02) -> None:
        self.num_steps = int(num_steps)
        betas = linear_beta_schedule(self.num_steps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward noising: x_t = sqrt(alpha_cumprod_t) x_0 + sqrt(1-alpha_cumprod_t) ε."""
        device = x_0.device
        ac = self.sqrt_alphas_cumprod.to(device)[t].view(-1, 1, 1, 1, 1)
        oc = self.sqrt_one_minus_alphas_cumprod.to(device)[t].view(-1, 1, 1, 1, 1)
        return ac * x_0 + oc * noise

    @torch.no_grad()
    def sample(
        self,
        model: LesionDDPM,
        *,
        shape: tuple[int, ...],
        log_vol: torch.Tensor,
        device: torch.device,
        n_steps_override: int | None = None,
    ) -> torch.Tensor:
        """Reverse process: start from N(0, I), denoise step by step."""
        x = torch.randn(shape, device=device)
        steps = self.num_steps if n_steps_override is None else int(n_steps_override)
        for t_int in reversed(range(steps)):
            t = torch.full((shape[0],), t_int, dtype=torch.long, device=device)
            eps = model(x, t, log_vol.to(device))
            alpha = self.alphas.to(device)[t_int]
            alpha_cumprod = self.alphas_cumprod.to(device)[t_int]
            beta = self.betas.to(device)[t_int]
            coef = (1.0 - alpha) / torch.sqrt(1.0 - alpha_cumprod)
            mean = (1.0 / torch.sqrt(alpha)) * (x - coef * eps)
            if t_int > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta) * noise
            else:
                x = mean
        return x


__all__ = ["LesionDDPM", "LesionDDPMSampler", "linear_beta_schedule", "sinusoidal_timestep_embedding"]
