"""Tests for the per-sample volume weight helper."""

from __future__ import annotations

import pytest
import torch
from nnunet_isles.losses._volume_weights import (
    bucket_for_volume,
    compute_per_sample_volume_ml,
    compute_per_sample_volume_weight,
    log_volume_for_conditioning,
)


def test_volume_ml_scales_with_voxel_count():
    seg = torch.zeros((2, 1, 8, 8, 8), dtype=torch.int64)
    seg[0, 0, 0:2, 0:2, 0:2] = 1  # 8 voxels
    seg[1, 0, 0:4, 0:4, 0:4] = 1  # 64 voxels
    vol = compute_per_sample_volume_ml(seg, (1.0, 1.0, 1.0))
    assert vol.shape == (2,)
    assert float(vol[0]) == pytest.approx(8 / 1000.0, rel=1e-4)
    assert float(vol[1]) == pytest.approx(64 / 1000.0, rel=1e-4)


def test_volume_weight_clips_and_inverts_volume():
    # Small lesion (0.5 mL) → high weight; large (10 mL) → low; empty → fallback.
    seg = torch.zeros((3, 1, 10, 10, 10), dtype=torch.int64)
    # Sample 0: 0.5 mL = 500 voxels at 1mm³
    seg[0, 0].view(-1)[:500] = 1
    # Sample 1: 10 mL = 1000 voxels at 1mm³  (actually 10 mL = 10000 voxels - fix)
    seg[1, 0].view(-1)[:10] = 1  # 10 voxels = 0.01 mL → very small → cap at w_max
    # Sample 2: empty
    w = compute_per_sample_volume_weight(seg, (1.0, 1.0, 1.0), target_ml=5.0)
    assert w.shape == (3,)
    # 0.5 mL: w = 5/0.5 = 10 → clipped to 4.0
    assert float(w[0]) == 4.0
    # 0.01 mL: very small, clipped to 4.0
    assert float(w[1]) == 4.0
    # empty: fallback 2.0
    assert float(w[2]) == 2.0


def test_volume_weight_unit_at_target():
    """At vol == target_ml, weight should be exactly 1.0 (well, before clip)."""
    seg = torch.zeros((1, 1, 100, 100, 100), dtype=torch.int64)
    # 5 mL = 5000 voxels at 1 mm³.
    seg[0, 0].view(-1)[:5000] = 1
    w = compute_per_sample_volume_weight(seg, (1.0, 1.0, 1.0), target_ml=5.0)
    assert abs(float(w[0]) - 1.0) < 1e-5


def test_volume_weight_floor_at_large_lesion():
    """vol = 50 mL → weight clipped down to w_min."""
    seg = torch.zeros((1, 1, 100, 100, 100), dtype=torch.int64)
    seg[0, 0].view(-1)[:50_000] = 1  # 50 mL
    w = compute_per_sample_volume_weight(seg, (1.0, 1.0, 1.0), target_ml=5.0)
    assert float(w[0]) == 0.5


def test_log_volume_for_conditioning_finite_on_empty():
    seg = torch.zeros((1, 1, 4, 4, 4), dtype=torch.int64)
    out = log_volume_for_conditioning(seg, (1.0, 1.0, 1.0))
    assert torch.isfinite(out).all()
    assert float(out[0]) == 0.0  # log(0 + 1) = 0


def test_bucket_for_volume_matches_eda_scheme():
    assert bucket_for_volume(0.1) == "<0.5mL"
    assert bucket_for_volume(2.0) == "0.5-5mL"
    assert bucket_for_volume(20.0) == "5-50mL"
    assert bucket_for_volume(100.0) == ">=50mL"


def test_volume_ml_respects_anisotropic_spacing():
    seg = torch.zeros((1, 1, 4, 4, 4), dtype=torch.int64)
    seg[0, 0] = 1  # 64 voxels
    vol = compute_per_sample_volume_ml(seg, (2.0, 1.0, 0.5))
    # Voxel volume = 2*1*0.5 = 1 mm³  → still 64 mm³ = 0.064 mL
    assert abs(float(vol[0]) - 0.064) < 1e-6
