"""Tests for the connected-component post-processing."""

from __future__ import annotations

import numpy as np
from nnunet_isles.inference.cc_postproc import apply_cc_filter, apply_lesion_size_suppression


def test_drops_small_cc_only():
    """A 5-voxel CC and a 100-voxel CC at min_voxels=10: small drops, large survives."""
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    # Small CC: 5 voxels in a corner.
    mask[1:2, 1:2, 1:6] = 1  # 5 voxels
    # Large CC: 5×5×5 = 125 voxels far away (no contact).
    mask[10:15, 10:15, 10:15] = 1
    out = apply_cc_filter(mask, min_voxels=10)
    assert out[1:2, 1:2, 1:6].sum() == 0  # small dropped
    assert out[10:15, 10:15, 10:15].sum() == 125  # large preserved


def test_empty_input_unchanged():
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    out = apply_cc_filter(mask, min_voxels=10)
    assert out.sum() == 0


def test_min_voxels_one_is_noop():
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[0, 0, 0] = 1
    out = apply_cc_filter(mask, min_voxels=1)
    assert out[0, 0, 0] == 1


def test_26_connectivity_merges_diagonals():
    """26-connectivity: corner-touching voxels form one CC. With min=2, they survive."""
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[1, 1, 1] = 1
    mask[2, 2, 2] = 1  # diagonal-touching
    out = apply_cc_filter(mask, min_voxels=2, connectivity=26)
    assert out.sum() == 2  # both preserved (one CC of size 2)
    # With 6-connectivity, these are separate CCs of size 1 each → both dropped.
    out6 = apply_cc_filter(mask, min_voxels=2, connectivity=6)
    assert out6.sum() == 0


def test_lesion_size_suppression_drops_small_bucket():
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[1:2, 1:2, 1:2] = 1  # 1 voxel = tiny CC (<0.5 mL at 1mm³)
    mask[10:15, 10:15, 10:15] = 1  # 125 voxels = 0.125 mL - still <0.5 mL
    mask[18:20, 18:20, 0:20] = 1  # 80 voxels - still <0.5 mL

    curve = {"<0.5mL": 0.0, "0.5-5mL": 1.0, "5-50mL": 1.0, ">=50mL": 1.0}
    out = apply_lesion_size_suppression(mask, voxel_volume_mm3=1.0, suppress_curve=curve)
    # All three CCs are <0.5 mL → all should be suppressed.
    assert out.sum() == 0


def test_lesion_size_suppression_keeps_large_bucket():
    mask = np.zeros((100, 100, 100), dtype=np.uint8)
    # 10 mL = 10000 voxels at 1mm³ → 0.5-5mL? No, 5-50mL.
    mask[0:25, 0:20, 0:20] = 1  # 10000 voxels = 10 mL → "5-50mL"
    curve = {"<0.5mL": 0.0, "0.5-5mL": 0.0, "5-50mL": 1.0, ">=50mL": 1.0}
    out = apply_lesion_size_suppression(mask, voxel_volume_mm3=1.0, suppress_curve=curve)
    assert out.sum() == 10000


def test_preserves_dtype():
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[2, 2, 2] = 1
    out = apply_cc_filter(mask, min_voxels=2)
    assert out.dtype == np.uint8
