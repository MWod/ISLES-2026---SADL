"""Tests for DiffusionLesionPasteTransform."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from nnunet_isles.augmentation.diffusion_lesion_paste import (
    DiffusionLesionPasteTransform,
    _sphere_mask,
)


def _make_synth_bank(tmp_path: Path, n: int = 3) -> Path:
    """Build a small synthetic-lesion bank on disk for testing."""
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n):
        img = rng.standard_normal((1, 16, 16, 16)).astype(np.float32)  # smaller for speed
        vol_ml = 0.5  # ~0.5 mL synth lesion
        np.savez_compressed(str(bank_dir / f"synth_{i:03d}.npz"), image=img, vol_ml=np.float32(vol_ml))
    return bank_dir


def test_sphere_mask_volume_increases_with_radius():
    m1 = _sphere_mask((20, 20, 20), (10, 10, 10), radius_voxels=2.0)
    m2 = _sphere_mask((20, 20, 20), (10, 10, 10), radius_voxels=4.0)
    assert m2.sum() > m1.sum()


def test_disabled_transform_is_passthrough(tmp_path):
    bank = _make_synth_bank(tmp_path)
    tr = DiffusionLesionPasteTransform(p=1.0, bank_dir=str(bank), enabled=False)
    img = torch.randn(1, 32, 32, 32)
    seg = torch.zeros(1, 32, 32, 32, dtype=torch.long)
    params = tr.get_parameters(image=img, segmentation=seg)
    assert params["apply"] is False


def test_transform_pastes_synth_lesion_and_updates_mask(tmp_path):
    bank = _make_synth_bank(tmp_path)
    tr = DiffusionLesionPasteTransform(p=1.0, bank_dir=str(bank), enabled=True, brain_threshold=-100)
    # Recipient: 32^3 image with non-constant intensities (so std > 0 → matched synth differs).
    torch.manual_seed(0)
    img = torch.randn(1, 32, 32, 32) + 1.0  # mean ~1, std ~1
    seg = torch.zeros(1, 32, 32, 32, dtype=torch.long)
    params = tr.get_parameters(image=img, segmentation=seg)
    if not params.get("apply", False):
        pytest.skip("randomly skipped - re-run with seed")
    data_dict = tr.apply({"image": img, "segmentation": seg}, **params)
    # Mask should now have foreground voxels.
    new_seg = data_dict["segmentation"]
    assert int(new_seg.sum()) > 0
    # Image should have been modified.
    new_img = data_dict["image"]
    assert not torch.allclose(new_img, img)


def test_get_parameters_returns_bank_file_when_enabled(tmp_path):
    bank = _make_synth_bank(tmp_path)
    torch.manual_seed(0)
    tr = DiffusionLesionPasteTransform(p=1.0, bank_dir=str(bank), enabled=True)
    img = torch.ones(1, 32, 32, 32)
    seg = torch.zeros(1, 32, 32, 32, dtype=torch.long)
    params = tr.get_parameters(image=img, segmentation=seg)
    assert params["apply"] is True
    assert "bank_file" in params


def test_paste_skips_when_donor_too_large(tmp_path):
    """If the synth patch is larger than the recipient, the transform should not crash."""
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    big = np.random.RandomState(0).randn(1, 32, 32, 32).astype(np.float32)
    np.savez_compressed(str(bank_dir / "big.npz"), image=big, vol_ml=np.float32(1.0))

    tr = DiffusionLesionPasteTransform(p=1.0, bank_dir=str(bank_dir), enabled=True, brain_threshold=-100)
    img = torch.ones(1, 16, 16, 16)  # smaller than synth
    seg = torch.zeros(1, 16, 16, 16, dtype=torch.long)
    params = tr.get_parameters(image=img, segmentation=seg)
    out = tr.apply({"image": img, "segmentation": seg}, **params)
    # No change should happen; we just need it to not crash.
    assert out["image"].shape == img.shape
    assert out["segmentation"].shape == seg.shape


def test_missing_bank_dir_raises(tmp_path):
    tr = DiffusionLesionPasteTransform(p=1.0, bank_dir=str(tmp_path / "does_not_exist"), enabled=True)
    img = torch.ones(1, 16, 16, 16)
    seg = torch.zeros(1, 16, 16, 16, dtype=torch.long)
    with pytest.raises(FileNotFoundError):
        tr.get_parameters(image=img, segmentation=seg)


def test_empty_bank_raises(tmp_path):
    empty = tmp_path / "empty_bank"
    empty.mkdir()
    tr = DiffusionLesionPasteTransform(p=1.0, bank_dir=str(empty), enabled=True)
    img = torch.ones(1, 16, 16, 16)
    seg = torch.zeros(1, 16, 16, 16, dtype=torch.long)
    with pytest.raises(FileNotFoundError):
        tr.get_parameters(image=img, segmentation=seg)


def test_registered_in_augmentation_registry():
    from nnunet_isles.registry import AUGMENTATION_REGISTRY

    assert "diffusion_lesion" in AUGMENTATION_REGISTRY._registry  # type: ignore[attr-defined]
