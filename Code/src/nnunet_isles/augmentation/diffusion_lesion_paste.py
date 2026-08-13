"""C3b - DiffusionLesionPasteTransform.

Sister transform to CarveMix, but the donor pool is the C3a synthetic
lesion bank (DDPM-generated 64^3 patches without an explicit mask). The
mask is derived geometrically: a sphere centered on the patch centroid
whose radius matches the patch's `vol_ml` field (= the target volume the
DDPM was conditioned on). This pairs DDPM-generated lesion intensity
patterns with a clean geometric mask, sidestepping the open problem of
predicting a segmentation mask from a generative model.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform

from nnunet_isles.registry import AUGMENTATION_REGISTRY


def _sphere_mask(
    shape: tuple[int, int, int], center: tuple[int, int, int], radius_voxels: float
) -> np.ndarray:
    """Boolean sphere of the given radius (voxels) at `center` inside `shape`."""
    cz, cy, cx = center
    z = np.arange(shape[0])[:, None, None] - cz
    y = np.arange(shape[1])[None, :, None] - cy
    x = np.arange(shape[2])[None, None, :] - cx
    return (z * z + y * y + x * x) <= radius_voxels * radius_voxels


@AUGMENTATION_REGISTRY.register("diffusion_lesion")
class DiffusionLesionPasteTransform(BasicTransform):
    """Paste a DDPM-synthesized small-lesion patch into the recipient with a
    geometric sphere mask sized to the patch's target volume."""

    def __init__(
        self,
        p: float = 0.3,
        bank_dir: str | None = None,
        feather_sigma: float = 1.5,
        enabled: bool = False,
        brain_threshold: float | None = None,
        # Voxel volume in mm^3 - used to convert vol_ml ↔ voxel count.
        # Defaults to 1.0 (1mm³ iso). Set this from the calling trainer if needed.
        voxel_volume_mm3: float = 1.0,
    ) -> None:
        super().__init__()
        self.p = float(p)
        self.enabled = bool(enabled)
        self.bank_dir = Path(bank_dir) if bank_dir else None
        self.feather_sigma = float(feather_sigma)
        self.brain_threshold = brain_threshold
        self.voxel_volume_mm3 = float(voxel_volume_mm3)
        self._bank_files: list[Path] | None = None

    def _ensure_bank_loaded(self) -> list[Path]:
        if self._bank_files is None:
            if self.bank_dir is None or not self.bank_dir.exists():
                raise FileNotFoundError(
                    f"DiffusionLesionPaste bank not found at {self.bank_dir}. "
                    "Run scripts/sample_synthetic_lesions.py first."
                )
            self._bank_files = sorted(self.bank_dir.glob("*.npz"))
            if not self._bank_files:
                raise FileNotFoundError(f"Synthetic bank {self.bank_dir} is empty.")
        return self._bank_files

    def get_parameters(self, **data_dict) -> dict:
        if not self.enabled or self.p <= 0.0 or torch.rand(()) > self.p:
            return {"apply": False}
        bank = self._ensure_bank_loaded()
        idx = int(torch.randint(0, len(bank), ()).item())
        return {"apply": True, "bank_file": bank[idx]}

    def apply(self, data_dict: dict, **params) -> dict:
        if not params.get("apply", False):
            return data_dict
        img = data_dict.get("image")
        seg = data_dict.get("segmentation")
        if img is None or seg is None:
            return data_dict
        try:
            with np.load(params["bank_file"]) as npz:
                synth_image = npz["image"]  # (1, 64, 64, 64) in DDPM normalised space ([-1, 1])
                vol_ml = float(npz["vol_ml"])
        except (OSError, KeyError):
            return data_dict

        img_np = img.detach().cpu().numpy()
        seg_np = seg.detach().cpu().numpy()
        new_img, new_seg = self._paste(img_np, seg_np, synth_image[0], vol_ml)
        data_dict["image"] = torch.from_numpy(new_img).to(img.device, dtype=img.dtype)
        data_dict["segmentation"] = torch.from_numpy(new_seg).to(seg.device, dtype=seg.dtype)
        return data_dict

    def _paste(
        self,
        img: np.ndarray,
        seg: np.ndarray,
        synth_image: np.ndarray,
        vol_ml: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        # img: (C, X, Y, Z), seg: (1, X, Y, Z), synth_image: (X', Y', Z').
        c = img.shape[0]
        spatial = img.shape[1:]
        synth_shape = synth_image.shape
        if any(d > s for d, s in zip(synth_shape, spatial, strict=False)):
            return img, seg

        thr = self.brain_threshold if self.brain_threshold is not None else float(img[0].min())
        brain = img[0] > thr
        if not brain.any():
            return img, seg

        # Geometric mask: sphere whose volume in mL matches vol_ml.
        radius_voxels = (vol_ml * 1000.0 / self.voxel_volume_mm3 * 3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
        radius_voxels = max(1.0, radius_voxels)
        center = tuple(s // 2 for s in synth_shape)
        synth_mask = _sphere_mask(synth_shape, center, radius_voxels)
        if not synth_mask.any():
            return img, seg

        # Sample a paste location inside the brain.
        donor_centre = center
        valid = brain.copy()
        for axis, (dc, ds) in enumerate(zip(donor_centre, synth_shape, strict=False)):
            lo = dc
            hi = spatial[axis] - (ds - dc)
            slicer = [slice(None)] * 3
            slicer[axis] = slice(0, lo)
            valid[tuple(slicer)] = False
            slicer[axis] = slice(hi, None)
            valid[tuple(slicer)] = False
        coords = np.argwhere(valid)
        if coords.size == 0:
            return img, seg
        cx, cy, cz = coords[np.random.randint(len(coords))]
        x0, y0, z0 = int(cx) - center[0], int(cy) - center[1], int(cz) - center[2]
        x1, y1, z1 = x0 + synth_shape[0], y0 + synth_shape[1], z0 + synth_shape[2]

        # Compute feathered weight from the sphere mask.
        try:
            from scipy.ndimage import distance_transform_edt, gaussian_filter

            dist = distance_transform_edt(synth_mask).astype(np.float32)
            if dist.max() > 0:
                dist = dist / dist.max()
            weight = gaussian_filter(dist, sigma=self.feather_sigma).astype(np.float32)
        except ImportError:
            weight = synth_mask.astype(np.float32)

        new_img = img.copy()
        for ch in range(c):
            recipient_crop = new_img[ch, x0:x1, y0:y1, z0:z1]
            # Match synth intensities to recipient stats.
            recipient_fg = recipient_crop[recipient_crop > thr]
            if recipient_fg.size > 0:
                r_mean = float(recipient_fg.mean())
                r_std = float(recipient_fg.std() + 1e-6)
            else:
                r_mean, r_std = 0.0, 1.0
            s_mean, s_std = float(synth_image.mean()), float(synth_image.std() + 1e-6)
            synth_matched = (synth_image - s_mean) / s_std * r_std + r_mean
            new_img[ch, x0:x1, y0:y1, z0:z1] = weight * synth_matched + (1.0 - weight) * recipient_crop

        new_seg = seg.copy()
        new_seg[0, x0:x1, y0:y1, z0:z1] = np.maximum(
            new_seg[0, x0:x1, y0:y1, z0:z1], synth_mask.astype(seg.dtype)
        )
        return new_img, new_seg


__all__ = ["DiffusionLesionPasteTransform"]
