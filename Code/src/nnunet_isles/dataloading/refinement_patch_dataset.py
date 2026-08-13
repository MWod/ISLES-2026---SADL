"""Refinement patch dataset.

Extracts 96³ patches centered on each ground-truth lesion CC plus a
configurable number of negative patches per case (from healthy brain
regions). Used to train a small 3D U-Net that refines the patch-level
prediction of any stage-1 segmentation network.

Lesion patches: 26-connectivity CC labeling on the GT mask, then for
each CC compute its centroid, crop a 96³ box around it (with reflect
padding if the centroid is near the volume edge).

Negative patches: sampled uniformly from voxels where the GT mask is
zero AND the image is non-zero (within the brain). One negative per
lesion patch by default.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RefinementPatch:
    """A single 96³ patch extracted around a lesion CC or a negative location."""

    image: np.ndarray  # (1, D, H, W) float32
    mask: np.ndarray  # (D, H, W) uint8
    case_id: str
    centroid_zyx: tuple[int, int, int]
    is_positive: bool
    cc_voxel_count: int  # 0 for negative patches


def _crop_with_padding(arr: np.ndarray, center_zyx: tuple[int, int, int], size: int) -> np.ndarray:
    """Crop a (..., D, H, W) array to `size**3` centered on `center_zyx`, padding with zeros."""
    half = size // 2
    ranges = []
    for _i, c in enumerate(center_zyx):
        lo = c - half
        hi = lo + size
        ranges.append((lo, hi))
    pad: list[tuple[int, int]] = [(0, 0)] * (arr.ndim - 3)
    src_slices: list[slice] = list(arr.shape[: arr.ndim - 3])  # type: ignore[arg-type]
    src_slices = [slice(None)] * (arr.ndim - 3)
    out_shape = list(arr.shape[: arr.ndim - 3]) + [size, size, size]
    out = np.zeros(out_shape, dtype=arr.dtype)
    for i, (lo, hi) in enumerate(ranges):
        ax_max = arr.shape[i + (arr.ndim - 3)]
        src_lo = max(lo, 0)
        src_hi = min(hi, ax_max)
        dst_lo = src_lo - lo
        dst_hi = dst_lo + (src_hi - src_lo)
        if src_hi <= src_lo:
            # Patch fully outside the array - return zeros.
            return out
        # Compose slices below - pad info recorded incidentally.
        pad.append((dst_lo, size - dst_hi))
    # Build slice tuples for source and destination.
    src_idx = tuple(src_slices) + tuple(
        slice(max(lo, 0), min(hi, arr.shape[(arr.ndim - 3) + i])) for i, (lo, hi) in enumerate(ranges)
    )
    dst_idx = tuple(src_slices) + tuple(
        slice(max(0, -lo), max(0, -lo) + (min(hi, arr.shape[(arr.ndim - 3) + i]) - max(lo, 0)))
        for i, (lo, hi) in enumerate(ranges)
    )
    out[dst_idx] = arr[src_idx]
    return out


def extract_lesion_patches(
    image: np.ndarray,
    mask: np.ndarray,
    case_id: str,
    *,
    patch_size: int = 96,
    min_voxels: int = 1,
    connectivity: int = 26,
) -> list[RefinementPatch]:
    """Extract one 96³ patch per GT lesion CC (centered on the CC centroid).

    Args:
        image: (1, D, H, W) or (D, H, W) - single-channel image.
        mask: (D, H, W) binary GT mask.
        case_id: identifier used for traceability.
        patch_size: spatial size of each patch (cubic).
        min_voxels: drop CCs smaller than this.
        connectivity: scipy.ndimage label connectivity (6/18/26).

    Returns:
        A list of RefinementPatch objects (one per CC ≥ min_voxels).
    """
    from scipy.ndimage import label

    if image.ndim == 3:
        image = image[None]
    elif image.ndim != 4:
        raise ValueError(f"image must be (D,H,W) or (1,D,H,W); got shape {image.shape}")

    struct = np.ones((3, 3, 3), dtype=bool) if connectivity == 26 else None
    labelled, n = label(mask.astype(bool), structure=struct)
    out: list[RefinementPatch] = []
    for cc_idx in range(1, n + 1):
        cc_mask = labelled == cc_idx
        vox = int(cc_mask.sum())
        if vox < min_voxels:
            continue
        coords = np.argwhere(cc_mask)
        centroid = tuple(int(round(c)) for c in coords.mean(axis=0))  # type: ignore[assignment]
        img_patch = _crop_with_padding(image, centroid, patch_size)  # type: ignore[arg-type]
        msk_patch = _crop_with_padding(mask, centroid, patch_size)
        out.append(
            RefinementPatch(
                image=img_patch.astype(np.float32),
                mask=msk_patch.astype(np.uint8),
                case_id=case_id,
                centroid_zyx=centroid,  # type: ignore[arg-type]
                is_positive=True,
                cc_voxel_count=vox,
            )
        )
    return out


def extract_negative_patches(
    image: np.ndarray,
    mask: np.ndarray,
    case_id: str,
    *,
    n_negatives: int = 1,
    patch_size: int = 96,
    seed: int | None = None,
    brain_threshold: float = 1e-3,
) -> list[RefinementPatch]:
    """Extract `n_negatives` random patches from non-lesion brain regions."""
    rng = np.random.default_rng(seed)
    if image.ndim == 3:
        image = image[None]
    img0 = image[0]
    brain = (img0.astype(np.float32).__abs__() > brain_threshold) & (mask == 0)
    coords = np.argwhere(brain)
    if coords.size == 0:
        return []
    picks = rng.choice(len(coords), size=min(n_negatives, len(coords)), replace=False)
    out: list[RefinementPatch] = []
    for idx in picks:
        center = tuple(int(c) for c in coords[idx])  # type: ignore[assignment]
        img_patch = _crop_with_padding(image, center, patch_size)  # type: ignore[arg-type]
        msk_patch = _crop_with_padding(mask, center, patch_size)
        out.append(
            RefinementPatch(
                image=img_patch.astype(np.float32),
                mask=msk_patch.astype(np.uint8),
                case_id=case_id,
                centroid_zyx=center,  # type: ignore[arg-type]
                is_positive=False,
                cc_voxel_count=0,
            )
        )
    return out


def save_patches_npz(patches: Iterable[RefinementPatch], output_dir: Path) -> int:
    """Save each RefinementPatch as `<case_id>_<idx>_{pos|neg}.npz`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    per_case: dict[str, int] = {}
    for p in patches:
        idx = per_case.get(p.case_id, 0)
        per_case[p.case_id] = idx + 1
        tag = "pos" if p.is_positive else "neg"
        out_path = output_dir / f"{p.case_id}_{idx:04d}_{tag}.npz"
        np.savez_compressed(
            str(out_path),
            image=p.image,
            mask=p.mask,
            centroid=np.array(p.centroid_zyx, dtype=np.int64),
            is_positive=np.array(p.is_positive),
            cc_voxel_count=np.array(p.cc_voxel_count),
        )
        n += 1
    return n


def load_patch_npz(path: Path) -> RefinementPatch:
    data = np.load(str(path))
    return RefinementPatch(
        image=data["image"],
        mask=data["mask"],
        case_id=path.stem.rsplit("_", 2)[0],
        centroid_zyx=tuple(int(c) for c in data["centroid"]),  # type: ignore[arg-type]
        is_positive=bool(data["is_positive"]),
        cc_voxel_count=int(data["cc_voxel_count"]),
    )


__all__ = [
    "RefinementPatch",
    "extract_lesion_patches",
    "extract_negative_patches",
    "save_patches_npz",
    "load_patch_npz",
]
