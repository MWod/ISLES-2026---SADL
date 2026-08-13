"""SimpleITK N4 bias field correction wrapper.

Used optionally inside IslesPreprocessor. Cached on disk by case to avoid
re-running on every nnU-Net preprocessing pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def run_n4_bias_correction(
    image: np.ndarray, *, shrink_factor: int = 2, n_iterations: int = 50
) -> np.ndarray:
    """Run N4 bias correction on a single 3D volume, returning the corrected float array."""
    import SimpleITK as sitk

    sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
    mask = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([n_iterations] * 4)
    if shrink_factor > 1:
        small = sitk.Shrink(sitk_image, [shrink_factor] * sitk_image.GetDimension())
        small_mask = sitk.Shrink(mask, [shrink_factor] * sitk_image.GetDimension())
        corrector.Execute(small, small_mask)
        log_bias = corrector.GetLogBiasFieldAsImage(sitk_image)
        corrected = sitk_image / sitk.Exp(log_bias)
    else:
        corrected = corrector.Execute(sitk_image, mask)
    return sitk.GetArrayFromImage(corrected).astype(np.float32)


def maybe_n4(
    image: np.ndarray,
    case_id: str | None = None,
    cache_dir: str | Path | None = None,
    *,
    shrink_factor: int = 2,
    n_iterations: int = 50,
) -> np.ndarray:
    """Wrapper that caches the N4 result per-case to disk."""
    if cache_dir is None or case_id is None:
        return run_n4_bias_correction(image, shrink_factor=shrink_factor, n_iterations=n_iterations)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{case_id}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    corrected = run_n4_bias_correction(image, shrink_factor=shrink_factor, n_iterations=n_iterations)
    np.save(cache_path, corrected)
    return corrected
