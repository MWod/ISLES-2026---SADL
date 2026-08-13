"""Intensity harmonisation hooks.

`harmonize` dispatches by name to a registered harmonizer. Implementations:
  - "none": identity (default)
  - "white_stripe": Shinohara et al. NeuroImage Clin. 2014. Locates the
    intensity band centred on the white-matter mode and z-scores the whole
    foreground using its mean / std. For T1-weighted MRI the white-stripe is
    the rightmost mode of the intensity histogram (WM is bright).
  - "iguane": IGUANe image-level GAN harmonization (Roca et al. MedIA 2024).
    Not wired - needs the generator checkpoint. Raises NotImplementedError so
    a bad config fails fast.
"""

from __future__ import annotations

import numpy as np

from nnunet_isles.registry import HARMONIZER_REGISTRY


@HARMONIZER_REGISTRY.register("none")
def harmonize_identity(image: np.ndarray, **_: object) -> np.ndarray:
    return image


@HARMONIZER_REGISTRY.register("white_stripe")
def harmonize_white_stripe(
    image: np.ndarray,
    *,
    modality: str = "t1",
    width: float = 0.05,
    foreground_threshold: float | None = None,
    **_: object,
) -> np.ndarray:
    """White-stripe normalisation (Shinohara 2014).

    1. Compute foreground mask: image > threshold (default: image > 0).
    2. Estimate KDE on the foreground intensity histogram.
    3. Find the appropriate modality-specific mode:
       - T1: rightmost local maximum of the KDE (white matter is hyperintense)
       - T2 / FLAIR: leftmost local maximum among the high-intensity peaks
    4. Define the white-stripe as the band of intensities within ±width
       of that mode (where width is a relative window - default 5%).
    5. Compute mean (mu_ws) and std (sigma_ws) over voxels inside the stripe.
    6. Return (image - mu_ws) / sigma_ws.

    For ATLAS-v2 T1w, this stabilises intensity scales across scanners while
    preserving the lesion-vs-WM contrast (lesions sit BELOW the WM peak).
    """
    img = np.asarray(image, dtype=np.float32)
    thr = float(foreground_threshold) if foreground_threshold is not None else 0.0
    fg = img[img > thr]
    if fg.size < 256:
        return img

    # Histogram-based mode finder (cheaper than full KDE and adequate at
    # 200 bins given typical brain foreground voxel counts of 1e5-1e6).
    n_bins = 200
    hist, edges = np.histogram(fg, bins=n_bins)
    # Smooth with a small triangular kernel.
    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    kernel = kernel / kernel.sum()
    smooth = np.convolve(hist.astype(np.float32), kernel, mode="same")

    # Mode selection.
    if modality.lower() == "t1":
        # Walk from the right and pick the first local maximum.
        mode_idx = int(np.argmax(smooth))
        for i in range(n_bins - 2, 0, -1):
            if smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]:
                mode_idx = i
                break
    else:
        mode_idx = int(np.argmax(smooth))

    mode_val = 0.5 * float(edges[mode_idx] + edges[mode_idx + 1])
    if mode_val == 0:
        return img

    # White-stripe band: ±width * (p99 - p1) around the mode.
    p1, p99 = np.percentile(fg, [1, 99])
    half = float(width) * float(p99 - p1)
    lo, hi = mode_val - half, mode_val + half
    band = fg[(fg >= lo) & (fg <= hi)]
    if band.size < 32:
        return img
    mu = float(band.mean())
    sigma = float(band.std() + 1e-6)
    return ((img - mu) / sigma).astype(np.float32)


@HARMONIZER_REGISTRY.register("iguane")
def harmonize_iguane(image: np.ndarray, **kwargs: object) -> np.ndarray:
    raise NotImplementedError("IGUANe harmoniser not yet wired.")


def harmonize(name: str, image: np.ndarray, **kwargs: object) -> np.ndarray:
    fn = HARMONIZER_REGISTRY.get(name)
    return fn(image, **kwargs)  # type: ignore[misc]
