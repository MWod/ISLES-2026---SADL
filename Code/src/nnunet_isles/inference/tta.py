"""Extended test-time augmentation - rotation + multi-scale on top of nnU-Net's
default mirror TTA.

Wraps any callable `predictor(patch: Tensor) -> logits: Tensor` and averages
predictions over a set of augmentations + their inverses. Designed to be
composed with nnU-Net's `_internal_maybe_mirror_and_predict` rather than
replace it - mirror TTA stays in upstream; we add rotation + scale on top.

Modes:
  * `mirror_only` (default) - passthrough; no extra TTA over the upstream mirror.
  * `mirror_rot` - adds ±5° rotation TTA around each of the 3 spatial axes.
  * `mirror_rot_scale` - adds rotation + multi-scale TTA (factors 0.9 and 1.1).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

_VALID_MODES = ("mirror_only", "mirror_rot", "mirror_rot_scale")


def _rotate3d_90multiples(x: torch.Tensor, k: int, dims: tuple[int, int]) -> torch.Tensor:
    """torch.rot90 wrapper - used for the special-case 90° step."""
    return torch.rot90(x, k=k, dims=dims)


def _affine_rotate3d(x: torch.Tensor, degrees: float, axis: int) -> torch.Tensor:
    """Affine rotation by `degrees` around the spatial axis `axis ∈ {0,1,2}`.

    `x` is shape `(B, C, D, H, W)`. Uses `torch.nn.functional.grid_sample`
    via an affine matrix for small-angle rotation. Bilinear interpolation
    on the input; the caller is responsible for inverting the rotation
    on the predictor's output (`_affine_rotate3d(out, -degrees, axis)`).
    """
    if x.ndim != 5:
        raise ValueError(f"_affine_rotate3d expects 5D (B,C,D,H,W); got {x.shape}")
    theta = torch.deg2rad(torch.tensor(degrees, device=x.device, dtype=x.dtype))
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    if axis == 0:  # rotate in (H, W) plane (around depth axis)
        R = torch.eye(3, device=x.device, dtype=x.dtype)
        R[1, 1], R[1, 2], R[2, 1], R[2, 2] = cos_t, -sin_t, sin_t, cos_t
    elif axis == 1:  # rotate in (D, W) plane
        R = torch.eye(3, device=x.device, dtype=x.dtype)
        R[0, 0], R[0, 2], R[2, 0], R[2, 2] = cos_t, -sin_t, sin_t, cos_t
    elif axis == 2:  # rotate in (D, H) plane
        R = torch.eye(3, device=x.device, dtype=x.dtype)
        R[0, 0], R[0, 1], R[1, 0], R[1, 1] = cos_t, -sin_t, sin_t, cos_t
    else:
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}")
    affine = torch.zeros(x.shape[0], 3, 4, device=x.device, dtype=x.dtype)
    affine[:, :3, :3] = R
    grid = F.affine_grid(affine, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def _scale3d(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Trilinear resize the 5D tensor by `factor`. Inverse: `_scale3d(out, 1/factor)`
    then trim/pad back to original spatial shape - the wrapper does the trim."""
    if abs(factor - 1.0) < 1e-6:
        return x
    return F.interpolate(x, scale_factor=factor, mode="trilinear", align_corners=False)


def _crop_or_pad_to(x: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    """Center crop or zero-pad a 5D tensor's spatial dims to `target_shape`."""
    cur = x.shape[2:]
    if cur == tuple(target_shape):
        return x
    pads_for_F: list[int] = []
    crops: list[tuple[int, int]] = []
    # F.pad takes (W_lo, W_hi, H_lo, H_hi, D_lo, D_hi).
    for _ax, (c, t) in enumerate(zip(reversed(cur), reversed(target_shape), strict=True)):
        if c < t:
            diff = t - c
            pads_for_F.extend([diff // 2, diff - diff // 2])
            crops.append((0, t))
        else:
            pads_for_F.extend([0, 0])
            start = (c - t) // 2
            crops.append((start, start + t))
    if any(p > 0 for p in pads_for_F):
        x = F.pad(x, pads_for_F)
    # Apply crops along spatial axes (in forward order).
    slices: list[slice] = [slice(None), slice(None)]
    for s, e in reversed(crops):
        slices.append(slice(s, e))
    return x[tuple(slices)]


class TTAWrapper:
    """Composable extension of nnU-Net's mirror TTA.

    Usage:
        tta = TTAWrapper(network_forward_fn, mode="mirror_rot")
        averaged_logits = tta.predict(patch)  # patch shape (B, C, D, H, W)
    """

    def __init__(
        self,
        predict_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        mode: str = "mirror_only",
        rotation_degrees: tuple[float, ...] = (5.0,),
        scale_factors: tuple[float, ...] = (0.9, 1.1),
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}; got {mode!r}")
        self.predict_fn = predict_fn
        self.mode = mode
        self.rotation_degrees = tuple(float(d) for d in rotation_degrees)
        self.scale_factors = tuple(float(s) for s in scale_factors)

    def predict(self, patch: torch.Tensor) -> torch.Tensor:
        """Return averaged predictions over the configured augmentation set.

        For numerical stability, we average in softmax-probability space
        (post-softmax) - assumes the upstream `predict_fn` returns logits and
        softmaxes internally, OR we softmax here. To keep the wrapper agnostic
        of what `predict_fn` returns, we average raw outputs and let the caller
        apply softmax / argmax afterwards.
        """
        outs = [self.predict_fn(patch)]

        if "rot" in self.mode:
            for axis in (0, 1, 2):
                for degrees in self.rotation_degrees:
                    for d in (degrees, -degrees):
                        rotated = _affine_rotate3d(patch, d, axis)
                        pred_rot = self.predict_fn(rotated)
                        # Undo the rotation on the output.
                        outs.append(_affine_rotate3d(pred_rot, -d, axis))

        if "scale" in self.mode:
            target_shape = patch.shape[2:]
            for s in self.scale_factors:
                scaled = _scale3d(patch, s)
                pred_s = self.predict_fn(scaled)
                unscaled = _scale3d(pred_s, 1.0 / s)
                outs.append(_crop_or_pad_to(unscaled, target_shape))

        return torch.stack(outs, dim=0).mean(dim=0)


__all__ = ["TTAWrapper", "_affine_rotate3d", "_scale3d", "_crop_or_pad_to"]
