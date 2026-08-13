"""Tests for the TTAWrapper."""

from __future__ import annotations

import pytest
import torch
from nnunet_isles.inference.tta import (
    TTAWrapper,
    _affine_rotate3d,
    _crop_or_pad_to,
    _scale3d,
)


def _identity_predictor(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


def test_mirror_only_is_passthrough():
    """With mode=mirror_only, TTAWrapper.predict should equal the single call."""
    tta = TTAWrapper(_identity_predictor, mode="mirror_only")
    patch = torch.randn(1, 1, 8, 8, 8)
    out = tta.predict(patch)
    torch.testing.assert_close(out, patch, rtol=1e-5, atol=1e-5)


def test_rot_tta_averages_over_inverse_rotations_for_identity_predictor():
    """For an identity predictor, the inverse rotation should bring the
    augmented prediction back near the original - averaging produces
    approximately the input modulo interpolation noise.
    """
    tta = TTAWrapper(_identity_predictor, mode="mirror_rot", rotation_degrees=(5.0,))
    patch = torch.randn(1, 1, 16, 16, 16)
    out = tta.predict(patch)
    # Output is the same shape as input.
    assert out.shape == patch.shape
    # For small rotations of an identity predictor with bilinear interpolation,
    # the averaged result loses high-frequency content but mean roughly matches.
    assert torch.abs(out.mean() - patch.mean()) < 0.05


def test_scale_tta_preserves_shape():
    tta = TTAWrapper(_identity_predictor, mode="mirror_rot_scale", scale_factors=(0.9, 1.1))
    patch = torch.randn(1, 2, 8, 8, 8)
    out = tta.predict(patch)
    assert out.shape == patch.shape


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        TTAWrapper(_identity_predictor, mode="invalid")


def test_affine_rotate_3d_zero_degrees_identity():
    x = torch.randn(1, 1, 8, 8, 8)
    out = _affine_rotate3d(x, 0.0, axis=0)
    # Zero rotation grid_sample = approximate identity (small interpolation slop ok).
    torch.testing.assert_close(out, x, rtol=1e-4, atol=1e-4)


def test_scale3d_factor_one_is_identity():
    x = torch.randn(1, 1, 8, 8, 8)
    out = _scale3d(x, 1.0)
    torch.testing.assert_close(out, x)


def test_crop_or_pad_to_handles_both_directions():
    x = torch.ones(1, 1, 6, 6, 6)
    # Pad up to 10
    padded = _crop_or_pad_to(x, (10, 10, 10))
    assert padded.shape[2:] == (10, 10, 10)
    # Crop down to 4
    cropped = _crop_or_pad_to(x, (4, 4, 4))
    assert cropped.shape[2:] == (4, 4, 4)


def test_predict_fn_called_more_times_under_rot_mode():
    """Mirror-only: 1 call. mirror_rot: 1 + (axes × angles × directions) = 1 + 3*1*2 = 7 calls."""
    calls = []

    def counting_predictor(x):
        calls.append(0)
        return x.clone()

    tta = TTAWrapper(counting_predictor, mode="mirror_rot", rotation_degrees=(5.0,))
    patch = torch.zeros(1, 1, 4, 4, 4)
    _ = tta.predict(patch)
    assert len(calls) == 1 + 3 * 1 * 2
