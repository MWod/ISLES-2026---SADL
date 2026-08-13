"""Tests for the per-CC probability statistics module."""

from __future__ import annotations

import numpy as np
from nnunet_isles.inference.cc_stats import (
    adaptive_size_gate,
    cc_probability_stats,
    drop_low_confidence_ccs,
)


def _build_two_cc_fixture() -> tuple[np.ndarray, np.ndarray]:
    """Two well-separated CCs in a (32, 32, 32) volume.

    CC-A: 10 voxels, max_prob=0.9, mean_prob=0.85, prob_mass=8.5
        (5 voxels at 0.9, 5 voxels at 0.8)
    CC-B: 200 voxels, max_prob=0.4, mean_prob=0.35, prob_mass=70.0
        (100 voxels at 0.4, 100 voxels at 0.3)
    """
    shape = (32, 32, 32)
    prob_fg = np.zeros(shape, dtype=np.float64)
    mask = np.zeros(shape, dtype=np.uint8)

    # CC-A: 1x1x10 line at [1, 1, 1:11].
    mask[1, 1, 1:11] = 1
    prob_fg[1, 1, 1:6] = 0.9
    prob_fg[1, 1, 6:11] = 0.8

    # CC-B: 10x10x2 slab at [10:20, 10:20, 10:12].
    mask[10:20, 10:20, 10:12] = 1
    prob_fg[10:20, 10:20, 10] = 0.4
    prob_fg[10:20, 10:20, 11] = 0.3

    return prob_fg, mask


def _find_by_voxels(stats: list[dict], voxels: int) -> dict:
    matches = [s for s in stats if s["voxels"] == voxels]
    assert len(matches) == 1, f"expected exactly one CC with {voxels} voxels, got {len(matches)}"
    return matches[0]


def test_cc_probability_stats_two_ccs():
    prob_fg, mask = _build_two_cc_fixture()
    stats = cc_probability_stats(prob_fg, mask)
    assert len(stats) == 2

    cc_a = _find_by_voxels(stats, 10)
    assert cc_a["max_prob"] == pytest_approx_close(0.9)
    assert cc_a["mean_prob"] == pytest_approx_close(0.85)
    assert cc_a["prob_mass"] == pytest_approx_close(8.5)
    assert cc_a["bbox"] == ((1, 2), (1, 2), (1, 11))
    assert isinstance(cc_a["label"], int)

    cc_b = _find_by_voxels(stats, 200)
    assert cc_b["max_prob"] == pytest_approx_close(0.4)
    assert cc_b["mean_prob"] == pytest_approx_close(0.35)
    assert cc_b["prob_mass"] == pytest_approx_close(70.0)
    assert cc_b["bbox"] == ((10, 20), (10, 20), (10, 12))


def test_cc_probability_stats_empty_mask():
    prob_fg = np.zeros((8, 8, 8), dtype=np.float32)
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    assert cc_probability_stats(prob_fg, mask) == []


def test_drop_low_confidence_min_max_prob_drops_cc_b():
    prob_fg, mask = _build_two_cc_fixture()
    out = drop_low_confidence_ccs(prob_fg, mask, min_max_prob=0.5)
    assert out.dtype == np.uint8
    assert out.shape == mask.shape
    # CC-A (max=0.9 >= 0.5) survives.
    assert out[1, 1, 1:11].sum() == 10
    # CC-B (max=0.4 < 0.5) dropped.
    assert out[10:20, 10:20, 10:12].sum() == 0


def test_drop_low_confidence_min_voxels_drops_cc_a():
    prob_fg, mask = _build_two_cc_fixture()
    out = drop_low_confidence_ccs(prob_fg, mask, min_voxels=50)
    assert out[1, 1, 1:11].sum() == 0  # CC-A (10 voxels) dropped
    assert out[10:20, 10:20, 10:12].sum() == 200  # CC-B kept


def test_drop_low_confidence_defaults_are_identity():
    prob_fg, mask = _build_two_cc_fixture()
    out = drop_low_confidence_ccs(prob_fg, mask)
    assert out.dtype == np.uint8
    assert out.shape == mask.shape
    assert int(out.sum()) == int(mask.astype(bool).sum())


def test_drop_low_confidence_empty_mask_returns_zeros():
    prob_fg = np.zeros((8, 8, 8), dtype=np.float32)
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    out = drop_low_confidence_ccs(prob_fg, mask, min_max_prob=0.5)
    assert out.shape == mask.shape
    assert out.dtype == np.uint8
    assert out.sum() == 0


def test_drop_low_confidence_all_thresholds_anded():
    """A CC must clear ALL min_* to survive."""
    prob_fg, mask = _build_two_cc_fixture()
    # CC-A meets min_max_prob but fails min_voxels.
    out = drop_low_confidence_ccs(prob_fg, mask, min_max_prob=0.5, min_voxels=50)
    assert out.sum() == 0  # both dropped: CC-A fails size, CC-B fails prob


def test_adaptive_size_gate_high_conf_low_floor():
    """size_curve=[(0.0, 500), (0.8, 5)]: CC-A keeps (10>=5), CC-B drops (200<500)."""
    prob_fg, mask = _build_two_cc_fixture()
    out = adaptive_size_gate(prob_fg, mask, size_curve=[(0.0, 500), (0.8, 5)])
    assert out.dtype == np.uint8
    assert out[1, 1, 1:11].sum() == 10
    assert out[10:20, 10:20, 10:12].sum() == 0


def test_adaptive_size_gate_empty_curve_is_identity():
    prob_fg, mask = _build_two_cc_fixture()
    out = adaptive_size_gate(prob_fg, mask, size_curve=[])
    assert out.dtype == np.uint8
    assert int(out.sum()) == int(mask.astype(bool).sum())


def test_adaptive_size_gate_empty_mask():
    prob_fg = np.zeros((8, 8, 8), dtype=np.float32)
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    out = adaptive_size_gate(prob_fg, mask, size_curve=[(0.0, 5)])
    assert out.shape == mask.shape
    assert out.dtype == np.uint8
    assert out.sum() == 0


def test_connectivity_26_merges_diagonal_singletons():
    """Two voxels touching only at a corner: 26-conn merges, 6-conn separates."""
    shape = (5, 5, 5)
    prob_fg = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[1, 1, 1] = 1
    mask[2, 2, 2] = 1
    prob_fg[1, 1, 1] = 0.9
    prob_fg[2, 2, 2] = 0.9

    stats_26 = cc_probability_stats(prob_fg, mask, connectivity=26)
    assert len(stats_26) == 1
    assert stats_26[0]["voxels"] == 2

    stats_6 = cc_probability_stats(prob_fg, mask, connectivity=6)
    assert len(stats_6) == 2
    for s in stats_6:
        assert s["voxels"] == 1


def test_shape_preserved_across_all_apis():
    prob_fg, mask = _build_two_cc_fixture()
    assert drop_low_confidence_ccs(prob_fg, mask, min_voxels=1).shape == mask.shape
    assert adaptive_size_gate(prob_fg, mask, size_curve=[(0.0, 1)]).shape == mask.shape


def pytest_approx_close(value: float, tol: float = 1e-6) -> _Approx:
    """Small local approx helper so we do not depend on pytest.approx signature."""
    return _Approx(value, tol)


class _Approx:
    __slots__ = ("value", "tol")

    def __init__(self, value: float, tol: float) -> None:
        self.value = value
        self.tol = tol

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (int, float, np.floating, np.integer)):
            return NotImplemented
        return abs(float(other) - self.value) <= self.tol

    def __repr__(self) -> str:
        return f"~{self.value}+/-{self.tol}"
