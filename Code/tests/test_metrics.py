"""Metric correctness on tiny hand-built cases."""

from __future__ import annotations

import numpy as np
from nnunet_isles.evaluation.metrics import (
    absolute_volume_difference_ml,
    dice_coefficient,
    lesion_count_f1,
    lesion_wise_f1,
)


def _box_mask(shape, ranges):
    m = np.zeros(shape, dtype=np.uint8)
    sl = tuple(slice(*r) for r in ranges)
    m[sl] = 1
    return m


def test_dice_identity():
    m = _box_mask((10, 10, 10), [(2, 8), (2, 8), (2, 8)])
    assert dice_coefficient(m, m) == 1.0


def test_dice_disjoint():
    a = _box_mask((10, 10, 10), [(0, 3), (0, 3), (0, 3)])
    b = _box_mask((10, 10, 10), [(7, 10), (7, 10), (7, 10)])
    assert dice_coefficient(a, b) < 1.0e-3


def test_dice_empty_both():
    z = np.zeros((4, 4, 4), dtype=np.uint8)
    assert dice_coefficient(z, z) == 1.0


def test_avd_zero_when_identical():
    m = _box_mask((10, 10, 10), [(2, 8), (2, 8), (2, 8)])
    assert absolute_volume_difference_ml(m, m, (1.0, 1.0, 1.0)) == 0.0


def test_lesion_count_f1_perfect():
    m = _box_mask((10, 10, 10), [(2, 4), (2, 4), (2, 4)])
    result = lesion_count_f1(m, m)
    assert result["count_f1"] == 1.0
    assert result["count_abs_diff"] == 0


def test_lesion_wise_f1_identity():
    m = _box_mask((10, 10, 10), [(2, 4), (2, 4), (2, 4)])
    assert lesion_wise_f1(m, m) == 1.0
