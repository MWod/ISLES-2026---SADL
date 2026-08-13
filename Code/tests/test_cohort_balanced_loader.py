"""Tests for CohortBalancedDataLoader3D + compute_cohort_balanced_probabilities."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from nnunet_isles.dataloading.cohort_balanced_loader import (
    compute_cohort_balanced_probabilities,
)


def test_uniform_cohort_weights_yield_per_cohort_uniform_mass():
    """20 Training cases + 8 ATLAS3 + 2 Testing → each cohort should sum to ~1/3."""
    indices = (
        [f"T{i:03d}" for i in range(20)] + [f"A{i:03d}" for i in range(8)] + [f"E{i:03d}" for i in range(2)]
    )
    case_to_cohort = {
        **{f"T{i:03d}": "Training" for i in range(20)},
        **{f"A{i:03d}": "ATLAS3" for i in range(8)},
        **{f"E{i:03d}": "Testing" for i in range(2)},
    }
    probs = compute_cohort_balanced_probabilities(indices=indices, case_to_cohort=case_to_cohort)
    assert probs.shape == (30,)
    assert abs(probs.sum() - 1.0) < 1e-9

    mass_per_cohort = {"Training": 0.0, "ATLAS3": 0.0, "Testing": 0.0}
    for cid, p in zip(indices, probs, strict=True):
        mass_per_cohort[case_to_cohort[cid]] += p
    for cohort, mass in mass_per_cohort.items():
        assert abs(mass - 1.0 / 3.0) < 1e-9, f"{cohort}: {mass}"


def test_within_cohort_cases_have_equal_probability():
    """Each case in the same cohort gets the same probability."""
    indices = [f"T{i}" for i in range(5)] + [f"A{i}" for i in range(3)]
    c2c = {**{f"T{i}": "Training" for i in range(5)}, **{f"A{i}": "ATLAS3" for i in range(3)}}
    probs = compute_cohort_balanced_probabilities(indices=indices, case_to_cohort=c2c)
    # Training cases (first 5) all equal.
    assert np.allclose(probs[:5], probs[0])
    # ATLAS3 cases (last 3) all equal.
    assert np.allclose(probs[5:], probs[5])
    # The ATLAS3 cases should each be > a Training case (smaller cohort).
    assert probs[5] > probs[0]


def test_explicit_cohort_weights_are_honoured():
    """Custom weights {Training: 1, ATLAS3: 2} → ATLAS3 mass should be 2/3."""
    indices = [f"T{i}" for i in range(10)] + [f"A{i}" for i in range(5)]
    c2c = {**{f"T{i}": "Training" for i in range(10)}, **{f"A{i}": "ATLAS3" for i in range(5)}}
    probs = compute_cohort_balanced_probabilities(
        indices=indices,
        case_to_cohort=c2c,
        cohort_weights={"Training": 1.0, "ATLAS3": 2.0},
    )
    mass_training = probs[:10].sum()
    mass_atlas3 = probs[10:].sum()
    assert abs(mass_training - 1.0 / 3.0) < 1e-9
    assert abs(mass_atlas3 - 2.0 / 3.0) < 1e-9


def test_missing_cases_fall_back_to_modal_cohort_with_warning():
    """Cases not in the cohort dict get the modal cohort and emit a warning."""
    indices = ["known1", "known2", "unknown1"]
    c2c = {"known1": "Training", "known2": "Training"}
    with pytest.warns(UserWarning, match="not in case_to_cohort"):
        probs = compute_cohort_balanced_probabilities(indices=indices, case_to_cohort=c2c)
    # All 3 indices end up assigned to "Training" (the only / modal cohort) →
    # uniform 1/3 each.
    assert np.allclose(probs, np.array([1 / 3.0] * 3))


def test_empty_indices_returns_empty():
    out = compute_cohort_balanced_probabilities(indices=[], case_to_cohort={})
    assert out.shape == (0,)


def test_sampling_distribution_matches_target_with_many_draws():
    """A simulated draw from `np.random.choice(indices, p=probs)` produces
    per-cohort empirical frequency ≈ 1/K within a tolerance."""
    rng = np.random.default_rng(seed=42)
    # 50 Training + 30 ATLAS3 + 10 Testing - same imbalance shape as V2 (~5:3:1).
    indices = [f"T{i}" for i in range(50)] + [f"A{i}" for i in range(30)] + [f"E{i}" for i in range(10)]
    c2c = {
        **{f"T{i}": "Training" for i in range(50)},
        **{f"A{i}": "ATLAS3" for i in range(30)},
        **{f"E{i}": "Testing" for i in range(10)},
    }
    probs = compute_cohort_balanced_probabilities(indices=indices, case_to_cohort=c2c)
    draws = rng.choice(indices, size=10000, replace=True, p=probs)
    counts = Counter(c2c[c] for c in draws)
    # Each cohort should be ~1/3 = 33.3%.
    for cohort in ("Training", "ATLAS3", "Testing"):
        freq = counts[cohort] / 10000
        assert abs(freq - 1.0 / 3.0) < 0.02, f"{cohort}: {freq:.3f}"


def test_cohort_with_zero_total_weight_falls_back_to_uniform():
    indices = ["a", "b", "c"]
    c2c = {"a": "X", "b": "Y", "c": "X"}
    probs = compute_cohort_balanced_probabilities(
        indices=indices, case_to_cohort=c2c, cohort_weights={"X": 0.0, "Y": 0.0}
    )
    # All-zero weights → fallback to uniform across present cohorts.
    # X cases: 2 cases × (0.5 / 2) = 0.5 cohort mass; Y: 1 × 0.5 = 0.5.
    assert abs(probs.sum() - 1.0) < 1e-9
    assert abs(probs[0] + probs[2] - 0.5) < 1e-9
    assert abs(probs[1] - 0.5) < 1e-9
