"""CohortBalancedDataLoader3D.

Default nnU-Net sampling picks cases uniformly from `self.indices`, which
in our V2 train pool gives the dominant cohort (Training: 657 cases, ~54%)
double the per-batch probability of ATLAS3 (~41%) and 20× the rate of
the smallest cohort (Testing, ~5%). This loader instead samples each
case with probability `1 / (K * |cohort(case)|)` so that, in expectation,
every cohort contributes the same number of samples per batch regardless
of its absolute size.

Why per-case probability and not per-batch cohort rotation: the batch
size is small (1-2 patches on a single GPU), so batch-level rotation produces
correlated samples within a batch from the same cohort, which hurts the
BatchNorm-free InstanceNorm baseline but matters for any future BN
variants. Per-case sampling weights are stable across runs once seeded
and don't require a wrapper that intercepts every `get_indices` call.

The cohort assignment is provided as a `case_to_cohort: dict[str, str]`
at construction time. Cases whose identifier is missing from the dict
fall back to the modal cohort (largest) with a warning. Cases mapped to
a cohort not represented in the loader's `indices` are silently dropped.
"""

from __future__ import annotations

import warnings
from collections import Counter
from typing import Any

import numpy as np

try:
    from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
except ImportError:
    nnUNetDataLoader = object  # type: ignore[assignment, misc]


class CohortBalancedDataLoader3D(nnUNetDataLoader):  # type: ignore[misc, valid-type]
    """nnUNetDataLoader variant with per-case sampling weights derived from cohort."""

    def __init__(
        self,
        *args: Any,
        case_to_cohort: dict[str, str] | None = None,
        cohort_weights: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if case_to_cohort is None:
            raise ValueError(
                "CohortBalancedDataLoader3D requires `case_to_cohort` (dict[case_id, cohort_name])"
            )
        # Pop our kwargs so they don't leak into the parent.
        kwargs.pop("case_to_cohort", None)
        super().__init__(*args, **kwargs)
        self._case_to_cohort = dict(case_to_cohort)
        # Build per-case sampling probabilities aligned with self.indices.
        probs = compute_cohort_balanced_probabilities(
            indices=list(self.indices),
            case_to_cohort=self._case_to_cohort,
            cohort_weights=cohort_weights,
        )
        self.sampling_probabilities = probs


def compute_cohort_balanced_probabilities(
    *,
    indices: list[str],
    case_to_cohort: dict[str, str],
    cohort_weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Per-case sampling probability such that each cohort's expected mass equals
    its `cohort_weights[cohort]` (default: uniform across present cohorts).

    Args:
        indices: case identifiers in loader order.
        case_to_cohort: cohort lookup; missing cases default to the modal cohort.
        cohort_weights: optional override `{cohort_name: relative_weight}`. None ⇒
            uniform weight on the cohorts that appear in `indices`.

    Returns:
        `np.ndarray[float, shape=(len(indices),)]` summing to 1.
    """
    if len(indices) == 0:
        return np.array([], dtype=np.float64)

    # Resolve modal cohort (used to backfill cases missing from the mapping).
    cohort_counts = Counter(case_to_cohort.values())
    modal_cohort = max(cohort_counts, key=lambda c: cohort_counts[c]) if cohort_counts else "Training"

    # Assign each index a cohort (with fallback).
    n_missing = 0
    cohorts = []
    for case_id in indices:
        c = case_to_cohort.get(case_id)
        if c is None:
            n_missing += 1
            c = modal_cohort
        cohorts.append(c)
    if n_missing > 0:
        warnings.warn(
            f"CohortBalanced loader: {n_missing}/{len(indices)} cases not in "
            f"case_to_cohort - fell back to modal cohort {modal_cohort!r}.",
            stacklevel=2,
        )

    # Per-cohort case-count among the loader's indices.
    present_counts = Counter(cohorts)
    present_cohorts = list(present_counts.keys())
    n_cohorts = len(present_cohorts)
    if n_cohorts == 0:
        return np.full(len(indices), 1.0 / len(indices), dtype=np.float64)

    if cohort_weights is None:
        cohort_target = dict.fromkeys(present_cohorts, 1.0 / n_cohorts)
    else:
        # Restrict to present cohorts; normalise.
        raw = {c: float(cohort_weights.get(c, 0.0)) for c in present_cohorts}
        total = sum(raw.values())
        if total <= 0.0:
            cohort_target = dict.fromkeys(present_cohorts, 1.0 / n_cohorts)
        else:
            cohort_target = {c: raw[c] / total for c in present_cohorts}

    # Each case in cohort c gets probability cohort_target[c] / count(c).
    probs = np.empty(len(indices), dtype=np.float64)
    for i, c in enumerate(cohorts):
        probs[i] = cohort_target[c] / present_counts[c]
    s = probs.sum()
    if s > 0:
        probs /= s
    return probs


__all__ = ["CohortBalancedDataLoader3D", "compute_cohort_balanced_probabilities"]
