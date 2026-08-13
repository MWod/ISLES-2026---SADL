"""Tests for `nnunet_isles.inference.policy.DecisionPolicy`."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from nnunet_isles.inference.policy import DecisionPolicy, default_policy


def _make_prob(n_pos_voxels: int, shape: tuple[int, ...], prob: float = 0.9) -> np.ndarray:
    """Return a `prob_fg` volume with exactly `n_pos_voxels` voxels above 0.5."""
    total = int(np.prod(shape))
    if n_pos_voxels > total:
        raise ValueError(f"n_pos_voxels={n_pos_voxels} exceeds total voxels {total} in shape {shape}")
    arr = np.zeros(total, dtype=np.float32)
    arr[:n_pos_voxels] = prob
    return arr.reshape(shape)


def test_default_policy_roundtrips_through_json(tmp_path: Path) -> None:
    policy = default_policy()
    path = tmp_path / "policy.json"
    policy.to_json(path)
    loaded = DecisionPolicy.from_json(path)
    assert asdict(loaded) == asdict(policy)


def test_default_policy_with_multiple_members_has_uniform_weights() -> None:
    policy = default_policy(n_members=5)
    assert policy.weights is not None
    assert len(policy.weights) == 5
    assert all(abs(w - 0.2) < 1e-9 for w in policy.weights)


def test_default_policy_single_member_has_none_weights() -> None:
    policy = default_policy(n_members=1)
    assert policy.weights is None


def test_custom_four_bucket_policy_roundtrips(tmp_path: Path) -> None:
    policy = DecisionPolicy(
        mode="noisy_or",
        k=None,
        member_threshold=0.5,
        weights=[0.2, 0.3, 0.5],
        bucket_edges_ml=[0.5, 5.0, 50.0],
        threshold_by_bucket=[0.30, 0.35, 0.40, 0.45],
        min_voxels_by_bucket=[0, 10, 25, 50],
        min_max_prob=0.4,
        min_mean_prob=0.2,
        min_prob_mass=1.5,
        never_empty=True,
        rescue_min_prob=0.05,
        connectivity=26,
        voxel_volume_ml=0.001,
    )
    path = tmp_path / "custom.json"
    policy.to_json(path)
    loaded = DecisionPolicy.from_json(path)
    assert asdict(loaded) == asdict(policy)
    assert loaded.mode == "noisy_or"
    assert loaded.threshold_by_bucket == [0.30, 0.35, 0.40, 0.45]


def _build_canonical_policy(voxel_volume_ml: float = 0.1) -> DecisionPolicy:
    """Canonical 4-bucket policy used across pick_threshold_for_case tests.

    `voxel_volume_ml=0.1` keeps test shapes small: 1 voxel = 0.1 mL.
    """
    return DecisionPolicy(
        bucket_edges_ml=[0.5, 5.0, 50.0],
        threshold_by_bucket=[0.30, 0.35, 0.40, 0.45],
        min_voxels_by_bucket=[0, 10, 25, 50],
        voxel_volume_ml=voxel_volume_ml,
    )


def test_pick_threshold_bucket_zero_smallest_lesion() -> None:
    """pred_vol = 3 voxels * 0.1 mL = 0.3 mL -> bucket 0 (<0.5mL)."""
    policy = _build_canonical_policy()
    prob = _make_prob(n_pos_voxels=3, shape=(10, 10, 10))
    threshold, min_voxels, label = policy.pick_threshold_for_case(prob)
    assert threshold == pytest.approx(0.30)
    assert min_voxels == 0
    assert label == "<0.5mL"


def test_pick_threshold_bucket_one_small_medium() -> None:
    """pred_vol = 20 voxels * 0.1 mL = 2.0 mL -> bucket 1 (0.5-5mL)."""
    policy = _build_canonical_policy()
    prob = _make_prob(n_pos_voxels=20, shape=(10, 10, 10))
    threshold, min_voxels, label = policy.pick_threshold_for_case(prob)
    assert threshold == pytest.approx(0.35)
    assert min_voxels == 10
    assert label == "0.5-5mL"


def test_pick_threshold_bucket_two_medium_large() -> None:
    """pred_vol = 300 voxels * 0.1 mL = 30.0 mL -> bucket 2 (5-50mL)."""
    policy = _build_canonical_policy()
    prob = _make_prob(n_pos_voxels=300, shape=(10, 10, 10))
    threshold, min_voxels, label = policy.pick_threshold_for_case(prob)
    assert threshold == pytest.approx(0.40)
    assert min_voxels == 25
    assert label == "5-50mL"


def test_pick_threshold_bucket_three_largest_lesion() -> None:
    """pred_vol = 1000 voxels * 0.1 mL = 100.0 mL -> bucket 3 (>=50mL)."""
    policy = _build_canonical_policy()
    prob = _make_prob(n_pos_voxels=1000, shape=(10, 10, 10))
    threshold, min_voxels, label = policy.pick_threshold_for_case(prob)
    assert threshold == pytest.approx(0.45)
    assert min_voxels == 50
    assert label == ">=50mL"


def test_pick_threshold_uses_nominal_threshold_for_binarisation() -> None:
    """The two-pass binarisation must respect `nominal_threshold`.

    If we set nominal_threshold above the probability level, no voxels
    survive and the case lands in bucket 0.
    """
    policy = _build_canonical_policy()
    prob = _make_prob(n_pos_voxels=300, shape=(10, 10, 10), prob=0.4)
    # At nominal_threshold=0.5, none of the 0.4-valued voxels count -> 0 mL -> bucket 0.
    threshold, _, label = policy.pick_threshold_for_case(prob, nominal_threshold=0.5)
    assert threshold == pytest.approx(0.30)
    assert label == "<0.5mL"
    # At nominal_threshold=0.3, all 300 voxels count -> 30 mL -> bucket 2.
    threshold_2, _, label_2 = policy.pick_threshold_for_case(prob, nominal_threshold=0.3)
    assert threshold_2 == pytest.approx(0.40)
    assert label_2 == "5-50mL"


def test_pick_threshold_synthesises_labels_for_non_canonical_edges() -> None:
    """Non-canonical edges must produce synthesised labels rather than crash."""
    policy = DecisionPolicy(
        bucket_edges_ml=[1.0, 10.0],
        threshold_by_bucket=[0.25, 0.40, 0.55],
        min_voxels_by_bucket=[0, 5, 20],
        voxel_volume_ml=0.1,
    )
    # 3 voxels * 0.1 = 0.3 mL -> bucket 0 -> '<1'
    _, _, label_lo = policy.pick_threshold_for_case(_make_prob(3, (10, 10, 10)))
    assert label_lo == "<1"
    # 50 voxels * 0.1 = 5 mL -> bucket 1 -> '1-10'
    _, _, label_mid = policy.pick_threshold_for_case(_make_prob(50, (10, 10, 10)))
    assert label_mid == "1-10"
    # 500 voxels * 0.1 = 50 mL -> bucket 2 -> '>=10'
    _, _, label_hi = policy.pick_threshold_for_case(_make_prob(500, (10, 10, 10)))
    assert label_hi == ">=10"


def test_validate_rejects_threshold_length_mismatch() -> None:
    with pytest.raises(ValueError, match="threshold_by_bucket"):
        DecisionPolicy(
            bucket_edges_ml=[0.5, 5.0, 50.0],
            threshold_by_bucket=[0.3, 0.4, 0.5],  # 3 entries, need 4
            min_voxels_by_bucket=[0, 10, 25, 50],
        )


def test_validate_rejects_min_voxels_length_mismatch() -> None:
    with pytest.raises(ValueError, match="min_voxels_by_bucket"):
        DecisionPolicy(
            bucket_edges_ml=[0.5, 5.0, 50.0],
            threshold_by_bucket=[0.3, 0.4, 0.5, 0.6],
            min_voxels_by_bucket=[0, 10, 25],  # 3 entries, need 4
        )


def test_validate_rejects_non_ascending_edges() -> None:
    with pytest.raises(ValueError, match="strictly ascending"):
        DecisionPolicy(
            bucket_edges_ml=[5.0, 0.5, 50.0],
            threshold_by_bucket=[0.3, 0.4, 0.5, 0.6],
            min_voxels_by_bucket=[0, 10, 25, 50],
        )


def test_validate_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        DecisionPolicy(
            mode="weighted_mean",  # not in {'mean', 'noisy_or', 'k_of_n'}
            bucket_edges_ml=[],
            threshold_by_bucket=[0.5],
            min_voxels_by_bucket=[0],
        )


def test_validate_rejects_invalid_connectivity() -> None:
    with pytest.raises(ValueError, match="connectivity"):
        DecisionPolicy(
            bucket_edges_ml=[],
            threshold_by_bucket=[0.5],
            min_voxels_by_bucket=[0],
            connectivity=4,  # not in {6, 18, 26}
        )


def test_from_json_rejects_wrong_schema_version(tmp_path: Path) -> None:
    """A policy file marked schema_version 2.0 must be refused."""
    policy = default_policy()
    payload = asdict(policy)
    payload["schema_version"] = "2.0"
    path = tmp_path / "wrong.json"
    import json

    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema_version"):
        DecisionPolicy.from_json(path)


def test_post_init_coerces_tuple_lists() -> None:
    """__post_init__ must coerce tuple inputs to lists for JSON round-trip."""
    policy = DecisionPolicy(
        bucket_edges_ml=(0.5, 5.0, 50.0),
        threshold_by_bucket=(0.30, 0.35, 0.40, 0.45),
        min_voxels_by_bucket=(0, 10, 25, 50),
    )
    assert isinstance(policy.bucket_edges_ml, list)
    assert isinstance(policy.threshold_by_bucket, list)
    assert isinstance(policy.min_voxels_by_bucket, list)
    assert policy.threshold_by_bucket == [0.30, 0.35, 0.40, 0.45]


def test_default_policy_is_valid() -> None:
    """default_policy must produce a policy that survives its own validate()."""
    policy = default_policy()
    # No exception implies validate() succeeded during __post_init__.
    policy.validate()
    assert policy.mode == "mean"
    assert policy.never_empty is True
    assert policy.bucket_edges_ml == []
    assert policy.threshold_by_bucket == [0.5]


def test_bucket_index_at_exact_edges() -> None:
    """Exact bucket edges must land in the upper bucket (matches bucket_for_volume).

    With canonical edges [0.5, 5.0, 50.0], a single positive voxel gives
    pred_vol_ml equal to `voxel_volume_ml`, so we can drive each exact edge
    by tweaking that field.
    """
    prob_one = _make_prob(n_pos_voxels=1, shape=(10, 10, 10))
    for voxel_vol, expected_label in ((0.5, "0.5-5mL"), (5.0, "5-50mL"), (50.0, ">=50mL")):
        policy = DecisionPolicy(
            bucket_edges_ml=[0.5, 5.0, 50.0],
            threshold_by_bucket=[0.30, 0.35, 0.40, 0.45],
            min_voxels_by_bucket=[0, 10, 25, 50],
            voxel_volume_ml=voxel_vol,
        )
        _, _, label = policy.pick_threshold_for_case(prob_one)
        assert label == expected_label, (
            f"voxel_volume_ml={voxel_vol}: got {label!r}, expected {expected_label!r}"
        )


def test_validate_rejects_k_none_when_kofn() -> None:
    """validate() on a k_of_n policy with k=None must raise ValueError."""
    with pytest.raises(ValueError, match="k must be set"):
        DecisionPolicy(
            mode="k_of_n",
            k=None,
            bucket_edges_ml=[],
            threshold_by_bucket=[0.5],
            min_voxels_by_bucket=[0],
        )


def test_soft_bucket_boundary_field_defaults_off_and_roundtrips(tmp_path: Path) -> None:
    """New soft_bucket_boundary field is False by default and JSON-roundtrips."""
    policy = default_policy(n_members=3)
    assert policy.soft_bucket_boundary is False
    path = tmp_path / "policy.json"
    policy.to_json(path)
    loaded = DecisionPolicy.from_json(path)
    assert loaded.soft_bucket_boundary is False

    policy_on = DecisionPolicy(
        bucket_edges_ml=[0.5, 5.0, 50.0],
        threshold_by_bucket=[0.3, 0.4, 0.5, 0.4],
        min_voxels_by_bucket=[0, 0, 5, 50],
        soft_bucket_boundary=True,
    )
    path_on = tmp_path / "policy_on.json"
    policy_on.to_json(path_on)
    loaded_on = DecisionPolicy.from_json(path_on)
    assert loaded_on.soft_bucket_boundary is True
    assert loaded_on.threshold_by_bucket == [0.3, 0.4, 0.5, 0.4]


def test_soft_bucket_boundary_backward_compat_with_v1_json(tmp_path: Path) -> None:
    """Policies written before soft_bucket_boundary existed load with default=False."""
    import json as _json

    legacy = {
        "schema_version": "1.0",
        "mode": "mean",
        "k": None,
        "member_threshold": 0.5,
        "weights": None,
        "bucket_edges_ml": [0.5],
        "threshold_by_bucket": [0.4, 0.5],
        "min_voxels_by_bucket": [0, 5],
        "min_max_prob": 0.0,
        "min_mean_prob": 0.0,
        "min_prob_mass": 0.0,
        "never_empty": True,
        "rescue_min_prob": 0.1,
        "connectivity": 26,
        "voxel_volume_ml": 0.001,
    }
    p = tmp_path / "legacy.json"
    p.write_text(_json.dumps(legacy))
    loaded = DecisionPolicy.from_json(p)
    assert loaded.soft_bucket_boundary is False


def test_soft_bucket_boundary_smooth_matches_hard_at_bucket_centres() -> None:
    """At a bucket centre, the smooth interpolation should approximately match
    the hard bucket threshold (with default log-space offsets)."""
    policy = DecisionPolicy(
        bucket_edges_ml=[0.5, 5.0, 50.0],
        threshold_by_bucket=[0.30, 0.40, 0.55, 0.45],
        min_voxels_by_bucket=[0, 0, 0, 0],
        soft_bucket_boundary=True,
        voxel_volume_ml=0.001,
    )
    # 1.58 mL is the log10 midpoint between 0.5 and 5.0 mL. Not hit exactly
    # by an integer voxel count so allow small drift.
    fg = _make_prob(1581, shape=(20, 20, 20), prob=0.9)
    thr, mv, label = policy.pick_threshold_for_case(fg)
    assert abs(thr - 0.40) < 5e-3
    assert label == "0.5-5mL"


def test_soft_bucket_boundary_smooth_defuses_edge_case() -> None:
    """Near a bucket edge, the smooth threshold sits between the two adjacent
    bucket thresholds - NOT snapped to either one."""
    policy_hard = DecisionPolicy(
        bucket_edges_ml=[0.5, 5.0, 50.0],
        threshold_by_bucket=[0.30, 0.40, 0.55, 0.45],
        min_voxels_by_bucket=[0, 0, 0, 0],
        soft_bucket_boundary=False,
        voxel_volume_ml=0.001,
    )
    policy_soft = DecisionPolicy(
        bucket_edges_ml=[0.5, 5.0, 50.0],
        threshold_by_bucket=[0.30, 0.40, 0.55, 0.45],
        min_voxels_by_bucket=[0, 0, 0, 0],
        soft_bucket_boundary=True,
        voxel_volume_ml=0.001,
    )
    # Predicted volume ~5.1 mL - right on the 5.0 mL bucket edge.
    fg = _make_prob(5100, shape=(30, 30, 30), prob=0.9)
    thr_hard, _, _ = policy_hard.pick_threshold_for_case(fg)
    thr_soft, _, _ = policy_soft.pick_threshold_for_case(fg)
    # Hard picks the strict 5-50 mL threshold (0.55). Soft interpolates
    # between 0.40 (0.5-5 mL centre) and 0.55 (5-50 mL centre) and must land
    # somewhere in [0.40, 0.55].
    assert thr_hard == 0.55
    assert 0.40 <= thr_soft <= 0.55
    # And the soft threshold should be STRICTLY less than the hard 0.55
    # (that's the whole point - dial back the strictness near the edge).
    assert thr_soft < 0.55
