"""Tests for the `site_disjoint_fixed` split strategy used by V2.

The V2 split must hold out R018+R027+R047 verbatim (no auto-pick) AND must
drop sessions where `train_eligible == False` (V2's 5 empty-mask + 1
header-only-meta cases).
"""

from __future__ import annotations

import pandas as pd
from nnunet_isles.registry import SPLIT_REGISTRY


def _mk_df() -> pd.DataFrame:
    rows = []
    # 10 sessions across 6 sites; 3 held out, 3 trained.
    for site, n in [("R001", 5), ("R009", 4), ("R018", 3), ("R027", 4), ("R047", 6), ("R069", 8)]:
        for i in range(n):
            rows.append(
                {"session_id": f"sub-{site.lower()}s{i:03d}_ses-1", "site": site, "train_eligible": True}
            )
    # 2 ineligible sessions (e.g. empty-mask cases) - one in test-site (R047), one in train pool (R009).
    rows.append({"session_id": "sub-r047s999_ses-1", "site": "R047", "train_eligible": False})
    rows.append({"session_id": "sub-r009s999_ses-1", "site": "R009", "train_eligible": False})
    return pd.DataFrame(rows)


def test_holds_out_exactly_the_named_sites():
    df = _mk_df()
    strat = SPLIT_REGISTRY.get("site_disjoint_fixed")
    train_ids, test_ids, stats = strat(df, test_sites=["R018", "R027", "R047"])
    test_sites = {sid.split("-")[1][:4].upper() for sid in test_ids}
    train_sites = {sid.split("-")[1][:4].upper() for sid in train_ids}
    assert test_sites == {"R018", "R027", "R047"}
    assert test_sites.isdisjoint(train_sites)
    assert stats["held_out_sites"] == ["R018", "R027", "R047"]


def test_ineligible_sessions_excluded_from_both_sides():
    df = _mk_df()
    strat = SPLIT_REGISTRY.get("site_disjoint_fixed")
    train_ids, test_ids, _ = strat(df, test_sites=["R018", "R027", "R047"])
    assert "sub-r047s999_ses-1" not in test_ids  # ineligible R047 case dropped
    assert "sub-r009s999_ses-1" not in train_ids  # ineligible R009 case dropped
    # Test set: 3 + 4 + 6 = 13 eligible cases.
    assert len(test_ids) == 13
    # Train pool: 5 + 4 + 8 = 17 eligible cases.
    assert len(train_ids) == 17


def test_train_and_test_disjoint():
    df = _mk_df()
    strat = SPLIT_REGISTRY.get("site_disjoint_fixed")
    train_ids, test_ids, _ = strat(df, test_sites=["R018", "R027", "R047"])
    assert set(train_ids).isdisjoint(set(test_ids))


def test_raises_if_test_site_missing_from_data():
    df = _mk_df()
    strat = SPLIT_REGISTRY.get("site_disjoint_fixed")
    try:
        strat(df, test_sites=["R018", "R027", "R999"])
    except ValueError as e:
        assert "R999" in str(e)
        return
    raise AssertionError("expected ValueError for missing test site")


def test_eligibility_column_can_be_disabled():
    """When train_eligibility_column is set to None or absent, every session passes."""
    df = _mk_df()
    strat = SPLIT_REGISTRY.get("site_disjoint_fixed")
    train_ids, test_ids, _ = strat(df, test_sites=["R018"], train_eligibility_column=None)
    # Now the ineligible R009 case stays in train.
    assert "sub-r009s999_ses-1" in train_ids
    assert len(test_ids) == 3  # all R018 sessions are eligible
