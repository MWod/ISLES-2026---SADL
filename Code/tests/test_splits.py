"""Sanity tests for outer split strategies."""

from __future__ import annotations

import pandas as pd
from nnunet_isles.splits.outer import build_outer_split


def _toy_df(n_sites=4, per_site=20):
    rows = []
    for s in range(n_sites):
        for i in range(per_site):
            rows.append(
                {
                    "session_id": f"sub-r{s + 1:03d}s{i:03d}",
                    "site": f"R{s + 1:03d}",
                    "lesion_bucket": "small" if i < per_site // 2 else "large",
                }
            )
    return pd.DataFrame(rows)


def test_random_split_sizes():
    df = _toy_df()
    m = build_outer_split(
        df, split_name="r10", strategy="random", params={"outer_test_fraction": 0.10, "seed": 42}
    )
    assert m.n_train + m.n_test == len(df)
    assert set(m.train_session_ids).isdisjoint(set(m.test_session_ids))


def test_site_stratified_keeps_all_sites_in_train_and_test():
    df = _toy_df()
    m = build_outer_split(
        df, split_name="ss10", strategy="site_stratified", params={"outer_test_fraction": 0.20, "seed": 42}
    )
    train_sites = {sid.split("s")[0] for sid in m.train_session_ids}
    test_sites = {sid.split("s")[0] for sid in m.test_session_ids}
    assert train_sites == test_sites


def test_site_disjoint_test_sites_absent_from_train():
    df = _toy_df()
    m = build_outer_split(
        df,
        split_name="sd2",
        strategy="site_disjoint",
        params={"n_test_sites": 2, "test_site_selection": "stratified_size", "seed": 1},
    )
    test_sites = set(m.stats["held_out_sites"])
    train_sites = {sid.split("s")[0] for sid in m.train_session_ids}
    assert train_sites.isdisjoint(test_sites)


def test_loso_emits_iterations():
    df = _toy_df()
    m = build_outer_split(df, split_name="loso", strategy="loso", params={})
    iterations = m.stats["iterations"]
    assert len(iterations) >= 1
    # All sessions in some iteration's test set.
    covered = set()
    for it in iterations:
        covered.update(it["test_session_ids"])
    assert covered == set(df["session_id"])
