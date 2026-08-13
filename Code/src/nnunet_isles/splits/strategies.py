"""Outer split strategies registered against SPLIT_REGISTRY.

Each strategy takes a DataFrame of session metadata + a config and returns
(train_ids, test_ids, stats_dict). The config is a plain dict mirroring the
Hydra config for the split group.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from nnunet_isles.registry import SPLIT_REGISTRY

# Sites with fewer than this many sessions are grouped into a single
# "small-sites" meta-fold for LOSO / site-disjoint splits.
TINY_SITE_THRESHOLD = 10


def _stratify_by(df: pd.DataFrame, key: str) -> list[str]:
    """Return per-row stratify labels, falling back to '<missing>' for null cells."""
    return [
        str(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else "<missing>"
        for x in df[key]
    ]


def _per_site_counts(df: pd.DataFrame, session_ids: list[str]) -> dict[str, int]:
    rows = df[df["session_id"].isin(session_ids)]
    return dict(Counter(rows["site"].tolist()))


def _per_bucket_counts(df: pd.DataFrame, session_ids: list[str]) -> dict[str, int]:
    if "lesion_bucket" not in df.columns:
        return {}
    rows = df[df["session_id"].isin(session_ids)]
    return dict(Counter(str(b) for b in rows["lesion_bucket"].tolist()))


def _stats(df: pd.DataFrame, train_ids: list[str], test_ids: list[str]) -> dict[str, Any]:
    return {
        "train_per_site": _per_site_counts(df, train_ids),
        "test_per_site": _per_site_counts(df, test_ids),
        "train_per_bucket": _per_bucket_counts(df, train_ids),
        "test_per_bucket": _per_bucket_counts(df, test_ids),
    }


def _ensure_session_ids(df: pd.DataFrame) -> pd.DataFrame:
    if "session_id" not in df.columns:
        raise ValueError("session DataFrame must have a 'session_id' column")
    return df.reset_index(drop=True)


@SPLIT_REGISTRY.register("random")
def split_random(
    df: pd.DataFrame,
    *,
    outer_test_fraction: float = 0.10,
    seed: int = 42,
    stratify: str | None = "lesion_bucket",
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Stratified random split. Default stratification is on `lesion_bucket`."""
    df = _ensure_session_ids(df)
    from sklearn.model_selection import train_test_split

    stratify_labels = _stratify_by(df, stratify) if stratify and stratify in df.columns else None
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=outer_test_fraction,
        random_state=seed,
        stratify=stratify_labels,
    )
    train_ids = df.iloc[train_idx]["session_id"].tolist()
    test_ids = df.iloc[test_idx]["session_id"].tolist()
    return train_ids, test_ids, _stats(df, train_ids, test_ids)


@SPLIT_REGISTRY.register("site_stratified")
def split_site_stratified(
    df: pd.DataFrame, *, outer_test_fraction: float = 0.10, seed: int = 42
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Per-site stratified hold-out - preserves site marginals in train and test."""
    df = _ensure_session_ids(df)
    rng = np.random.default_rng(seed)
    train_ids: list[str] = []
    test_ids: list[str] = []
    for _site, group in df.groupby("site"):
        ids = group["session_id"].tolist()
        n_test = max(1, int(round(len(ids) * outer_test_fraction)))
        order = rng.permutation(len(ids)).tolist()
        site_test = [ids[i] for i in order[:n_test]]
        site_train = [ids[i] for i in order[n_test:]]
        test_ids.extend(site_test)
        train_ids.extend(site_train)
    return train_ids, test_ids, _stats(df, train_ids, test_ids)


@SPLIT_REGISTRY.register("site_disjoint")
def split_site_disjoint(
    df: pd.DataFrame,
    *,
    n_test_sites: int = 3,
    test_site_selection: str = "stratified_size",
    seed: int = 42,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Hold out N entire sites as test. Train pool contains zero sessions from those sites.

    `test_site_selection` options:
      - 'stratified_size': pick 1 small (smallest tier) + 1 medium + 1 large site when n_test_sites==3,
        otherwise pick N sites stratified by size quantile.
      - 'random': uniformly random over sites with size >= TINY_SITE_THRESHOLD.
    """
    df = _ensure_session_ids(df)
    rng = np.random.default_rng(seed)

    site_sizes = df.groupby("site").size().sort_values()
    eligible = site_sizes[site_sizes >= TINY_SITE_THRESHOLD]
    if len(eligible) < n_test_sites:
        raise ValueError(
            f"Not enough sites with size >= {TINY_SITE_THRESHOLD} for n_test_sites={n_test_sites}; "
            f"got {len(eligible)}. Lower n_test_sites or relax TINY_SITE_THRESHOLD."
        )

    if test_site_selection == "stratified_size":
        # Split sites by size quantiles and pick one per bin.
        chosen: list[str] = []
        quantile_bins = np.array_split(eligible.index.to_numpy(), n_test_sites)
        for bin_sites in quantile_bins:
            idx = int(rng.integers(0, len(bin_sites)))
            chosen.append(str(bin_sites[idx]))
    elif test_site_selection == "random":
        idxs = rng.choice(len(eligible), size=n_test_sites, replace=False)
        chosen = [str(eligible.index[i]) for i in idxs]
    else:
        raise ValueError(f"Unknown test_site_selection: {test_site_selection!r}")

    test_mask = df["site"].isin(chosen)
    test_ids = df[test_mask]["session_id"].tolist()
    train_ids = df[~test_mask]["session_id"].tolist()
    stats = _stats(df, train_ids, test_ids)
    stats["held_out_sites"] = chosen
    return train_ids, test_ids, stats


@SPLIT_REGISTRY.register("site_disjoint_fixed")
def split_site_disjoint_fixed(
    df: pd.DataFrame,
    *,
    test_sites: list[str],
    train_eligibility_column: str | None = "train_eligible",
    seed: int = 42,  # noqa: ARG001 - deterministic by construction; arg present for schema parity
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Hold out a fixed, explicitly-named list of sites as test.

    Used by V2 splits to keep the test-site list pinned across data refreshes,
    so V2 numbers stay directly comparable to earlier leaderboard rows that
    held out the same three sites (R018+R027+R047).

    If `train_eligibility_column` is set and present in the DataFrame, rows where
    that column is False are dropped from BOTH train and test (V2 empty-mask /
    header-only-metadata cases).
    """
    df = _ensure_session_ids(df)
    if train_eligibility_column and train_eligibility_column in df.columns:
        df = df[df[train_eligibility_column].astype(bool)].reset_index(drop=True)
    test_sites_norm = [str(s) for s in test_sites]
    missing = [s for s in test_sites_norm if s not in df["site"].astype(str).unique()]
    if missing:
        raise ValueError(f"site_disjoint_fixed: requested test_sites not present in sessions.tsv: {missing}")
    test_mask = df["site"].astype(str).isin(test_sites_norm)
    test_ids = df[test_mask]["session_id"].tolist()
    train_ids = df[~test_mask]["session_id"].tolist()
    stats = _stats(df, train_ids, test_ids)
    stats["held_out_sites"] = test_sites_norm
    if train_eligibility_column and train_eligibility_column in df.columns:
        stats["train_eligibility_column"] = train_eligibility_column
    return train_ids, test_ids, stats


@SPLIT_REGISTRY.register("loso")
def split_loso(
    df: pd.DataFrame, *, tiny_site_threshold: int = TINY_SITE_THRESHOLD, **_: Any
) -> tuple[list[str], list[str], dict[str, Any]]:
    """LOSO is not a single train/test split - it's an iteration scheme.

    This function returns an empty train/test split and a stats dict listing all
    outer iterations (each holds one site, or the tiny-sites meta-fold, as test).
    `generate_splits.py` reads `stats['iterations']` to drive the outer loop.
    """
    df = _ensure_session_ids(df)
    site_sizes = df.groupby("site").size().sort_values()
    tiny_sites = [str(s) for s in site_sizes[site_sizes < tiny_site_threshold].index.tolist()]
    big_sites = [str(s) for s in site_sizes[site_sizes >= tiny_site_threshold].index.tolist()]

    iterations: list[dict[str, Any]] = []
    for site in big_sites:
        iterations.append(
            {
                "name": f"holdout_{site}",
                "test_sites": [site],
                "test_session_ids": df[df["site"] == site]["session_id"].tolist(),
            }
        )
    if tiny_sites:
        iterations.append(
            {
                "name": "holdout_small_sites_meta",
                "test_sites": tiny_sites,
                "test_session_ids": df[df["site"].isin(tiny_sites)]["session_id"].tolist(),
            }
        )

    return (
        [],
        [],
        {
            "iterations": iterations,
            "tiny_site_threshold": tiny_site_threshold,
            "tiny_sites_grouped": tiny_sites,
        },
    )


@SPLIT_REGISTRY.register("grouped_kfold_site")
def split_grouped_kfold_site(
    df: pd.DataFrame, *, n_outer_folds: int = 5, fold_index: int = 0, seed: int = 42
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Pick fold `fold_index` of a `n_outer_folds`-way GroupKFold over site."""
    from sklearn.model_selection import GroupKFold

    df = _ensure_session_ids(df)
    gkf = GroupKFold(n_splits=n_outer_folds)
    groups = df["site"].tolist()
    splits = list(gkf.split(np.arange(len(df)), groups=groups))
    if fold_index >= n_outer_folds:
        raise ValueError(f"fold_index={fold_index} >= n_outer_folds={n_outer_folds}")
    train_idx, test_idx = splits[fold_index]
    train_ids = df.iloc[train_idx]["session_id"].tolist()
    test_ids = df.iloc[test_idx]["session_id"].tolist()
    return train_ids, test_ids, _stats(df, train_ids, test_ids)
