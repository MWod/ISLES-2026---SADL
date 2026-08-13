"""Inner 5-fold CV - emits nnU-Net's expected splits_final.json shape."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from nnunet_isles.splits.manifest import InnerSplitManifest, write_nnunet_inner_splits


def _per_fold_stats(df: pd.DataFrame, val_ids: list[str]) -> dict[str, Any]:
    rows = df[df["session_id"].isin(val_ids)]
    return {
        "n_val": len(val_ids),
        "val_per_site": dict(Counter(rows["site"].tolist())) if "site" in rows.columns else {},
        "val_per_bucket": (
            dict(Counter(str(b) for b in rows["lesion_bucket"].tolist()))
            if "lesion_bucket" in rows.columns
            else {}
        ),
    }


def make_inner_splits(
    train_pool_df: pd.DataFrame,
    *,
    n_folds: int = 5,
    seed: int = 42,
    group_by: str | None = None,
) -> tuple[list[dict[str, list[str]]], list[dict[str, Any]]]:
    """Return (folds, per_fold_stats) where folds is the nnU-Net splits_final.json shape."""
    if "session_id" not in train_pool_df.columns:
        raise ValueError("train pool DataFrame must have a 'session_id' column")

    df = train_pool_df.reset_index(drop=True)
    ids = df["session_id"].tolist()

    if group_by is not None and group_by in df.columns:
        from sklearn.model_selection import GroupKFold

        gkf = GroupKFold(n_splits=n_folds)
        groups = df[group_by].tolist()
        splits_iter = gkf.split(np.arange(len(df)), groups=groups)
    else:
        from sklearn.model_selection import StratifiedKFold

        # Stratify on `lesion_bucket` when available; otherwise on `site`; otherwise plain KFold.
        if "lesion_bucket" in df.columns:
            stratify_labels = df["lesion_bucket"].astype(str).fillna("<missing>").tolist()
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            splits_iter = skf.split(np.arange(len(df)), y=stratify_labels)
        elif "site" in df.columns:
            stratify_labels = df["site"].astype(str).fillna("<missing>").tolist()
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            splits_iter = skf.split(np.arange(len(df)), y=stratify_labels)
        else:
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            splits_iter = kf.split(np.arange(len(df)))

    folds: list[dict[str, list[str]]] = []
    stats: list[dict[str, Any]] = []
    for fold_i, (train_idx, val_idx) in enumerate(splits_iter):
        train_ids = sorted(ids[i] for i in train_idx)
        val_ids = sorted(ids[i] for i in val_idx)
        folds.append({"train": train_ids, "val": val_ids})
        s = _per_fold_stats(df, val_ids)
        s["fold"] = fold_i
        stats.append(s)
    return folds, stats


def write_inner_splits(
    train_pool_df: pd.DataFrame,
    *,
    out_dir: str,
    split_name: str,
    n_folds: int = 5,
    seed: int = 42,
    group_by: str | None = None,
    git_sha: str = "unknown",
) -> tuple[str, str]:
    folds, per_fold_stats = make_inner_splits(train_pool_df, n_folds=n_folds, seed=seed, group_by=group_by)
    meta = InnerSplitManifest(
        split_name=split_name,
        n_folds=n_folds,
        seed=seed,
        group_by=group_by,
        per_fold_stats=per_fold_stats,
        git_sha=git_sha,
    )
    splits_path, meta_path = write_nnunet_inner_splits(folds, meta, out_dir)
    return str(splits_path), str(meta_path)
