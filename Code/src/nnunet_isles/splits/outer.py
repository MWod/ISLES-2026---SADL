"""Outer split builder - dispatches by strategy name through SPLIT_REGISTRY."""

from __future__ import annotations

from typing import Any

import pandas as pd

from nnunet_isles.registry import SPLIT_REGISTRY
from nnunet_isles.splits.manifest import OuterSplitManifest


def build_outer_split(
    df: pd.DataFrame,
    *,
    split_name: str,
    strategy: str,
    params: dict[str, Any],
    git_sha: str = "unknown",
) -> OuterSplitManifest:
    """Build an outer-split manifest using a registered strategy."""
    fn = SPLIT_REGISTRY.get(strategy)
    train_ids, test_ids, stats = fn(df, **params)  # type: ignore[operator]
    return OuterSplitManifest(
        split_name=split_name,
        strategy=strategy,
        params=dict(params),
        git_sha=git_sha,
        n_total_sessions=len(df),
        n_train=len(train_ids),
        n_test=len(test_ids),
        train_session_ids=sorted(train_ids),
        test_session_ids=sorted(test_ids),
        stats=stats,
    )
