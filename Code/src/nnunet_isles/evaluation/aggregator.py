"""Cross-fold + ensemble aggregation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

METRIC_COLUMNS = ("dice", "hd95", "avd_ml", "lesion_f1", "count_f1")


def _finite(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def _summary(series: pd.Series) -> dict[str, float]:
    s = _finite(series)
    if len(s) == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "p10": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
        }
    return {
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "p10": float(s.quantile(0.10)),
        "p50": float(s.quantile(0.50)),
        "p90": float(s.quantile(0.90)),
    }


def aggregate_cv(per_case_per_fold: list[pd.DataFrame]) -> dict[str, Any]:
    """Aggregate per-case metric tables across folds (validation CV)."""
    if not per_case_per_fold:
        return {}
    by_metric: dict[str, dict[str, dict[str, float]]] = {}
    fold_means: dict[str, list[float]] = {m: [] for m in METRIC_COLUMNS}
    for fold_i, df in enumerate(per_case_per_fold):
        by_metric[f"fold_{fold_i}"] = {m: _summary(df[m]) for m in METRIC_COLUMNS if m in df.columns}
        for m in METRIC_COLUMNS:
            if m in df.columns:
                fold_means[m].append(float(_finite(df[m]).mean())) if not df[m].empty else None
    cv_summary = {
        m: {"mean": float(np.mean(v)) if v else float("nan"), "std": float(np.std(v)) if v else float("nan")}
        for m, v in fold_means.items()
    }
    return {"per_fold": by_metric, "cv_summary": cv_summary}


def aggregate_ensemble(per_case: pd.DataFrame) -> dict[str, Any]:
    """Aggregate per-case metrics for the held-out test ensemble pass."""
    out: dict[str, Any] = {}
    for m in METRIC_COLUMNS:
        if m in per_case.columns:
            out[m] = _summary(per_case[m])
    # Per-site mean Dice
    if "site" in per_case.columns and "dice" in per_case.columns:
        out["per_site_dice"] = {
            site: float(_finite(group["dice"]).mean()) for site, group in per_case.groupby("site")
        }
    # Per-bucket mean Dice
    if "lesion_bucket" in per_case.columns and "dice" in per_case.columns:
        out["per_bucket_dice"] = {
            str(bucket): float(_finite(group["dice"]).mean())
            for bucket, group in per_case.groupby("lesion_bucket")
        }
    return out
