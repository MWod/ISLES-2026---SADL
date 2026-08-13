"""Per-session metadata helpers used by the cohort-aware, bucket-weighted,
curriculum, and MoE trainers.

Provides a single function `load_session_metadata` that reads a per-session
metadata TSV (or Parquet) produced by the EDA step and returns a flat dict
keyed by session_id. The function is cached so multiple trainer subclasses
can call it without repeated I/O.

If the metadata file is absent, `load_session_metadata` raises
`FileNotFoundError` with instructions for producing it - it is required
by the trainers that consume it, but not by inference.

Cohort collapse rule:
  Training, Training_ATLAS2  -> "Training"
  Testing,  Testing_ATLAS2   -> "Testing"
  ATLAS3                     -> "ATLAS3"
  empty / unknown            -> "Training"  (modal fallback for header-only-meta cases)
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

_COHORT_REMAP: dict[str, str] = {
    "Training": "Training",
    "Training_ATLAS2": "Training",
    "Testing": "Testing",
    "Testing_ATLAS2": "Testing",
    "ATLAS3": "ATLAS3",
    "": "Training",
}


def _collapse_cohort(raw: str | None) -> str:
    if raw is None or not isinstance(raw, str):
        return "Training"
    return _COHORT_REMAP.get(raw.strip(), "Training")


@functools.lru_cache(maxsize=8)
def load_session_metadata(sessions_tsv_path: str | Path) -> dict[str, dict[str, Any]]:
    """Read sessions.tsv → {session_id: {"cohort", "lesion_volume_ml", "site", "train_eligible"}}.

    Cached via lru_cache so repeated trainer-init calls don't re-read disk.
    Pass the absolute path (cache key) - relative paths will miss the cache.

    Args:
        sessions_tsv_path: path to sessions.tsv (or sessions.parquet - both supported).

    Returns:
        dict mapping session_id → metadata dict with the 4 fields listed above.
        Cohort is collapsed per the rule in this module's docstring.
    """
    import pandas as pd

    p = Path(sessions_tsv_path)
    if not p.exists():
        raise FileNotFoundError(
            f"sessions metadata not found: {p}. "
            "This file is produced by the project's EDA step; re-run the EDA "
            "pipeline (or set ISLES_SESSIONS_TSV to a pre-built copy) before "
            "using the cohort-aware, bucket-weighted, curriculum, or MoE trainers."
        )
    df = pd.read_parquet(p) if p.suffix in (".parquet", ".pq") else pd.read_csv(p, sep="\t")

    required = {"session_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sessions.tsv missing required columns: {missing}")

    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        sid = str(row["session_id"])
        cohort_raw = row.get("cohort") if "cohort" in df.columns else None
        out[sid] = {
            "cohort": _collapse_cohort(cohort_raw),
            "lesion_volume_ml": float(row.get("lesion_volume_ml", float("nan"))),
            "site": str(row.get("site", "")),
            "train_eligible": bool(row.get("train_eligible", True))
            if "train_eligible" in df.columns
            else True,
        }
    return out


def session_ids_by_cohort(meta: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Invert `meta` to {cohort: [session_id, ...]}.

    Sorts session_ids within each cohort for determinism.
    """
    out: dict[str, list[str]] = {}
    for sid, m in meta.items():
        out.setdefault(m["cohort"], []).append(sid)
    for k in out:
        out[k].sort()
    return out


def session_ids_by_volume_bucket(
    meta: dict[str, dict[str, Any]],
    buckets_ml: tuple[float, ...] = (0.5, 5.0, 50.0),
) -> dict[str, list[str]]:
    """Invert `meta` to {bucket_name: [session_id, ...]}. Bucket names match
    the V1/V2 EDA convention: `<0.5mL`, `0.5-5mL`, `5-50mL`, `>=50mL`.
    """
    cuts = sorted(buckets_ml)
    names: list[str] = []
    prev = 0.0
    for cut in cuts:
        names.append(f"<{cut:g}mL" if prev == 0 else f"{prev:g}-{cut:g}mL")
        prev = cut
    names.append(f">={prev:g}mL")

    def _bucket_of(vol_ml: float) -> str:
        for cut, name in zip(cuts, names[:-1], strict=False):
            if vol_ml < cut:
                return name
        return names[-1]

    out: dict[str, list[str]] = {n: [] for n in names}
    for sid, m in meta.items():
        out[_bucket_of(m["lesion_volume_ml"])].append(sid)
    for k in out:
        out[k].sort()
    return out


__all__ = [
    "load_session_metadata",
    "session_ids_by_cohort",
    "session_ids_by_volume_bucket",
]
