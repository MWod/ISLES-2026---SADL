"""Split manifest schemas + JSON IO.

Outer manifest is OUR responsibility; inner manifest is nnU-Net's expected
`splits_final.json` shape (a list of dicts with `train`/`val` keys).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class OuterSplitManifest:
    schema_version: str = "1.0"
    split_name: str = ""
    strategy: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    git_sha: str = "unknown"
    n_total_sessions: int = 0
    n_train: int = 0
    n_test: int = 0
    train_session_ids: list[str] = field(default_factory=list)
    test_session_ids: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class InnerSplitManifest:
    """Companion side-car to nnU-Net's splits_final.json - records how it was generated."""

    schema_version: str = "1.0"
    split_name: str = ""
    n_folds: int = 5
    seed: int = 42
    group_by: str | None = None
    per_fold_stats: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    git_sha: str = "unknown"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_outer_manifest(manifest: OuterSplitManifest, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not manifest.created_at:
        manifest.created_at = _now_iso()
    target = out_dir / "outer.json"
    target.write_text(json.dumps(asdict(manifest), indent=2))
    return target


def load_outer_manifest(out_dir: str | Path) -> OuterSplitManifest:
    out_dir = Path(out_dir)
    payload = json.loads((out_dir / "outer.json").read_text())
    return OuterSplitManifest(**payload)


def write_nnunet_inner_splits(
    folds: list[dict[str, list[str]]],
    inner_meta: InnerSplitManifest,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """Write `inner_splits_final.json` (nnU-Net format) and `inner_meta.json`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not inner_meta.created_at:
        inner_meta.created_at = _now_iso()
    splits_path = out_dir / "inner_splits_final.json"
    meta_path = out_dir / "inner_meta.json"
    splits_path.write_text(json.dumps(folds, indent=2))
    meta_path.write_text(json.dumps(asdict(inner_meta), indent=2))
    return splits_path, meta_path
