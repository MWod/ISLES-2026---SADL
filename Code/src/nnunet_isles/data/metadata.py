"""Per-session metadata records parsed from ATLAS CSV side-cars."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MetadataRecord:
    """Per-session metadata as parsed from `<sub>_<ses>_metadata.csv`.

    Fields parallel ATLAS v2.0 + ISLES 2026 metadata schema.
    """

    atlas2_dataset: str  # "Training" / "Generalizability" / "Hidden"
    session_id: str
    days_post_stroke: float | None
    chronicity: str | None  # may be empty in the raw CSV
    site: str


def load_metadata_csv(csv_path: str | Path) -> MetadataRecord:
    """Parse the single-row metadata CSV that accompanies each session."""
    csv_path = Path(csv_path)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        try:
            row = next(reader)
        except StopIteration as e:
            raise ValueError(f"Empty metadata CSV: {csv_path}") from e

    def _parse_float(value: str) -> float | None:
        value = (value or "").strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _parse_str(value: str) -> str | None:
        value = (value or "").strip()
        return value or None

    return MetadataRecord(
        atlas2_dataset=row.get("ATLAS2_DATASET", "").strip(),
        session_id=row.get("SESSION_ID", "").strip(),
        days_post_stroke=_parse_float(row.get("DAYS_POST_STROKE", "")),
        chronicity=_parse_str(row.get("CHRONICITY", "")),
        site=row.get("SITE", "").strip(),
    )
