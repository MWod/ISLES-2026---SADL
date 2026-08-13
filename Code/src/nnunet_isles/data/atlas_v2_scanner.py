"""BIDS walker over the ATLAS v2.0 / ISLES 2026 training pool.

Discovers per-session triples (T1w, lesion mask, metadata CSV) under
`<root>/<SITE>/<sub-rNNNsNNN>/ses-1/anat/`, returning a list of
`SessionRecord`s ready for downstream EDA / split generation / nnU-Net
conversion.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from nnunet_isles.data.metadata import MetadataRecord, load_metadata_csv


@dataclass(frozen=True)
class SessionRecord:
    session_id: str  # e.g. "sub-r009s001_ses-1"
    subject: str  # e.g. "sub-r009s001"
    session: str  # e.g. "ses-1"
    site: str  # e.g. "R009"
    t1_path: Path
    mask_path: Path
    metadata_path: Path
    metadata: MetadataRecord | None  # None if the CSV was missing or malformed


def _site_dirs(root: Path) -> list[Path]:
    """Return site dirs sorted. Walks any directory whose immediate children are BIDS
    `sub-*` subdirectories - this picks up ATLAS R-codes AND the V2 SOOP cohort (and
    any future non-R-prefix cohorts) without having to hardcode the prefix list."""
    out: list[Path] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        # Cheap probe: does the dir contain at least one `sub-*` child?
        if any(child.name.startswith("sub-") for child in p.iterdir() if child.is_dir()):
            out.append(p)
    return sorted(out)


def _subject_dirs(site_dir: Path) -> list[Path]:
    return sorted(p for p in site_dir.iterdir() if p.is_dir() and p.name.startswith("sub-"))


def _session_dirs(subject_dir: Path) -> list[Path]:
    return sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name.startswith("ses-"))


def _find_one(anat_dir: Path, suffix: str) -> Path | None:
    matches = sorted(anat_dir.glob(f"*{suffix}"))
    if len(matches) == 0:
        return None
    if len(matches) > 1:
        raise RuntimeError(f"Expected exactly one '{suffix}' file in {anat_dir}, found {len(matches)}")
    return matches[0]


def iter_atlas_v2(root: str | Path, exclude_sites: tuple[str, ...] = ()) -> Iterator[SessionRecord]:
    """Iterate session records under root in deterministic (site, subject, session) order.

    Sites listed in `exclude_sites` are skipped. Default is empty: the permissive
    `_site_dirs()` already filters out directories without `sub-*` children
    (e.g. V1's R016 with only `.DS_Store` is naturally excluded). V1 callers that
    relied on the explicit R016 exclusion still work - R016 just isn't picked up
    by the structure check.
    Sessions missing the T1 or the mask are skipped with no error (logged by the caller).
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"ATLAS root does not exist: {root}")

    excluded = set(exclude_sites)
    for site_dir in _site_dirs(root):
        if site_dir.name in excluded:
            continue
        for subject_dir in _subject_dirs(site_dir):
            for session_dir in _session_dirs(subject_dir):
                anat = session_dir / "anat"
                if not anat.is_dir():
                    continue
                t1 = _find_one(anat, "_space-orig_desc-brain_T1w.nii.gz")
                mask = _find_one(anat, "_space-orig_label-lesion_desc-T1lesion_mask.nii.gz")
                meta_csv = _find_one(anat, "_metadata.csv")
                if t1 is None or mask is None:
                    continue
                metadata: MetadataRecord | None
                try:
                    metadata = load_metadata_csv(meta_csv) if meta_csv is not None else None
                except Exception:
                    metadata = None
                yield SessionRecord(
                    session_id=f"{subject_dir.name}_{session_dir.name}",
                    subject=subject_dir.name,
                    session=session_dir.name,
                    site=site_dir.name,
                    t1_path=t1,
                    mask_path=mask,
                    metadata_path=meta_csv if meta_csv is not None else anat,
                    metadata=metadata,
                )


def scan_atlas_v2(root: str | Path, exclude_sites: tuple[str, ...] = ()) -> list[SessionRecord]:
    """Materialize the iterator as a list for convenience."""
    return list(iter_atlas_v2(root, exclude_sites=exclude_sites))
