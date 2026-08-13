"""Lightweight tests for the ATLAS BIDS walker - uses a temp directory tree."""

from __future__ import annotations

from pathlib import Path

from nnunet_isles.data.atlas_v2_scanner import scan_atlas_v2


def _make_fake_session(root: Path, site: str, subject_id: str, session: str = "ses-1") -> None:
    anat = root / site / subject_id / session / "anat"
    anat.mkdir(parents=True, exist_ok=True)
    base = f"{subject_id}_{session}"
    (anat / f"{base}_space-orig_desc-brain_T1w.nii.gz").write_bytes(b"")
    (anat / f"{base}_space-orig_label-lesion_desc-T1lesion_mask.nii.gz").write_bytes(b"")
    (anat / f"{base}_metadata.csv").write_text(
        f"ATLAS2_DATASET,SESSION_ID,DAYS_POST_STROKE,CHRONICITY,SITE\nTraining,{base},5.0,,{site}\n"
    )


def test_scanner_discovers_expected_sessions(tmp_path: Path):
    root = tmp_path / "Training_Raw"
    _make_fake_session(root, "R001", "sub-r001s001")
    _make_fake_session(root, "R001", "sub-r001s002")
    _make_fake_session(root, "R016", "sub-r016s001")  # in V2 R016 has 1 session - included by default
    _make_fake_session(root, "R009", "sub-r009s001")

    sessions = scan_atlas_v2(root)
    site_ids = sorted({s.site for s in sessions})
    # V2: R016 is non-empty and SHOULD be picked up by default. V1 callers that
    # need it gone can still pass `exclude_sites=("R016",)` explicitly.
    assert "R016" in site_ids
    assert "R001" in site_ids
    assert "R009" in site_ids
    assert len(sessions) == 4

    s = next(s for s in sessions if s.subject == "sub-r001s001")
    assert s.metadata is not None
    assert s.metadata.atlas2_dataset == "Training"
    assert s.metadata.days_post_stroke == 5.0
    assert s.metadata.chronicity is None


def test_scanner_skips_dirs_without_bids_subjects(tmp_path: Path):
    """Empty-on-disk site dirs (the V1 R016 pattern with only .DS_Store) are
    naturally excluded by the structure-based `_site_dirs` filter."""
    root = tmp_path / "Training_Raw"
    _make_fake_session(root, "R001", "sub-r001s001")
    # Site directory without any `sub-*` child - must be skipped.
    (root / "R016").mkdir(parents=True)
    (root / "R016" / ".DS_Store").write_text("")

    sessions = scan_atlas_v2(root)
    assert {s.site for s in sessions} == {"R001"}


def test_scanner_picks_up_non_r_prefix_cohorts(tmp_path: Path):
    """V2 introduces SOOP cohort. The structure-based detector must find it."""
    root = tmp_path / "Training_Raw_V2"
    _make_fake_session(root, "R001", "sub-r001s001")
    _make_fake_session(root, "SOOP", "sub-soop0001")

    sessions = scan_atlas_v2(root)
    site_ids = sorted({s.site for s in sessions})
    assert site_ids == ["R001", "SOOP"]


def test_scanner_explicit_exclude_still_honored(tmp_path: Path):
    """Backward compat with V1 callers that explicitly passed `exclude_sites=("R016",)`."""
    root = tmp_path / "Training_Raw"
    _make_fake_session(root, "R001", "sub-r001s001")
    _make_fake_session(root, "R016", "sub-r016s001")

    sessions = scan_atlas_v2(root, exclude_sites=("R016",))
    assert {s.site for s in sessions} == {"R001"}
