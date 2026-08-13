"""Tests for the session metadata loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from nnunet_isles.utils.session_metadata import (
    load_session_metadata,
    session_ids_by_cohort,
    session_ids_by_volume_bucket,
)


def _write_tsv(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "sessions.tsv"
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    # Bust lru_cache between tests - same path key would return stale entry.
    load_session_metadata.cache_clear()
    return p


def test_basic_load(tmp_path: Path):
    p = _write_tsv(
        tmp_path,
        [
            {
                "session_id": "sub-a",
                "cohort": "Training",
                "lesion_volume_ml": 2.5,
                "site": "R001",
                "train_eligible": True,
            },
            {
                "session_id": "sub-b",
                "cohort": "ATLAS3",
                "lesion_volume_ml": 0.3,
                "site": "SOOP",
                "train_eligible": True,
            },
        ],
    )
    m = load_session_metadata(p)
    assert m["sub-a"]["cohort"] == "Training"
    assert m["sub-a"]["lesion_volume_ml"] == pytest.approx(2.5)
    assert m["sub-b"]["cohort"] == "ATLAS3"
    assert m["sub-b"]["site"] == "SOOP"


def test_cohort_collapse_rules(tmp_path: Path):
    p = _write_tsv(
        tmp_path,
        [
            {
                "session_id": "a",
                "cohort": "Training_ATLAS2",
                "lesion_volume_ml": 1.0,
                "site": "X",
                "train_eligible": True,
            },
            {
                "session_id": "b",
                "cohort": "Testing",
                "lesion_volume_ml": 1.0,
                "site": "X",
                "train_eligible": True,
            },
            {
                "session_id": "c",
                "cohort": "Testing_ATLAS2",
                "lesion_volume_ml": 1.0,
                "site": "X",
                "train_eligible": True,
            },
            {"session_id": "d", "cohort": "", "lesion_volume_ml": 1.0, "site": "X", "train_eligible": True},
            {
                "session_id": "e",
                "cohort": "ATLAS3",
                "lesion_volume_ml": 1.0,
                "site": "X",
                "train_eligible": True,
            },
            {
                "session_id": "f",
                "cohort": "Some-Unknown-Cohort",
                "lesion_volume_ml": 1.0,
                "site": "X",
                "train_eligible": True,
            },
        ],
    )
    m = load_session_metadata(p)
    assert m["a"]["cohort"] == "Training"
    assert m["b"]["cohort"] == "Testing"
    assert m["c"]["cohort"] == "Testing"
    assert m["d"]["cohort"] == "Training"
    assert m["e"]["cohort"] == "ATLAS3"
    assert m["f"]["cohort"] == "Training"  # unknown → Training fallback


def test_session_ids_by_cohort(tmp_path: Path):
    p = _write_tsv(
        tmp_path,
        [
            {
                "session_id": "a",
                "cohort": "Training",
                "lesion_volume_ml": 1.0,
                "site": "X",
                "train_eligible": True,
            },
            {
                "session_id": "b",
                "cohort": "ATLAS3",
                "lesion_volume_ml": 1.0,
                "site": "X",
                "train_eligible": True,
            },
            {
                "session_id": "c",
                "cohort": "Training",
                "lesion_volume_ml": 1.0,
                "site": "X",
                "train_eligible": True,
            },
        ],
    )
    m = load_session_metadata(p)
    by = session_ids_by_cohort(m)
    assert set(by["Training"]) == {"a", "c"}
    assert by["ATLAS3"] == ["b"]
    # Sorted within each cohort.
    assert by["Training"] == sorted(by["Training"])


def test_session_ids_by_volume_bucket(tmp_path: Path):
    p = _write_tsv(
        tmp_path,
        [
            {
                "session_id": "tiny",
                "cohort": "Training",
                "lesion_volume_ml": 0.1,
                "site": "X",
                "train_eligible": True,
            },
            {
                "session_id": "mid",
                "cohort": "Training",
                "lesion_volume_ml": 2.0,
                "site": "X",
                "train_eligible": True,
            },
            {
                "session_id": "big",
                "cohort": "Training",
                "lesion_volume_ml": 20.0,
                "site": "X",
                "train_eligible": True,
            },
            {
                "session_id": "huge",
                "cohort": "Training",
                "lesion_volume_ml": 100.0,
                "site": "X",
                "train_eligible": True,
            },
        ],
    )
    m = load_session_metadata(p)
    by = session_ids_by_volume_bucket(m)
    assert by["<0.5mL"] == ["tiny"]
    assert by["0.5-5mL"] == ["mid"]
    assert by["5-50mL"] == ["big"]
    assert by[">=50mL"] == ["huge"]


def test_missing_tsv_raises(tmp_path: Path):
    load_session_metadata.cache_clear()
    with pytest.raises(FileNotFoundError):
        load_session_metadata(tmp_path / "nope.tsv")
