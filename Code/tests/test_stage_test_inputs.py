"""Regression tests for finalize._stage_test_inputs multi-channel staging.

Catches the bug where finalize.py only symlinked channel 0 (`<sid>_0000.nii.gz`)
and the contralateral channel was never staged for hpcv4, causing inference to
crash with "expected 2 channels, but got 1" because the network was trained on
2 input channels (T1 + flipped T1).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _import_finalize():
    """Load finalize.py as a module without executing its `main()`."""
    src = Path(__file__).resolve().parents[1] / "scripts" / "finalize.py"
    spec = importlib.util.spec_from_file_location("_finalize_under_test", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stage_single_channel(tmp_path):
    finalize = _import_finalize()
    raw = tmp_path / "imagesTr"
    raw.mkdir()
    sid = "sub-r001s001_ses-1"
    (raw / f"{sid}_0000.nii.gz").write_bytes(b"dummy")

    staging = tmp_path / "staging"
    finalize._stage_test_inputs([sid], raw, staging)
    assert (staging / f"{sid}_0000.nii.gz").is_symlink()
    assert not (staging / f"{sid}_0001.nii.gz").exists()


def test_stage_two_channels(tmp_path):
    """Regression: Dataset503 (T1 + flipped T1) must stage BOTH channels."""
    finalize = _import_finalize()
    raw = tmp_path / "imagesTr"
    raw.mkdir()
    sid = "sub-r001s001_ses-1"
    (raw / f"{sid}_0000.nii.gz").write_bytes(b"t1")
    (raw / f"{sid}_0001.nii.gz").write_bytes(b"flipped")

    staging = tmp_path / "staging"
    finalize._stage_test_inputs([sid], raw, staging)
    assert (staging / f"{sid}_0000.nii.gz").is_symlink()
    assert (staging / f"{sid}_0001.nii.gz").is_symlink()
    assert os.readlink(staging / f"{sid}_0000.nii.gz") == str(raw / f"{sid}_0000.nii.gz")
    assert os.readlink(staging / f"{sid}_0001.nii.gz") == str(raw / f"{sid}_0001.nii.gz")


def test_stage_three_channels(tmp_path):
    """Regression: Dataset504 (T1 + flipped + difference) stages all three."""
    finalize = _import_finalize()
    raw = tmp_path / "imagesTr"
    raw.mkdir()
    sid = "sub-r001s001_ses-1"
    for ch in (0, 1, 2):
        (raw / f"{sid}_{ch:04d}.nii.gz").write_bytes(f"ch{ch}".encode())

    staging = tmp_path / "staging"
    finalize._stage_test_inputs([sid], raw, staging)
    for ch in (0, 1, 2):
        assert (staging / f"{sid}_{ch:04d}.nii.gz").is_symlink()


def test_stage_missing_channel_0_raises(tmp_path):
    finalize = _import_finalize()
    raw = tmp_path / "imagesTr"
    raw.mkdir()
    sid = "sub-r001s001_ses-1"
    # No files at all.
    staging = tmp_path / "staging"
    try:
        finalize._stage_test_inputs([sid], raw, staging)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError for missing channel 0")
