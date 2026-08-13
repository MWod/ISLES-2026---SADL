"""Tests for the output-space ensemble."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from nnunet_isles.inference.output_ensemble import (
    ensemble_one_case,
    find_common_cases,
)


def _save_softmax(path: Path, prob_fg: np.ndarray) -> None:
    full = np.stack([1.0 - prob_fg, prob_fg], axis=0).astype(np.float32)
    np.savez(str(path), probabilities=full)


def test_ensemble_uniform_averaging(tmp_path: Path):
    """Uniform-weighted average of three softmax arrays equals their mean."""
    p1 = np.array([[[0.2, 0.4], [0.6, 0.8]]], dtype=np.float32)
    p2 = np.array([[[0.4, 0.6], [0.8, 0.2]]], dtype=np.float32)
    p3 = np.array([[[0.6, 0.8], [0.2, 0.4]]], dtype=np.float32)

    _save_softmax(tmp_path / "e1.npz", p1)
    _save_softmax(tmp_path / "e2.npz", p2)
    _save_softmax(tmp_path / "e3.npz", p3)

    combined = ensemble_one_case([tmp_path / "e1.npz", tmp_path / "e2.npz", tmp_path / "e3.npz"])
    # combined shape: (2, 1, 2, 2). FG channel = mean of (p1, p2, p3).
    expected_fg = (p1 + p2 + p3) / 3
    np.testing.assert_array_almost_equal(combined[1], expected_fg)


def test_ensemble_weighted_averaging(tmp_path: Path):
    """Non-uniform weights are applied in proportion."""
    p1 = np.array([[[0.0]]], dtype=np.float32)
    p2 = np.array([[[1.0]]], dtype=np.float32)
    _save_softmax(tmp_path / "e1.npz", p1)
    _save_softmax(tmp_path / "e2.npz", p2)

    # 3:1 weighting → 0.25 expected.
    combined = ensemble_one_case(
        [tmp_path / "e1.npz", tmp_path / "e2.npz"],
        weights=[3.0, 1.0],
    )
    assert combined[1].item() == pytest.approx(0.25, rel=1e-5)


def test_ensemble_rejects_shape_mismatch(tmp_path: Path):
    p1 = np.zeros((2, 2, 2), dtype=np.float32)
    p2 = np.zeros((3, 3, 3), dtype=np.float32)
    _save_softmax(tmp_path / "e1.npz", p1)
    _save_softmax(tmp_path / "e2.npz", p2)
    with pytest.raises(ValueError, match="shape mismatch"):
        ensemble_one_case([tmp_path / "e1.npz", tmp_path / "e2.npz"])


def test_ensemble_requires_nonempty_list():
    with pytest.raises(ValueError, match="at least one"):
        ensemble_one_case([])


def test_ensemble_rejects_misaligned_weights(tmp_path: Path):
    p = np.zeros((2, 2, 2), dtype=np.float32)
    _save_softmax(tmp_path / "e1.npz", p)
    with pytest.raises(ValueError, match="align 1:1"):
        ensemble_one_case([tmp_path / "e1.npz"], weights=[1.0, 0.5])


def test_output_space_ensemble_cli_importable():
    """Smoke: `scripts/output_space_ensemble.py` parses without import errors -
    catches the obvious "missing import / missing attribute" class of regression
    without needing to bootstrap a full nnUNet_raw/preprocessed/results fixture
    (the CLI itself touches those, so end-to-end is checked on HPC)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_ose_importable",
        Path(__file__).resolve().parents[1] / "scripts" / "output_space_ensemble.py",
    )
    ose = importlib.util.module_from_spec(spec)
    # Loading the module (not invoking main()) exercises top-level imports.
    spec.loader.exec_module(ose)
    assert callable(ose.main)


def test_find_common_cases(tmp_path: Path):
    d1 = tmp_path / "exp1"
    d2 = tmp_path / "exp2"
    d3 = tmp_path / "exp3"
    for d in (d1, d2, d3):
        d.mkdir()
    for case in ("a", "b", "c"):
        (d1 / f"{case}.npz").write_bytes(b"")
    for case in ("a", "b"):
        (d2 / f"{case}.npz").write_bytes(b"")
    for case in ("a", "b", "d"):
        (d3 / f"{case}.npz").write_bytes(b"")
    common = find_common_cases([d1, d2, d3])
    assert common == ["a", "b"]
