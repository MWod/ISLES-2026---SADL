"""Fold-ensemble - mean softmax across the 5 fold models.

Operates either on disk-cached probability maps (preferred - saves memory and
allows multi-experiment ensembling later) or on a list of in-memory numpy arrays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def softmax_mean_ensemble(probs_per_fold: list[np.ndarray]) -> np.ndarray:
    """Mean softmax across folds. All arrays must share shape."""
    if not probs_per_fold:
        raise ValueError("softmax_mean_ensemble requires at least one fold probability map")
    stacked = np.stack(probs_per_fold, axis=0)
    return stacked.mean(axis=0)


def load_fold_probs(prob_dir: str | Path, case_id: str) -> np.ndarray:
    """Load a per-case probability map from disk (.npz or .npy)."""
    prob_dir = Path(prob_dir)
    npz_path = prob_dir / f"{case_id}.npz"
    if npz_path.exists():
        with np.load(npz_path) as data:
            # nnU-Net's predictor stores under key 'probabilities' when save_probabilities=True
            return data["probabilities"]
    npy_path = prob_dir / f"{case_id}.npy"
    if npy_path.exists():
        return np.load(npy_path)
    raise FileNotFoundError(f"No probability map for {case_id} under {prob_dir}")


def ensemble_case_from_disk(prob_dirs: list[str | Path], case_id: str) -> np.ndarray:
    """Load this case from each fold dir, average, return ensemble probability map."""
    return softmax_mean_ensemble([load_fold_probs(d, case_id) for d in prob_dirs])
