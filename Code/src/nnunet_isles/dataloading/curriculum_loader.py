"""CurriculumDataLoader3D.

Implements a size-based curriculum: at epoch 0 only the top-X% of cases
by lesion volume are sampleable (default warmup floor: 90th percentile,
i.e. the largest 10%); over `warmup_epochs` the visible pool grows
linearly until the full case list is unmasked, after which the loader
behaves identically to the base nnUNetDataLoader.

The case-pool restriction is implemented via per-case sampling
probabilities: cases below the current percentile get probability 0,
cases at or above get uniform probability. This is mathematically
equivalent to a hard mask and uses the same plumbing as
CohortBalancedDataLoader3D, so multi-threaded behaviour is identical.

The trainer (`IslesTrainerCurriculum`) sets `self.current_epoch` on the
loader at each `on_train_epoch_start`, which recomputes the
`sampling_probabilities` array.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
except ImportError:
    nnUNetDataLoader = object  # type: ignore[assignment, misc]


def compute_curriculum_visible_threshold(
    lesion_volumes_ml: np.ndarray,
    *,
    current_epoch: int,
    warmup_epochs: int,
    floor_percentile: float = 90.0,
) -> float:
    """Return the volume threshold below which cases are MASKED OUT.

    At `current_epoch=0`, threshold = percentile(volumes, floor_percentile)
    so only the top (100-floor_percentile)% by volume are visible.
    At `current_epoch >= warmup_epochs`, threshold = -inf (all visible).
    Linear interpolation in between.
    """
    if current_epoch >= warmup_epochs or warmup_epochs <= 0:
        return -float("inf")
    if len(lesion_volumes_ml) == 0:
        return -float("inf")
    frac = max(0.0, min(1.0, current_epoch / float(warmup_epochs)))
    current_percentile = float(floor_percentile) * (1.0 - frac)
    if current_percentile <= 0.0:
        return -float("inf")
    return float(np.percentile(lesion_volumes_ml, current_percentile))


def compute_curriculum_probabilities(
    case_volumes_ml: dict[str, float],
    indices: list[str],
    *,
    current_epoch: int,
    warmup_epochs: int,
    floor_percentile: float = 90.0,
) -> np.ndarray:
    """Return uniform sampling probability over visible cases at `current_epoch`."""
    if len(indices) == 0:
        return np.array([], dtype=np.float64)
    vols = np.array([case_volumes_ml.get(cid, 0.0) for cid in indices], dtype=np.float64)
    thr = compute_curriculum_visible_threshold(
        vols, current_epoch=current_epoch, warmup_epochs=warmup_epochs, floor_percentile=floor_percentile
    )
    visible = vols >= thr
    n_visible = int(visible.sum())
    if n_visible == 0:
        # Curriculum collapsed (no cases) - fall back to uniform across all.
        return np.full(len(indices), 1.0 / len(indices), dtype=np.float64)
    probs = np.zeros(len(indices), dtype=np.float64)
    probs[visible] = 1.0 / n_visible
    return probs


class CurriculumDataLoader3D(nnUNetDataLoader):  # type: ignore[misc, valid-type]
    """nnUNetDataLoader that exposes a size-based curriculum schedule.

    Multi-process safety: nnU-Net wraps the loader with `NonDetMultiThreadedAugmenter`,
    which forks N worker processes - each gets its own copy of this loader. A
    plain Python attribute set via `set_current_epoch()` on the parent process
    would never propagate to workers (their data and `sampling_probabilities`
    were captured at fork time). To fix this we keep the epoch counter in a
    `multiprocessing.Value` (mmap-backed shared memory) and have `get_indices`
    recompute `sampling_probabilities` whenever the shared epoch changes.
    """

    def __init__(
        self,
        *args: Any,
        case_volumes_ml: dict[str, float] | None = None,
        warmup_epochs: int = 150,
        floor_percentile: float = 90.0,
        **kwargs: Any,
    ) -> None:
        if case_volumes_ml is None:
            raise ValueError("CurriculumDataLoader3D requires `case_volumes_ml` (dict[case_id, volume_ml])")
        kwargs.pop("case_volumes_ml", None)
        super().__init__(*args, **kwargs)
        self._case_volumes_ml = dict(case_volumes_ml)
        self._warmup_epochs = int(warmup_epochs)
        self._floor_percentile = float(floor_percentile)
        # Shared epoch counter - survives fork into the augmenter's worker procs.
        # `multiprocessing.Value('i', 0)` is mmap-backed and visible across forks.
        import multiprocessing as _mp

        self._epoch_counter = _mp.Value("i", 0)
        # Per-loader-copy cache so workers only recompute probs when epoch changes.
        self._cached_epoch_for_probs: int = -1
        # Initialise probabilities at epoch 0.
        self.set_current_epoch(0)

    def set_current_epoch(self, epoch: int) -> None:
        """Update the shared epoch counter. Called by the trainer's
        `on_train_epoch_start` hook on the PARENT process. Workers re-read the
        counter on each `get_indices` call and recompute probabilities lazily."""
        with self._epoch_counter.get_lock():
            self._epoch_counter.value = int(epoch)
        # Refresh the parent's local copy too (matters for SingleThreadedAugmenter).
        self._refresh_sampling_probabilities()

    def _refresh_sampling_probabilities(self) -> None:
        """Recompute `sampling_probabilities` from the shared counter - cached so
        each batch is cheap once the epoch stops changing."""
        current_epoch = int(self._epoch_counter.value)
        if current_epoch == self._cached_epoch_for_probs:
            return
        self.sampling_probabilities = compute_curriculum_probabilities(
            self._case_volumes_ml,
            list(self.indices),
            current_epoch=current_epoch,
            warmup_epochs=self._warmup_epochs,
            floor_percentile=self._floor_percentile,
        )
        self._cached_epoch_for_probs = current_epoch

    def get_indices(self):  # type: ignore[override]
        # Called PER BATCH by the underlying batchgenerators DataLoader. We
        # cheaply check the shared epoch counter and only recompute the
        # probability vector when the epoch actually advances.
        self._refresh_sampling_probabilities()
        return super().get_indices()

    def visible_case_count(self) -> int:
        """Return the number of cases currently visible at this epoch (for logging)."""
        if self.sampling_probabilities is None:
            return len(self.indices)
        return int((np.asarray(self.sampling_probabilities) > 0.0).sum())


__all__ = [
    "CurriculumDataLoader3D",
    "compute_curriculum_probabilities",
    "compute_curriculum_visible_threshold",
]
