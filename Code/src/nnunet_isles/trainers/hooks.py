"""Trainer-side hooks: TB logging, throughput, grad norm."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[assignment, misc]


class TensorboardHook:
    """Wraps a SummaryWriter and writes per-epoch / per-step scalars + viz images.

    The trainer subclass owns this hook; viz cases are picked deterministically
    by `visualization.case_select` after fold 0 of the first val epoch.
    """

    def __init__(self, log_dir: str | Path, fold: int):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir
        self.fold = fold
        self._writer = SummaryWriter(str(log_dir)) if SummaryWriter is not None else None

    def log_scalars(self, step: int, scalars: dict[str, float]) -> None:
        if self._writer is None:
            return
        for k, v in scalars.items():
            self._writer.add_scalar(k, float(v), step)
        self._writer.flush()

    def log_image(self, tag: str, image_chw: Any, step: int) -> None:
        if self._writer is None:
            return
        self._writer.add_image(tag, image_chw, step, dataformats="CHW")
        self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()


class ThroughputHook:
    """Tracks samples/sec and batches/sec across an epoch."""

    def __init__(self) -> None:
        self._epoch_start: float | None = None

    def epoch_start(self) -> None:
        self._epoch_start = time.monotonic()

    def epoch_end(self, n_iterations: int, batch_size: int) -> dict[str, float]:
        if self._epoch_start is None:
            return {"batches_per_sec": 0.0, "samples_per_sec": 0.0}
        elapsed = max(1e-6, time.monotonic() - self._epoch_start)
        return {
            "batches_per_sec": float(n_iterations) / elapsed,
            "samples_per_sec": float(n_iterations * batch_size) / elapsed,
        }


class GradNormHook:
    """Captures global grad norm after the backward pass each iteration."""

    def __init__(self) -> None:
        self.last: float = 0.0

    def capture(self, parameters: Any) -> None:
        import torch

        total_sq = 0.0
        for p in parameters:
            if p.grad is None:
                continue
            total_sq += float(p.grad.detach().pow(2).sum().item())
        self.last = float(torch.tensor(total_sq).sqrt().item())
