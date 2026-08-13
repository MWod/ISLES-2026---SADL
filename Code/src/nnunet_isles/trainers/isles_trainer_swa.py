"""IslesTrainerSWA - stochastic weight averaging (SWA / SWAD).

Maintains a running average of `network.state_dict()` from `swa_start_epoch`
onwards (default 350 of 700). At the end of training, saves the SWA params
to `<checkpoint_dir>/swa.pth`. `finalize.py --use-swa` loads it instead of
`checkpoint_best.pth` for the test-set inference passes.

Why this matters: ATLAS-style chronic-stroke nnU-Net training tends to
oscillate around a basin in the final ~100 epochs; SWA averages those
oscillations and often picks up +0.005-0.015 Dice for free (NeurIPS 2021
SWAD reports similar on domain-generalisation tasks).

InstanceNorm note: we use InstanceNorm throughout (per nnU-Net plans),
which has no running statistics. So there's no BN-recompute step needed
after weight averaging - SWA params are directly usable at inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from nnunet_isles.registry import TRAINER_REGISTRY
from nnunet_isles.trainers.isles_trainer import IslesTrainer


@TRAINER_REGISTRY.register("IslesTrainerSWA")
class IslesTrainerSWA(IslesTrainer):
    """IslesTrainer + Stochastic Weight Averaging."""

    # Config knobs (settable via train.py before instantiation):
    isles_swa_start_epoch: int = 350
    isles_swa_freq: int = 1
    isles_swa_filename: str = "swa.pth"
    # Per-epoch pickled snapshot of `_swa_state` + `_n_swa`. Survives a crash
    # in `on_train_end` and lets us re-emit `swa.pth` offline without
    # retraining. ~600 MB per file, overwritten each epoch.
    isles_swa_state_cache_filename: str = "_swa_state_cache.pt"

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: Any = None,
    ) -> None:
        # Signature MUST mirror upstream nnUNetTrainer.__init__ exactly - upstream
        # introspects `inspect.signature(self.__init__).parameters` then looks up
        # each name in `locals()`. Using *args/**kwargs leaves my_init_kwargs as
        # `{'args': (...), 'kwargs': {}}` instead of the named init args.
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )
        self._swa_state: dict[str, torch.Tensor] | None = None
        self._n_swa: int = 0
        # Set on first call to on_train_epoch_end (after upstream's initialise()
        # populates self.output_folder). We use this flag to attempt cache
        # warm-start exactly once.
        self._swa_cache_restored: bool = False

    def _swa_cache_path(self) -> Path:
        return Path(self.output_folder) / self.isles_swa_state_cache_filename

    def _maybe_restore_swa_cache(self) -> None:
        """Warm-start `_swa_state` + `_n_swa` from disk if a prior epoch's
        snapshot survives. Lets a resumed/recovered run pick up where it left
        off without re-accumulating from epoch swa_start."""
        if self._swa_cache_restored:
            return
        self._swa_cache_restored = True
        cache_path = self._swa_cache_path()
        if not cache_path.exists():
            return
        try:
            payload = torch.load(str(cache_path), map_location="cpu", weights_only=False)
            self._swa_state = payload["swa_state"]
            self._n_swa = int(payload["n_swa"])
            self.print_to_log_file(f"[SWA] restored cache from {cache_path} (n_swa={self._n_swa})")
        except (OSError, KeyError, RuntimeError) as e:
            self.print_to_log_file(
                f"[SWA] WARNING: could not load cache at {cache_path}: {e}. "
                "Starting SWA accumulation from scratch."
            )

    def _save_swa_cache(self) -> None:
        """Pickle the current `_swa_state` + `_n_swa` to disk. Overwrites the
        previous epoch's snapshot. Called every accumulation step so a crash
        anywhere after epoch swa_start can be recovered without retraining."""
        if self._swa_state is None:
            return
        cache_path = self._swa_cache_path()
        payload = {
            "swa_state": self._swa_state,
            "n_swa": int(self._n_swa),
            "swa_start_epoch": int(self.isles_swa_start_epoch),
        }
        # tmp-then-rename so a crash mid-write doesn't corrupt a previous good cache.
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        torch.save(payload, str(tmp_path))
        tmp_path.replace(cache_path)

    def on_train_epoch_end(self, train_outputs):  # type: ignore[override]
        super().on_train_epoch_end(train_outputs)
        cur_epoch = int(self.current_epoch)
        if cur_epoch < int(self.isles_swa_start_epoch):
            return
        if (cur_epoch - int(self.isles_swa_start_epoch)) % max(1, int(self.isles_swa_freq)) != 0:
            return
        # First eligible epoch: try to warm-start from a prior crash's cache.
        self._maybe_restore_swa_cache()
        # Update running average. The first SWA-eligible epoch initialises the buffer.
        with torch.no_grad():
            current = {k: v.detach().clone().float() for k, v in self.network.state_dict().items()}
            if self._swa_state is None:
                self._swa_state = current
                self._n_swa = 1
            else:
                self._n_swa += 1
                for k, v in current.items():
                    # Running mean: avg_n = avg_{n-1} + (v - avg_{n-1}) / n
                    self._swa_state[k].add_(v - self._swa_state[k], alpha=1.0 / self._n_swa)
        # Persist after every accumulation step so a crash before `on_train_end`
        # doesn't lose the in-memory average.
        self._save_swa_cache()
        if cur_epoch % 50 == 0:
            self.print_to_log_file(f"[SWA] epoch={cur_epoch}, n_swa={self._n_swa}")

    def on_train_end(self):  # type: ignore[override]
        super().on_train_end()
        if self._swa_state is None:
            self.print_to_log_file(
                f"[SWA] WARNING: no SWA snapshots accumulated (training ended before "
                f"swa_start_epoch={self.isles_swa_start_epoch}). swa.pth not written."
            )
            return
        # Cast back to the original parameter dtype (we accumulated in float32).
        orig_state = self.network.state_dict()
        swa_state_typed = {k: self._swa_state[k].to(orig_state[k].dtype) for k in orig_state}
        # `self.output_folder` is a `str` from upstream nnU-Net's `join()`, not a
        # Path - wrap before the `/` operator. Otherwise `str / str` raises
        # TypeError and a 700-epoch run loses its swa.pth at the finish line.
        out_path = Path(self.output_folder) / self.isles_swa_filename

        # nnU-Net's predictor loads checkpoints by name and expects the full
        # upstream schema - `trainer_name`, `init_args['configuration']`,
        # `inference_allowed_mirroring_axes`, `network_weights`, etc. See
        # `third_party/nnUNet/nnunetv2/inference/predict_from_raw_data.py:90-93`
        # and upstream's own save_checkpoint at
        # `third_party/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:1194-1214`.
        # We take `checkpoint_best.pth` as a schema template, replace only the
        # `network_weights` entry with the SWA state, then save as swa.pth. That
        # way `finalize.py --use-swa` can load swa.pth through the same
        # `initialize_from_trained_model_folder` code path without any custom
        # deserialisation logic.
        template_path = Path(self.output_folder) / "checkpoint_best.pth"
        if template_path.exists():
            payload = torch.load(str(template_path), map_location="cpu", weights_only=False)
        else:
            # Fallback if checkpoint_best.pth is missing (unusual - but keep the
            # SWA write non-fatal). Populate the minimum keys the predictor reads.
            self.print_to_log_file(
                f"[SWA] WARNING: checkpoint_best.pth not found at {template_path}; "
                "writing bare-payload swa.pth. finalize --use-swa may fail to load it."
            )
            payload = {
                "trainer_name": type(self).__name__,
                "init_args": getattr(self, "my_init_kwargs", {}),
                "inference_allowed_mirroring_axes": getattr(self, "inference_allowed_mirroring_axes", None),
            }
        payload["network_weights"] = swa_state_typed
        # SWA-specific provenance - nnU-Net ignores unknown keys.
        payload["swa_n_avg"] = int(self._n_swa)
        payload["swa_start_epoch"] = int(self.isles_swa_start_epoch)
        torch.save(payload, str(out_path))
        self.print_to_log_file(f"[SWA] wrote {out_path} (n_avg={self._n_swa}) - nnU-Net-schema payload")


def average_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Utility: arithmetic mean of a list of state_dicts. Exposed for tests + soup."""
    if not state_dicts:
        raise ValueError("average_state_dicts requires at least one input")
    keys = state_dicts[0].keys()
    avg = {k: torch.zeros_like(state_dicts[0][k], dtype=torch.float32) for k in keys}
    for sd in state_dicts:
        if set(sd.keys()) != set(keys):
            raise ValueError("all state_dicts must share the same keys")
        for k in keys:
            avg[k].add_(sd[k].to(torch.float32))
    n = len(state_dicts)
    for k in keys:
        avg[k].div_(n)
        avg[k] = avg[k].to(state_dicts[0][k].dtype)
    return avg


__all__ = ["IslesTrainerSWA", "average_state_dicts"]
