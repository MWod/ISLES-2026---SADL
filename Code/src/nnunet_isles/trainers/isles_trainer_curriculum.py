"""IslesTrainerCurriculum - size-based curriculum learning.

Wraps the training loader with `CurriculumDataLoader3D` and ticks the
curriculum forward each epoch via `on_train_epoch_start`. Validation
loader stays vanilla. Per-case volume comes from sessions.tsv via the
shared `load_session_metadata` helper.

Curriculum schedule:
  epoch 0      → only the top-10% by volume visible (floor_percentile=90)
  epoch w/2    → top-55% visible (linear interpolation)
  epoch ≥ w    → all cases visible (loader behaves like vanilla)

Default warmup_epochs=150. Configurable via class attrs (set by train.py
from cfg.trainer.* knobs).
"""

from __future__ import annotations

import os
from pathlib import Path

from nnunet_isles.dataloading.curriculum_loader import CurriculumDataLoader3D
from nnunet_isles.registry import TRAINER_REGISTRY
from nnunet_isles.trainers.isles_trainer import IslesTrainer
from nnunet_isles.utils.session_metadata import load_session_metadata

try:
    from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
        NonDetMultiThreadedAugmenter,
    )
    from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
    from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
    from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
except ImportError:
    NonDetMultiThreadedAugmenter = None  # type: ignore[assignment]
    SingleThreadedAugmenter = None  # type: ignore[assignment]
    nnUNetDataLoader = None  # type: ignore[assignment]
    get_allowed_n_proc_DA = None  # type: ignore[assignment]


def _resolve_sessions_tsv() -> Path:
    env = os.environ.get("ISLES_SESSIONS_TSV")
    if env:
        p = Path(env)
        if not p.exists():
            raise FileNotFoundError(f"ISLES_SESSIONS_TSV points to non-existent path: {p}")
        return p
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "data_analysis" / "sessions.tsv"
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "Could not find sessions.tsv. Set $ISLES_SESSIONS_TSV or ensure "
        "data_analysis/sessions.tsv exists at the repo root."
    )


@TRAINER_REGISTRY.register("IslesTrainerCurriculum")
class IslesTrainerCurriculum(IslesTrainer):
    """IslesTrainer with size-based curriculum on the training loader."""

    isles_curriculum_warmup_epochs: int = 150
    isles_curriculum_floor_percentile: float = 90.0

    def _build_case_volumes(self, indices: list[str]) -> dict[str, float]:
        sessions_tsv = _resolve_sessions_tsv()
        meta = load_session_metadata(sessions_tsv)
        out: dict[str, float] = {}
        for cid in indices:
            m = meta.get(cid)
            if m is None:
                continue
            vol = m.get("lesion_volume_ml", float("nan"))
            try:
                out[cid] = float(vol) if vol == vol else 0.0  # NaN → 0
            except (TypeError, ValueError):
                out[cid] = 0.0
        return out

    def on_train_epoch_start(self) -> None:  # type: ignore[override]
        super().on_train_epoch_start()
        # Tick the curriculum forward. The MultiThreadedAugmenter wraps the
        # data loader; we kept a reference to the underlying loader so we
        # can talk to it directly.
        loader = getattr(self, "_isles_train_dataloader", None)
        if loader is not None and hasattr(loader, "set_current_epoch"):
            loader.set_current_epoch(int(self.current_epoch))

    def get_dataloaders(self):  # type: ignore[override]
        if self.dataset_class is None:
            from nnunetv2.training.dataloading.utils import infer_dataset_class

            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        tr_indices = list(dataset_tr.identifiers)
        case_volumes = self._build_case_volumes(tr_indices)
        cls = type(self)
        if len(case_volumes) == 0:
            raise RuntimeError(
                f"Curriculum: no train case identifiers ({len(tr_indices)}) matched any "
                f"session_id in sessions.tsv - cannot build per-case volume map."
            )

        dl_tr = CurriculumDataLoader3D(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
            case_volumes_ml=case_volumes,
            warmup_epochs=cls.isles_curriculum_warmup_epochs,
            floor_percentile=cls.isles_curriculum_floor_percentile,
        )
        # Save loader reference so on_train_epoch_start can tick it.
        self._isles_train_dataloader = dl_tr

        dl_val = nnUNetDataLoader(  # type: ignore[misc]
            dataset_val,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )

        allowed_num_processes = get_allowed_n_proc_DA()  # type: ignore[misc]
        if allowed_num_processes == 0:
            return SingleThreadedAugmenter(dl_tr, None), SingleThreadedAugmenter(dl_val, None)
        mt_gen_train = NonDetMultiThreadedAugmenter(  # type: ignore[misc]
            data_loader=dl_tr,
            transform=None,
            num_processes=allowed_num_processes,
            num_cached=max(6, allowed_num_processes // 2),
            seeds=None,
            pin_memory=self.device.type == "cuda",
            wait_time=0.002,
        )
        mt_gen_val = NonDetMultiThreadedAugmenter(  # type: ignore[misc]
            data_loader=dl_val,
            transform=None,
            num_processes=max(1, allowed_num_processes // 2),
            num_cached=max(3, allowed_num_processes // 4),
            seeds=None,
            pin_memory=self.device.type == "cuda",
            wait_time=0.002,
        )
        return mt_gen_train, mt_gen_val
