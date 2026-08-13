"""IslesTrainerCohortBalanced - per-cohort uniform sampling on training.

Replaces the standard nnUNetDataLoader with `CohortBalancedDataLoader3D`,
which assigns per-case sampling probabilities `1 / (K * |cohort(case)|)`
so each of the 3 collapsed cohorts gets ~uniform mass in expectation over
many batches, regardless of absolute cohort size.

Cohort collapse rule:
    Training + Training_ATLAS2 -> Training
    Testing  + Testing_ATLAS2  -> Testing
    ATLAS3                     -> ATLAS3
    SOOP is folded into ATLAS3.

Validation loader stays vanilla (no balanced sampling) so val metrics
remain comparable to other ablations.

Cohort lookup uses `nnunet_isles.utils.session_metadata.load_session_metadata`
on `data_analysis/sessions.tsv`. Path discovery: tries the env-var
`ISLES_SESSIONS_TSV` first, then `<repo_root>/data_analysis/sessions.tsv`,
then a hard error.
"""

from __future__ import annotations

import os
from pathlib import Path

from nnunet_isles.dataloading.cohort_balanced_loader import CohortBalancedDataLoader3D
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
    """Locate `data_analysis/sessions.tsv` from env or repo root."""
    env = os.environ.get("ISLES_SESSIONS_TSV")
    if env:
        p = Path(env)
        if not p.exists():
            raise FileNotFoundError(f"ISLES_SESSIONS_TSV points to non-existent path: {p}")
        return p
    # Walk up from this file to find a `data_analysis/sessions.tsv`.
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "data_analysis" / "sessions.tsv"
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "Could not find sessions.tsv. Set $ISLES_SESSIONS_TSV or ensure "
        "data_analysis/sessions.tsv exists at the repo root."
    )


def _case_id_to_session_id(case_id: str) -> str:
    """Strip nnU-Net's per-case suffix to get the original session_id.

    nnU-Net's `data.identifiers` are typically of the form
    `<session_id>` for our datasets (we ship raw with names matching
    sessions.tsv); should the convention diverge, this hook is the
    one place to patch.
    """
    return case_id


@TRAINER_REGISTRY.register("IslesTrainerCohortBalanced")
class IslesTrainerCohortBalanced(IslesTrainer):
    """IslesTrainer with cohort-balanced training-side sampling."""

    # Class knob: optional explicit cohort weight override (e.g. {"Training": 1, "Testing": 1, "ATLAS3": 1}).
    # None ⇒ uniform across present cohorts.
    isles_cohort_weights: dict[str, float] | None = None

    def _build_case_to_cohort(self, indices: list[str]) -> dict[str, str]:
        sessions_tsv = _resolve_sessions_tsv()
        meta = load_session_metadata(sessions_tsv)
        out: dict[str, str] = {}
        for cid in indices:
            sid = _case_id_to_session_id(cid)
            if sid in meta:
                out[cid] = meta[sid]["cohort"]
        return out

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

        # Build cohort lookup for train indices.
        tr_indices = list(dataset_tr.identifiers)
        case_to_cohort = self._build_case_to_cohort(tr_indices)
        cls = type(self)

        if len(case_to_cohort) == 0:
            raise RuntimeError(
                f"CohortBalanced: no train case identifiers ({len(tr_indices)}) "
                f"matched any session_id in sessions.tsv - cannot build sampling weights."
            )

        dl_tr = CohortBalancedDataLoader3D(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,  # overwritten by the cohort logic in __init__
            pad_sides=None,
            transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
            case_to_cohort=case_to_cohort,
            cohort_weights=cls.isles_cohort_weights,
        )
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
