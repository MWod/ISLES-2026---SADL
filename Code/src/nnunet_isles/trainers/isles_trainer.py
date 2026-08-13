"""IslesTrainer - nnUNetTrainer subclass with config-driven hooks.

Minimal subclass to start. Custom loss / optimizer / scheduler / DS toggle
plug in via class attributes set from the Hydra config BEFORE nnU-Net
instantiates the trainer (nnU-Net's `nnUNetv2_train` interface doesn't accept
arbitrary kwargs).

The convention: scripts/train.py reads the Hydra config, sets the desired
class attributes on IslesTrainer, then invokes nnU-Net's training entrypoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nnunet_isles.registry import TRAINER_REGISTRY

try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
except ImportError:
    nnUNetTrainer = object  # type: ignore[assignment, misc]


@TRAINER_REGISTRY.register("IslesTrainer")
class IslesTrainer(nnUNetTrainer):  # type: ignore[misc, valid-type]
    """Default ISLES trainer. Subclass per-experiment to override individual hooks."""

    # Config knobs set by scripts/train.py before instantiation.
    isles_num_epochs: int | None = None
    isles_num_iterations_per_epoch: int | None = None
    isles_num_val_iterations_per_epoch: int | None = None
    isles_oversample_foreground_percent: float | None = None
    isles_enable_deep_supervision: bool | None = None
    isles_log_dir: str | None = None
    isles_loss_key: str | None = None  # LOSS_REGISTRY key
    isles_loss_kwargs: dict[str, Any] = {}
    # Augmentation extensions - train.py populates these from cfg.augmentation.
    # Each is None when disabled. When set, IslesTrainer's get_training_transforms
    # override inserts the transform near the front of the upstream pipeline.
    isles_gin_kwargs: dict[str, Any] | None = None
    isles_carvemix_kwargs: dict[str, Any] | None = None
    isles_hemiswap_kwargs: dict[str, Any] | None = None
    isles_diffusion_lesion_kwargs: dict[str, Any] | None = None

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: Any = None,
    ) -> None:
        # Signature MUST mirror upstream nnUNetTrainer.__init__ exactly. Upstream
        # captures init args via `inspect.signature(self.__init__).parameters`
        # and then `locals()[name]` - meaning every parameter name on the most-
        # derived class must exist as a local in the chained super().__init__
        # frame. Using *args/**kwargs here breaks that introspection with a
        # KeyError on 'args'.
        try:
            import torch  # local import: avoid hard dep when nnU-Net absent.

            if device is None:
                device = torch.device("cuda")
        except ImportError:
            pass
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )
        cls = type(self)
        if cls.isles_num_epochs is not None:
            self.num_epochs = cls.isles_num_epochs
        if cls.isles_num_iterations_per_epoch is not None:
            self.num_iterations_per_epoch = cls.isles_num_iterations_per_epoch
        if cls.isles_num_val_iterations_per_epoch is not None:
            self.num_val_iterations_per_epoch = cls.isles_num_val_iterations_per_epoch
        if cls.isles_oversample_foreground_percent is not None:
            self.oversample_foreground_percent = cls.isles_oversample_foreground_percent
        if cls.isles_enable_deep_supervision is not None:
            self.enable_deep_supervision = cls.isles_enable_deep_supervision

        self._isles_tb_hook = None
        self._isles_throughput_hook = None

    def initialise(self) -> None:  # type: ignore[override]
        super().initialise()
        cls = type(self)
        if cls.isles_log_dir is not None:
            from nnunet_isles.trainers.hooks import TensorboardHook, ThroughputHook

            log_dir = Path(cls.isles_log_dir) / f"fold_{self.fold}"
            self._isles_tb_hook = TensorboardHook(log_dir=log_dir, fold=int(self.fold))
            self._isles_throughput_hook = ThroughputHook()

    def _build_loss(self):  # type: ignore[override]
        # Upstream nnUNetTrainer._build_loss wraps the loss with
        # DeepSupervisionWrapper when self.enable_deep_supervision is True so it
        # can consume the network's list-of-tensors output (one per scale).
        # When we override to substitute a custom loss from the registry we
        # must reproduce that wrap, otherwise the raw loss receives a list and
        # crashes (`AttributeError: 'list' object has no attribute 'shape'`).
        cls = type(self)
        if cls.isles_loss_key is None:
            return super()._build_loss()

        from nnunet_isles.registry import LOSS_REGISTRY

        loss = LOSS_REGISTRY.build(cls.isles_loss_key, **cls.isles_loss_kwargs)

        if not self.enable_deep_supervision:
            return loss

        # Mirror upstream weighting: weight_i = 1 / 2**i (higher-res gets more
        # weight); drop the lowest-res output (weights[-1] = 0); normalise to
        # sum=1; then wrap.
        import numpy as np
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

        ds_scales = self._get_deep_supervision_scales()
        weights = np.array([1.0 / (2**i) for i in range(len(ds_scales))])
        if self.is_ddp and not self._do_i_compile():
            weights[-1] = 1e-6
        else:
            weights[-1] = 0.0
        weights = weights / weights.sum()
        return DeepSupervisionWrapper(loss, weights)

    @staticmethod
    def _build_isles_extras() -> list:
        """Build the optional GIN / CarveMix transforms based on class config."""
        extras: list = []
        if IslesTrainer.isles_gin_kwargs:
            from nnunet_isles.augmentation.gin import GINTransform

            extras.append(GINTransform(**IslesTrainer.isles_gin_kwargs))
        if IslesTrainer.isles_carvemix_kwargs:
            from nnunet_isles.augmentation.carvemix import CarveMixTransform

            extras.append(CarveMixTransform(**IslesTrainer.isles_carvemix_kwargs))
        if IslesTrainer.isles_hemiswap_kwargs:
            from nnunet_isles.augmentation.hemiswap import IslesHemiSwapTransform

            extras.append(IslesHemiSwapTransform(**IslesTrainer.isles_hemiswap_kwargs))
        if IslesTrainer.isles_diffusion_lesion_kwargs:
            from nnunet_isles.augmentation.diffusion_lesion_paste import DiffusionLesionPasteTransform

            extras.append(DiffusionLesionPasteTransform(**IslesTrainer.isles_diffusion_lesion_kwargs))
        return extras

    @staticmethod
    def get_training_transforms(  # type: ignore[override]
        patch_size,
        rotation_for_DA,
        deep_supervision_scales,
        mirror_axes,
        do_dummy_2d_data_aug,
        use_mask_for_norm=None,
        is_cascaded: bool = False,
        foreground_labels=None,
        regions=None,
        ignore_label=None,
    ):
        # nnUNetTrainer's get_training_transforms is a @staticmethod. We rebuild
        # the upstream transform pipeline via super() and then splice ours in
        # AFTER the spatial transform but BEFORE the intensity-aug stack, so
        # GIN/CarveMix see crop-resampled patches but feed downstream
        # intensity augmentation.
        base = nnUNetTrainer.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=use_mask_for_norm,
            is_cascaded=is_cascaded,
            foreground_labels=foreground_labels,
            regions=regions,
            ignore_label=ignore_label,
        )
        extras = IslesTrainer._build_isles_extras()
        if not extras:
            return base

        # ComposeTransforms exposes a `.transforms` list - splice our extras
        # right after the spatial transform (index 1 in the upstream order).
        try:
            insert_at = 1 if len(base.transforms) > 1 else 0
            base.transforms = base.transforms[:insert_at] + extras + base.transforms[insert_at:]
        except AttributeError:
            # If the upstream container layout changes, fall back to appending.
            base.transforms = list(base.transforms) + extras
        return base

    def on_train_epoch_start(self) -> None:  # type: ignore[override]
        super().on_train_epoch_start()
        if self._isles_throughput_hook is not None:
            self._isles_throughput_hook.epoch_start()

    def on_train_epoch_end(self, train_outputs):  # type: ignore[override]
        super().on_train_epoch_end(train_outputs)
        if self._isles_tb_hook is None:
            return
        scalars: dict[str, float] = {
            "train/loss_total": float(self.logger.my_fantastic_logging["train_losses"][-1])
            if hasattr(self, "logger") and self.logger.my_fantastic_logging.get("train_losses")
            else 0.0,
            "train/lr": float(self.optimizer.param_groups[0]["lr"]) if self.optimizer is not None else 0.0,
        }
        if self._isles_throughput_hook is not None:
            scalars.update(
                self._isles_throughput_hook.epoch_end(self.num_iterations_per_epoch, self.batch_size or 0)
            )
        self._isles_tb_hook.log_scalars(self.current_epoch, scalars)

    def on_validation_epoch_end(self, val_outputs):  # type: ignore[override]
        super().on_validation_epoch_end(val_outputs)
        if self._isles_tb_hook is None:
            return
        if hasattr(self, "logger"):
            mean_fg_dice = self.logger.my_fantastic_logging.get("mean_fg_dice", [])
            if mean_fg_dice:
                self._isles_tb_hook.log_scalars(
                    self.current_epoch, {"val/dice/lesion": float(mean_fg_dice[-1])}
                )

    def on_train_end(self) -> None:  # type: ignore[override]
        super().on_train_end()
        if self._isles_tb_hook is not None:
            self._isles_tb_hook.close()
