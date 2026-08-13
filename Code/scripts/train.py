"""Training entrypoint - invokes nnU-Net's run_training with Hydra-resolved config.

Sets env vars + class attributes BEFORE importing nnunetv2 (paths.py reads env at
import-time). For the baseline experiment we use the upstream nnUNetTrainer.

To apply our custom epoch/iteration counts we cannot rely on class-attribute
mutation (nnUNetTrainer.__init__ overwrites those with hardcoded defaults).
Instead we call get_trainer_from_args ourselves, mutate the returned instance,
then drive the same code paths run_training would.

Usage:
    python scripts/train.py --config-name config fold=0
    python scripts/train.py --config-name experiment/baseline_nnunet_random_test10 fold=0
    python scripts/train.py --config-name experiment/baseline_nnunet_random_test10 fold=0 trainer=short
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))

# Set env vars BEFORE any nnunet_isles or nnunetv2 imports.
import paths as _paths  # noqa: E402

for _var, _val in (
    ("nnUNet_raw", _paths.nnunet_raw),
    ("nnUNet_preprocessed", _paths.nnunet_preprocessed),
    ("nnUNet_results", _paths.nnunet_results),
):
    Path(_val).mkdir(parents=True, exist_ok=True)
    os.environ[_var] = str(_val)

from scripts._autopath_resolver import register_autopath_resolver  # noqa: E402

register_autopath_resolver()

import hydra  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> int:
    print("=" * 70)
    print("ISLES 2026 - training")
    print("=" * 70)
    print(OmegaConf.to_yaml(cfg, resolve=True))

    from nnunet_isles.utils import set_seed

    set_seed(int(cfg.seed))

    # Read trainer class from cfg.model.trainer_class. For the baseline this is
    # upstream `nnUNetTrainer`; for the DA5 ablation it's `nnUNetTrainerDA5`.
    # Our IslesTrainer is registered in nnunet_isles.trainers but is not yet
    # discoverable by nnU-Net's recursive_find_python_class.
    trainer_class_name = str(cfg.model.trainer_class)
    dataset_id = int(cfg.nnunet_dataset_id)
    plans_identifier = str(cfg.preprocessing.plans_identifier)
    configuration = str(cfg.model.configuration)
    fold = int(cfg.fold)

    # Bail early if the preprocessed cache or the plans file don't exist -
    # avoids a confusing nnU-Net stack trace 10 seconds into the run.
    nnunet_preprocessed_dir = Path(os.environ["nnUNet_preprocessed"]) / cfg.nnunet_dataset_name  # noqa: SIM112
    plans_file = nnunet_preprocessed_dir / f"{plans_identifier}.json"
    if not plans_file.exists():
        print(f"[train] FATAL: plans file not found at {plans_file}", file=sys.stderr)
        print("[train] run: python scripts/preprocess_dataset.py --step all", file=sys.stderr)
        return 2
    splits_file = nnunet_preprocessed_dir / "splits_final.json"
    if not splits_file.exists():
        print(f"[train] FATAL: splits file not found at {splits_file}", file=sys.stderr)
        print(f"[train] run: python scripts/generate_splits.py split={cfg.split.name}", file=sys.stderr)
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If the trainer class is one of ours (registered in nnunet_isles.trainers),
    # propagate the Hydra-config knobs onto the class BEFORE instantiation, then
    # build the instance via the same plans/dataset_json plumbing as upstream.
    from nnunet_isles.registry import TRAINER_REGISTRY
    from nnunet_isles.trainers.isles_trainer import IslesTrainer
    from nnunetv2.run.run_training import get_trainer_from_args, maybe_load_checkpoint
    from torch.backends import cudnn

    # Loss extension.
    loss_cfg = cfg.get("loss")
    if loss_cfg is not None:
        loss_type = loss_cfg.get("type", "dice_ce")
        loss_kwargs = {
            k: v
            for k, v in OmegaConf.to_container(loss_cfg, resolve=True).items()
            if k not in {"name", "type"}
        }
        IslesTrainer.isles_loss_key = str(loss_type)
        IslesTrainer.isles_loss_kwargs = loss_kwargs
    # Augmentation extension.
    aug_cfg = cfg.get("augmentation")
    if aug_cfg is not None:
        aug_dict = OmegaConf.to_container(aug_cfg, resolve=True)
        gin = aug_dict.get("gin") or {}
        if isinstance(gin, dict) and gin.get("enabled", False):
            IslesTrainer.isles_gin_kwargs = {k: v for k, v in gin.items() if k != "enabled"} | {
                "enabled": True
            }
        else:
            IslesTrainer.isles_gin_kwargs = None
        cmx = aug_dict.get("carvemix") or {}
        if isinstance(cmx, dict) and cmx.get("enabled", False):
            IslesTrainer.isles_carvemix_kwargs = {k: v for k, v in cmx.items() if k != "enabled"} | {
                "enabled": True
            }
        else:
            IslesTrainer.isles_carvemix_kwargs = None
        hs = aug_dict.get("hemiswap") or {}
        if isinstance(hs, dict) and hs.get("enabled", False):
            IslesTrainer.isles_hemiswap_kwargs = {k: v for k, v in hs.items() if k != "enabled"} | {
                "enabled": True
            }
        else:
            IslesTrainer.isles_hemiswap_kwargs = None
        dlp = aug_dict.get("diffusion_lesion") or {}
        if isinstance(dlp, dict) and dlp.get("enabled", False):
            IslesTrainer.isles_diffusion_lesion_kwargs = {k: v for k, v in dlp.items() if k != "enabled"} | {
                "enabled": True
            }
        else:
            IslesTrainer.isles_diffusion_lesion_kwargs = None

    use_custom = trainer_class_name in TRAINER_REGISTRY._registry  # type: ignore[attr-defined]
    if use_custom:
        trainer_cls = TRAINER_REGISTRY.get(trainer_class_name)
        from batchgenerators.utilities.file_and_folder_operations import join as bgjoin
        from batchgenerators.utilities.file_and_folder_operations import load_json
        from nnunetv2.paths import nnUNet_preprocessed
        from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

        ds_folder = bgjoin(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_id))
        plans_dict = load_json(bgjoin(ds_folder, plans_identifier + ".json"))
        plans_dict["continue_training"] = False
        ds_json = load_json(bgjoin(ds_folder, "dataset.json"))
        nnunet_trainer = trainer_cls(
            plans=plans_dict, configuration=configuration, fold=fold, dataset_json=ds_json, device=device
        )
    else:
        nnunet_trainer = get_trainer_from_args(
            str(dataset_id),
            configuration,
            fold,
            trainer_class_name,
            plans_identifier,
            continue_training=False,
            device=device,
        )

    # Re-root the trainer's output folder under the EXPERIMENT NAME so multiple
    # experiments that share the same (trainer_class, plans, configuration)
    # triple don't trample each other's checkpoints.
    #
    # Upstream constructs:
    #   nnUNet_results/<dataset>/<TrainerClass>__<plans>__<config>/fold_<N>/
    # which collides across e.g. focal_tversky, dice_topk, gin, carvemix -
    # all five use IslesTrainer + nnUNetPlans_iso10 + 3d_fullres.
    # Our scheme:
    #   nnUNet_results/<dataset>/<experiment_name>/fold_<N>/
    #
    # nnUNetTrainer.__init__ already (a) mkdir'd the OLD folder, (b) bound
    # self.log_file there, (c) built MetaLogger there, (d) wrote a citation
    # to the OLD log file. We have to redo all of that: move the partial log
    # file across, retarget the logger, then delete the empty old folder.
    import shutil  # noqa: PLC0415

    experiment_name_for_path = str(cfg.experiment_name)
    new_output_base = (
        Path(os.environ["nnUNet_results"])  # noqa: SIM112
        / str(cfg.nnunet_dataset_name)
        / experiment_name_for_path
    )
    old_output_folder = Path(nnunet_trainer.output_folder)
    old_output_folder_base = Path(nnunet_trainer.output_folder_base)
    new_output_folder = new_output_base / f"fold_{fold}"
    new_output_folder.mkdir(parents=True, exist_ok=True)

    # 1) Move the partial log file (if any). nnUNetTrainer.__init__ wrote a
    #    "thank you for citing" line into it, so it almost certainly exists.
    old_log_file = Path(nnunet_trainer.log_file) if nnunet_trainer.log_file else None
    if old_log_file is not None and old_log_file.exists():
        new_log_file = new_output_folder / old_log_file.name
        shutil.move(str(old_log_file), str(new_log_file))
        nnunet_trainer.log_file = str(new_log_file)

    # 2) Retarget the MetaLogger (used for progress.png, my_fantastic_logging,
    #    optional WandB sink, etc.) at the new folder.
    if getattr(nnunet_trainer, "logger", None) is not None:
        nnunet_trainer.logger.output_folder = str(new_output_folder)

    # 3) Now flip the trainer's own path attributes - done last so the moves
    #    above see the original values.
    nnunet_trainer.output_folder_base = str(new_output_base)
    nnunet_trainer.output_folder = str(new_output_folder)

    # 4) Best-effort cleanup of the now-empty upstream-style folders. Only
    #    remove if empty - never clobber a real run.
    for stale in (old_output_folder, old_output_folder_base):
        try:
            if stale.exists() and stale.is_dir() and not any(stale.iterdir()):
                stale.rmdir()
        except OSError:
            pass

    print(f"[train] rerooted nnU-Net output folder → {new_output_folder} (was {old_output_folder})")

    # Apply Hydra-config epoch / iteration overrides AFTER nnU-Net's __init__
    # finished writing its own defaults to the instance.
    nnunet_trainer.num_epochs = int(cfg.trainer.num_epochs)
    nnunet_trainer.num_iterations_per_epoch = int(cfg.trainer.num_iterations_per_epoch)
    nnunet_trainer.num_val_iterations_per_epoch = int(cfg.trainer.num_val_iterations_per_epoch)
    nnunet_trainer.oversample_foreground_percent = float(cfg.trainer.oversample_foreground_percent)
    nnunet_trainer.enable_deep_supervision = bool(cfg.trainer.deep_supervision)

    # Bucket-weighted trainer knobs. `cfg.trainer.bucket` is an optional
    # sub-section that ONLY `IslesTrainerBucketWeighted` (and subclasses like
    # `IslesTrainerBucketWeightedSWA`) understands - the `hasattr` guard makes
    # this a no-op for other trainers. Follows the same instance-attr pattern as
    # `num_epochs` above.
    bucket_cfg = cfg.trainer.get("bucket") if hasattr(cfg.trainer, "get") else None
    if bucket_cfg is not None:
        for attr, key in (
            ("isles_bucket_target_ml", "target_ml"),
            ("isles_bucket_w_min", "w_min"),
            ("isles_bucket_w_max", "w_max"),
            ("isles_bucket_empty_weight", "empty_weight"),
        ):
            if hasattr(nnunet_trainer, attr) and key in bucket_cfg:
                setattr(nnunet_trainer, attr, float(bucket_cfg[key]))
                nnunet_trainer.print_to_log_file(
                    f"[isles] bucket.{key}={bucket_cfg[key]} → {type(nnunet_trainer).__name__}.{attr}"
                )
    if not bool(cfg.trainer.deep_supervision):
        nnunet_trainer.print_to_log_file(
            "[isles] deep supervision DISABLED via Hydra config (cfg.trainer.deep_supervision=false)."
        )

    # Propagate cfg.trainer.compile to nnU-Net's compile-toggle env var.
    # nnU-Net v2.7+ enables torch.compile by default; if triton isn't available
    # (e.g. on aarch64 without a triton wheel), this crashes at the first
    # train_step. _do_i_compile() reads `nnUNet_compile` env var on the trainer
    # instance, so setting it here BEFORE run_training() is sufficient.
    raw_compile = cfg.trainer.get("compile", False)
    if isinstance(raw_compile, str):
        do_compile = raw_compile.strip().lower() in ("true", "1", "t", "yes")
    else:
        do_compile = bool(raw_compile)
    os.environ["nnUNet_compile"] = "true" if do_compile else "false"  # noqa: SIM112
    nnunet_trainer.print_to_log_file(
        f"[isles] nnUNet_compile={os.environ['nnUNet_compile']} "  # noqa: SIM112
        f"(from cfg.trainer.compile={raw_compile!r})"
    )

    # Optional warm-start. `pretrained_weights` is null by default (identical to
    # training from scratch). When set (path or ${autopath:...}), nnU-Net's
    # load_pretrained_weights transfers shape-matching layers and skips the seg head;
    # multi-channel-pretrained stems must be collapsed to 1 channel first (adapt_pretrained_stem.py).
    pretrained_weights = cfg.get("pretrained_weights")
    pw_file = str(pretrained_weights) if pretrained_weights else None
    if pw_file:
        nnunet_trainer.print_to_log_file(f"[isles] warm-starting from pretrained weights: {pw_file}")
    maybe_load_checkpoint(
        nnunet_trainer, continue_training=False, validation_only=False, pretrained_weights_file=pw_file
    )

    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    print(
        f"[train] dataset={dataset_id} configuration={configuration} fold={fold} "
        f"trainer={trainer_class_name} plans={plans_identifier} "
        f"num_epochs={nnunet_trainer.num_epochs} iter/ep={nnunet_trainer.num_iterations_per_epoch}"
    )

    nnunet_trainer.run_training()
    nnunet_trainer.perform_actual_validation(save_probabilities=True)
    print("[train] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
