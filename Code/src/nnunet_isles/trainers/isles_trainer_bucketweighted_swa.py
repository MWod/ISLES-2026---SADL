"""IslesTrainerBucketWeightedSWA - bucket-weighted DC+TopK combined with SWA.

Composes per-sample volume-bucket-weighted DC+TopK loss (`IslesTrainerBucketWeighted`)
with stochastic weight averaging (`IslesTrainerSWA`) via Python MRO. Both parents
override disjoint hook sets:

  * `IslesTrainerBucketWeighted` overrides `train_step`, `validation_step`,
    `_call_loss`, `_collect_val_metrics` - all loss/metric-side hooks.
  * `IslesTrainerSWA` overrides `__init__`, `_maybe_restore_swa_cache`,
    `_save_swa_cache`, `on_train_epoch_end`, `on_train_end` - all weight-
    averaging hooks.

MRO with `class IslesTrainerBucketWeightedSWA(IslesTrainerBucketWeighted, IslesTrainerSWA)`
therefore picks up BucketWeighted's overrides for the loss path (from the
first parent) and SWA's overrides for the averaging path (from the second),
with `super()` chains that terminate at `IslesTrainer → nnUNetTrainer` cleanly.

The combo class body is intentionally empty - the composition IS the value.
"""

from __future__ import annotations

from nnunet_isles.registry import TRAINER_REGISTRY
from nnunet_isles.trainers.isles_trainer_bucket_weighted import IslesTrainerBucketWeighted
from nnunet_isles.trainers.isles_trainer_swa import IslesTrainerSWA


@TRAINER_REGISTRY.register("IslesTrainerBucketWeightedSWA")
class IslesTrainerBucketWeightedSWA(IslesTrainerBucketWeighted, IslesTrainerSWA):
    """Per-sample volume-bucket-weighted DC+TopK with SWA on top.

    * `train_step` / `validation_step` come from `IslesTrainerBucketWeighted`
      (first in MRO) - they compute per-sample volume weights and thread them
      through the loss.
    * `on_train_epoch_end` / `on_train_end` come from `IslesTrainerSWA`
      (second in MRO, but BucketWeighted doesn't override those) - they
      accumulate the SWA running mean and write `swa.pth` at end-of-training.
    * `__init__` comes from `IslesTrainerSWA` (only parent with a custom `__init__`).

    Paired at inference time with `finalize.py --use-swa` (loads `swa.pth`
    instead of `checkpoint_best.pth`).
    """


__all__ = ["IslesTrainerBucketWeightedSWA"]
