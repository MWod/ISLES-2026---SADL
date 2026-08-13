"""Tests for the bucket-weighted + SWA combo trainer.

The trainer body is empty by design - all behaviour comes from MRO
composition. What we verify here:

  1. TRAINER_REGISTRY carries the "IslesTrainerBucketWeightedSWA" key
     so `train.py` can look it up from the experiment YAML.
  2. Verify `__init__` signature matches upstream nnUNetTrainer's.
     nnU-Net's `my_init_kwargs` bookkeeping introspects
     `inspect.signature(self.__init__).parameters` and then looks up
     each name in `locals()`; using `*args/**kwargs` breaks that lookup.
  3. MRO order: BucketWeighted first (so `train_step` picks up the
     bucket-weight injection), SWA second (so `on_train_epoch_end` +
     `on_train_end` fire for weight averaging).
  4. The hooks each parent overrides are still present on the composed
     class (BucketWeighted's `train_step` and SWA's `on_train_end` in
     particular).
"""

from __future__ import annotations

import inspect


def test_combo_trainer_registered_in_registry():
    """train.py looks up `cls.model.trainer_class` in TRAINER_REGISTRY.
    Verify the combo class is there under the exact string the YAML uses."""
    from nnunet_isles.registry import TRAINER_REGISTRY

    assert "IslesTrainerBucketWeightedSWA" in TRAINER_REGISTRY._registry  # type: ignore[attr-defined]


def test_combo_trainer_mro_order_is_bucketweighted_then_swa():
    """`IslesTrainerBucketWeighted` MUST come first in MRO so its `train_step`
    (which computes per-sample volume weights) is the one Python resolves.
    `IslesTrainerSWA` second contributes `on_train_epoch_end` + `on_train_end`
    (which BucketWeighted doesn't override, so falls through cleanly)."""
    from nnunet_isles.trainers.isles_trainer_bucket_weighted import IslesTrainerBucketWeighted
    from nnunet_isles.trainers.isles_trainer_bucketweighted_swa import IslesTrainerBucketWeightedSWA
    from nnunet_isles.trainers.isles_trainer_swa import IslesTrainerSWA

    mro = IslesTrainerBucketWeightedSWA.__mro__
    names = [c.__name__ for c in mro]
    bw_idx = names.index("IslesTrainerBucketWeighted")
    swa_idx = names.index("IslesTrainerSWA")
    assert bw_idx < swa_idx, f"IslesTrainerBucketWeighted must precede IslesTrainerSWA in MRO; got {names}"
    # And these are the two direct parents (via `class Foo(A, B):`).
    assert IslesTrainerBucketWeightedSWA.__bases__ == (
        IslesTrainerBucketWeighted,
        IslesTrainerSWA,
    )


def test_combo_trainer_train_step_comes_from_bucketweighted():
    """MRO resolves `train_step` to BucketWeighted (the loss-side path). If
    someone accidentally overrides `train_step` in SWA in the future, this
    test surfaces the collision."""
    from nnunet_isles.trainers.isles_trainer_bucket_weighted import IslesTrainerBucketWeighted
    from nnunet_isles.trainers.isles_trainer_bucketweighted_swa import IslesTrainerBucketWeightedSWA

    # `IslesTrainerBucketWeightedSWA.train_step` should be the same function
    # object as `IslesTrainerBucketWeighted.train_step` (both classes resolve
    # to that unbound function via MRO).
    assert IslesTrainerBucketWeightedSWA.train_step is IslesTrainerBucketWeighted.train_step, (
        "train_step must resolve to IslesTrainerBucketWeighted's implementation"
    )


def test_combo_trainer_on_train_end_comes_from_swa():
    """`on_train_end` is where SWA writes swa.pth. BucketWeighted doesn't
    override it - must resolve to SWA's implementation for the SWA weights
    to actually land on disk."""
    from nnunet_isles.trainers.isles_trainer_bucketweighted_swa import IslesTrainerBucketWeightedSWA
    from nnunet_isles.trainers.isles_trainer_swa import IslesTrainerSWA

    assert IslesTrainerBucketWeightedSWA.on_train_end is IslesTrainerSWA.on_train_end, (
        "on_train_end must resolve to IslesTrainerSWA's implementation"
    )


def test_combo_trainer_on_train_epoch_end_comes_from_swa():
    """SWA's `on_train_epoch_end` is where per-epoch weight averaging happens
    (+ state-cache pickling). Must resolve via MRO to SWA."""
    from nnunet_isles.trainers.isles_trainer_bucketweighted_swa import IslesTrainerBucketWeightedSWA
    from nnunet_isles.trainers.isles_trainer_swa import IslesTrainerSWA

    assert IslesTrainerBucketWeightedSWA.on_train_epoch_end is IslesTrainerSWA.on_train_epoch_end


def test_combo_trainer_init_signature_matches_upstream():
    """nnUNetTrainer bookkeeping uses `inspect.signature(self.__init__).parameters`
    then `locals()[name]` to pickle `my_init_kwargs`. Using *args/**kwargs
    (or any signature not matching upstream's explicit `plans, configuration,
    fold, dataset_json, device`) leaves `my_init_kwargs` malformed and breaks
    checkpoint resume."""
    from nnunet_isles.trainers.isles_trainer_bucketweighted_swa import IslesTrainerBucketWeightedSWA

    sig = inspect.signature(IslesTrainerBucketWeightedSWA.__init__)
    param_names = [p for p in sig.parameters if p != "self"]
    assert param_names == ["plans", "configuration", "fold", "dataset_json", "device"], (
        f"__init__ signature must match upstream nnUNetTrainer.__init__ "
        f"(plans, configuration, fold, dataset_json, device); got {param_names}"
    )


def test_combo_trainer_swa_class_attrs_visible():
    """SWA-side class attributes (`isles_swa_start_epoch`, etc.) MUST be
    accessible on the composed class so `on_train_end` reads the right
    filename and `on_train_epoch_end` gates on the right epoch."""
    from nnunet_isles.trainers.isles_trainer_bucketweighted_swa import IslesTrainerBucketWeightedSWA

    assert hasattr(IslesTrainerBucketWeightedSWA, "isles_swa_start_epoch")
    assert hasattr(IslesTrainerBucketWeightedSWA, "isles_swa_filename")
    assert hasattr(IslesTrainerBucketWeightedSWA, "isles_swa_state_cache_filename")


def test_combo_trainer_bucketweighted_class_attrs_visible():
    """BucketWeighted-side knobs (`isles_bucket_target_ml`, etc.) must be
    accessible so the target_ml ablation can override them via the
    train.py Hydra plumbing (though this combo sweep uses the default 5.0)."""
    from nnunet_isles.trainers.isles_trainer_bucketweighted_swa import IslesTrainerBucketWeightedSWA

    assert hasattr(IslesTrainerBucketWeightedSWA, "isles_bucket_target_ml")
    assert hasattr(IslesTrainerBucketWeightedSWA, "isles_bucket_w_min")
    assert hasattr(IslesTrainerBucketWeightedSWA, "isles_bucket_w_max")
    assert hasattr(IslesTrainerBucketWeightedSWA, "isles_bucket_empty_weight")
