"""Trainer subclasses. Importing this module registers all IslesTrainer variants."""

from nnunet_isles.trainers.hooks import GradNormHook, TensorboardHook, ThroughputHook
from nnunet_isles.trainers.isles_trainer import IslesTrainer
from nnunet_isles.trainers.isles_trainer_bucket_weighted import IslesTrainerBucketWeighted
from nnunet_isles.trainers.isles_trainer_bucketweighted_swa import IslesTrainerBucketWeightedSWA
from nnunet_isles.trainers.isles_trainer_cohort_balanced import IslesTrainerCohortBalanced
from nnunet_isles.trainers.isles_trainer_cohort_moe import IslesTrainerCohortMoE
from nnunet_isles.trainers.isles_trainer_curriculum import IslesTrainerCurriculum
from nnunet_isles.trainers.isles_trainer_resenc import IslesTrainerResEnc
from nnunet_isles.trainers.isles_trainer_swa import IslesTrainerSWA

__all__ = [
    "GradNormHook",
    "IslesTrainer",
    "IslesTrainerBucketWeighted",
    "IslesTrainerBucketWeightedSWA",
    "IslesTrainerCohortBalanced",
    "IslesTrainerCohortMoE",
    "IslesTrainerCurriculum",
    "IslesTrainerResEnc",
    "IslesTrainerSWA",
    "TensorboardHook",
    "ThroughputHook",
]
