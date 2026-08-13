"""IslesTrainerResEnc - variant forcing the ResidualEncoderUNet architecture.

nnU-Net v2.7.0 exposes ResEnc via the dataset's plans file (a separate
`-pl nnUNetPlannerResEncM/L/XL` planner is invoked at planning time). This
trainer subclass exists so we can override behaviour beyond what plans encode.
"""

from __future__ import annotations

from nnunet_isles.registry import TRAINER_REGISTRY
from nnunet_isles.trainers.isles_trainer import IslesTrainer


@TRAINER_REGISTRY.register("IslesTrainerResEnc")
class IslesTrainerResEnc(IslesTrainer):
    """ResEnc variant placeholder. Architecture selection is driven by the plans file."""
