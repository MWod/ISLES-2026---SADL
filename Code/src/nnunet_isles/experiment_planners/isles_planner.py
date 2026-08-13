"""IslesPlanner - forces an isotropic target_spacing tuple from Hydra config.

This enforces locked decision #8: segmentation operates at a reasonable
isotropic physical resolution. nnU-Net's median-per-axis default (which can
be anisotropic) is rejected.

Other ExperimentPlanner decisions (patch size, batch size, network
topology, normalisation scheme selection) are left to nnU-Net.
"""

from __future__ import annotations

import numpy as np

from nnunet_isles.registry import PLANNER_REGISTRY

try:
    from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import (
        ExperimentPlanner,
    )
except ImportError:
    ExperimentPlanner = object  # type: ignore[assignment, misc]


@PLANNER_REGISTRY.register("IslesPlanner")
class IslesPlanner(ExperimentPlanner):  # type: ignore[misc, valid-type]
    """ExperimentPlanner with a fixed isotropic target_spacing.

    Configure by class attribute (set by the entrypoint script before
    instantiating). Defaults to 1.0 mm iso; overridden by Hydra config:

        IslesPlanner.isles_target_spacing = (0.8, 0.8, 0.8)
    """

    isles_target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    overwrite_plans_name: str = "nnUNetPlans_iso10"

    def determine_fullres_target_spacing(self) -> np.ndarray:  # type: ignore[override]
        return np.asarray(self.isles_target_spacing, dtype=float)
