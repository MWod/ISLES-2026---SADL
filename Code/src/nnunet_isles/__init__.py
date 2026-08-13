"""nnunet_isles - ISLES 2026 extensions on top of vendored nnU-Net v2.

Importing this module triggers component registration via the side-effect imports below.
"""

from nnunet_isles.registry import (
    AUGMENTATION_REGISTRY,
    HARMONIZER_REGISTRY,
    LOSS_REGISTRY,
    META_COND_REGISTRY,
    NETWORK_REGISTRY,
    PLANNER_REGISTRY,
    PREPROCESSOR_REGISTRY,
    SPLIT_REGISTRY,
    TRAINER_REGISTRY,
)

__version__ = "0.1.0"

# Side-effect imports to populate registries. Order matters: dependencies first.
from nnunet_isles import (
    augmentation,  # noqa: E402, F401
    dataloading,  # noqa: E402, F401
    experiment_planners,  # noqa: E402, F401
    losses,  # noqa: E402, F401
    preprocessing,  # noqa: E402, F401
    splits,  # noqa: E402, F401
    trainers,  # noqa: E402, F401
)

__all__ = [
    "AUGMENTATION_REGISTRY",
    "HARMONIZER_REGISTRY",
    "LOSS_REGISTRY",
    "META_COND_REGISTRY",
    "NETWORK_REGISTRY",
    "PLANNER_REGISTRY",
    "PREPROCESSOR_REGISTRY",
    "SPLIT_REGISTRY",
    "TRAINER_REGISTRY",
    "__version__",
]
