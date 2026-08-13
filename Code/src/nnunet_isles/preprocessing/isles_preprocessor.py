"""IslesPreprocessor - DefaultPreprocessor subclass injecting N4 + harmonisation.

Plug-in point for image harmonisation at preprocessing time. Defaults are
no-ops, so this passes through to upstream nnU-Net behaviour unless the
Hydra config enables N4 / harmonisation.

The actual constructor signature is fixed by nnU-Net (no kwargs accepted);
we configure the subclass via class-level attributes set by a factory.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nnunet_isles.preprocessing.harmonization import harmonize
from nnunet_isles.preprocessing.n4_bias import run_n4_bias_correction
from nnunet_isles.registry import PREPROCESSOR_REGISTRY

try:
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
except ImportError:  # nnU-Net not installed (e.g. EDA-only environment)
    DefaultPreprocessor = object  # type: ignore[assignment, misc]


@PREPROCESSOR_REGISTRY.register("IslesPreprocessor")
class IslesPreprocessor(DefaultPreprocessor):  # type: ignore[misc, valid-type]
    """nnU-Net DefaultPreprocessor with optional N4 + harmonization hooks.

    Configure by setting class attributes before nnU-Net calls run_case_npy:
        IslesPreprocessor.n4_enabled = True
        IslesPreprocessor.harmonization_name = "white_stripe"
    """

    n4_enabled: bool = False
    harmonization_name: str = "none"
    harmonizer_kwargs: dict[str, Any] = {}

    def _normalize(
        self,
        data: np.ndarray,
        seg: np.ndarray,
        configuration_manager: Any,
        foreground_intensity_properties_per_channel: Any,
    ) -> np.ndarray:  # type: ignore[override]
        # Apply our hooks BEFORE nnU-Net's z-scoring.
        if self.n4_enabled or self.harmonization_name != "none":
            for c in range(data.shape[0]):
                channel = data[c]
                if self.n4_enabled:
                    channel = run_n4_bias_correction(channel)
                if self.harmonization_name != "none":
                    channel = harmonize(self.harmonization_name, channel, **self.harmonizer_kwargs)
                data[c] = channel.astype(data.dtype, copy=False)
        return super()._normalize(
            data, seg, configuration_manager, foreground_intensity_properties_per_channel
        )
