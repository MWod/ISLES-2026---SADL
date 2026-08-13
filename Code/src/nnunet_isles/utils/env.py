"""Set the nnU-Net env vars from a Hydra paths config.

nnU-Net v2 reads nnUNet_raw, nnUNet_preprocessed, nnUNet_results from the
process environment at import time. Call this once at the top of every
entrypoint script BEFORE importing nnunetv2.
"""

from __future__ import annotations

import os
from pathlib import Path


def set_nnunet_env_vars(
    nnunet_raw: str | Path,
    nnunet_preprocessed: str | Path,
    nnunet_results: str | Path,
) -> None:
    """Populate nnU-Net env vars and ensure the directories exist."""
    for var_name, value in (
        ("nnUNet_raw", nnunet_raw),
        ("nnUNet_preprocessed", nnunet_preprocessed),
        ("nnUNet_results", nnunet_results),
    ):
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[var_name] = str(path)
