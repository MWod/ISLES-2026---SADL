"""Tiny helper used by submit_sweep_template.sh to read experiment fields.

Reads a Hydra experiment overlay (e.g. `experiment/hpcv2_baseline`) and prints
shell-evaluable assignments for the fields the SLURM driver needs:

    ISLES_EXP_NAME=hpcv2_baseline
    ISLES_TRAINER_CLASS=nnUNetTrainer
    ISLES_PLANS_ID=nnUNetPlans_iso10
    ISLES_CONFIGURATION=3d_fullres
    ISLES_SPLIT_NAME=site_disjoint_test3

Usage:
    eval "$(python scripts/_resolve_experiment_config.py experiment/hpcv2_baseline)"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_CODE_DIR = _THIS.parents[1]
sys.path.insert(0, str(_CODE_DIR))

# Set safe defaults so importing scripts._autopath_resolver doesn't fail.
for _var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):  # noqa: SIM112
    os.environ.setdefault(_var, "/tmp")


def main() -> int:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <experiment_overlay_name>", file=sys.stderr)
        return 2
    overlay = sys.argv[1]
    # Accept both "experiment/foo" (path form) and "foo" (bare name).
    overlay_name = overlay[len("experiment/") :] if overlay.startswith("experiment/") else overlay

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from scripts._autopath_resolver import register_autopath_resolver

    register_autopath_resolver()
    GlobalHydra.instance().clear()
    config_dir = str((_CODE_DIR / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=[f"+experiment={overlay_name}"])

    fields = {
        "ISLES_EXP_NAME": str(cfg.experiment_name),
        "ISLES_TRAINER_CLASS": str(cfg.model.trainer_class),
        "ISLES_PLANS_ID": str(cfg.preprocessing.plans_identifier),
        "ISLES_CONFIGURATION": str(cfg.model.configuration),
        "ISLES_SPLIT_NAME": str(cfg.split.name),
        "ISLES_DATASET_ID": str(cfg.nnunet_dataset_id),
        "ISLES_DATASET_NAME": str(cfg.nnunet_dataset_name),
    }
    for k, v in fields.items():
        # POSIX shell-safe single-quote escape.
        safe = v.replace("'", "'\"'\"'")
        print(f"export {k}='{safe}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
