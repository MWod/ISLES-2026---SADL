"""nnU-Net plan_and_preprocess driver for ISLES 2026.

Three sub-steps invoked from this script:
  1. extract_fingerprints - scan dataset and write dataset_fingerprint.json
  2. plan_experiments    - write nnUNetPlans_iso10.json (with our isotropic target)
  3. preprocess          - resample + cache all 3d_fullres patches under nnUNet_preprocessed/

Usage:
    python scripts/preprocess_dataset.py --step all
    python scripts/preprocess_dataset.py --step fingerprint
    python scripts/preprocess_dataset.py --step plan
    python scripts/preprocess_dataset.py --step preprocess --num-processes 8
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))

# nnunetv2.paths reads these env vars at module-import time, so we must set
# them BEFORE any nnunet_isles or nnunetv2 imports (the registry imports in
# nnunet_isles/__init__.py transitively pull nnunetv2).
import paths as _paths  # noqa: E402

for _var, _val in (
    ("nnUNet_raw", _paths.nnunet_raw),
    ("nnUNet_preprocessed", _paths.nnunet_preprocessed),
    ("nnUNet_results", _paths.nnunet_results),
):
    Path(_val).mkdir(parents=True, exist_ok=True)
    os.environ[_var] = str(_val)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        choices=["fingerprint", "plan", "preprocess", "all"],
        default="all",
    )
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--target-spacing", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    parser.add_argument("--plans-name", default="nnUNetPlans_iso10")
    parser.add_argument("--preprocessor-class", default="DefaultPreprocessor")
    parser.add_argument(
        "--planner-class",
        default="ExperimentPlanner",
        help="Use 'ExperimentPlanner' for the baseline (target spacing is enforced via "
        "overwrite_target_spacing). Switch to 'IslesPlanner' once we need extra hooks.",
    )
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--configurations", nargs="+", default=["3d_fullres"])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--n4-bias-correction",
        action="store_true",
        help="Enable N4 bias correction in IslesPreprocessor (only honored when --preprocessor-class=IslesPreprocessor).",
    )
    parser.add_argument(
        "--harmonization",
        default="none",
        choices=["none", "white_stripe", "iguane"],
        help="Intensity harmonizer applied by IslesPreprocessor before nnU-Net z-scoring.",
    )
    args = parser.parse_args()

    # Env vars set at module top (before any nnunetv2 import).
    # Names are mixed-case upstream - keep them; noqa for ruff SIM112.
    print(f"[env] nnUNet_raw          = {os.environ['nnUNet_raw']}")  # noqa: SIM112
    print(f"[env] nnUNet_preprocessed = {os.environ['nnUNet_preprocessed']}")  # noqa: SIM112
    print(f"[env] nnUNet_results      = {os.environ['nnUNet_results']}")  # noqa: SIM112

    from nnunetv2.experiment_planning.plan_and_preprocess_api import (
        extract_fingerprints,
        plan_experiments,
        preprocess,
    )

    # If IslesPreprocessor is selected, set its class attrs so the N4/harmonization
    # hooks actually fire. nnU-Net's plan_and_preprocess instantiates the class
    # by name without kwargs, so this is the configuration channel.
    if args.preprocessor_class == "IslesPreprocessor":
        # Importing nnunet_isles populates the registry and (transitively) the
        # IslesPreprocessor module.
        import nnunet_isles  # noqa: F401
        from nnunet_isles.preprocessing.isles_preprocessor import IslesPreprocessor

        IslesPreprocessor.n4_enabled = bool(args.n4_bias_correction)
        IslesPreprocessor.harmonization_name = str(args.harmonization)
        print(
            f"[isles_preprocessor] n4_enabled={IslesPreprocessor.n4_enabled} "
            f"harmonization={IslesPreprocessor.harmonization_name}"
        )

    if args.step in ("fingerprint", "all"):
        print("=" * 70)
        print(f"[1/3] extract_fingerprints dataset_id={args.dataset_id}")
        print("=" * 70)
        extract_fingerprints(
            dataset_ids=[args.dataset_id],
            num_processes=args.num_processes,
        )

    if args.step in ("plan", "all"):
        print("=" * 70)
        print(
            f"[2/3] plan_experiments planner={args.planner_class} "
            f"target_spacing={args.target_spacing} plans_name={args.plans_name}"
        )
        print("=" * 70)
        plans_identifier = plan_experiments(
            dataset_ids=[args.dataset_id],
            experiment_planner_class_name=args.planner_class,
            preprocess_class_name=args.preprocessor_class,
            overwrite_target_spacing=tuple(args.target_spacing),
            overwrite_plans_name=args.plans_name,
        )
        print(f"[plan] plans_identifier returned: {plans_identifier}")

    if args.step in ("preprocess", "all"):
        print("=" * 70)
        print(f"[3/3] preprocess configurations={args.configurations} num_processes={args.num_processes}")
        print("=" * 70)
        # `num_processes` must match the number of configurations (per-config worker count).
        per_config_workers = [args.num_processes] * len(args.configurations)
        preprocess(
            dataset_ids=[args.dataset_id],
            plans_identifier=args.plans_name,
            configurations=tuple(args.configurations),
            num_processes=per_config_workers,
            verbose=args.verbose,
        )

    print("[preprocess_dataset] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
