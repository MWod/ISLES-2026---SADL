"""Output-space ensemble CLI.

Composes per-case softmax NPZs across N experiments into a single
combined softmax + binarised NIfTI mask, then evaluates against the V2
holdout and writes a `leaderboard_row.json` matching `finalize.py`'s
schema (so `aggregate_leaderboard.py` ingests it unchanged).

Inputs (must exist on disk first):
  * Per-experiment ensemble softmax NPZs at
    `Evaluation_Results/<exp>/test_predictions_ensemble/<case>.npz`.
    Produced by `finalize.py --save-softmax` on each experiment.
  * Ground truth at `<nnUNet_preprocessed>/<dataset>/gt_segmentations/`.

Pipeline:
  1. Discover the case-id intersection across all experiment dirs.
  2. (optional) `--learn-weights`: coordinate-ascent on val Dice for per-experiment
     weights `w_i >= 0, sum w_i = 1`. Without the flag, weights are uniform
     (or read from `--weights w1 w2 ...`).
  3. For each test case: `combined = sum_i w_i * softmax_i`, write
     `<output>/test_predictions_ensemble/<case>.{npz, nii.gz}`.
  4. Re-use `finalize._compute_per_case` + `aggregate_ensemble` to get
     Dice / HD95 / Lesion-F1 / per-bucket; write `leaderboard_row.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parents[1] / "src"))


def _now() -> str:
    import datetime as _dt

    return _dt.datetime.utcnow().isoformat() + "Z"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="experiment names to ensemble (each must have test_predictions_ensemble/<case>.npz on disk)",
    )
    parser.add_argument(
        "--output-name",
        required=True,
        help="Evaluation_Results/<output_name>/ - destination for combined predictions + leaderboard row",
    )
    parser.add_argument(
        "--weights",
        nargs="*",
        type=float,
        default=None,
        help="Per-experiment weights (aligned with --experiments). Mutually exclusive with --learn-weights.",
    )
    parser.add_argument(
        "--learn-weights",
        action="store_true",
        help="Coordinate-ascent on val Dice to learn per-experiment weights.",
    )
    parser.add_argument("--split-name", default="v2_site_disjoint_test3")
    parser.add_argument("--dataset-name", default="Dataset510_AtlasV2_V2")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--learn-weights-steps", type=int, default=50)
    parser.add_argument("--learn-weights-lr", type=float, default=0.05)
    # Composite decision layer (leak-free winner from the OOF fusion sandbox:
    # mean + never_empty + cc10). Defaults reproduce the old plain-threshold behaviour.
    parser.add_argument(
        "--never-empty",
        action="store_true",
        help="never emit an all-zero mask - rescue the top prob-mass component "
        "(chronic-stroke GT is never empty; parameter-free domain rule)",
    )
    parser.add_argument(
        "--min-voxels",
        type=int,
        default=0,
        help="drop connected components smaller than this many voxels (cc filter; 0 = off)",
    )
    parser.add_argument("--rescue-min-prob", type=float, default=0.10)
    parser.add_argument(
        "--val-case-limit",
        type=int,
        default=None,
        help="Randomly subsample this many val cases before caching (fits large OOF pools in RAM). "
        "None (default) = use all cases.",
    )
    parser.add_argument(
        "--val-random-seed",
        type=int,
        default=42,
        help="Seed for the val-case subsample (reproducibility).",
    )
    parser.add_argument(
        "--val-subdir",
        default="val_softmax_oof",
        help="per-experiment subdir with LEAK-FREE out-of-fold val softmax "
        "(populate via gather_oof_val_softmax.py). Weights are learned here, applied to test.",
    )
    parser.add_argument(
        "--allow-test-weight-learning",
        action="store_true",
        help="fallback ONLY: learn weights on the holdout itself (LEAKY) when OOF val softmax is absent",
    )
    args = parser.parse_args()

    import numpy as np
    import paths as _paths
    import SimpleITK as sitk
    from finalize import _compute_per_case, _load_sessions_meta
    from nnunet_isles.evaluation.aggregator import aggregate_ensemble
    from nnunet_isles.evaluation.reports import build_ensemble_pdf
    from nnunet_isles.inference import fusion
    from nnunet_isles.inference.output_ensemble import (
        ensemble_one_case,
        find_common_cases,
        learn_weights_from_val,
    )
    from nnunet_isles.utils import current_git_sha, hash_omegaconf

    if args.weights and args.learn_weights:
        print("[a5] FATAL: --weights and --learn-weights are mutually exclusive", file=sys.stderr)
        return 2

    eval_root_base = Path(_paths.evaluation_results_path)
    exp_npz_dirs = [eval_root_base / e / "test_predictions_ensemble" for e in args.experiments]
    for p in exp_npz_dirs:
        if not p.is_dir():
            print(f"[a5] FATAL: missing softmax dir for ensemble input: {p}", file=sys.stderr)
            print(
                "[a5] hint: re-run finalize.py --save-softmax for that experiment first",
                file=sys.stderr,
            )
            return 2

    test_case_ids = find_common_cases(exp_npz_dirs)
    if not test_case_ids:
        print("[a5] FATAL: no case-ids in common across the experiment NPZ dirs", file=sys.stderr)
        return 2
    print(f"[a5] {len(test_case_ids)} cases in common across {len(args.experiments)} experiments")

    # Resolve per-experiment weights.
    if args.weights is not None:
        if len(args.weights) != len(args.experiments):
            print("[a5] FATAL: --weights must align 1:1 with --experiments", file=sys.stderr)
            return 2
        weights = list(args.weights)
    elif args.learn_weights:
        # LEAK-FREE: learn weights on out-of-fold val softmax (the training pool,
        # disjoint from the site-disjoint holdout), then APPLY to the test set.
        # The old code learned + scored on the holdout itself - an overfit the
        # red-team flagged. Populate <exp>/val_softmax_oof/ via gather_oof_val_softmax.py.
        gt_dir = Path(_paths.nnunet_preprocessed) / args.dataset_name / "gt_segmentations"
        val_npz_dirs = [eval_root_base / e / args.val_subdir for e in args.experiments]
        missing = [str(d) for d in val_npz_dirs if not d.is_dir()]
        if missing:
            if args.allow_test_weight_learning:
                print(
                    "[a5] WARNING: OOF val softmax missing - LEARNING WEIGHTS ON THE HOLDOUT "
                    "(leaky, via --allow-test-weight-learning). Missing: " + ", ".join(missing),
                    file=sys.stderr,
                )
                val_npz_dirs, val_case_ids = exp_npz_dirs, test_case_ids
            else:
                print(
                    "[a5] FATAL: leak-free weight learning needs OOF val softmax at "
                    f"<exp>/{args.val_subdir}/. Missing: {missing}",
                    file=sys.stderr,
                )
                print(
                    "[a5] fix: python Code/scripts/gather_oof_val_softmax.py --experiments "
                    + " ".join(args.experiments)
                    + f" --dataset-name {args.dataset_name}",
                    file=sys.stderr,
                )
                print(
                    "[a5] (or pass --allow-test-weight-learning to reproduce the old leaky behaviour)",
                    file=sys.stderr,
                )
                return 2
        else:
            val_case_ids = find_common_cases(val_npz_dirs)
            if not val_case_ids:
                print("[a5] FATAL: no common OOF cases across the val_softmax_oof dirs", file=sys.stderr)
                return 2
            print(
                f"[a5] learning weights leak-free on {len(val_case_ids)} OOF val cases "
                f"(disjoint from the {len(test_case_ids)} holdout cases)"
            )
        weights = learn_weights_from_val(
            val_npz_dirs,
            gt_dir,
            val_case_ids,
            n_steps=args.learn_weights_steps,
            lr=args.learn_weights_lr,
            val_case_limit=args.val_case_limit,
            val_random_seed=args.val_random_seed,
        )
        print(
            "[a5] learned weights: "
            + " ".join(f"{e}={w:.3f}" for e, w in zip(args.experiments, weights, strict=False))
        )
    else:
        weights = [1.0] * len(args.experiments)

    # --- Ensemble each test case + write NPZ + binarised NIfTI ---
    eval_root = eval_root_base / args.output_name
    pred_dir = eval_root / "test_predictions_ensemble"
    pred_dir.mkdir(parents=True, exist_ok=True)
    print(f"[a5] ensembling {len(test_case_ids)} cases → {pred_dir}", flush=True)

    import time as _time

    t_write0 = _time.time()
    for case_idx, sid in enumerate(test_case_ids):
        t_case = _time.time()
        npz_paths = [d / f"{sid}.npz" for d in exp_npz_dirs]
        combined = ensemble_one_case(npz_paths, weights=weights)
        # Persist combined softmax for downstream post-processing.
        np.savez_compressed(str(pred_dir / f"{sid}.npz"), probabilities=combined.astype(np.float32))
        # Binarise + clone geometry from the first available sibling NIfTI in any
        # of the input experiment dirs.
        fg_prob = combined[1]
        # Decision layer over the weighted-mean fg field. never_empty=False +
        # min_voxels=0 → identical to the old (fg_prob > threshold) binarisation.
        mask = fusion.apply_decision_layer(
            fg_prob,
            mode="mean",
            threshold=args.threshold,
            never_empty=args.never_empty,
            rescue_min_prob=args.rescue_min_prob,
            min_voxels=args.min_voxels,
        )
        ref_nifti: Path | None = None
        for d in exp_npz_dirs:
            cand = d / f"{sid}.nii.gz"
            if cand.exists():
                ref_nifti = cand
                break
        if ref_nifti is not None:
            ref_sitk = sitk.ReadImage(str(ref_nifti))
            out = sitk.GetImageFromArray(mask)
            out.CopyInformation(ref_sitk)
            sitk.WriteImage(out, str(pred_dir / f"{sid}.nii.gz"))
        else:
            # No NIfTI template available - write identity-affine NIfTI as last resort.
            sitk.WriteImage(sitk.GetImageFromArray(mask), str(pred_dir / f"{sid}.nii.gz"))
        # Progress ping every 10 cases (or on the last case). Each case is
        # ~15-25 s so 10-case ticks give ~2-4 min feedback windows.
        if (case_idx + 1) % 10 == 0 or (case_idx + 1) == len(test_case_ids):
            elapsed = _time.time() - t_write0
            rate = (case_idx + 1) / max(elapsed, 1e-6)
            eta = (len(test_case_ids) - (case_idx + 1)) / max(rate, 1e-6)
            print(
                f"[a5] wrote {case_idx + 1}/{len(test_case_ids)}  "
                f"last_dt={_time.time() - t_case:.1f}s  "
                f"elapsed={elapsed:.0f}s  eta≈{eta:.0f}s",
                flush=True,
            )

    # --- Metrics + leaderboard row (mirrors finalize._build_ensemble) ---
    splits_dir = Path(_paths.splits_path) / args.split_name
    outer = json.loads((splits_dir / "outer.json").read_text())
    sessions_df = _load_sessions_meta(Path(_paths.project_path))
    gt_dir = Path(_paths.nnunet_preprocessed) / args.dataset_name / "gt_segmentations"

    per_case = _compute_per_case(pred_dir, gt_dir, test_case_ids, sessions_df)
    per_case.to_csv(eval_root / "test_per_case.tsv", sep="\t", index=False)
    summary = aggregate_ensemble(per_case)

    header = {
        "experiment_name": args.output_name,
        "split_name": args.split_name,
        "trainer_class": f"OutputSpaceEnsemble({','.join(args.experiments)})",
        "plans_identifier": "n/a",
        "configuration": "n/a",
        "folds": [],
        "n_test_cases": len(test_case_ids),
        "n_train": int(outer.get("n_train", 0)),
        "git_sha": current_git_sha(Path(_paths.project_path)),
        "timestamp": _now(),
    }
    build_ensemble_pdf(eval_root / "test_ensemble.pdf", header, per_case, summary)
    print(f"[a5] ensemble PDF → {eval_root / 'test_ensemble.pdf'}")

    def _safe(key: str) -> float:
        s = summary.get(key)
        if isinstance(s, dict) and "mean" in s:
            return float(s["mean"])
        return float("nan")

    row = {
        "schema_version": "1.1",
        "dataset_version": "v2" if args.split_name.startswith("v2_") else "v1",
        "experiment_name": args.output_name,
        "git_sha": header["git_sha"],
        "config_hash": hash_omegaconf({"experiments": args.experiments, "weights": weights}),
        "split_name": args.split_name,
        "dataset_name": args.dataset_name,
        "ensemble_dice": _safe("dice"),
        "ensemble_lesion_f1": _safe("lesion_f1"),
        "ensemble_hd95": _safe("hd95"),
        "ensemble_avd": _safe("avd_ml"),
        "ensemble_lesion_count_f1": _safe("count_f1"),
        "dice_cv_mean": float("nan"),
        "dice_cv_std": float("nan"),
        "per_site_dice": summary.get("per_site_dice", {}),
        "per_bucket_dice": summary.get("per_bucket_dice", {}),
        "n_train": int(outer.get("n_train", 0)),
        "n_test": len(per_case),
        "trainer_class": header["trainer_class"],
        "plans_identifier": header["plans_identifier"],
        "configuration": header["configuration"],
        "checkpoint_name": "output_space_ensemble",
        "folds": [],
        "timestamp": header["timestamp"],
        # Output-space ensemble provenance.
        "ensemble_inputs": args.experiments,
        "ensemble_weights": weights,
        "ensemble_threshold": args.threshold,
    }
    (eval_root / "leaderboard_row.json").write_text(json.dumps(row, indent=2))
    print(f"[a5] wrote leaderboard row → {eval_root / 'leaderboard_row.json'}")
    print(
        f"[a5] ensemble_dice={row['ensemble_dice']:.4f} "
        f"hd95={row['ensemble_hd95']:.3f} "
        f"lesion_f1={row['ensemble_lesion_f1']:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
