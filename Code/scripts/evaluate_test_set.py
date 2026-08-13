"""Test-set evaluation for a 5-fold-trained experiment.

After all 5 folds finish training, this script:
  1. Reads outer.test session ids from Splits/<split>/outer.json
  2. Stages the test T1w images to a temp dir as `<case>_0000.nii.gz`
  3. Loads the 5-fold ensemble via nnUNet's predict_from_files (mean softmax + mirror TTA)
  4. For each test case, computes Dice / HD95 / AVD / lesion-F1 / count-F1
  5. Aggregates per-site + per-bucket; writes test_ensemble.pdf + leaderboard_row.json
     under Evaluation_Results/<experiment_name>/.

Usage (called from sweep driver, after the 5 folds are done):
    python scripts/evaluate_test_set.py \\
        --experiment-name baseline_nnunet_v2_site_disjoint_test3_dice_ce \\
        --split-name site_disjoint_test3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))

# Env vars set BEFORE nnunetv2 imports.
import paths as _paths  # noqa: E402

for _var, _val in (
    ("nnUNet_raw", _paths.nnunet_raw),
    ("nnUNet_preprocessed", _paths.nnunet_preprocessed),
    ("nnUNet_results", _paths.nnunet_results),
):
    Path(_val).mkdir(parents=True, exist_ok=True)
    os.environ[_var] = str(_val)


def _read_outer_manifest(split_dir: Path) -> dict:
    outer_path = split_dir / "outer.json"
    if not outer_path.exists():
        raise FileNotFoundError(f"outer.json not found at {outer_path}")
    return json.loads(outer_path.read_text())


def _stage_test_inputs(test_session_ids: list[str], raw_images_dir: Path, staging_dir: Path) -> None:
    """Symlink test T1w images into a flat dir matching nnU-Net's expected
    `<case>_0000.nii.gz` convention. The nnUNet_raw layout has them already named
    that way under imagesTr/, so we just symlink across.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    for sid in test_session_ids:
        src = raw_images_dir / f"{sid}_0000.nii.gz"
        dst = staging_dir / f"{sid}_0000.nii.gz"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if not src.exists():
            raise FileNotFoundError(f"Test T1w image not found: {src}")
        os.symlink(src, dst)


def _load_gt_for_case(session_id: str, gt_dir: Path):
    """Load the binary GT mask for a case from nnUNet_preprocessed/gt_segmentations/."""
    import SimpleITK as sitk

    gt_path = gt_dir / f"{session_id}.nii.gz"
    if not gt_path.exists():
        raise FileNotFoundError(f"GT mask not found: {gt_path}")
    gt_img = sitk.ReadImage(str(gt_path))
    return sitk.GetArrayFromImage(gt_img), gt_img.GetSpacing()  # (x, y, z)


def _load_pred_for_case(session_id: str, pred_dir: Path):
    import SimpleITK as sitk

    pred_path = pred_dir / f"{session_id}.nii.gz"
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction not found: {pred_path}")
    img = sitk.ReadImage(str(pred_path))
    return sitk.GetArrayFromImage(img)


def _lookup_per_session_meta(session_id: str, sessions_df) -> tuple[str, str | None]:
    """Return (site, lesion_bucket) for a session_id from sessions.tsv."""
    row = sessions_df[sessions_df["session_id"] == session_id]
    if len(row) == 0:
        return "UNKNOWN", None
    site = str(row.iloc[0]["site"])
    bucket = row.iloc[0].get("lesion_bucket")
    return site, str(bucket) if bucket and bucket == bucket else None  # NaN check


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--dataset-name", default="Dataset501_AtlasV2")
    parser.add_argument("--trainer-class", default="nnUNetTrainer")
    parser.add_argument("--plans-identifier", default="nnUNetPlans_iso10")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--checkpoint-name", default="checkpoint_best.pth")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--no-tta", action="store_true", help="Disable sagittal-mirror TTA (default: enabled)."
    )
    parser.add_argument("--keep-predictions", action="store_true")
    args = parser.parse_args()

    import pandas as pd

    # Inputs.
    project_path = Path(_paths.project_path)
    splits_dir = Path(_paths.splits_path) / args.split_name
    outer = _read_outer_manifest(splits_dir)
    test_ids: list[str] = outer["test_session_ids"]
    print(f"[eval_test] split={args.split_name} | n_test={len(test_ids)}")

    raw_images_dir = Path(_paths.nnunet_raw) / args.dataset_name / "imagesTr"
    gt_dir = Path(_paths.nnunet_preprocessed) / args.dataset_name / "gt_segmentations"

    sessions_tsv = project_path / "data_analysis" / "sessions.tsv"
    if sessions_tsv.exists():
        sessions_df = pd.read_csv(sessions_tsv, sep="\t")
    else:
        raise FileNotFoundError(
            f"Expected {sessions_tsv} (per-session cohort/site table). "
            "Regenerate it with your EDA pipeline or provide the file manually; "
            "the file must contain at least the columns `session_id`, `site`, "
            "and `lesion_bucket`."
        )

    # nnU-Net model folder.
    model_folder = (
        Path(_paths.nnunet_results)
        / args.dataset_name
        / f"{args.trainer_class}__{args.plans_identifier}__{args.configuration}"
    )
    if not model_folder.is_dir():
        print(f"[eval_test] FATAL: model folder not found: {model_folder}", file=sys.stderr)
        return 2

    # Check that all required fold checkpoints exist.
    missing = [f for f in args.folds if not (model_folder / f"fold_{f}" / args.checkpoint_name).exists()]
    if missing:
        print(
            f"[eval_test] FATAL: missing fold checkpoints {missing} under {model_folder}",
            file=sys.stderr,
        )
        return 2

    eval_root = Path(_paths.evaluation_results_path) / args.experiment_name
    eval_root.mkdir(parents=True, exist_ok=True)
    pred_dir = eval_root / "test_predictions"

    # Stage inputs + run prediction.
    with tempfile.TemporaryDirectory(prefix="isles_test_inputs_") as tmp_inputs:
        tmp_inputs_path = Path(tmp_inputs)
        _stage_test_inputs(test_ids, raw_images_dir, tmp_inputs_path)
        print(f"[eval_test] staged {len(test_ids)} test inputs at {tmp_inputs_path}")

        from nnunet_isles.inference.predictor import IslesPredictor

        predictor = IslesPredictor(
            use_mirroring=not args.no_tta,
            allowed_mirroring_axes=(0,),  # sagittal only
            tile_step_size=0.5,
            use_gaussian=True,
            perform_everything_on_device=True,
            device="cuda",
            verbose=False,
        )
        predictor.initialize_from_model_folder(
            model_folder,
            use_folds=tuple(args.folds),
            checkpoint_name=args.checkpoint_name,
        )
        if pred_dir.exists():
            shutil.rmtree(pred_dir)
        pred_dir.mkdir(parents=True, exist_ok=True)
        print(f"[eval_test] running ensemble prediction → {pred_dir}")
        predictor.predict_folder(
            tmp_inputs_path,
            pred_dir,
            save_probabilities=False,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
        )
    print("[eval_test] prediction done")

    # Compute metrics per case.
    from nnunet_isles.evaluation.aggregator import aggregate_ensemble
    from nnunet_isles.evaluation.case_runner import compute_case_metrics
    from nnunet_isles.evaluation.reports import build_ensemble_pdf
    from nnunet_isles.utils import current_git_sha, hash_omegaconf

    rows = []
    for sid in test_ids:
        try:
            gt, spacing_xyz = _load_gt_for_case(sid, gt_dir)
            pred = _load_pred_for_case(sid, pred_dir)
        except FileNotFoundError as e:
            print(f"[eval_test] WARN: skipping {sid}: {e}", file=sys.stderr)
            continue
        # SITK spacing is (x, y, z); our metric helpers expect (sz, sy, sx).
        sx, sy, sz = spacing_xyz
        spacing_zyx = (float(sz), float(sy), float(sx))
        site, bucket = _lookup_per_session_meta(sid, sessions_df)
        metrics = compute_case_metrics(
            pred=pred,
            gt=gt,
            spacing_mm=spacing_zyx,
            session_id=sid,
            site=site,
            lesion_bucket=bucket,
        )
        rows.append(metrics.to_row())

    per_case = pd.DataFrame(rows)
    per_case_path = eval_root / "test_per_case.tsv"
    per_case.to_csv(per_case_path, sep="\t", index=False)
    print(f"[eval_test] wrote per-case metrics → {per_case_path}")

    summary = aggregate_ensemble(per_case)

    # PDF report.
    header = {
        "experiment_name": args.experiment_name,
        "split_name": args.split_name,
        "trainer_class": args.trainer_class,
        "plans_identifier": args.plans_identifier,
        "configuration": args.configuration,
        "checkpoint_name": args.checkpoint_name,
        "folds": list(args.folds),
        "tta": "sagittal_mirror" if not args.no_tta else "none",
        "n_test_cases": len(per_case),
        "git_sha": current_git_sha(project_path),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    pdf_path = eval_root / "test_ensemble.pdf"
    build_ensemble_pdf(pdf_path, header, per_case, summary)
    print(f"[eval_test] wrote PDF → {pdf_path}")

    # leaderboard_row.json
    def _safe_mean(key: str) -> float:
        s = summary.get(key)
        if isinstance(s, dict) and "mean" in s:
            return float(s["mean"])
        return float("nan")

    row = {
        "schema_version": "1.0",
        "experiment_name": args.experiment_name,
        "git_sha": header["git_sha"],
        "config_hash": hash_omegaconf({"experiment_name": args.experiment_name, "split": args.split_name}),
        "split_name": args.split_name,
        "ensemble_dice": _safe_mean("dice"),
        "ensemble_lesion_f1": _safe_mean("lesion_f1"),
        "ensemble_hd95": _safe_mean("hd95"),
        "ensemble_avd": _safe_mean("avd_ml"),
        "ensemble_lesion_count_f1": _safe_mean("count_f1"),
        "dice_cv_mean": float("nan"),  # populated by finalize.py once CV summary is wired
        "dice_cv_std": float("nan"),
        "per_site_dice": summary.get("per_site_dice", {}),
        "per_bucket_dice": summary.get("per_bucket_dice", {}),
        "n_train": int(outer.get("n_train", 0)),
        "n_test": len(per_case),
        "trainer_class": args.trainer_class,
        "plans_identifier": args.plans_identifier,
        "configuration": args.configuration,
        "checkpoint_name": args.checkpoint_name,
        "folds": list(args.folds),
        "timestamp": header["timestamp"],
    }
    leaderboard_row_path = eval_root / "leaderboard_row.json"
    leaderboard_row_path.write_text(json.dumps(row, indent=2))
    print(f"[eval_test] wrote leaderboard row → {leaderboard_row_path}")

    # Clean up predictions unless --keep-predictions.
    if not args.keep_predictions:
        shutil.rmtree(pred_dir, ignore_errors=True)
        print(f"[eval_test] cleaned predictions ({pred_dir})")

    print("[eval_test] done")
    print(f"  Mean ensemble Dice:       {row['ensemble_dice']:.4f}")
    print(f"  Mean ensemble Lesion F1:  {row['ensemble_lesion_f1']:.4f}")
    print(f"  Mean ensemble HD95:       {row['ensemble_hd95']:.4f}")
    print(f"  Mean ensemble AVD (mL):   {row['ensemble_avd']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
