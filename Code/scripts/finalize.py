"""End-of-sweep finalize step - produces the full set of PDFs and the leaderboard row.

Runs three things in sequence after all 5 folds have trained:

  1. CV summary
     Reads each fold's `nnUNet_results/.../fold_N/validation/summary.json`
     (written automatically by nnU-Net's `perform_actual_validation`), aggregates
     mean ± std across folds, writes Evaluation_Results/<exp>/cv_summary.pdf.

  2. Per-fold test ensemble
     For each fold N (0..4), runs IslesPredictor with `use_folds=(N,)` on the
     held-out test set (read from Splits/<split>/outer.json), computes Dice /
     HD95 / AVD / lesion-F1 / count-F1 per case, writes
     Evaluation_Results/<exp>/test_per_fold/fold_N.pdf and the per-case TSV.

  3. Ensemble test
     Runs IslesPredictor with `use_folds=(0,1,2,3,4)` + sagittal-mirror TTA,
     writes Evaluation_Results/<exp>/test_ensemble.pdf and leaderboard_row.json,
     then refreshes leaderboard.csv.

Usage:
    python scripts/finalize.py \\
        --experiment-name baseline_nnunet_v2_site_disjoint_test3_dice_ce \\
        --split-name site_disjoint_test3 \\
        --trainer-class nnUNetTrainer \\
        --plans-identifier nnUNetPlans_iso10 \\
        --configuration 3d_fullres

Flags to skip portions (useful for iterative debugging):
    --skip-cv-summary
    --skip-per-fold
    --skip-ensemble
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
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


# ----------------------------- helpers -----------------------------


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_sessions_meta(project_path: Path):
    import pandas as pd

    tsv = project_path / "data_analysis" / "sessions.tsv"
    if not tsv.exists():
        raise FileNotFoundError(
            f"Expected {tsv} (per-session cohort/site table). "
            "Regenerate it with your EDA pipeline or provide the file manually; "
            "the file must contain at least the columns `session_id`, `site`, "
            "and `lesion_bucket`."
        )
    return pd.read_csv(tsv, sep="\t")


def _lookup_meta(session_id: str, sessions_df) -> tuple[str, str | None]:
    if len(sessions_df) == 0:
        return "UNKNOWN", None
    row = sessions_df[sessions_df["session_id"] == session_id]
    if len(row) == 0:
        return "UNKNOWN", None
    site = str(row.iloc[0]["site"])
    bucket = row.iloc[0].get("lesion_bucket")
    return site, str(bucket) if bucket and bucket == bucket else None  # NaN check


def _stage_test_inputs(test_ids: list[str], raw_images_dir: Path, staging_dir: Path) -> None:
    """Symlink every channel of each test case into the staging dir.

    Channel 0 (`<sid>_0000.nii.gz`) is mandatory; channels 1..3 are linked
    opportunistically (Dataset503 has 2 channels, Dataset504 has 3, etc.).
    Single-channel datasets (Dataset501, 502) simply have no channels 1+ on
    disk and the loop skips them.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    for sid in test_ids:
        src = raw_images_dir / f"{sid}_0000.nii.gz"
        dst = staging_dir / f"{sid}_0000.nii.gz"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if not src.exists():
            raise FileNotFoundError(f"Test T1w image not found: {src}")
        os.symlink(src, dst)
        for ch in (1, 2, 3):
            src_ch = raw_images_dir / f"{sid}_{ch:04d}.nii.gz"
            if src_ch.exists():
                dst_ch = staging_dir / f"{sid}_{ch:04d}.nii.gz"
                if dst_ch.exists() or dst_ch.is_symlink():
                    dst_ch.unlink()
                os.symlink(src_ch, dst_ch)


def _read_pred_gt(session_id: str, pred_dir: Path, gt_dir: Path):
    import SimpleITK as sitk

    pred_path = pred_dir / f"{session_id}.nii.gz"
    gt_path = gt_dir / f"{session_id}.nii.gz"
    if not pred_path.exists():
        raise FileNotFoundError(f"prediction not found: {pred_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"GT not found: {gt_path}")
    pred_img = sitk.ReadImage(str(pred_path))
    gt_img = sitk.ReadImage(str(gt_path))
    return (
        sitk.GetArrayFromImage(pred_img),
        sitk.GetArrayFromImage(gt_img),
        gt_img.GetSpacing(),  # (x, y, z)
    )


def _compute_per_case(pred_dir: Path, gt_dir: Path, test_ids: list[str], sessions_df):
    import pandas as pd
    from nnunet_isles.evaluation.case_runner import compute_case_metrics

    rows = []
    for sid in test_ids:
        try:
            pred, gt, spacing_xyz = _read_pred_gt(sid, pred_dir, gt_dir)
        except FileNotFoundError as e:
            print(f"[finalize] WARN: skipping {sid}: {e}", file=sys.stderr)
            continue
        sx, sy, sz = spacing_xyz
        site, bucket = _lookup_meta(sid, sessions_df)
        m = compute_case_metrics(
            pred=pred,
            gt=gt,
            spacing_mm=(float(sz), float(sy), float(sx)),
            session_id=sid,
            site=site,
            lesion_bucket=bucket,
        )
        rows.append(m.to_row())
    return pd.DataFrame(rows)


# ----------------------------- cv_summary -----------------------------


def _build_cv_summary(
    model_folder: Path, folds: list[int], sessions_df, eval_root: Path, header: dict
) -> dict:
    """Aggregate per-fold validation summaries → cv_summary.pdf + per-fold dicts."""
    import pandas as pd
    from nnunet_isles.evaluation.aggregator import aggregate_cv
    from nnunet_isles.evaluation.reports import build_cv_summary_pdf

    per_case_per_fold: list[pd.DataFrame] = []
    for f in folds:
        summary_json = model_folder / f"fold_{f}" / "validation" / "summary.json"
        if not summary_json.exists():
            print(
                f"[finalize] cv_summary: fold {f} summary.json missing at {summary_json} - skipping",
                file=sys.stderr,
            )
            per_case_per_fold.append(pd.DataFrame())
            continue
        payload = json.loads(summary_json.read_text())
        rows = []
        for case in payload.get("metric_per_case", []):
            metrics_1 = case.get("metrics", {}).get("1", {})
            ref_path = Path(case.get("reference_file", ""))
            sid = ref_path.stem.replace(".nii", "")
            site, bucket = _lookup_meta(sid, sessions_df)
            rows.append(
                {
                    "session_id": sid,
                    "site": site,
                    "lesion_bucket": bucket,
                    "dice": float(metrics_1.get("Dice", float("nan"))),
                    "iou": float(metrics_1.get("IoU", float("nan"))),
                    "n_pred": float(metrics_1.get("n_pred", float("nan"))),
                    "n_ref": float(metrics_1.get("n_ref", float("nan"))),
                }
            )
        per_case_per_fold.append(pd.DataFrame(rows))

    cv = aggregate_cv(per_case_per_fold)
    cv_header = {**header, "report_type": "cv_summary"}
    cv_pdf = eval_root / "cv_summary.pdf"
    build_cv_summary_pdf(cv_pdf, cv_header, per_case_per_fold, cv.get("cv_summary", {}))
    print(f"[finalize] cv_summary → {cv_pdf}")
    # Also persist the aggregate dict for downstream use (leaderboard row).
    (eval_root / "cv_summary.json").write_text(json.dumps(cv, indent=2, default=str))
    return cv


# ----------------------------- per-fold + ensemble test -----------------------------


def _run_inference(
    model_folder: Path,
    use_folds: tuple[int, ...],
    input_dir: Path,
    output_dir: Path,
    *,
    save_softmax: bool = False,
    checkpoint_name: str = "checkpoint_best.pth",
) -> None:
    """One call to nnUNetPredictor - used by both per-fold (use_folds=(N,)) and ensemble paths.

    `save_softmax=True` triggers nnU-Net's `save_probabilities` flag: per-case
    `<id>.npz` written alongside the discretized NIfTI. Required by downstream
    inference tooling (threshold tuning, output-space ensemble).
    """
    from nnunet_isles.inference.predictor import IslesPredictor

    predictor = IslesPredictor(
        use_mirroring=True,
        allowed_mirroring_axes=(0,),
        tile_step_size=0.5,
        use_gaussian=True,
        perform_everything_on_device=True,
        device="cuda",
        verbose=False,
    )
    predictor.initialize_from_model_folder(model_folder, use_folds=use_folds, checkpoint_name=checkpoint_name)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor.predict_folder(
        input_dir,
        output_dir,
        save_probabilities=save_softmax,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
    )


def _build_per_fold(
    model_folder: Path,
    folds: list[int],
    staged_inputs: Path,
    gt_dir: Path,
    test_ids: list[str],
    sessions_df,
    eval_root: Path,
    header: dict,
    *,
    save_softmax: bool = False,
    checkpoint_name: str = "checkpoint_best.pth",
) -> None:
    """For each fold, predict the test set with that fold alone + write a per-fold PDF."""
    from nnunet_isles.evaluation.aggregator import aggregate_ensemble
    from nnunet_isles.evaluation.reports import build_per_fold_pdf

    per_fold_dir = eval_root / "test_per_fold"
    per_fold_dir.mkdir(parents=True, exist_ok=True)
    pred_root = eval_root / "test_predictions_per_fold"

    for f in folds:
        print(f"[finalize] per-fold: fold {f} - predicting test set")
        pred_dir_f = pred_root / f"fold_{f}"
        _run_inference(
            model_folder,
            (f,),
            staged_inputs,
            pred_dir_f,
            save_softmax=save_softmax,
            checkpoint_name=checkpoint_name,
        )

        per_case = _compute_per_case(pred_dir_f, gt_dir, test_ids, sessions_df)
        (per_fold_dir / f"fold_{f}.tsv").write_text(
            "\n".join(
                [
                    "\t".join(per_case.columns),
                    *("\t".join(str(v) for v in row) for row in per_case.values),
                ]
            )
        )
        summary = aggregate_ensemble(per_case)
        pdf_path = per_fold_dir / f"fold_{f}.pdf"
        build_per_fold_pdf(pdf_path, f, header, per_case, summary)
        print(
            f"[finalize] per-fold: fold {f} → {pdf_path}  (Dice mean={summary.get('dice', {}).get('mean', 'nan')})"
        )

    # Clean per-fold predictions unless user wants to keep them; they're regeneratable.
    shutil.rmtree(pred_root, ignore_errors=True)


def _build_ensemble(
    model_folder: Path,
    folds: list[int],
    staged_inputs: Path,
    gt_dir: Path,
    test_ids: list[str],
    sessions_df,
    eval_root: Path,
    header: dict,
    cv: dict,
    outer: dict,
    args,
) -> dict:
    from nnunet_isles.evaluation.aggregator import aggregate_ensemble
    from nnunet_isles.evaluation.reports import build_ensemble_pdf
    from nnunet_isles.utils import hash_omegaconf

    pred_dir = eval_root / "test_predictions_ensemble"
    print(f"[finalize] ensemble: predicting test set with folds {folds}")
    save_softmax = getattr(args, "save_softmax", False)
    checkpoint_name = "swa.pth" if getattr(args, "use_swa", False) else "checkpoint_best.pth"
    _run_inference(
        model_folder,
        tuple(folds),
        staged_inputs,
        pred_dir,
        save_softmax=save_softmax,
        checkpoint_name=checkpoint_name,
    )

    per_case = _compute_per_case(pred_dir, gt_dir, test_ids, sessions_df)
    per_case_path = eval_root / "test_per_case.tsv"
    per_case.to_csv(per_case_path, sep="\t", index=False)
    print(f"[finalize] ensemble: per-case → {per_case_path}")

    summary = aggregate_ensemble(per_case)
    pdf_path = eval_root / "test_ensemble.pdf"
    build_ensemble_pdf(pdf_path, header, per_case, summary)
    print(f"[finalize] ensemble → {pdf_path}")

    # leaderboard_row.json
    def _safe(key: str) -> float:
        s = summary.get(key)
        if isinstance(s, dict) and "mean" in s:
            return float(s["mean"])
        return float("nan")

    cv_dice = cv.get("cv_summary", {}).get("dice", {}) if cv else {}
    # `dataset_version` discriminator helps aggregate_leaderboard.py sort V1 vs V2
    # rows; "v2" iff the split name carries the canonical V2 prefix.
    dataset_version = "v2" if args.split_name.startswith("v2_") else "v1"
    row = {
        "schema_version": "1.1",
        "dataset_version": dataset_version,
        "experiment_name": args.experiment_name,
        "git_sha": header["git_sha"],
        "config_hash": hash_omegaconf({"experiment_name": args.experiment_name, "split": args.split_name}),
        "split_name": args.split_name,
        "dataset_name": args.dataset_name,
        "ensemble_dice": _safe("dice"),
        "ensemble_lesion_f1": _safe("lesion_f1"),
        "ensemble_hd95": _safe("hd95"),
        "ensemble_avd": _safe("avd_ml"),
        "ensemble_lesion_count_f1": _safe("count_f1"),
        "dice_cv_mean": float(cv_dice.get("mean", float("nan"))) if cv_dice else float("nan"),
        "dice_cv_std": float(cv_dice.get("std", float("nan"))) if cv_dice else float("nan"),
        "per_site_dice": summary.get("per_site_dice", {}),
        "per_bucket_dice": summary.get("per_bucket_dice", {}),
        "n_train": int(outer.get("n_train", 0)),
        "n_test": len(per_case),
        "trainer_class": args.trainer_class,
        "plans_identifier": args.plans_identifier,
        "configuration": args.configuration,
        "checkpoint_name": "checkpoint_best.pth",
        "folds": list(folds),
        "timestamp": header["timestamp"],
    }
    (eval_root / "leaderboard_row.json").write_text(json.dumps(row, indent=2))
    print(f"[finalize] wrote leaderboard row → {eval_root / 'leaderboard_row.json'}")

    # `--save-softmax` implies retaining the ensemble prediction dir; otherwise
    # the per-case NPZs we just wrote get wiped before downstream tools
    # (threshold tuning, output-space ensemble) can read them.
    if not (args.keep_predictions or args.save_softmax):
        shutil.rmtree(pred_dir, ignore_errors=True)
    return row


# ----------------------------- main -----------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--dataset-name", default="Dataset501_AtlasV2")
    parser.add_argument("--trainer-class", default="nnUNetTrainer")
    parser.add_argument("--plans-identifier", default="nnUNetPlans_iso10")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--skip-cv-summary", action="store_true")
    parser.add_argument("--skip-per-fold", action="store_true")
    parser.add_argument("--skip-ensemble", action="store_true")
    parser.add_argument("--keep-predictions", action="store_true")
    # Softmax export, SWA checkpoint, and TTA-mode flags:
    parser.add_argument(
        "--save-softmax",
        action="store_true",
        help="Save per-case softmax NPZ alongside the discretized NIfTI "
        "(needed by threshold tuning and output-space ensembling).",
    )
    parser.add_argument(
        "--use-swa",
        action="store_true",
        help="Load swa.pth instead of checkpoint_best.pth.",
    )
    parser.add_argument(
        "--tta-mode",
        default="mirror_only",
        choices=("mirror_only", "mirror_rot", "mirror_rot_scale"),
        help="Test-time augmentation mode for the inference passes.",
    )
    args = parser.parse_args()

    from nnunet_isles.utils import current_git_sha

    project_path = Path(_paths.project_path)
    splits_dir = Path(_paths.splits_path) / args.split_name
    outer_path = splits_dir / "outer.json"
    if not outer_path.exists():
        print(f"[finalize] FATAL: outer manifest not found: {outer_path}", file=sys.stderr)
        return 2
    outer = json.loads(outer_path.read_text())
    test_ids: list[str] = outer["test_session_ids"]

    raw_images_dir = Path(_paths.nnunet_raw) / args.dataset_name / "imagesTr"
    gt_dir = Path(_paths.nnunet_preprocessed) / args.dataset_name / "gt_segmentations"
    # train.py reroots nnU-Net's output_folder_base to <dataset>/<experiment_name>/
    # so checkpoints from sibling experiments (same trainer/plans/config) don't
    # overwrite each other. Look up the model folder by experiment_name.
    model_folder = Path(_paths.nnunet_results) / args.dataset_name / args.experiment_name

    if not model_folder.is_dir():
        print(f"[finalize] FATAL: model folder not found: {model_folder}", file=sys.stderr)
        return 2

    # Verify all requested fold checkpoints exist.
    missing = [f for f in args.folds if not (model_folder / f"fold_{f}" / "checkpoint_best.pth").exists()]
    if missing:
        print(
            f"[finalize] FATAL: missing fold checkpoints {missing} under {model_folder}",
            file=sys.stderr,
        )
        return 2

    sessions_df = _load_sessions_meta(project_path)
    eval_root = Path(_paths.evaluation_results_path) / args.experiment_name
    eval_root.mkdir(parents=True, exist_ok=True)

    header = {
        "experiment_name": args.experiment_name,
        "split_name": args.split_name,
        "trainer_class": args.trainer_class,
        "plans_identifier": args.plans_identifier,
        "configuration": args.configuration,
        "folds": list(args.folds),
        "n_test_cases": len(test_ids),
        "n_train": int(outer.get("n_train", 0)),
        "git_sha": current_git_sha(project_path),
        "timestamp": _now(),
    }

    # 1. CV summary (does not need GPU inference - purely reads existing summary.jsons).
    cv: dict = {}
    if not args.skip_cv_summary:
        print("=" * 70)
        print("[finalize] STAGE 1 - CV summary")
        print("=" * 70)
        cv = _build_cv_summary(model_folder, args.folds, sessions_df, eval_root, header)

    # Stage test inputs once for stages 2 + 3.
    if args.skip_per_fold and args.skip_ensemble:
        print("[finalize] both --skip-per-fold and --skip-ensemble set; nothing else to do.")
        return 0

    with tempfile.TemporaryDirectory(prefix="isles_finalize_inputs_") as tmp_inputs:
        staged = Path(tmp_inputs)
        _stage_test_inputs(test_ids, raw_images_dir, staged)
        print(f"[finalize] staged {len(test_ids)} test inputs at {staged}")

        # 2. Per-fold test (5 separate predictions).
        if not args.skip_per_fold:
            print("=" * 70)
            print("[finalize] STAGE 2 - Per-fold test predictions")
            print("=" * 70)
            _build_per_fold(
                model_folder,
                args.folds,
                staged,
                gt_dir,
                test_ids,
                sessions_df,
                eval_root,
                header,
                save_softmax=args.save_softmax,
                checkpoint_name=("swa.pth" if args.use_swa else "checkpoint_best.pth"),
            )

        # 3. Ensemble.
        if not args.skip_ensemble:
            print("=" * 70)
            print("[finalize] STAGE 3 - Ensemble test prediction")
            print("=" * 70)
            row = _build_ensemble(
                model_folder,
                args.folds,
                staged,
                gt_dir,
                test_ids,
                sessions_df,
                eval_root,
                header,
                cv,
                outer,
                args,
            )
            print()
            print(f"  Ensemble Dice:       {row['ensemble_dice']:.4f}")
            print(f"  Ensemble Lesion F1:  {row['ensemble_lesion_f1']:.4f}")
            print(f"  Ensemble HD95:       {row['ensemble_hd95']:.4f}")
            print(f"  Ensemble AVD (mL):   {row['ensemble_avd']:.4f}")
            print(f"  CV Dice (mean±std):  {row['dice_cv_mean']:.4f} ± {row['dice_cv_std']:.4f}")

    # Refresh global leaderboard.
    subprocess.run([sys.executable, str(_THIS.parent / "aggregate_leaderboard.py")], check=False)
    print("[finalize] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
