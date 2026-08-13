"""Apply a tuned threshold (+ optional temperature) to the ensemble
softmax NPZs, re-binarise per case, recompute metrics, and write a `_thr`
leaderboard row.

Consumes `Results/<exp>/threshold_sweep.json` produced by `tune_threshold.py`.
Writes:
  * `Results/<exp>/test_predictions_thr/<case>.nii.gz` - rebinarised masks.
  * `Results/<exp>/test_per_case_thr.tsv` - per-case metrics.
  * `Results/<exp>/leaderboard_row_thr.json` - threshold-variant row.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--dataset-name", default="Dataset510_AtlasV2_V2")
    parser.add_argument("--voxel-volume-mm3", type=float, default=1.0)
    args = parser.parse_args()

    import numpy as np
    import paths as _paths
    import SimpleITK as sitk
    from finalize import _compute_per_case, _load_sessions_meta
    from nnunet_isles.evaluation.aggregator import aggregate_ensemble
    from nnunet_isles.inference.threshold_tuner import apply_temperature, load_softmax_npz

    eval_root = Path(_paths.evaluation_results_path) / args.experiment
    sweep_path = eval_root / "threshold_sweep.json"
    if not sweep_path.exists():
        print(f"[a3'] FATAL: {sweep_path} missing - run tune_threshold.py first.", file=sys.stderr)
        return 2
    sweep = json.loads(sweep_path.read_text())
    best_global = float(sweep["best_threshold"])
    temperature = float(sweep.get("temperature", 1.0))
    per_bucket = sweep.get("per_bucket")  # dict[str, float] or None

    npz_dir = eval_root / "test_predictions_ensemble"
    npz_paths = sorted(npz_dir.glob("*.npz"))
    if not npz_paths:
        print(f"[a3'] FATAL: no NPZs at {npz_dir}", file=sys.stderr)
        return 2

    out_dir = eval_root / "test_predictions_thr"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _bucket(prob_fg) -> str:
        from nnunet_isles.losses._volume_weights import bucket_for_volume
        from scipy.ndimage import label

        pred = (prob_fg > 0.5).astype("uint8")
        if pred.sum() == 0:
            return "<0.5mL"
        labelled, n = label(pred, structure=np.ones((3, 3, 3), dtype=bool))
        if n == 0:
            return "<0.5mL"
        largest_voxels = int(max((labelled == i).sum() for i in range(1, n + 1)))
        vol_ml = largest_voxels * args.voxel_volume_mm3 / 1000.0
        return bucket_for_volume(vol_ml)

    for sp in npz_paths:
        sid = sp.stem
        prob = load_softmax_npz(sp)
        if temperature != 1.0:
            prob = apply_temperature(prob, temperature)
        # Per-bucket threshold path: pick threshold from this case's own predicted
        # bucket (bootstrap at 0.5). Falls back to global if per_bucket key missing.
        if per_bucket:
            b = _bucket(prob)
            thr = float(per_bucket.get(b, best_global))
        else:
            thr = best_global
        mask = (prob > thr).astype("uint8")
        # Clone geometry from the sibling NIfTI in the ensemble dir (finalize
        # writes both .npz and .nii.gz per case; the .nii.gz preserves sform/qform).
        ref = npz_dir / f"{sid}.nii.gz"
        out_sitk = sitk.GetImageFromArray(mask)
        if ref.exists():
            out_sitk.CopyInformation(sitk.ReadImage(str(ref)))
        sitk.WriteImage(out_sitk, str(out_dir / f"{sid}.nii.gz"))

    # Recompute metrics via finalize's helper.
    gt_dir = Path(_paths.nnunet_preprocessed) / args.dataset_name / "gt_segmentations"
    sessions_df = _load_sessions_meta(Path(_paths.project_path))
    test_ids = [sp.stem for sp in npz_paths]
    per_case = _compute_per_case(out_dir, gt_dir, test_ids, sessions_df)
    per_case.to_csv(eval_root / "test_per_case_thr.tsv", sep="\t", index=False)
    summary = aggregate_ensemble(per_case)

    # Load the base leaderboard row (if it exists) as a template so the _thr
    # row keeps every provenance field (git_sha, timestamp, etc.).
    base_row_path = eval_root / "leaderboard_row.json"
    base_row = json.loads(base_row_path.read_text()) if base_row_path.exists() else {}

    def _safe(key: str) -> float:
        s = summary.get(key)
        if isinstance(s, dict) and "mean" in s:
            return float(s["mean"])
        return float("nan")

    row = dict(base_row)
    row["experiment_name"] = f"{args.experiment}_thr"
    row["ensemble_dice"] = _safe("dice")
    row["ensemble_lesion_f1"] = _safe("lesion_f1")
    row["ensemble_hd95"] = _safe("hd95")
    row["ensemble_avd"] = _safe("avd_ml")
    row["ensemble_lesion_count_f1"] = _safe("count_f1")
    row["per_site_dice"] = summary.get("per_site_dice", {})
    row["per_bucket_dice"] = summary.get("per_bucket_dice", {})
    row["threshold"] = best_global
    row["temperature"] = temperature
    row["per_bucket_threshold"] = per_bucket
    (eval_root / "leaderboard_row_thr.json").write_text(json.dumps(row, indent=2))
    print(
        f"[a3'] wrote leaderboard_row_thr.json → dice={_safe('dice'):.4f}  hd95={_safe('hd95'):.3f}  f1={_safe('lesion_f1'):.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
