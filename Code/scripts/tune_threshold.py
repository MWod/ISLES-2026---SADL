"""Threshold sweep + optional temperature scaling.

Reads per-case softmax NPZs saved by `finalize.py --save-softmax` and picks
the threshold (globally or per-bucket) that maximises mean val Dice. Writes
the chosen threshold + temperature to `Results/<exp>/threshold_sweep.json`
which `apply_threshold.py` consumes downstream.

All numerical logic lives in `Code/src/nnunet_isles/inference/threshold_tuner.py`;
this CLI is a thin argparse wrapper + I/O layer over that library plus a small
temperature-fitter that reuses `apply_temperature` internally.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parents[1] / "src"))


def _fit_temperature(
    softmax_paths: list[Path],
    gt_paths: list[Path],
    *,
    candidates: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0),
) -> float:
    """Pick the temperature T ∈ candidates that minimises mean per-case NLL on val.

    For 2-class softmax with foreground prob `p`, the per-voxel NLL under GT
    is `-log(p)` where GT=1, `-log(1-p)` where GT=0. Mean over cases; pick argmin.
    """
    import numpy as np
    from nnunet_isles.inference.threshold_tuner import apply_temperature, load_softmax_npz

    def _load_gt(path: Path):
        import SimpleITK as sitk

        return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype("uint8")

    # Load once, apply each candidate temperature per case, average NLL.
    cached_pairs = []
    for sp, gp in zip(softmax_paths, gt_paths, strict=True):
        prob = load_softmax_npz(sp)
        gt = _load_gt(gp)
        if prob.shape != gt.shape:
            continue
        cached_pairs.append((prob, gt))
    if not cached_pairs:
        return 1.0

    def nll_at(T: float) -> float:
        losses = []
        for prob, gt in cached_pairs:
            p = apply_temperature(prob, T)
            eps = 1.0e-7
            p_clip = np.clip(p, eps, 1.0 - eps)
            nll = -(gt * np.log(p_clip) + (1 - gt) * np.log(1.0 - p_clip))
            losses.append(float(nll.mean()))
        return float(np.mean(losses)) if losses else float("inf")

    best_T, best_nll = 1.0, float("inf")
    for T in candidates:
        v = nll_at(float(T))
        if v < best_nll:
            best_nll = v
            best_T = float(T)
    return best_T


def _bucket_for_pred_cc_volume(prob_fg, threshold: float, voxel_volume_mm3: float) -> str:
    """Assign a case to a lesion-size bucket based on the volume of its LARGEST
    predicted CC at the given threshold. Bootstrap - never touches GT. Matches
    the lesion_bucket names from `_volume_weights.bucket_for_volume`.
    """
    import numpy as np
    from nnunet_isles.losses._volume_weights import bucket_for_volume
    from scipy.ndimage import label

    pred = (prob_fg > threshold).astype("uint8")
    if pred.sum() == 0:
        return "<0.5mL"  # empty prediction → treat as tiny
    labelled, n = label(pred, structure=np.ones((3, 3, 3), dtype=bool))
    if n == 0:
        return "<0.5mL"
    largest_voxels = int(max((labelled == i).sum() for i in range(1, n + 1)))
    vol_ml = largest_voxels * voxel_volume_mm3 / 1000.0
    return bucket_for_volume(vol_ml)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--dataset-name", default="Dataset510_AtlasV2_V2")
    parser.add_argument(
        "--per-bucket", action="store_true", help="Fit a separate threshold per lesion-size bucket."
    )
    parser.add_argument(
        "--with-temperature",
        action="store_true",
        help="Fit Mask-TS temperature on val NLL before threshold sweep.",
    )
    parser.add_argument(
        "--min-bucket-cases",
        type=int,
        default=10,
        help="If a bucket has fewer val cases than this, fall back to the global threshold.",
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        type=float,
        default=None,
        help="Explicit threshold candidates; default: np.arange(0.20, 0.65, 0.025).",
    )
    parser.add_argument(
        "--voxel-volume-mm3",
        type=float,
        default=1.0,
        help="For bucket assignment (Dataset510 is 1mm iso → 1.0).",
    )
    args = parser.parse_args()

    import numpy as np
    import paths as _paths
    from nnunet_isles.inference.threshold_tuner import (
        apply_temperature,
        load_softmax_npz,
        save_sweep_result,
        sweep_threshold,
    )

    if args.candidates is None:
        args.candidates = list(np.arange(0.20, 0.65, 0.025).round(3))

    eval_root = Path(_paths.evaluation_results_path) / args.experiment
    pred_dir = eval_root / "test_predictions_ensemble"
    if not pred_dir.is_dir():
        print(
            f"[a3] FATAL: no softmax NPZs at {pred_dir}. Re-run finalize.py --save-softmax first.",
            file=sys.stderr,
        )
        return 2

    npz_paths = sorted(pred_dir.glob("*.npz"))
    if not npz_paths:
        print(f"[a3] FATAL: {pred_dir} exists but has 0 NPZ files.", file=sys.stderr)
        return 2

    gt_dir = Path(_paths.nnunet_preprocessed) / args.dataset_name / "gt_segmentations"
    if not gt_dir.is_dir():
        print(f"[a3] FATAL: GT dir not found: {gt_dir}", file=sys.stderr)
        return 2

    # Align softmax + GT by case-id stem.
    aligned = []
    for sp in npz_paths:
        sid = sp.stem
        gp = gt_dir / f"{sid}.nii.gz"
        if gp.exists():
            aligned.append((sp, gp))
    if not aligned:
        print("[a3] FATAL: no softmax/GT pairs matched by case_id.", file=sys.stderr)
        return 2
    softmax_paths = [p[0] for p in aligned]
    gt_paths = [p[1] for p in aligned]
    print(
        f"[a3] {args.experiment}: {len(aligned)} val/holdout pairs → sweeping {len(args.candidates)} thresholds"
    )

    temperature = 1.0
    if args.with_temperature:
        temperature = _fit_temperature(softmax_paths, gt_paths)
        print(f"[a3] fitted temperature: T={temperature:.3f}")

    # Global sweep - always run, gives a fallback + a headline metric.
    scores = sweep_threshold(
        softmax_paths, gt_paths, candidates=tuple(args.candidates), temperature=temperature
    )
    best_global = max(scores, key=scores.get)  # type: ignore[arg-type]
    print(f"[a3] best global threshold: {best_global:.3f}  (mean_dice={scores[best_global]:.4f})")

    per_bucket = None
    bucket_population = None
    if args.per_bucket:
        # Assign each case to a bucket by its own predicted-CC-volume at 0.5 (bootstrap; never GT).
        case_buckets: dict[str, str] = {}
        for sp, gp in aligned:
            sid = gp.name.replace(".nii.gz", "")
            prob = load_softmax_npz(sp)
            if temperature != 1.0:
                prob = apply_temperature(prob, temperature)
            case_buckets[sid] = _bucket_for_pred_cc_volume(prob, 0.5, args.voxel_volume_mm3)
        from collections import Counter

        bucket_population = dict(Counter(case_buckets.values()))
        print(f"[a3] bucket populations: {bucket_population}")

        # Fit per-bucket only for buckets with enough val cases; fall back to
        # global elsewhere. Prevents overfitting on tiny buckets.
        from nnunet_isles.inference.threshold_tuner import fit_per_bucket_threshold

        per_bucket_raw = fit_per_bucket_threshold(
            softmax_paths, gt_paths, case_buckets, candidates=tuple(args.candidates)
        )
        per_bucket = {}
        for b, t in per_bucket_raw.items():
            n = bucket_population.get(b, 0)
            if n < args.min_bucket_cases:
                per_bucket[b] = float(best_global)
                print(
                    f"[a3] bucket {b!r}: only {n} cases (<{args.min_bucket_cases}); falling back to global {best_global:.3f}"
                )
            else:
                per_bucket[b] = float(t)
                print(f"[a3] bucket {b!r}: {n} cases → threshold {t:.3f}")

    out_path = eval_root / "threshold_sweep.json"
    save_sweep_result(
        out_path,
        candidates=[float(t) for t in args.candidates],
        scores=scores,
        best_threshold=float(best_global),
        temperature=float(temperature),
        per_bucket=per_bucket,
    )
    # Also persist the bucket-population count for auditing.
    if bucket_population is not None:
        payload = json.loads(out_path.read_text())
        payload["bucket_population"] = bucket_population
        out_path.write_text(json.dumps(payload, indent=2))
    print(f"[a3] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
