"""Sweep `min_cc_voxels` on val predictions and pick the best threshold.

Liu et al. arXiv:2408.02929 reports +1.3 Dice / +2.4 Lesion-F1 on ATLAS
R2.0 from CC-based post-processing. This tool runs the sweep against the
per-fold val predictions that nnU-Net produces during training (under
`<nnUNet_results>/<dataset>/<experiment_name>/fold_<N>/validation/` for
the renamed layout, or the legacy
`<Trainer>__<plans>__<config>/fold_<N>/validation/`).

Usage:
    python Code/scripts/tune_cc_postproc.py \
        --experiment hpcv6_baseline_v2_dicetopk_500ep \
        --dataset-name Dataset510_AtlasV2_V2 \
        --trainer-class IslesTrainer \
        --plans-identifier nnUNetPlans_iso10 \
        --configuration 3d_fullres \
        --folds 0 1 2 3 4 \
        --metric lesion_f1 \
        [--candidates 0 5 10 25 50 100 200] \
        [--num-workers 24]

Wall-clock optimisations:
  * Parallel across cases via `multiprocessing.Pool` - near-linear speedup
    over 24 SBATCH CPUs. Prior single-threaded version took ~160 min per
    experiment on 1357 val cases; parallel version ~7 min.
  * Per case: compute CC label arrays once (pred + gt), then filter by
    `min_voxels` in-memory via a bincount lookup - no per-candidate
    re-labelling.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parents[1] / "src"))


def _read_mask(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.uint8)


def _dice_bin(pred_bool: np.ndarray, gt_bool: np.ndarray) -> float:
    denom = int(pred_bool.sum()) + int(gt_bool.sum())
    return float(2 * int((pred_bool & gt_bool).sum()) / denom) if denom > 0 else 1.0


# Backward-compat wrappers - downstream scripts (apply_cc_postproc.py) still
# import `_dice`/`_lesion_f1` by the pre-parallelisation names. Keep these
# thin adapters so the public API stays stable.
def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Public alias - accepts uint8/bool masks; dispatches to `_dice_bin`."""
    return _dice_bin(pred > 0, gt > 0)


def _lesion_f1(pred: np.ndarray, gt: np.ndarray, iou_thresh: float = 0.2) -> float:
    """Public alias - labels the input masks then dispatches to `_lesion_f1_from_labels`."""
    from scipy.ndimage import label

    struct = np.ones((3, 3, 3), dtype=bool)
    pred_lab, npred = label(pred > 0, structure=struct)
    gt_lab, ngt = label(gt > 0, structure=struct)
    # keep_pred = all foreground CCs pass (no min_voxels filter for the public API).
    keep_pred = np.zeros(max(npred + 1, 1), dtype=bool)
    keep_pred[1 : npred + 1] = True
    return _lesion_f1_from_labels(pred_lab, keep_pred, gt_lab, ngt, iou_thresh=iou_thresh)


def _lesion_f1_from_labels(
    pred_lab: np.ndarray,
    keep_pred: np.ndarray,
    gt_lab: np.ndarray,
    ngt: int,
    iou_thresh: float = 0.2,
) -> float:
    """Lesion-wise F1 given precomputed CC labels + which pred CCs survive.

    Args:
        pred_lab: labelled pred CC array, background=0.
        keep_pred: bool array of shape (npred+1,) marking which pred CC IDs
            pass the current `min_voxels` filter. keep_pred[0] must be False.
        gt_lab: labelled gt CC array, background=0.
        ngt: number of gt CCs.

    Matching semantics identical to the old `_lesion_f1`: TP when best-IoU
    match >= `iou_thresh` and unmatched-gt-so-far.
    """
    active_pred_ids = np.where(keep_pred)[0]
    # keep_pred[0] should be False, but defensively drop background if slipped.
    active_pred_ids = active_pred_ids[active_pred_ids > 0]
    n_active = int(active_pred_ids.size)
    if n_active == 0 and ngt == 0:
        return 1.0
    if n_active == 0 or ngt == 0:
        return 0.0
    tp = 0
    matched_gt: set[int] = set()
    for pi in active_pred_ids.tolist():
        pmask = pred_lab == pi
        best_iou, best_g = 0.0, -1
        for gi in range(1, ngt + 1):
            if gi in matched_gt:
                continue
            gmask = gt_lab == gi
            inter = int((pmask & gmask).sum())
            if inter == 0:
                continue
            union = int((pmask | gmask).sum())
            iou = float(inter) / float(union)
            if iou > best_iou:
                best_iou, best_g = iou, int(gi)
        if best_iou >= iou_thresh and best_g > 0:
            tp += 1
            matched_gt.add(best_g)
    fp = n_active - tp
    fn = ngt - len(matched_gt)
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def _score_case(
    pred_path: Path,
    gt_path: Path,
    candidates: tuple[int, ...],
    metric: str,
) -> list[float]:
    """Compute the metric across all `candidates` for a single (pred, gt) pair.

    Runs in a worker process. Reads both files once, computes CC labels
    once, then loops candidates in-memory. Returns a list of scores aligned
    with `candidates`.
    """
    from scipy.ndimage import label

    pred = _read_mask(pred_path)
    gt = _read_mask(gt_path)

    struct = np.ones((3, 3, 3), dtype=bool)
    pred_lab, npred = label(pred > 0, structure=struct)
    gt_lab, ngt = label(gt > 0, structure=struct)

    if npred > 0:
        pred_sizes = np.bincount(pred_lab.ravel(), minlength=npred + 1)
    else:
        pred_sizes = np.zeros(1, dtype=np.int64)
    gt_bool = gt > 0

    scores: list[float] = []
    for mv in candidates:
        # Fast path: min_voxels <= 1 → no filtering; use original pred.
        if mv <= 1 or npred == 0:
            keep = np.zeros(pred_sizes.size, dtype=bool)
            keep[1:] = True  # every non-background CC survives (npred may be 0)
            # If npred == 0 the array has one slot (background); keep stays all-False.
        else:
            keep = pred_sizes >= mv
            keep[0] = False  # never keep background
        if metric == "dice":
            filt_bool = np.zeros_like(pred, dtype=bool) if (npred == 0 or not keep.any()) else keep[pred_lab]
            scores.append(_dice_bin(filt_bool, gt_bool))
        else:  # lesion_f1
            scores.append(_lesion_f1_from_labels(pred_lab, keep, gt_lab, ngt))
    return scores


def _score_case_star(args_tuple):
    """Pool.imap adapter - unpacks the single tuple arg."""
    return _score_case(*args_tuple)


def _default_num_workers() -> int:
    """Prefer SLURM's per-task CPU allocation; fall back to cpu_count."""
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        try:
            return max(1, int(slurm))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def main() -> int:
    import paths as _paths

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--trainer-class", required=True)
    parser.add_argument("--plans-identifier", required=True)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--metric", choices=("dice", "lesion_f1"), default="lesion_f1")
    parser.add_argument("--candidates", nargs="+", type=int, default=[0, 5, 10, 25, 50, 100, 200])
    parser.add_argument(
        "--num-workers",
        type=int,
        default=_default_num_workers(),
        help="Multiprocessing pool size for the case loop; default = SLURM_CPUS_PER_TASK or cpu_count().",
    )
    args = parser.parse_args()

    results_root = Path(_paths.evaluation_results_path) / args.experiment
    nnunet_results = Path(_paths.nnunet_results) / args.dataset_name
    # Two on-disk layouts:
    #   1. Renamed layout:         <dataset>/<experiment_name>/fold_N/validation/
    #   2. Legacy nnU-Net layout:  <dataset>/<Trainer>__<plans>__<config>/fold_N/validation/
    # The train.py rename block re-parents fold output at load time; some older
    # runs still use the legacy layout. Try both.
    legacy_trainer_dir = f"{args.trainer_class}__{args.plans_identifier}__{args.configuration}"
    val_root_renamed = nnunet_results / args.experiment
    val_root_legacy = nnunet_results / legacy_trainer_dir

    print(f"[tune_cc_postproc] experiment={args.experiment}")
    print(f"[tune_cc_postproc] metric={args.metric}")
    print(f"[tune_cc_postproc] candidates={args.candidates}")
    print(f"[tune_cc_postproc] workers={args.num_workers}")
    print("[tune_cc_postproc] val roots (renamed → legacy):")
    print(f"    {val_root_renamed}")
    print(f"    {val_root_legacy}")

    def _resolve_val_dir(fold: int) -> Path | None:
        for root in (val_root_renamed, val_root_legacy):
            candidate = root / f"fold_{fold}" / "validation"
            if candidate.exists():
                return candidate
        return None

    case_pairs: list[tuple[Path, Path]] = []
    for f in args.folds:
        val_dir = _resolve_val_dir(f)
        if val_dir is None:
            print(
                f"  WARN: fold {f} val dir missing in both layouts under "
                f"{nnunet_results}/{{{args.experiment},{legacy_trainer_dir}}}/fold_{f}/validation",
                file=sys.stderr,
            )
            continue
        for pred_path in sorted(val_dir.glob("*.nii.gz")):
            sid = pred_path.stem.removesuffix(".nii")
            gt_path = (
                Path(_paths.nnunet_preprocessed) / args.dataset_name / "gt_segmentations" / f"{sid}.nii.gz"
            )
            if not gt_path.exists():
                continue
            case_pairs.append((pred_path, gt_path))
    if not case_pairs:
        print("[tune_cc_postproc] FATAL: no val cases found", file=sys.stderr)
        return 2
    print(f"[tune_cc_postproc] val cases: {len(case_pairs)}")

    candidates = tuple(args.candidates)
    work = [(pp, gp, candidates, args.metric) for pp, gp in case_pairs]

    # Parallel case loop. `imap_unordered` streams progress; we accumulate an
    # (n_cases, n_candidates) score matrix and average per column.
    per_case_scores: list[list[float]] = []
    n_workers = max(1, min(args.num_workers, len(work)))
    if n_workers == 1:
        for w in work:
            per_case_scores.append(_score_case_star(w))
    else:
        # Import here so the pool workers don't need to inherit argparse state.
        from multiprocessing import get_context

        # `spawn` is safer than `fork` on HPC (avoids inherited fd/RNG state);
        # startup is a few extra seconds but the case loop is minutes.
        ctx = get_context("spawn")
        done = 0
        report_every = max(1, len(work) // 20)  # ~20 progress ticks
        with ctx.Pool(processes=n_workers) as pool:
            for scores in pool.imap_unordered(_score_case_star, work, chunksize=1):
                per_case_scores.append(scores)
                done += 1
                if done % report_every == 0 or done == len(work):
                    print(f"  [progress] {done}/{len(work)} cases scored", flush=True)

    arr = np.asarray(per_case_scores, dtype=np.float64)  # (n_cases, n_candidates)
    results: dict[int, float] = {}
    for i, mv in enumerate(candidates):
        mean_score = float(arr[:, i].mean())
        results[int(mv)] = mean_score
        print(f"  min_voxels={mv:4d}  mean_{args.metric}={mean_score:.4f}")

    best_min = max(results, key=results.get)  # type: ignore[arg-type]
    print(f"[tune_cc_postproc] best: min_voxels={best_min}  {args.metric}={results[best_min]:.4f}")

    results_root.mkdir(parents=True, exist_ok=True)
    out = {
        "metric": args.metric,
        "candidates": list(candidates),
        "scores": results,
        "best_min_voxels": int(best_min),
        "best_score": float(results[best_min]),
        "n_val_cases": len(case_pairs),
    }
    (results_root / "postproc_sweep.json").write_text(json.dumps(out, indent=2))
    print(f"[tune_cc_postproc] wrote {results_root / 'postproc_sweep.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
