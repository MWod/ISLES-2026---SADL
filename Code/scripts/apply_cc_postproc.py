"""Apply a tuned CC threshold to test predictions, write postproc'd metrics.

Reads `Results/<experiment>/postproc_sweep.json` (produced by `tune_cc_postproc.py`),
applies the chosen `min_voxels` to the per-fold test predictions in
`Results/<experiment>/test_per_fold/*.tsv` (or the ensemble predictions if
`--use-ensemble`), recomputes per-case Dice + Lesion-F1, and writes a
postproc-variant TSV + leaderboard row.

The test predictions themselves are not stored as NIfTI per-case in our
finalize output (only metrics survive). For full re-evaluation we need
the raw prediction NIfTIs. If `--predictions-dir` is given, we read NIfTIs
from there; otherwise we look under the experiment's standard nnU-Net
results folder.
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
    import pandas as pd
    import paths as _paths
    from nnunet_isles.inference.cc_postproc import apply_cc_filter
    from tune_cc_postproc import _dice, _lesion_f1, _read_mask

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="Path containing per-case test .nii.gz prediction files",
    )
    parser.add_argument(
        "--gt-dir", type=Path, required=True, help="Path containing per-case GT .nii.gz files"
    )
    parser.add_argument(
        "--min-voxels", type=int, default=None, help="Override the value in postproc_sweep.json"
    )
    args = parser.parse_args()

    results_root = Path(_paths.evaluation_results_path) / args.experiment
    sweep_path = results_root / "postproc_sweep.json"
    if args.min_voxels is not None:
        min_voxels = args.min_voxels
        print(f"[apply_cc_postproc] using override min_voxels={min_voxels}")
    else:
        if not sweep_path.exists():
            print(f"[apply_cc_postproc] FATAL: no postproc_sweep.json at {sweep_path}", file=sys.stderr)
            return 2
        sweep = json.loads(sweep_path.read_text())
        min_voxels = int(sweep["best_min_voxels"])
        print(f"[apply_cc_postproc] using min_voxels={min_voxels} from {sweep_path}")

    rows = []
    pred_files = sorted(args.predictions_dir.glob("*.nii.gz"))
    if not pred_files:
        print(f"[apply_cc_postproc] FATAL: no .nii.gz in {args.predictions_dir}", file=sys.stderr)
        return 2

    for pred_path in pred_files:
        sid = pred_path.name.replace(".nii.gz", "")
        gt_path = args.gt_dir / f"{sid}.nii.gz"
        if not gt_path.exists():
            print(f"  skip (no GT): {sid}", file=sys.stderr)
            continue
        pred = _read_mask(pred_path)
        gt = _read_mask(gt_path)
        filt = apply_cc_filter(pred, min_voxels=min_voxels)
        rows.append(
            {
                "session_id": sid,
                "dice_raw": _dice(pred, gt),
                "dice_postproc": _dice(filt, gt),
                "lesion_f1_raw": _lesion_f1(pred, gt),
                "lesion_f1_postproc": _lesion_f1(filt, gt),
            }
        )

    df = pd.DataFrame(rows)
    out_tsv = results_root / "test_per_case_postproc.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"[apply_cc_postproc] wrote {out_tsv}")
    print(f"  mean Dice:       raw={df.dice_raw.mean():.4f}  postproc={df.dice_postproc.mean():.4f}")
    print(
        f"  mean Lesion-F1:  raw={df.lesion_f1_raw.mean():.4f}  postproc={df.lesion_f1_postproc.mean():.4f}"
    )

    leaderboard_row = {
        "experiment_name": args.experiment,
        "postproc": {"min_cc_voxels": min_voxels},
        "ensemble_dice_postproc": float(df.dice_postproc.mean()),
        "ensemble_lesion_f1_postproc": float(df.lesion_f1_postproc.mean()),
        "n_test": len(df),
    }
    (results_root / "leaderboard_row_postproc.json").write_text(json.dumps(leaderboard_row, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
