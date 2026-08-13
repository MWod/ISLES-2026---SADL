"""Collect each experiment's out-of-fold (OOF) validation softmax into one flat dir.

nnU-Net writes per-fold OOF predictions to
    <nnUNet_results>/<dataset>/<experiment>/fold_<k>/validation/<case>.npz
Each training-pool case is validated by exactly one fold, so the union across
folds is a leak-free prediction for the whole train pool (disjoint from the
site-disjoint test holdout). This script links them into
    <evaluation_results>/<experiment>/val_softmax_oof/<case>.npz
so `output_space_ensemble.py --learn-weights` (and any threshold tuner) can learn
on genuine OOF instead of leaking the holdout.

Symlinks by default (cheap; everything stays on the same filesystem where the
ensemble runs); use --copy to materialise real files. Handles both the
`<experiment>/` layout and the legacy `<Trainer>__<plans>__<config>/` layout via
--legacy-dir.

Example (HPC):
  python Code/scripts/gather_oof_val_softmax.py \
    --experiments hpcv8_bucketweighted_swa_v2_dicetopk_700ep hpcv7_da5_v2_dicetopk_700ep \
    --dataset-name Dataset510_AtlasV2_V2
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parents[1] / "src"))


def _member_dir(nnunet_results: Path, dataset: str, experiment: str, legacy_dir: str | None) -> Path | None:
    """Resolve the nnU-Net results dir holding this experiment's fold_*/validation."""
    candidates = [nnunet_results / dataset / experiment]
    if legacy_dir:
        candidates.append(nnunet_results / dataset / legacy_dir)
    for c in candidates:
        if any((c / f"fold_{k}" / "validation").is_dir() for k in range(10)):
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", nargs="+", required=True)
    ap.add_argument("--dataset-name", default="Dataset510_AtlasV2_V2")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--val-subdir", default="val_softmax_oof")
    ap.add_argument(
        "--legacy-dir",
        default=None,
        help="fallback results-dir name if <experiment>/ is absent, e.g. "
        "IslesTrainer__nnUNetPlans_iso10__3d_fullres",
    )
    ap.add_argument("--copy", action="store_true", help="copy files instead of symlinking")
    ap.add_argument("--also-nii", action="store_true", help="also link the sibling <case>.nii.gz")
    args = ap.parse_args()

    import paths as _paths

    nnunet_results = Path(_paths.nnunet_results)
    eval_root = Path(_paths.evaluation_results_path)
    folds = [int(x) for x in args.folds.split(",")]

    rc = 0
    for exp in args.experiments:
        member = _member_dir(nnunet_results, args.dataset_name, exp, args.legacy_dir)
        if member is None:
            print(
                f"[gather] SKIP {exp}: no fold_*/validation under {nnunet_results / args.dataset_name}",
                file=sys.stderr,
            )
            rc = 1
            continue
        out_dir = eval_root / exp / args.val_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        exts = [".npz"] + ([".nii.gz"] if args.also_nii else [])
        for fold in folds:
            vdir = member / f"fold_{fold}" / "validation"
            for npz in sorted(glob.glob(str(vdir / "*.npz"))):
                stem = Path(npz).name[:-4]
                for ext in exts:
                    src = vdir / f"{stem}{ext}"
                    if not src.exists():
                        continue
                    dst = out_dir / f"{stem}{ext}"
                    if dst.exists() or dst.is_symlink():
                        dst.unlink()
                    if args.copy:
                        import shutil

                        shutil.copy2(src, dst)
                    else:
                        os.symlink(os.path.relpath(src, out_dir), dst)
                n += 1
        print(f"[gather] {exp}: {n} OOF cases -> {out_dir} (from {member.name})")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
