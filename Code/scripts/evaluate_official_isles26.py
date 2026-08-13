"""Score a set of predictions against the official ISLES 2026 evaluator.

Two prediction sources are supported:

* ``--experiment <name>`` - reads a single experiment's per-case masks
  (``Results/<name>/test_predictions_ensemble/*.nii.gz``) and, for PR-AUC,
  the sibling softmax NPZs (``*.npz`` with key ``probabilities`` shape
  ``(C, *spatial)``). Works for both individual sweeps and any output of
  ``run_inference_postproc_pipeline.sh`` (e.g. ``v8_ensemble_top11_thr_cc``).
* ``--pillar1-members <names> --policy-json <path>`` - composes the
  Pillar-1 decision policy on the fly from a list of member sweeps
  (order MUST match ``policy.weights``). PR-AUC is computed on the
  weighted-mean fused softmax; Dice / Lesion-F1 / Volume-diff / LCD are
  computed on the mask returned by :func:`fusion.apply_policy`.

Ground truth is looked up in ``paths.nnunet_raw / <gt-dataset>/labelsTr``
(default ``Dataset501_AtlasV2`` - every R018/R027/R047 session is
present there; V2's Dataset510 is HPC-only). Voxel size in mL is read
per-case from the GT NIfTI header.

Writes:
* ``Results/<output>/official_metrics_isles26.tsv`` - one row per case
* ``Results/<output>/official_metrics_isles26.json`` - summary means,
  by-bucket means, and provenance (source mode, member list, policy hash).

Exit code: 0 if all requested sessions were scored, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parents[1] / "src"))


def _ensemble_dir(results_root: Path, exp: str) -> Path:
    return Path(results_root) / exp / "test_predictions_ensemble"


def _read_uint8_mask(nii_path: Path):
    import SimpleITK as sitk

    im = sitk.ReadImage(str(nii_path))
    import numpy as np

    arr = sitk.GetArrayFromImage(im).astype(np.uint8)
    # spacing is returned in (x, y, z) from ITK; arr is (z, y, x).
    sx, sy, sz = im.GetSpacing()
    voxel_size_ml = float(sx * sy * sz) / 1000.0
    return arr, voxel_size_ml


def _resolve_sessions(pred_dir: Path, whitelist: list[str] | None) -> list[str]:
    # ``Path.stem`` only strips one extension so ``foo.nii.gz`` -> ``foo.nii``.
    # Peel both suffixes so session IDs match the on-disk naming.
    all_ids = sorted(p.name[: -len(".nii.gz")] for p in pred_dir.glob("*.nii.gz"))
    if whitelist:
        keep = set(whitelist)
        missing = keep - set(all_ids)
        if missing:
            print(
                f"[eval_official] WARN: {len(missing)} whitelisted sessions absent "
                f"from {pred_dir}: {sorted(missing)[:5]}...",
                file=sys.stderr,
            )
        return [s for s in all_ids if s in keep]
    return all_ids


def _write_outputs(rows: list[dict], summary: dict, out_dir: Path) -> None:
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "official_metrics_isles26.tsv", sep="\t", index=False)
    (out_dir / "official_metrics_isles26.json").write_text(json.dumps(summary, indent=2))
    print(f"[eval_official] wrote {out_dir / 'official_metrics_isles26.tsv'}")
    print(f"[eval_official] wrote {out_dir / 'official_metrics_isles26.json'}")


def _bucket_means(df, cols):
    return {
        bucket: {col: float(sub[col].mean()) for col in cols}
        for bucket, sub in df.groupby("bucket_ml")
        if not sub.empty
    }


def _score_single_experiment(args, paths_mod) -> int:
    import numpy as np
    import pandas as pd
    from nnunet_isles.evaluation.isles26_official import evaluate_case_official
    from nnunet_isles.inference.threshold_tuner import load_softmax_npz

    results_root = Path(args.results_root or paths_mod.evaluation_results_path)
    pred_dir = _ensemble_dir(results_root, args.experiment)
    if not pred_dir.is_dir():
        print(f"[eval_official] FATAL: {pred_dir} does not exist", file=sys.stderr)
        return 2

    gt_root = Path(args.gt_dir or (Path(paths_mod.nnunet_raw) / args.gt_dataset / "labelsTr"))
    if not gt_root.is_dir():
        print(f"[eval_official] FATAL: GT dir {gt_root} does not exist", file=sys.stderr)
        return 2

    sessions = _resolve_sessions(pred_dir, args.sessions)
    if not sessions:
        print(f"[eval_official] FATAL: no sessions to score in {pred_dir}", file=sys.stderr)
        return 2

    print(
        f"[eval_official] scoring {len(sessions)} session(s) from experiment "
        f"{args.experiment!r} against {gt_root}"
    )

    rows = []
    n_missing_gt = 0
    n_missing_soft = 0
    for i, sid in enumerate(sessions, start=1):
        gt_path = gt_root / f"{sid}.nii.gz"
        if not gt_path.exists():
            n_missing_gt += 1
            print(f"  MISSING GT for {sid}: {gt_path}", file=sys.stderr)
            continue

        gt_arr, voxel_ml = _read_uint8_mask(gt_path)
        pr_arr, _ = _read_uint8_mask(pred_dir / f"{sid}.nii.gz")

        soft = None
        npz_path = pred_dir / f"{sid}.npz"
        if npz_path.exists():
            soft = load_softmax_npz(npz_path).astype(np.float32)
            if soft.shape != gt_arr.shape:
                print(
                    f"  WARN {sid}: soft-map shape {soft.shape} != GT {gt_arr.shape}; "
                    f"skipping PR-AUC for this case",
                    file=sys.stderr,
                )
                soft = None
        else:
            n_missing_soft += 1

        m = evaluate_case_official(sid, gt_arr, pr_arr, soft, voxel_ml)
        row = m.as_dict()
        row["bucket_ml"] = m.bucket_ml
        row["voxel_size_ml"] = voxel_ml
        rows.append(row)
        if i % 20 == 0 or i == len(sessions):
            print(f"  [{i}/{len(sessions)}] scored {sid}")

    if not rows:
        return 1

    df = pd.DataFrame(rows)
    metric_cols = ["dice", "lesion_f1", "abs_lesion_count_diff", "abs_volume_diff_ml", "pr_auc"]
    summary = {
        "source_mode": "single_experiment",
        "experiment": args.experiment,
        "results_root": str(results_root),
        "gt_dir": str(gt_root),
        "n_cases_scored": int(len(rows)),
        "n_missing_gt": int(n_missing_gt),
        "n_missing_softmax": int(n_missing_soft),
        "overall_means": {col: float(df[col].mean()) for col in metric_cols},
        "by_bucket_means": _bucket_means(df, metric_cols),
    }
    out_dir = Path(args.out_dir or (results_root / args.experiment))
    _write_outputs(rows, summary, out_dir)
    print(json.dumps(summary["overall_means"], indent=2))
    return 0 if (n_missing_gt == 0) else 1


def _score_pillar1(args, paths_mod) -> int:
    import numpy as np
    import pandas as pd
    from nnunet_isles.evaluation.isles26_official import evaluate_case_official
    from nnunet_isles.inference import fusion
    from nnunet_isles.inference.policy import DecisionPolicy
    from nnunet_isles.inference.threshold_tuner import load_softmax_npz

    results_root = Path(args.results_root or paths_mod.evaluation_results_path)
    members = args.pillar1_members
    policy = DecisionPolicy.from_json(Path(args.policy_json))
    policy_bytes = Path(args.policy_json).read_bytes()
    policy_hash = hashlib.blake2b(policy_bytes, digest_size=8).hexdigest()

    ref_dir = _ensemble_dir(results_root, members[0])
    if not ref_dir.is_dir():
        print(f"[eval_official] FATAL: ref {ref_dir} does not exist", file=sys.stderr)
        return 2
    extra_dirs = [_ensemble_dir(results_root, m) for m in members[1:]]
    for m, d in zip(members[1:], extra_dirs, strict=True):
        if not d.is_dir():
            print(f"[eval_official] FATAL: extra {m!r}: {d} not a directory", file=sys.stderr)
            return 2

    gt_root = Path(args.gt_dir or (Path(paths_mod.nnunet_raw) / args.gt_dataset / "labelsTr"))
    sessions = _resolve_sessions(ref_dir, args.sessions)
    if not sessions:
        print(f"[eval_official] FATAL: no sessions under {ref_dir}", file=sys.stderr)
        return 2

    expected_m = len(policy.weights) if policy.weights is not None else len(members)
    if expected_m != len(members):
        print(
            f"[eval_official] FATAL: policy.weights has {expected_m} entries but "
            f"--pillar1-members has {len(members)} - arity mismatch",
            file=sys.stderr,
        )
        return 2

    print(
        f"[eval_official] Pillar-1 fusion: {len(members)} members, policy_hash={policy_hash}, "
        f"scoring {len(sessions)} session(s) against {gt_root}"
    )

    rows = []
    n_missing_gt = 0
    n_member_mismatch = 0
    for i, sid in enumerate(sessions, start=1):
        gt_path = gt_root / f"{sid}.nii.gz"
        if not gt_path.exists():
            n_missing_gt += 1
            print(f"  MISSING GT for {sid}: {gt_path}", file=sys.stderr)
            continue

        gt_arr, voxel_ml = _read_uint8_mask(gt_path)

        ref_npz = ref_dir / f"{sid}.npz"
        if not ref_npz.exists():
            print(f"  MISSING ref softmax {ref_npz}; skipping", file=sys.stderr)
            n_member_mismatch += 1
            continue
        ref_fg = load_softmax_npz(ref_npz).astype(np.float32)
        fgs = [ref_fg]
        for _m_name, d in zip(members[1:], extra_dirs, strict=True):
            p = d / f"{sid}.npz"
            if not p.exists() or load_softmax_npz(p).shape != ref_fg.shape:
                break
            fgs.append(load_softmax_npz(p).astype(np.float32))
        if len(fgs) != expected_m:
            print(
                f"  WARN {sid}: {len(fgs)}/{expected_m} members survived; skipping",
                file=sys.stderr,
            )
            n_member_mismatch += 1
            continue

        stack = np.stack(fgs, axis=0)
        mask = fusion.apply_policy(stack, policy).astype(np.uint8)
        # Soft map for PR-AUC = weighted-mean fused foreground softmax
        # (the exact field the policy uses before thresholding). For
        # noisy_or / k_of_n it's still a defensible soft summary since the
        # released code accepts any continuous map, not just calibrated
        # probabilities.
        soft_fg = fusion.mean_prob_weighted(stack, policy.weights).astype(np.float32)

        if mask.shape != gt_arr.shape:
            # Same-space assumption is documented at the NPZ layer - abort
            # loudly rather than silently mis-scoring.
            print(
                f"  FATAL {sid}: fused mask shape {mask.shape} != GT {gt_arr.shape}",
                file=sys.stderr,
            )
            return 2

        m = evaluate_case_official(sid, gt_arr, mask, soft_fg, voxel_ml)
        row = m.as_dict()
        row["bucket_ml"] = m.bucket_ml
        row["voxel_size_ml"] = voxel_ml
        rows.append(row)
        if i % 20 == 0 or i == len(sessions):
            print(f"  [{i}/{len(sessions)}] scored {sid}")

    if not rows:
        return 1

    df = pd.DataFrame(rows)
    metric_cols = ["dice", "lesion_f1", "abs_lesion_count_diff", "abs_volume_diff_ml", "pr_auc"]
    summary = {
        "source_mode": "pillar1_fusion",
        "members": list(members),
        "policy_json_path": str(args.policy_json),
        "policy_hash": policy_hash,
        "policy_mode": policy.mode,
        "n_members_expected": int(expected_m),
        "results_root": str(results_root),
        "gt_dir": str(gt_root),
        "n_cases_scored": int(len(rows)),
        "n_missing_gt": int(n_missing_gt),
        "n_member_mismatch": int(n_member_mismatch),
        "overall_means": {col: float(df[col].mean()) for col in metric_cols},
        "by_bucket_means": _bucket_means(df, metric_cols),
    }
    out_dir = Path(args.out_dir)
    _write_outputs(rows, summary, out_dir)
    print(json.dumps(summary["overall_means"], indent=2))
    return 0 if (n_missing_gt == 0 and n_member_mismatch == 0) else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--experiment",
        help="Score a single experiment's test_predictions_ensemble/ under --results-root.",
    )
    parser.add_argument(
        "--pillar1-members",
        nargs="+",
        help=(
            "Fuse via Pillar-1 policy. First name is the reference (geometry + "
            "case list); additional names contribute per-case softmax. Order must "
            "match policy.weights."
        ),
    )
    parser.add_argument("--policy-json", type=Path, help="Path to DecisionPolicy JSON (Pillar-1 mode).")
    parser.add_argument("--results-root", type=Path, default=None, help="Override results root.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for TSV + JSON.")
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=None,
        help="Ground-truth directory. Default: paths.nnunet_raw / --gt-dataset / labelsTr.",
    )
    parser.add_argument(
        "--gt-dataset",
        default="Dataset501_AtlasV2",
        help="Dataset name under nnUNet_raw (default: Dataset501_AtlasV2).",
    )
    parser.add_argument("--sessions", nargs="+", default=None, help="Optional whitelist of session IDs.")
    args = parser.parse_args()

    if bool(args.experiment) == bool(args.pillar1_members):
        print(
            "[eval_official] FATAL: pass exactly one of --experiment or --pillar1-members",
            file=sys.stderr,
        )
        return 2
    if args.pillar1_members and args.policy_json is None:
        print("[eval_official] FATAL: --pillar1-members requires --policy-json", file=sys.stderr)
        return 2
    if args.pillar1_members and args.out_dir is None:
        print("[eval_official] FATAL: --pillar1-members requires --out-dir", file=sys.stderr)
        return 2

    import paths as paths_mod

    if args.experiment:
        return _score_single_experiment(args, paths_mod)
    return _score_pillar1(args, paths_mod)


if __name__ == "__main__":
    sys.exit(main())
