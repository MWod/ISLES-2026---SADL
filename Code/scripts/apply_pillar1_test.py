"""Apply a tuned Pillar-1 DecisionPolicy to the test softmax and emit final masks.

A separate tuning step searches over fusion + adaptive threshold + FP
cleanup + never-empty rescue on leak-free OOF and writes the winner to a
`DecisionPolicy` JSON. This script is the applier: it consumes that JSON, walks
the reference experiment's `test_predictions_ensemble/*.npz`, stacks any
additional experiments' aligned foreground softmax, runs
`fusion.apply_policy(...)`, and writes one `<session>.nii.gz` mask per case to
`--out-dir` with the reference prediction's geometry.

Only the reference experiment (first `--experiments` entry) is authoritative
for grid + geometry. Additional experiments may drop out of a session's fusion
pool if their NPZ is missing or shape-mismatched - but because the tuned
`policy.weights` (and any `k_of_n` k) was fitted for the full member set,
sessions with fewer surviving members than expected are SKIPPED with a WARN
rather than silently reweighted. At startup we sanity-check each extra
experiment's ensemble directory and print its NPZ count so typos in
`--experiments` do not silently collapse the pool to ref-only.

Exit 0 on success; exit 1 if any reference session's softmax NPZ is missing
or if any session was skipped due to a surviving-member-count mismatch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parents[1] / "src"))


def _ensemble_dir(results_root: Path, experiment: str) -> Path:
    return Path(results_root) / experiment / "test_predictions_ensemble"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help=(
            "One or more experiment names. The first is the reference: its grid, "
            "geometry, and session list are authoritative. Additional experiments "
            "contribute aligned foreground softmax to the fusion stack."
        ),
    )
    parser.add_argument(
        "--policy-json",
        type=Path,
        required=True,
        help="Path to the tuner-produced DecisionPolicy JSON.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write <session>.nii.gz masks.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Override the results root (default: paths.evaluation_results_path).",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=None,
        help=(
            "Optional whitelist of session IDs. If omitted, all sessions with a "
            "softmax NPZ under the reference's test_predictions_ensemble/ are used."
        ),
    )
    args = parser.parse_args()

    import numpy as np
    import paths as _paths
    import SimpleITK as sitk
    from nnunet_isles.inference import fusion
    from nnunet_isles.inference.policy import DecisionPolicy
    from nnunet_isles.inference.threshold_tuner import load_softmax_npz

    results_root = (
        Path(args.results_root) if args.results_root is not None else Path(_paths.evaluation_results_path)
    )
    policy = DecisionPolicy.from_json(args.policy_json)
    print(f"[apply_pillar1_test] loaded policy from {args.policy_json}")
    print(f"[apply_pillar1_test]   mode={policy.mode}  never_empty={policy.never_empty}")

    ref_exp = args.experiments[0]
    ref_dir = _ensemble_dir(results_root, ref_exp)
    if not ref_dir.is_dir():
        print(f"[apply_pillar1_test] FATAL: {ref_dir} does not exist", file=sys.stderr)
        return 2

    sessions = list(args.sessions) if args.sessions else sorted(p.stem for p in ref_dir.glob("*.npz"))
    if not sessions:
        print(f"[apply_pillar1_test] FATAL: no sessions found under {ref_dir}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[apply_pillar1_test] writing masks to {args.out_dir}")
    print(
        f"[apply_pillar1_test] ref experiment={ref_exp}  extras={args.experiments[1:]}  n_sessions={len(sessions)}"
    )

    extra_names = list(args.experiments[1:])
    extra_dirs = [_ensemble_dir(results_root, e) for e in extra_names]

    # Startup sanity check on each extra experiment directory. A typo in
    # --experiments would otherwise silently collapse the fusion pool to
    # ref-only. WARN loudly and print the NPZ count for each extra.
    for name, extra_dir in zip(extra_names, extra_dirs, strict=True):
        if not extra_dir.is_dir():
            print(
                f"[apply_pillar1_test] WARN extra experiment {name!r}: {extra_dir} is not a directory",
                file=sys.stderr,
            )
        else:
            n_npz = sum(1 for _ in extra_dir.glob("*.npz"))
            print(f"[apply_pillar1_test]   extra {name!r}: {n_npz} NPZ(s) at {extra_dir}")

    # Expected surviving member count. If the policy carries per-member weights
    # they define the arity; otherwise fall back to the CLI arity. A tuned
    # policy that expected M members but receives fewer would silently change
    # meaning, so we skip such sessions instead of reweighting.
    n_experiments = len(args.experiments)
    expected_m = len(policy.weights) if policy.weights is not None else n_experiments

    n_missing = 0
    n_shape_skipped = 0
    n_member_mismatch = 0
    _warned_missing_extras: set[str] = set()
    for i, sid in enumerate(sessions, start=1):
        ref_npz = ref_dir / f"{sid}.npz"
        if not ref_npz.exists():
            n_missing += 1
            print(f"  MISSING ref softmax: {ref_npz}", file=sys.stderr)
            continue
        ref_fg = load_softmax_npz(ref_npz).astype(np.float32)

        # Stack the extras that are present AND shape-aligned to the reference.
        fgs = [ref_fg]
        for name, extra_dir in zip(extra_names, extra_dirs, strict=True):
            p = extra_dir / f"{sid}.npz"
            if not p.exists():
                if name not in _warned_missing_extras:
                    print(
                        f"  WARN extra {name!r}: NPZ missing for session {sid} at {p} "
                        f"(further missing sessions for this extra will be silent)",
                        file=sys.stderr,
                    )
                    _warned_missing_extras.add(name)
                continue
            fg = load_softmax_npz(p).astype(np.float32)
            if fg.shape != ref_fg.shape:
                print(
                    f"  WARN {sid}: extra {p} shape {fg.shape} != ref {ref_fg.shape}; skipping",
                    file=sys.stderr,
                )
                n_shape_skipped += 1
                continue
            fgs.append(fg)

        # Guard: a tuned policy was built for `expected_m` members. If any
        # were WARN'd-out above the semantics of `policy.weights` and of
        # `k_of_n` change silently. Skip such sessions loudly rather than
        # producing quietly-degraded masks.
        if len(fgs) != expected_m:
            n_member_mismatch += 1
            n_missing += 1
            print(
                f"  WARN {sid}: surviving members {len(fgs)} != expected {expected_m} "
                f"(policy.weights arity or --experiments arity); skipping session",
                file=sys.stderr,
            )
            continue

        # Always stack: apply_policy's k_of_n branch requires ndim==4, and the
        # weighted-mean / noisy-OR branches accept an (M=1, *spatial) stack
        # equivalently to a 3-D single map.
        probs = np.stack(fgs, axis=0)

        mask = fusion.apply_policy(probs, policy).astype(np.uint8)

        # Clone geometry from the sibling NIfTI (finalize emits both .npz and .nii.gz;
        # the .nii.gz preserves spacing/direction/origin). If it is missing, warn and
        # fall back to an identity geometry (spacing=1, origin=0, RAS/LPS direction).
        out_img = sitk.GetImageFromArray(mask)
        ref_nii = ref_dir / f"{sid}.nii.gz"
        if ref_nii.exists():
            out_img.CopyInformation(sitk.ReadImage(str(ref_nii)))
        else:
            print(
                f"  WARN {sid}: reference geometry {ref_nii} missing; using identity spacing",
                file=sys.stderr,
            )
        sitk.WriteImage(out_img, str(args.out_dir / f"{sid}.nii.gz"))

        if i % 10 == 0:
            print(f"  [{i}/{len(sessions)}] wrote {sid}.nii.gz")

    print(
        f"[apply_pillar1_test] done: {len(sessions) - n_missing}/{len(sessions)} masks written, "
        f"{n_missing} skipped "
        f"(member-mismatch skips={n_member_mismatch}, shape-skipped extras={n_shape_skipped})"
    )
    return 1 if n_missing > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
