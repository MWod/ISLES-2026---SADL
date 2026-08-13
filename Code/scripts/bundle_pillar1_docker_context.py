"""Bundle the 11-member Pillar-1 ensemble into ``docker/build_context/`` for
the ISLES 2026 grand-challenge inference container.

Reads a manifest (see [docker/manifest_pillar1_v2.json](docker/manifest_pillar1_v2.json))
that lists 11 experiments in the order that matches ``policy.weights`` in
``policy.json``. For each experiment, prunes every checkpoint payload
(``checkpoint_best.pth`` for 10 members, ``swa.pth`` for the SWAD member),
optionally casting ``network_weights`` to ``float16``, and copies the
result into ``docker/build_context/checkpoints/<exp>/`` alongside the
per-experiment ``plans.json`` / ``dataset.json`` / ``dataset_fingerprint.json``.
The Pillar-1 ``policy.json`` and the manifest itself are copied to
``docker/build_context/policy/`` and the context root respectively.

Layout produced:

    docker/build_context/
        checkpoints/
            <exp_1>/
                plans.json
                dataset.json
                dataset_fingerprint.json
                fold_0/{checkpoint_best.pth | swa.pth}
                ...
                fold_4/{checkpoint_best.pth | swa.pth}
            ...
            <exp_11>/
        policy/
            policy.json
        manifest_pillar1_v2.json

This matches the shape the in-container ``predict_pillar1`` module expects:
one flat directory per experiment, ready to be passed to
``IslesPredictor.initialize_from_model_folder(model_folder, checkpoint_name=...)``.

nnU-Net's checkpoint-consuming keys are (upstream
``predict_from_raw_data.initialize_from_trained_model_folder``):
``trainer_name``, ``init_args``, ``inference_allowed_mirroring_axes``,
``network_weights``. Everything else is safe to drop.

Usage:

    python Code/scripts/bundle_pillar1_docker_context.py \\
        --manifest docker/manifest_pillar1_v2.json \\
        --out docker/build_context/ \\
        [--fp16] [--dry-run]

Exits non-zero if any required source file is missing, if the manifest
cannot be parsed, or if a pruned checkpoint fails a round-trip sanity
check (torch.load of the emitted file).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))  # Code/
sys.path.insert(0, str(_THIS.parents[1] / "src"))  # Code/src/

# Keys the pruner keeps in every checkpoint payload. Anything outside this
# set is dropped. Verified against upstream nnU-Net at
# third_party/nnUNet/nnunetv2/inference/predict_from_raw_data.py:87-95.
_KEEP_KEYS: tuple[str, ...] = (
    "trainer_name",
    "init_args",
    "inference_allowed_mirroring_axes",
    "network_weights",
    "current_epoch",  # harmless metadata; useful for provenance
)

_METADATA_FILES: tuple[str, ...] = (
    "plans.json",
    "dataset.json",
    "dataset_fingerprint.json",
)


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or unit == "TB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes} B"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"  # unreachable - kept for the type checker


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _prune_checkpoint(src: Path, dst: Path, *, fp16: bool) -> tuple[int, int]:
    """Prune ``src`` to ``dst`` keeping only inference-relevant fields.

    If ``fp16`` is set, casts every tensor in ``network_weights`` to
    ``torch.float16``. Returns (in_bytes, out_bytes).
    """
    import torch

    payload = torch.load(str(src), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"unexpected checkpoint payload type at {src}: {type(payload)}")

    missing = [k for k in ("trainer_name", "init_args", "network_weights") if k not in payload]
    if missing:
        raise KeyError(f"checkpoint {src} missing required key(s): {missing}")

    pruned = {k: payload[k] for k in _KEEP_KEYS if k in payload}

    if fp16:
        weights = pruned["network_weights"]
        if not isinstance(weights, dict):
            raise TypeError(f"network_weights is not a dict-like state_dict at {src}: {type(weights)}")
        pruned["network_weights"] = {
            name: (tensor.to(torch.float16) if tensor.is_floating_point() else tensor)
            for name, tensor in weights.items()
        }

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pruned, str(dst))

    # Round-trip sanity check: load the pruned file back and confirm it
    # contains the same set of required keys. Catches on-disk corruption
    # before it can foil a downstream Docker build.
    check = torch.load(str(dst), map_location="cpu", weights_only=False)
    for k in ("trainer_name", "init_args", "network_weights"):
        if k not in check:
            raise RuntimeError(f"round-trip check FAILED: {dst} lost required key '{k}'")

    return src.stat().st_size, dst.stat().st_size


def _bundle_experiment(
    exp: dict,
    dataset: str,
    src_root: Path,
    out_root: Path,
    *,
    fp16: bool,
    dry_run: bool,
) -> tuple[int, int, list[str]]:
    """Bundle a single experiment. Returns (in_bytes, out_bytes, warnings)."""
    exp_name = exp["name"]
    checkpoint_name = exp["checkpoint_name"]
    src_exp = src_root / dataset / exp_name
    out_exp = out_root / "checkpoints" / exp_name

    if not src_exp.is_dir():
        raise FileNotFoundError(f"source experiment dir not found: {src_exp}")

    # Metadata files at the experiment root.
    for name in _METADATA_FILES:
        src = src_exp / name
        if not src.is_file():
            raise FileNotFoundError(f"missing {name}: {src}")

    # All 5 folds must have the required checkpoint.
    missing_folds: list[str] = []
    for fold in range(5):
        if not (src_exp / f"fold_{fold}" / checkpoint_name).is_file():
            missing_folds.append(f"fold_{fold}/{checkpoint_name}")
    if missing_folds:
        raise FileNotFoundError(
            f"experiment {exp_name!r}: missing checkpoints under {src_exp}: {missing_folds}"
        )

    warnings: list[str] = []
    in_bytes = 0
    out_bytes = 0

    if dry_run:
        for name in _METADATA_FILES:
            src = src_exp / name
            in_bytes += src.stat().st_size
            out_bytes += src.stat().st_size
        for fold in range(5):
            src = src_exp / f"fold_{fold}" / checkpoint_name
            in_bytes += src.stat().st_size
            # Approximate pruned size: fp16 halves floats, prune drops <5 %
            approx = int(src.stat().st_size * (0.5 if fp16 else 0.95))
            out_bytes += approx
        return in_bytes, out_bytes, warnings

    out_exp.mkdir(parents=True, exist_ok=True)
    # Copy metadata (unchanged).
    for name in _METADATA_FILES:
        src = src_exp / name
        dst = out_exp / name
        shutil.copy2(src, dst)
        in_bytes += src.stat().st_size
        out_bytes += dst.stat().st_size

    # Prune + copy each fold's checkpoint.
    for fold in range(5):
        src = src_exp / f"fold_{fold}" / checkpoint_name
        dst = out_exp / f"fold_{fold}" / checkpoint_name
        i, o = _prune_checkpoint(src, dst, fp16=fp16)
        in_bytes += i
        out_bytes += o

    return in_bytes, out_bytes, warnings


def _copy_policy(policy_json: Path, out_root: Path, *, dry_run: bool) -> int:
    if not policy_json.is_file():
        raise FileNotFoundError(f"policy JSON not found: {policy_json}")
    dst = out_root / "policy" / "policy.json"
    if dry_run:
        return policy_json.stat().st_size
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(policy_json, dst)
    return dst.stat().st_size


def _copy_manifest(manifest: Path, out_root: Path, *, dry_run: bool) -> int:
    dst = out_root / manifest.name
    if dry_run:
        return manifest.stat().st_size
    out_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest, dst)
    return dst.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, default=Path("docker/manifest_pillar1_v2.json"))
    parser.add_argument("--out", type=Path, default=Path("docker/build_context/"))
    parser.add_argument(
        "--policy-json",
        type=Path,
        default=Path("Results/_diagnostics_pillar1_v2/policy.json"),
        help="Path to the tuned Pillar-1 DecisionPolicy JSON to bundle.",
    )
    parser.add_argument(
        "--nnunet-results-root",
        type=Path,
        default=None,
        help="Override the source nnunet_results root (default: paths.nnunet_results).",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Cast network_weights to float16 in the pruned checkpoints."
    )
    parser.add_argument("--dry-run", action="store_true", help="Report sizes without writing files.")
    parser.add_argument(
        "--wipe-out",
        action="store_true",
        help="Remove <out> before bundling. Off by default so re-runs are idempotent.",
    )
    args = parser.parse_args()

    if not args.manifest.is_file():
        print(f"[bundle] FATAL: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    manifest = json.loads(args.manifest.read_text())

    src_root: Path
    if args.nnunet_results_root is not None:
        src_root = args.nnunet_results_root
    else:
        import paths as _paths

        src_root = Path(_paths.nnunet_results)
    if not src_root.is_dir():
        print(f"[bundle] FATAL: nnunet_results root not found: {src_root}", file=sys.stderr)
        return 2

    dataset = manifest["dataset"]
    experiments = manifest["experiments"]
    if len(experiments) != 11:
        print(
            f"[bundle] WARNING: manifest has {len(experiments)} experiments; expected 11 for Pillar-1 v2",
            file=sys.stderr,
        )

    if args.wipe_out and args.out.exists() and not args.dry_run:
        print(f"[bundle] wiping existing {args.out}")
        shutil.rmtree(args.out)

    mode = "fp16" if args.fp16 else "fp32-pruned"
    print(f"[bundle] mode                = {mode}")
    print(f"[bundle] source              = {src_root}/{dataset}")
    print(f"[bundle] destination         = {args.out}")
    print(f"[bundle] manifest            = {args.manifest}")
    print(f"[bundle] policy              = {args.policy_json}")
    print(f"[bundle] experiments (order) = {len(experiments)}")
    print(f"[bundle] dry-run             = {args.dry_run}")
    print()

    total_in = 0
    total_out = 0
    print(f"{'#':>2}  {'experiment':<52}  {'ckpt_name':<20}  {'in':>8}  {'out':>8}  ratio")
    print("-" * 108)
    for i, exp in enumerate(experiments, start=1):
        try:
            in_b, out_b, _warnings = _bundle_experiment(
                exp, dataset, src_root, args.out, fp16=args.fp16, dry_run=args.dry_run
            )
        except (FileNotFoundError, KeyError, RuntimeError, TypeError) as e:
            print(f"\n[bundle] FATAL bundling {exp['name']!r}: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        ratio = (out_b / in_b) if in_b else 0.0
        print(
            f"{i:>2}  {exp['name']:<52}  {exp['checkpoint_name']:<20}  "
            f"{_human_size(in_b):>8}  {_human_size(out_b):>8}  {ratio:>4.2f}x"
        )
        total_in += in_b
        total_out += out_b

    # Policy + manifest copies.
    policy_bytes = _copy_policy(args.policy_json, args.out, dry_run=args.dry_run)
    manifest_bytes = _copy_manifest(args.manifest, args.out, dry_run=args.dry_run)
    total_out += policy_bytes + manifest_bytes

    print("-" * 108)
    print(
        f"{'total':>2}  {'(11 experiments + policy + manifest)':<52}  {'':<20}  "
        f"{_human_size(total_in):>8}  {_human_size(total_out):>8}"
    )

    if not args.dry_run:
        # Write a small bundle-side manifest with mode + timestamp so
        # downstream tooling can confirm what got shipped.
        bundle_info = {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "mode": mode,
            "policy_hash_in_manifest": manifest.get("provenance", {}).get("policy_hash"),
            "n_experiments": len(experiments),
            "source_root": str(src_root),
            "dataset": dataset,
            "total_out_bytes": total_out,
        }
        (args.out / "bundle_info.json").write_text(json.dumps(bundle_info, indent=2))

    print("\n[bundle] done" if not args.dry_run else "\n[bundle] dry-run - no files written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
