"""CLI entrypoint: convert ATLAS v2.0 / ISLES BIDS layout to nnU-Net raw dataset.

Thin wrapper around `nnunet_isles.data.atlas_v2_to_nnunet.convert_atlas_v2_to_nnunet`.
Reads source/destination paths from `paths/`, so the same command works locally
and on any SLURM cluster (filesystem auto-detected via paths/__init__.py).

Usage:
    python scripts/convert_atlas_v2_to_nnunet.py
    python scripts/convert_atlas_v2_to_nnunet.py --mode copy   # for HPC scratch with no symlink across mounts
    python scripts/convert_atlas_v2_to_nnunet.py --limit 8     # smoke
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))

# Set nnU-Net env vars BEFORE any nnunet_isles / nnunetv2 imports - nnunetv2.paths
# reads these at module-import time.
import paths as _paths  # noqa: E402

for _var, _val in (
    ("nnUNet_raw", _paths.nnunet_raw),
    ("nnUNet_preprocessed", _paths.nnunet_preprocessed),
    ("nnUNet_results", _paths.nnunet_results),
):
    Path(_val).mkdir(parents=True, exist_ok=True)
    os.environ[_var] = str(_val)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--dataset-name", default="AtlasV2")
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument(
        "--exclude-sites",
        nargs="+",
        default=["R016"],
        help="Site IDs to skip (default: R016 - empty on disk).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N sessions (debug).")
    args = parser.parse_args()

    from nnunet_isles.data.atlas_v2_to_nnunet import convert_atlas_v2_to_nnunet

    atlas_root = Path(_paths.raw_data_path)
    nnunet_raw_root = Path(_paths.nnunet_raw)

    if not atlas_root.exists():
        print(f"[convert] FATAL: ATLAS root not found: {atlas_root}", file=sys.stderr)
        return 2

    print(f"[convert] atlas_root      = {atlas_root}")
    print(f"[convert] nnunet_raw_root = {nnunet_raw_root}")
    print(f"[convert] dataset         = Dataset{args.dataset_id:03d}_{args.dataset_name}")
    print(f"[convert] mode            = {args.mode}")
    print(f"[convert] exclude_sites   = {args.exclude_sites}")
    if args.limit is not None:
        print(f"[convert] limit           = {args.limit}")

    dataset_dir, sessions = convert_atlas_v2_to_nnunet(
        atlas_root=atlas_root,
        nnunet_raw_root=nnunet_raw_root,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        mode=args.mode,
        exclude_sites=tuple(args.exclude_sites),
        limit=args.limit,
    )

    print(f"[convert] wrote {len(sessions)} sessions to {dataset_dir}")
    print(f"[convert]   imagesTr: {sum(1 for _ in (dataset_dir / 'imagesTr').iterdir())} files")
    print(f"[convert]   labelsTr: {sum(1 for _ in (dataset_dir / 'labelsTr').iterdir())} files")
    print(f"[convert]   dataset.json: {(dataset_dir / 'dataset.json').is_file()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
