"""Convert BIDS-style ATLAS v2.0 / ISLES 2026 training pool to nnU-Net v2 dataset layout.

Output layout (under ``$nnUNet_raw``):

    Dataset<ID>_<Name>/
        dataset.json
        imagesTr/
            <session_id>_0000.nii.gz
        labelsTr/
            <session_id>.nii.gz

The 4-digit channel suffix `_0000` matches nnU-Net's single-modality T1w
convention. Files are symlinked (default) or copied based on ``mode``.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from nnunet_isles.data.atlas_v2_scanner import SessionRecord, scan_atlas_v2


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'symlink' or 'copy'.")


def convert_atlas_v2_to_nnunet(
    atlas_root: str | Path,
    nnunet_raw_root: str | Path,
    dataset_id: int = 501,
    dataset_name: str = "AtlasV2",
    mode: str = "symlink",
    exclude_sites: tuple[str, ...] = ("R016",),
    limit: int | None = None,
) -> tuple[Path, list[SessionRecord]]:
    """Materialize an nnU-Net raw dataset folder backed by ATLAS sessions.

    Returns the dataset directory and the list of `SessionRecord`s that were exported.
    """
    atlas_root = Path(atlas_root)
    nnunet_raw_root = Path(nnunet_raw_root)
    dataset_dir = nnunet_raw_root / f"Dataset{dataset_id:03d}_{dataset_name}"
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    sessions = scan_atlas_v2(atlas_root, exclude_sites=exclude_sites)
    if limit is not None:
        sessions = sessions[:limit]

    for session in sessions:
        _link_or_copy(session.t1_path, images_dir / f"{session.session_id}_0000.nii.gz", mode)
        _link_or_copy(session.mask_path, labels_dir / f"{session.session_id}.nii.gz", mode)

    dataset_json = {
        "channel_names": {"0": "T1w"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(sessions),
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "description": "ISLES 2026 - ATLAS v2.0 derived T1w stroke lesion training pool.",
        "reference": "https://isles-26.grand-challenge.org/",
        "release": "0.1",
        "licence": "CC-BY-4.0 (ATLAS v2.0)",
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    return dataset_dir, sessions
