"""Precompute a per-component lesion bank for CarveMix.

For each training session (union of all 5 inner folds' train+val for the
given split - i.e. everything that is NOT in the held-out test set), runs
26-connectivity labelling on the binary mask, then for every CC with volume
>= --min-voxels writes one .npz containing:
  - image: float32 crop of the T1w channel covering bbox + margin
  - mask:  uint8 crop of the binary mask for that CC (other CCs masked out)
  - spacing: tuple of voxel spacings (z, y, x) read from the nnUNet
    preprocessed metadata for traceability
  - source_session: original session id

Files land at
  <nnunet_preprocessed>/Dataset501_AtlasV2/lesion_bank/<session>_<cc>.npz

Run once per preprocessing pass (iso10, iso10_n4ws, etc.) if you intend to
mix donor and recipient at the same spacing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))

import paths as _paths  # noqa: E402

for _var, _val in (
    ("nnUNet_raw", _paths.nnunet_raw),
    ("nnUNet_preprocessed", _paths.nnunet_preprocessed),
    ("nnUNet_results", _paths.nnunet_results),
):
    Path(_val).mkdir(parents=True, exist_ok=True)
    os.environ[_var] = str(_val)


def _load_b2nd(npz_or_b2nd: Path) -> np.ndarray:
    """Load a preprocessed nnU-Net case (blosc2 b2nd) into a numpy array."""
    try:
        import blosc2  # type: ignore
    except ImportError as exc:
        raise ImportError("precompute_lesion_bank.py needs blosc2 (pip install blosc2)") from exc
    arr = blosc2.open(str(npz_or_b2nd), mode="r")
    return arr[:]


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """26-connectivity labelling on a 3D binary mask."""
    try:
        from scipy.ndimage import label as ndi_label
    except ImportError as exc:
        raise ImportError("precompute_lesion_bank.py needs scipy") from exc
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, n = ndi_label(mask.astype(bool), structure=structure)
    return labeled, int(n)


def _crop_with_margin(
    arr: np.ndarray, bbox: tuple[slice, slice, slice], margin: int
) -> tuple[np.ndarray, tuple[slice, slice, slice]]:
    """Expand bbox by `margin` voxels (clipped to volume bounds) and crop."""
    expanded = []
    for sl, dim in zip(bbox, arr.shape, strict=False):
        lo = max(0, sl.start - margin)
        hi = min(dim, sl.stop + margin)
        expanded.append(slice(lo, hi))
    expanded_t = tuple(expanded)
    return arr[expanded_t], expanded_t


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--dataset-name", default="Dataset501_AtlasV2")
    parser.add_argument("--plans-identifier", default="nnUNetPlans_iso10")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument(
        "--split-name",
        default="site_disjoint_test3",
        help="Outer split whose train+val pool the bank is built from. The held-out test cases are excluded.",
    )
    parser.add_argument("--margin-voxels", type=int, default=6)
    parser.add_argument("--min-voxels", type=int, default=50)
    parser.add_argument("--output-name", default="lesion_bank")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional cap for debug runs")
    args = parser.parse_args()

    splits_dir = Path(_paths.splits_path) / args.split_name
    outer = json.loads((splits_dir / "outer.json").read_text())
    test_ids = set(outer.get("test", []))

    preprocessed_root = (
        Path(_paths.nnunet_preprocessed) / args.dataset_name / f"{args.plans_identifier}_{args.configuration}"
    )
    out_dir = Path(_paths.nnunet_preprocessed) / args.dataset_name / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    case_files = sorted(preprocessed_root.glob("*_seg.b2nd"))
    if not case_files:
        # Fall back to v1-style .npz
        case_files = sorted(preprocessed_root.glob("*_seg.npz"))
    if not case_files:
        print(f"[lesion_bank] no preprocessed cases under {preprocessed_root}", file=sys.stderr)
        return 1

    print(f"[lesion_bank] found {len(case_files)} preprocessed cases; held-out test = {len(test_ids)}")

    n_written = 0
    n_skipped_test = 0
    n_cases_seen = 0
    for seg_path in case_files:
        # nnU-Net writes "<case_id>_seg.b2nd"; image is "<case_id>.b2nd".
        case_id = seg_path.name.replace("_seg.b2nd", "").replace("_seg.npz", "")
        if case_id in test_ids:
            n_skipped_test += 1
            continue
        n_cases_seen += 1
        if args.max_cases is not None and n_cases_seen > args.max_cases:
            break

        img_path = seg_path.with_name(case_id + (".b2nd" if seg_path.suffix == ".b2nd" else ".npz"))
        if not img_path.exists():
            print(f"[lesion_bank] missing image for {case_id}", file=sys.stderr)
            continue

        try:
            seg = _load_b2nd(seg_path)
            img = _load_b2nd(img_path)
        except Exception as e:  # noqa: BLE001
            print(f"[lesion_bank] failed loading {case_id}: {e}", file=sys.stderr)
            continue

        # Preprocessed arrays are (C, X, Y, Z) for image and (1, X, Y, Z) for seg.
        seg3d = seg[0] if seg.ndim == 4 else seg
        img3d = img[0] if img.ndim == 4 else img

        labeled, n_cc = _label_components(seg3d > 0)
        if n_cc == 0:
            continue

        for cc_idx in range(1, n_cc + 1):
            mask_cc = labeled == cc_idx
            n_vox = int(mask_cc.sum())
            if n_vox < args.min_voxels:
                continue
            coords = np.argwhere(mask_cc)
            bbox = tuple(slice(int(coords[:, ax].min()), int(coords[:, ax].max()) + 1) for ax in range(3))
            img_crop, expanded = _crop_with_margin(img3d, bbox, args.margin_voxels)
            mask_crop = mask_cc[expanded].astype(np.uint8)

            out_path = out_dir / f"{case_id}_cc{cc_idx:03d}.npz"
            np.savez_compressed(
                out_path,
                image=img_crop.astype(np.float32),
                mask=mask_crop,
                source_session=case_id,
                cc_voxels=n_vox,
            )
            n_written += 1

    print(
        f"[lesion_bank] cases processed: {n_cases_seen}, skipped (test): {n_skipped_test}, components written: {n_written}"
    )
    print(f"[lesion_bank] output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
