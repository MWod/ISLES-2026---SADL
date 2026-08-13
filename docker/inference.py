"""ISLES 2026 SADL - Grand Challenge invoke API adapter.

Runs an 11-member nnU-Net ensemble with sagittal-only test-time augmentation
in fp32 and inverse-resamples the plans-space prediction back to native
geometry on GPU via torch.nn.functional.interpolate (trilinear). Fusion
follows the frozen Pillar-1 decision policy shipped at
/opt/ml/model/policy/policy.json.

Interface (single interface - sorted socket-slug tuple):
    ("stroke-metadata", "t1-brain-mri")
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_ROOT = Path(os.environ.get("SADL_MODEL_ROOT", "/opt/ml/model"))

_SUPPORTED_INPUT_EXTS: tuple[str, ...] = (".mha", ".nii.gz", ".nii")


def _set_nnunet_env() -> None:
    scratch = Path(os.environ.get("SADL_TMP_ROOT", "/tmp/nnunet"))
    for var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        target = scratch / var
        target.mkdir(parents=True, exist_ok=True)
        os.environ[var] = str(target)


@dataclass
class SADLModel:
    predictors: list
    experiments: list
    policy: object
    manifest: dict


def init_model() -> SADLModel:
    _set_nnunet_env()

    import torch

    from nnunet_isles.inference.policy import DecisionPolicy
    from nnunet_isles.inference.predictor import IslesPredictor

    print(f"[SADL] MODEL_ROOT = {MODEL_ROOT}  exists={MODEL_ROOT.exists()}  "
          f"is_dir={MODEL_ROOT.is_dir()}", file=sys.stderr, flush=True)
    if MODEL_ROOT.is_dir():
        entries = sorted(MODEL_ROOT.iterdir())
        print(f"[SADL] MODEL_ROOT contains {len(entries)} top-level entries:",
              file=sys.stderr, flush=True)
        for p in entries[:20]:
            kind = "dir " if p.is_dir() else "file"
            size = f"{p.stat().st_size:>12}" if p.is_file() else " " * 12
            print(f"[SADL]   [{kind}] {size}  {p.name}", file=sys.stderr, flush=True)

    if not MODEL_ROOT.is_dir():
        raise FileNotFoundError(
            f"SADL model root not found at {MODEL_ROOT}. "
            f"Grand Challenge should extract model.tar.gz here at runtime."
        )

    manifest_path = MODEL_ROOT / "manifest_pillar1_v2.json"
    if not manifest_path.is_file():
        for sub in MODEL_ROOT.iterdir():
            if sub.is_dir() and (sub / "manifest_pillar1_v2.json").is_file():
                print(f"[SADL] found manifest under nested dir {sub.name}/ - re-rooting",
                      file=sys.stderr, flush=True)
                globals()["MODEL_ROOT"] = sub
                manifest_path = sub / "manifest_pillar1_v2.json"
                break
        else:
            raise FileNotFoundError(
                f"manifest missing: {manifest_path}. "
                f"Contents: {[p.name for p in MODEL_ROOT.iterdir()]}"
            )
    manifest = json.loads(manifest_path.read_text())

    policy_path = MODEL_ROOT / manifest["policy_json"]
    if not policy_path.is_file():
        raise FileNotFoundError(f"policy JSON missing: {policy_path}")
    policy = DecisionPolicy.from_json(policy_path)

    if policy.weights is None or len(policy.weights) != len(manifest["experiments"]):
        raise ValueError(
            f"policy.weights arity {len(policy.weights) if policy.weights else None} != "
            f"n_experiments {len(manifest['experiments'])}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SADL] init: device={device}, MODEL_ROOT={MODEL_ROOT}",
          file=sys.stderr, flush=True)
    print(f"[SADL] init: loading {len(manifest['experiments'])} predictors "
          f"(sag-only TTA, fp32, GPU-side inverse resample)", file=sys.stderr, flush=True)

    predictors: list = []
    experiments: list = []
    for i, exp in enumerate(manifest["experiments"]):
        model_folder = MODEL_ROOT / "checkpoints" / exp["name"]
        _verify_model_folder(model_folder, exp["checkpoint_name"])

        pred = IslesPredictor(
            use_mirroring=True,
            allowed_mirroring_axes=(0,),
            tile_step_size=0.5,
            use_gaussian=True,
            perform_everything_on_device=True,
            device=device,
            verbose=False,
        )
        pred.initialize_from_model_folder(
            model_folder,
            use_folds=(0, 1, 2, 3, 4),
            checkpoint_name=exp["checkpoint_name"],
        )
        predictors.append(pred)
        experiments.append({**exp, "model_folder": model_folder})
        print(
            f"[SADL] loaded [{i + 1:>2}/{len(manifest['experiments'])}] "
            f"{exp['name']}  axes=(0,)",
            file=sys.stderr, flush=True,
        )

    print("[SADL] init: all predictors loaded - model ready", file=sys.stderr, flush=True)
    return SADLModel(
        predictors=predictors,
        experiments=experiments,
        policy=policy,
        manifest=manifest,
    )


def _verify_model_folder(model_folder: Path, checkpoint_name: str) -> None:
    if not model_folder.is_dir():
        raise FileNotFoundError(f"experiment folder missing: {model_folder}")
    for req in ("plans.json", "dataset.json", "dataset_fingerprint.json"):
        if not (model_folder / req).is_file():
            raise FileNotFoundError(f"{model_folder}: missing {req}")
    for fold in range(5):
        ckpt = model_folder / f"fold_{fold}" / checkpoint_name
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"{model_folder}: missing fold_{fold}/{checkpoint_name}"
            )


def _get_interface_key() -> tuple[str, ...]:
    inputs = json.loads((INPUT_PATH / "inputs.json").read_text())
    return tuple(sorted(sv["socket"]["slug"] for sv in inputs))


def _load_t1w() -> tuple[sitk.Image, str]:
    src_dir = INPUT_PATH / "images/t1-brain-mri"
    if not src_dir.is_dir():
        raise FileNotFoundError(f"expected input dir missing: {src_dir}")
    for ext in _SUPPORTED_INPUT_EXTS:
        matches = sorted(src_dir.glob(f"*{ext}"))
        if matches:
            return sitk.ReadImage(str(matches[0])), matches[0].name
    raise FileNotFoundError(f"no input image under {src_dir}")


def _load_metadata() -> dict:
    path = INPUT_PATH / "stroke-metadata.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def _run_case(model: SADLModel, t1_image: sitk.Image) -> tuple[np.ndarray, np.ndarray]:
    """End-to-end SADL pipeline. Uses fuse_in_native_space_gpu_resample."""
    from nnunet_isles.inference import fusion
    from nnunet_isles.inference.native_space_fuser import fuse_in_native_space_gpu_resample

    case_id = "case"

    with tempfile.TemporaryDirectory(prefix="sadl_") as tmp_str:
        tmp = Path(tmp_str)
        staged_dir = tmp / "staged"
        staged_dir.mkdir()
        sitk.WriteImage(t1_image, str(staged_dir / f"{case_id}_0000.nii.gz"))

        fg_native, _properties, _ref = fuse_in_native_space_gpu_resample(
            model.predictors,
            model.policy.weights,
            [str(staged_dir / f"{case_id}_0000.nii.gz")],
            autocast_dtype=None,      # fp32 (no autocast) - arithmetic-preserving
            verbose=False,
        )
        np.clip(fg_native, 0.0, 1.0, out=fg_native)
        prob_map = fg_native
        mask = fusion.apply_policy(fg_native, model.policy).astype(np.uint8)

    return mask, prob_map


def _write_output(array: np.ndarray, reference: sitk.Image, location: Path) -> None:
    location.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(array)
    img.SetSpacing(reference.GetSpacing())
    img.SetOrigin(reference.GetOrigin())
    img.SetDirection(reference.GetDirection())
    sitk.WriteImage(img, str(location / "output.mha"), useCompression=True)


def run(model: SADLModel) -> int:
    key = _get_interface_key()
    expected = ("stroke-metadata", "t1-brain-mri")
    if key != expected:
        raise ValueError(f"unexpected interface key: {key} (expected {expected})")

    metadata = _load_metadata()
    print(f"[SADL] metadata: {json.dumps(metadata, default=str)}", file=sys.stderr, flush=True)

    t1_image, src_name = _load_t1w()
    print(
        f"[SADL] input: {src_name}, size={t1_image.GetSize()}, "
        f"spacing={t1_image.GetSpacing()}",
        file=sys.stderr, flush=True,
    )

    mask, prob_map = _run_case(model, t1_image)

    _write_output(mask, t1_image, OUTPUT_PATH / "images/stroke-lesion-segmentation")
    _write_output(prob_map, t1_image, OUTPUT_PATH / "images/lesion-probability-map")
    print(
        "[SADL] wrote /output/images/{stroke-lesion-segmentation,lesion-probability-map}/output.mha",
        file=sys.stderr, flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run(model=init_model()))
