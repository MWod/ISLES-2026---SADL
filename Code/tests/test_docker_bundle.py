"""Tests for ``Code/scripts/bundle_pillar1_docker_context.py``.

The bundler's two load-bearing pieces are:

* ``_prune_checkpoint`` - must preserve the exact key set nnU-Net's
  ``initialize_from_trained_model_folder`` reads (``trainer_name``,
  ``init_args``, ``inference_allowed_mirroring_axes``, ``network_weights``)
  and MUST drop optimizer/scheduler/grad-scaler state. fp16 mode must
  cast every float tensor in ``network_weights`` to ``torch.float16``
  without touching non-float tensors (e.g. int64 buffer indices).
* ``_bundle_experiment`` - must fail loudly on any missing source file
  and must lay files out as ``<out>/checkpoints/<exp>/{plans.json,
  dataset.json, dataset_fingerprint.json, fold_N/<ckpt_name>}``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

# ruff: noqa: E402
import bundle_pillar1_docker_context as bundler  # type: ignore

# ---------- fixture builders ----------


def _mock_state_dict() -> dict[str, torch.Tensor]:
    """State_dict with a mix of float and non-float tensors, mirroring
    what an nnU-Net PlainConvUNet checkpoint looks like at the top level.

    Sized large enough (~2 MB fp32) that torch.save's per-payload metadata
    overhead (~2 KB) does not swamp the fp16 halving in the size-reduction
    assertion below.
    """
    return {
        # Two "chunky" conv weights totalling ~2 MB - realistic-ish for a
        # small network layer, and large enough that fp16 halving is
        # visible above torch.save's constant overhead.
        "encoder.stages.0.conv.weight": torch.randn(128, 64, 3, 3, 3),  # ~882 KB fp32
        "encoder.stages.1.conv.weight": torch.randn(128, 128, 3, 3, 3),  # ~1.7 MB fp32
        "encoder.stages.0.conv.bias": torch.randn(128),  # fp32 (tiny)
        "encoder.stages.0.norm.weight": torch.ones(128),  # fp32 (tiny)
        "num_batches_tracked": torch.tensor(0, dtype=torch.int64),  # non-float, must be preserved
    }


def _mock_full_checkpoint() -> dict:
    return {
        "network_weights": _mock_state_dict(),
        "trainer_name": "IslesTrainer",
        "init_args": {
            "configuration": "3d_fullres",
            "plans": {"foo": "bar"},
        },
        "inference_allowed_mirroring_axes": (0,),
        "current_epoch": 499,
        # These must be dropped by the pruner.
        "optimizer_state": {"param_groups": [{"lr": 1e-4}]},
        "grad_scaler_state": {"scale": 1024.0},
        "lr_scheduler_state": {"last_epoch": 499},
        "_best_ema": 0.6849,
        "logging_info": ["a" * 1000],
    }


def _write_checkpoint(tmp_path: Path, name: str = "checkpoint_best.pth") -> Path:
    path = tmp_path / name
    torch.save(_mock_full_checkpoint(), str(path))
    return path


# ---------- _prune_checkpoint ----------


def test_prune_keeps_required_keys_and_drops_others(tmp_path: Path) -> None:
    src = _write_checkpoint(tmp_path, "checkpoint_best.pth")
    dst = tmp_path / "pruned" / "checkpoint_best.pth"
    bundler._prune_checkpoint(src, dst, fp16=False)

    payload = torch.load(str(dst), map_location="cpu", weights_only=False)
    assert set(payload.keys()) == {
        "network_weights",
        "trainer_name",
        "init_args",
        "inference_allowed_mirroring_axes",
        "current_epoch",
    }
    # Dropped fields must not appear.
    for dropped in (
        "optimizer_state",
        "grad_scaler_state",
        "lr_scheduler_state",
        "_best_ema",
        "logging_info",
    ):
        assert dropped not in payload
    assert payload["trainer_name"] == "IslesTrainer"
    assert payload["init_args"]["configuration"] == "3d_fullres"


def test_prune_fp16_casts_only_floats(tmp_path: Path) -> None:
    src = _write_checkpoint(tmp_path, "checkpoint_best.pth")
    dst = tmp_path / "pruned_fp16" / "checkpoint_best.pth"
    bundler._prune_checkpoint(src, dst, fp16=True)
    payload = torch.load(str(dst), map_location="cpu", weights_only=False)
    weights = payload["network_weights"]
    # All float tensors are fp16 now.
    assert weights["encoder.stages.0.conv.weight"].dtype == torch.float16
    assert weights["encoder.stages.0.conv.bias"].dtype == torch.float16
    assert weights["encoder.stages.0.norm.weight"].dtype == torch.float16
    # Non-float buffer index is untouched.
    assert weights["num_batches_tracked"].dtype == torch.int64
    assert int(weights["num_batches_tracked"]) == 0


def test_prune_fp16_significantly_reduces_size(tmp_path: Path) -> None:
    src = _write_checkpoint(tmp_path, "checkpoint_best.pth")
    dst32 = tmp_path / "fp32" / "checkpoint_best.pth"
    dst16 = tmp_path / "fp16" / "checkpoint_best.pth"
    bundler._prune_checkpoint(src, dst32, fp16=False)
    bundler._prune_checkpoint(src, dst16, fp16=True)
    fp32_bytes = dst32.stat().st_size
    fp16_bytes = dst16.stat().st_size
    # Not exactly half (torch.save has metadata overhead), but must be
    # noticeably smaller - treat 30 % reduction as the floor.
    assert fp16_bytes < 0.7 * fp32_bytes, (fp16_bytes, fp32_bytes)


def test_prune_rejects_payload_missing_required_key(tmp_path: Path) -> None:
    payload = _mock_full_checkpoint()
    del payload["trainer_name"]
    src = tmp_path / "bad.pth"
    torch.save(payload, str(src))
    with pytest.raises(KeyError, match="trainer_name"):
        bundler._prune_checkpoint(src, tmp_path / "out.pth", fp16=False)


def test_prune_rejects_non_dict_payload(tmp_path: Path) -> None:
    src = tmp_path / "bad.pth"
    torch.save([1, 2, 3], str(src))
    with pytest.raises(TypeError, match="unexpected checkpoint payload"):
        bundler._prune_checkpoint(src, tmp_path / "out.pth", fp16=False)


# ---------- _bundle_experiment ----------


def _seed_experiment_layout(root: Path, exp_name: str, dataset: str, *, ckpt_name: str) -> Path:
    exp_dir = root / dataset / exp_name
    exp_dir.mkdir(parents=True)
    for name in ("plans.json", "dataset.json", "dataset_fingerprint.json"):
        (exp_dir / name).write_text(json.dumps({"note": name}))
    for fold in range(5):
        fold_dir = exp_dir / f"fold_{fold}"
        fold_dir.mkdir()
        torch.save(_mock_full_checkpoint(), str(fold_dir / ckpt_name))
    return exp_dir


def test_bundle_experiment_produces_expected_layout(tmp_path: Path) -> None:
    src_root = tmp_path / "src"
    out_root = tmp_path / "out"
    dataset = "Dataset510_AtlasV2_V2"
    _seed_experiment_layout(
        src_root, "hpcv7_bucketweighted_v2_dicetopk_500ep", dataset, ckpt_name="checkpoint_best.pth"
    )

    exp = {"name": "hpcv7_bucketweighted_v2_dicetopk_500ep", "checkpoint_name": "checkpoint_best.pth"}
    in_b, out_b, warns = bundler._bundle_experiment(
        exp, dataset, src_root, out_root, fp16=False, dry_run=False
    )
    assert in_b > 0
    assert out_b > 0
    assert warns == []

    out_exp = out_root / "checkpoints" / "hpcv7_bucketweighted_v2_dicetopk_500ep"
    for name in ("plans.json", "dataset.json", "dataset_fingerprint.json"):
        assert (out_exp / name).is_file(), f"missing {name}"
    for fold in range(5):
        assert (out_exp / f"fold_{fold}" / "checkpoint_best.pth").is_file()


def test_bundle_experiment_uses_manifest_checkpoint_name(tmp_path: Path) -> None:
    """SWAD experiment ships ``swa.pth`` - bundler must not fall back to ``checkpoint_best.pth``."""
    src_root = tmp_path / "src"
    out_root = tmp_path / "out"
    dataset = "Dataset510_AtlasV2_V2"
    exp_name = "hpcv8_bucketweighted_swa_v2_dicetopk_700ep"
    _seed_experiment_layout(src_root, exp_name, dataset, ckpt_name="swa.pth")

    exp = {"name": exp_name, "checkpoint_name": "swa.pth"}
    bundler._bundle_experiment(exp, dataset, src_root, out_root, fp16=False, dry_run=False)
    for fold in range(5):
        assert (out_root / "checkpoints" / exp_name / f"fold_{fold}" / "swa.pth").is_file()
        assert not (out_root / "checkpoints" / exp_name / f"fold_{fold}" / "checkpoint_best.pth").exists()


def test_bundle_experiment_fails_on_missing_metadata(tmp_path: Path) -> None:
    src_root = tmp_path / "src"
    dataset = "Dataset510_AtlasV2_V2"
    exp_name = "hpcv7_bucketweighted_v2_dicetopk_500ep"
    exp_dir = _seed_experiment_layout(src_root, exp_name, dataset, ckpt_name="checkpoint_best.pth")
    (exp_dir / "plans.json").unlink()  # simulate missing metadata

    exp = {"name": exp_name, "checkpoint_name": "checkpoint_best.pth"}
    with pytest.raises(FileNotFoundError, match="plans.json"):
        bundler._bundle_experiment(exp, dataset, src_root, tmp_path / "out", fp16=False, dry_run=False)


def test_bundle_experiment_fails_on_missing_fold(tmp_path: Path) -> None:
    src_root = tmp_path / "src"
    dataset = "Dataset510_AtlasV2_V2"
    exp_name = "hpcv7_bucketweighted_v2_dicetopk_500ep"
    exp_dir = _seed_experiment_layout(src_root, exp_name, dataset, ckpt_name="checkpoint_best.pth")
    (exp_dir / "fold_3" / "checkpoint_best.pth").unlink()

    exp = {"name": exp_name, "checkpoint_name": "checkpoint_best.pth"}
    with pytest.raises(FileNotFoundError, match="fold_3"):
        bundler._bundle_experiment(exp, dataset, src_root, tmp_path / "out", fp16=False, dry_run=False)


def test_bundle_experiment_dry_run_writes_nothing(tmp_path: Path) -> None:
    src_root = tmp_path / "src"
    out_root = tmp_path / "out"
    dataset = "Dataset510_AtlasV2_V2"
    _seed_experiment_layout(
        src_root, "hpcv7_bucketweighted_v2_dicetopk_500ep", dataset, ckpt_name="checkpoint_best.pth"
    )
    exp = {"name": "hpcv7_bucketweighted_v2_dicetopk_500ep", "checkpoint_name": "checkpoint_best.pth"}
    in_b, out_b, _warns = bundler._bundle_experiment(
        exp, dataset, src_root, out_root, fp16=False, dry_run=True
    )
    assert in_b > 0
    assert out_b > 0
    assert not out_root.exists()
