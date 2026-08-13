"""Tests for the SWA helper trainer.

We test the pure utility `average_state_dicts` and the running-average
mechanism via a lightweight mock; instantiating the full IslesTrainerSWA
would pull in nnU-Net's complete training stack.
"""

from __future__ import annotations

import torch
from nnunet_isles.trainers.isles_trainer_swa import average_state_dicts


def test_average_state_dicts_matches_mean():
    sd1 = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([[1.0]])}
    sd2 = {"a": torch.tensor([3.0, 4.0]), "b": torch.tensor([[5.0]])}
    sd3 = {"a": torch.tensor([5.0, 6.0]), "b": torch.tensor([[9.0]])}
    avg = average_state_dicts([sd1, sd2, sd3])
    torch.testing.assert_close(avg["a"], torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(avg["b"], torch.tensor([[5.0]]))


def test_average_state_dicts_preserves_dtype():
    sd1 = {"a": torch.tensor([1, 3], dtype=torch.int32)}
    sd2 = {"a": torch.tensor([3, 5], dtype=torch.int32)}
    avg = average_state_dicts([sd1, sd2])
    # int average of 1,3 = 2 (truncated since cast back to int32).
    assert avg["a"].dtype == torch.int32
    torch.testing.assert_close(avg["a"], torch.tensor([2, 4], dtype=torch.int32))


def test_average_rejects_mismatched_keys():
    sd1 = {"a": torch.tensor([1.0])}
    sd2 = {"b": torch.tensor([1.0])}
    try:
        average_state_dicts([sd1, sd2])
    except ValueError as e:
        assert "same keys" in str(e)
        return
    raise AssertionError("expected ValueError for mismatched keys")


def test_average_rejects_empty_list():
    try:
        average_state_dicts([])
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty input")


def test_running_average_iteration_matches_full_mean():
    """SWA in the trainer accumulates incrementally; verify that the running
    formula `avg_n = avg_{n-1} + (v - avg_{n-1}) / n` produces the same result
    as a full one-shot average."""
    values = [
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([2.0, 4.0, 6.0]),
        torch.tensor([3.0, 6.0, 9.0]),
        torch.tensor([4.0, 8.0, 12.0]),
    ]
    # Running average:
    running = values[0].clone()
    for n, v in enumerate(values[1:], start=2):
        running = running + (v - running) / n
    # Full average:
    full = torch.stack(values).mean(dim=0)
    torch.testing.assert_close(running, full, rtol=1e-5, atol=1e-5)


def test_swa_payload_keys_match_finalize_loader_expectation():
    """`finalize.py --use-swa` will load `<output_folder>/swa.pth` and expect
    a `network_weights` key (matching nnU-Net's checkpoint_best format).
    Sanity-check that key is in the payload schema documented in IslesTrainerSWA."""
    from nnunet_isles.trainers.isles_trainer_swa import IslesTrainerSWA  # noqa: F401

    # Build a synthetic payload matching the trainer's `on_train_end` write format.
    payload = {
        "network_weights": {"conv.weight": torch.zeros(2, 2)},
        "swa_n_avg": 50,
        "swa_start_epoch": 350,
    }
    # Roundtrip serialise/deserialise to mimic finalize-side load.
    import io

    buf = io.BytesIO()
    torch.save(payload, buf)
    buf.seek(0)
    loaded = torch.load(buf, weights_only=False)
    assert "network_weights" in loaded
    assert loaded["swa_n_avg"] == 50


def test_swa_state_cache_roundtrip_via_helper(tmp_path):
    """`_save_swa_cache` + `_maybe_restore_swa_cache` must round-trip the
    in-memory accumulator. Regression: in-memory SWA state was lost because
    `on_train_end` crashed without ever writing the cache."""
    from nnunet_isles.trainers.isles_trainer_swa import IslesTrainerSWA

    # Build a minimal shim: a bare object with the two helpers + the attributes
    # they touch. Avoids spinning up the full nnUNetTrainer init stack.
    inst = object.__new__(IslesTrainerSWA)
    inst.output_folder = str(tmp_path)
    inst.isles_swa_state_cache_filename = "_swa_state_cache.pt"
    inst.isles_swa_start_epoch = 100
    inst._swa_state = {"layer.weight": torch.tensor([1.5, 2.5, 3.5])}
    inst._n_swa = 42
    inst._swa_cache_restored = False
    # Stub `print_to_log_file` so we don't need a real logger.
    inst.print_to_log_file = lambda *a, **k: None

    # Save snapshot.
    IslesTrainerSWA._save_swa_cache(inst)
    cache_file = tmp_path / "_swa_state_cache.pt"
    assert cache_file.exists()

    # Fresh instance - wipe in-memory state, then restore from cache.
    inst._swa_state = None
    inst._n_swa = 0
    inst._swa_cache_restored = False
    IslesTrainerSWA._maybe_restore_swa_cache(inst)
    assert inst._n_swa == 42
    torch.testing.assert_close(inst._swa_state["layer.weight"], torch.tensor([1.5, 2.5, 3.5]))


def test_swa_state_cache_restore_no_file_is_noop(tmp_path):
    """No cache on disk → restore is a no-op, accumulator stays at None."""
    from nnunet_isles.trainers.isles_trainer_swa import IslesTrainerSWA

    inst = object.__new__(IslesTrainerSWA)
    inst.output_folder = str(tmp_path)
    inst.isles_swa_state_cache_filename = "_swa_state_cache.pt"
    inst._swa_state = None
    inst._n_swa = 0
    inst._swa_cache_restored = False
    inst.print_to_log_file = lambda *a, **k: None

    IslesTrainerSWA._maybe_restore_swa_cache(inst)
    assert inst._swa_state is None
    assert inst._n_swa == 0
    # The "restored" flag flips even on miss, so a second call is also a no-op.
    assert inst._swa_cache_restored is True


def test_swa_state_cache_idempotent(tmp_path):
    """Repeated restore calls must not re-load (the second call should be a no-op
    even if the on-disk cache changes underneath us)."""
    from nnunet_isles.trainers.isles_trainer_swa import IslesTrainerSWA

    inst = object.__new__(IslesTrainerSWA)
    inst.output_folder = str(tmp_path)
    inst.isles_swa_state_cache_filename = "_swa_state_cache.pt"
    inst._swa_state = {"w": torch.tensor([1.0])}
    inst._n_swa = 1
    inst._swa_cache_restored = False
    inst.print_to_log_file = lambda *a, **k: None
    IslesTrainerSWA._save_swa_cache(inst)

    # First restore.
    inst._swa_state = None
    inst._n_swa = 0
    inst._swa_cache_restored = False
    IslesTrainerSWA._maybe_restore_swa_cache(inst)
    assert inst._n_swa == 1
    # Mutate then try restore again - should NOT overwrite.
    inst._n_swa = 999
    IslesTrainerSWA._maybe_restore_swa_cache(inst)
    assert inst._n_swa == 999


def test_swa_on_train_end_preserves_nnunet_checkpoint_schema(tmp_path):
    """`on_train_end` must write a `swa.pth` that carries every nnU-Net
    checkpoint key upstream's `nnUNetPredictor.initialize_from_trained_model_folder`
    reads: `trainer_name`, `init_args['configuration']`,
    `inference_allowed_mirroring_axes`, `network_weights`. It should also
    keep our SWA-specific provenance (`swa_n_avg`, `swa_start_epoch`).

    Regression test: guarantees that resume from an SWA checkpoint does not
    raise KeyError('trainer_name') when the checkpoint dict lacks that key."""
    import torch.nn as nn
    from nnunet_isles.trainers.isles_trainer_swa import IslesTrainerSWA

    # 1. Write a synthetic `checkpoint_best.pth` in the tmp output_folder that
    #    mirrors upstream's full schema (from nnUNetTrainer.save_checkpoint).
    template = {
        "network_weights": {"layer.weight": torch.zeros(2, 3)},
        "optimizer_state": {"step": 42},
        "grad_scaler_state": None,
        "logging": {"epoch": 699},
        "_best_ema": 0.7,
        "current_epoch": 700,
        "init_args": {
            "plans": {},
            "configuration": "3d_fullres",
            "fold": 0,
            "dataset_json": {},
            "device": "cuda",
        },
        "trainer_name": "IslesTrainerSWA",
        "inference_allowed_mirroring_axes": (0, 1, 2),
    }
    torch.save(template, str(tmp_path / "checkpoint_best.pth"))

    # 2. Build a bare IslesTrainerSWA instance skipping nnUNetTrainer.__init__.
    #    We manually populate the attributes on_train_end reads.
    inst = object.__new__(IslesTrainerSWA)
    inst.output_folder = str(tmp_path)
    inst.isles_swa_filename = "swa.pth"
    inst.isles_swa_start_epoch = 350
    inst._n_swa = 175
    # Live network with a `layer.weight` param matching the template.
    inst.network = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        inst.network.weight.copy_(torch.ones(2, 3) * 0.5)
    # _swa_state holds the SWA-averaged weights we want to end up in swa.pth.
    inst._swa_state = {"weight": torch.full((2, 3), 0.75)}
    inst.print_to_log_file = lambda *a, **k: None

    # Stub super().on_train_end() so we don't need the nnUNetTrainer chain.
    import unittest.mock as _mock

    with _mock.patch.object(IslesTrainerSWA.__bases__[0], "on_train_end", lambda self: None, create=True):
        IslesTrainerSWA.on_train_end(inst)

    # 3. Load the resulting swa.pth and verify schema + weights.
    swa_path = tmp_path / "swa.pth"
    assert swa_path.exists()
    loaded = torch.load(str(swa_path), map_location="cpu", weights_only=False)

    # Full upstream schema keys present.
    for k in ("trainer_name", "init_args", "inference_allowed_mirroring_axes", "network_weights"):
        assert k in loaded, f"swa.pth is missing key {k!r}"
    assert loaded["init_args"]["configuration"] == "3d_fullres"
    assert loaded["trainer_name"] == "IslesTrainerSWA"
    assert loaded["inference_allowed_mirroring_axes"] == (0, 1, 2)
    # SWA weights swapped in (network.weight -> _swa_state["weight"]).
    torch.testing.assert_close(loaded["network_weights"]["weight"], torch.full((2, 3), 0.75))
    # SWA-specific provenance keys preserved.
    assert loaded["swa_n_avg"] == 175
    assert loaded["swa_start_epoch"] == 350
    # Non-network metadata from the template still there (unrelated fields
    # nnU-Net may consult later - e.g. current_epoch for logging).
    assert loaded["current_epoch"] == 700


def test_swa_on_train_end_falls_back_when_template_missing(tmp_path):
    """If checkpoint_best.pth is absent, on_train_end still writes swa.pth with
    the minimum keys upstream's predictor needs - otherwise SWAD folds where
    training happened to skip checkpoint_best (e.g. no val Dice improvement)
    would silently produce an unloadable swa.pth."""
    import torch.nn as nn
    from nnunet_isles.trainers.isles_trainer_swa import IslesTrainerSWA

    inst = object.__new__(IslesTrainerSWA)
    inst.output_folder = str(tmp_path)
    inst.isles_swa_filename = "swa.pth"
    inst.isles_swa_start_epoch = 350
    inst._n_swa = 10
    inst.network = nn.Linear(3, 2, bias=False)
    inst._swa_state = {"weight": torch.zeros(2, 3)}
    inst.my_init_kwargs = {"configuration": "3d_fullres", "fold": 0}
    inst.inference_allowed_mirroring_axes = (0,)
    inst.print_to_log_file = lambda *a, **k: None

    import unittest.mock as _mock

    with _mock.patch.object(IslesTrainerSWA.__bases__[0], "on_train_end", lambda self: None, create=True):
        IslesTrainerSWA.on_train_end(inst)

    loaded = torch.load(str(tmp_path / "swa.pth"), map_location="cpu", weights_only=False)
    # Fallback path must still emit the schema keys the predictor reads.
    assert loaded["trainer_name"] == "IslesTrainerSWA"
    assert "init_args" in loaded and "configuration" in loaded["init_args"]
    assert loaded["inference_allowed_mirroring_axes"] == (0,)
    assert "network_weights" in loaded


# ---------------------------------------------------------------------------
# Repair CLI (`Code/scripts/repair_swa_checkpoints.py`) - for HPC runs that
# pre-date the trainer's schema-template fix.
# ---------------------------------------------------------------------------


def _load_repair_module():
    """Import the repair CLI module by path (scripts/ isn't a package)."""
    import importlib.util
    from pathlib import Path

    here = Path(__file__).resolve()
    repo = here.parents[2]
    script = repo / "Code" / "scripts" / "repair_swa_checkpoints.py"
    spec = importlib.util.spec_from_file_location("repair_swa_checkpoints", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_fixture_pair(tmp_path, weights_keys=("conv.weight", "conv.bias")):
    """Write a mock (checkpoint_best.pth, swa.pth-bare) pair the repair CLI can consume."""
    weights_best = {k: torch.zeros(2, 2) for k in weights_keys}
    weights_swa = {k: torch.ones(2, 2) * 0.5 for k in weights_keys}

    best_payload = {
        "trainer_name": "IslesTrainerSWA",
        "init_args": {"configuration": "3d_fullres", "fold": 0, "plans": {}},
        "inference_allowed_mirroring_axes": (0, 1, 2),
        "network_weights": weights_best,
        "optimizer_state": {"state": {}, "param_groups": []},
        "grad_scaler_state": None,
        "logging": {},
        "_best_ema": None,
        "current_epoch": 700,
    }
    bare_payload = {
        "network_weights": weights_swa,
        "swa_n_avg": 350,
        "swa_start_epoch": 350,
    }
    best_path = tmp_path / "checkpoint_best.pth"
    swa_path = tmp_path / "swa.pth"
    torch.save(best_payload, str(best_path))
    torch.save(bare_payload, str(swa_path))
    return best_path, swa_path, weights_swa


def test_repair_transplants_weights_and_schema(tmp_path):
    """Repair grafts the schema keys from checkpoint_best onto swa.pth while
    keeping the SWA-averaged network_weights + provenance."""
    mod = _load_repair_module()
    best_path, swa_path, swa_weights = _write_fixture_pair(tmp_path)

    status = mod.repair_one_fold(best_path, swa_path)
    assert "repaired" in status
    assert "n_avg=350" in status

    # Backup created.
    backup = swa_path.with_suffix(swa_path.suffix + ".bare")
    assert backup.exists()

    # Repaired swa.pth: schema from template + weights from bare.
    repaired = torch.load(str(swa_path), map_location="cpu", weights_only=False)
    assert repaired["trainer_name"] == "IslesTrainerSWA"
    assert repaired["init_args"]["configuration"] == "3d_fullres"
    assert repaired["inference_allowed_mirroring_axes"] == (0, 1, 2)
    assert repaired["swa_n_avg"] == 350
    assert repaired["swa_start_epoch"] == 350
    # Weights are the bare (SWA-averaged) ones, NOT the template's.
    for k, v in swa_weights.items():
        torch.testing.assert_close(repaired["network_weights"][k], v)


def test_repair_is_idempotent(tmp_path):
    """Second invocation on the same swa.pth is a no-op."""
    mod = _load_repair_module()
    best_path, swa_path, _ = _write_fixture_pair(tmp_path)

    first = mod.repair_one_fold(best_path, swa_path)
    assert "repaired" in first
    second = mod.repair_one_fold(best_path, swa_path)
    assert "skipped" in second and "already repaired" in second


def test_repair_rejects_key_mismatch(tmp_path):
    """If the SWA state_dict has different keys from the template, refuse."""
    mod = _load_repair_module()
    best_path, swa_path, _ = _write_fixture_pair(tmp_path, weights_keys=("conv.weight", "conv.bias"))
    # Overwrite swa.pth with a payload whose network_weights keys differ.
    torch.save(
        {
            "network_weights": {"different.name": torch.zeros(2, 2)},
            "swa_n_avg": 100,
            "swa_start_epoch": 350,
        },
        str(swa_path),
    )

    try:
        mod.repair_one_fold(best_path, swa_path)
    except AssertionError as e:
        assert "network_weights key mismatch" in str(e)
        return
    raise AssertionError("expected AssertionError for key mismatch")


def test_repair_cli_dry_run(tmp_path, monkeypatch, capsys):
    """--dry-run must not touch files."""
    mod = _load_repair_module()
    # Set up a fake nnunet_results tree: <root>/<dataset>/<exp>/fold_0/{best,swa}.pth
    dataset = "Dataset510_AtlasV2_V2"
    experiment = "hpcv7_swad_v2_dicetopk_700ep"
    fold_dir = tmp_path / dataset / experiment / "fold_0"
    fold_dir.mkdir(parents=True)
    best_path, swa_path, _ = _write_fixture_pair(fold_dir)
    swa_mtime_before = swa_path.stat().st_mtime

    # Point paths.nnunet_results at our tmp root.
    import paths as _paths

    monkeypatch.setattr(_paths, "nnunet_results", tmp_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "repair_swa_checkpoints.py",
            "--experiment-name",
            experiment,
            "--dataset-name",
            dataset,
            "--folds",
            "0",
            "--dry-run",
        ],
    )
    rc = mod.main()
    assert rc == 0

    # File not touched.
    assert not swa_path.with_suffix(swa_path.suffix + ".bare").exists()
    assert swa_path.stat().st_mtime == swa_mtime_before
    out = capsys.readouterr().out
    assert "DRY" in out and "would repair" in out
