"""Output-space (softmax-averaging) ensemble - replaces weight-space model soup,
which can NaN when member LR schedules diverge.

Averages per-case softmax NPZ across experiments. Architectures with different
parameter shapes cannot be weight-averaged, but their outputs all live in the
same nnU-Net plans space and CAN be averaged.

Reads NPZ files written by nnU-Net's `predict_from_files(save_probabilities=True)`
(channel-0 = background, channel-1 = foreground). Per case, computes
`sum_i w_i * softmax_i / sum_i w_i` and emits the combined NPZ + a
binarised NIfTI at the standard threshold (0.5 by default).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def ensemble_one_case(
    npz_paths: list[Path],
    weights: list[float] | None = None,
) -> np.ndarray:
    """Average softmax across experiments for a single case.

    Returns the combined softmax with shape `(C, *spatial)`. The caller binarises.
    """
    if not npz_paths:
        raise ValueError("ensemble_one_case requires at least one input NPZ")
    if weights is None:
        weights = [1.0] * len(npz_paths)
    if len(weights) != len(npz_paths):
        raise ValueError("weights must align 1:1 with npz_paths")

    w_total = float(sum(weights))
    if w_total <= 0:
        raise ValueError(f"sum of weights must be positive; got {weights}")

    combined: np.ndarray | None = None
    for path, w in zip(npz_paths, weights, strict=True):
        with np.load(str(path)) as f:
            prob = f["probabilities"]  # (C, *spatial), float32
        if combined is None:
            combined = (w / w_total) * prob.astype(np.float32)
        else:
            if prob.shape != combined.shape:
                raise ValueError(
                    f"shape mismatch - first input: {combined.shape}, current ({path.name}): {prob.shape}"
                )
            combined = combined + (w / w_total) * prob.astype(np.float32)
    assert combined is not None
    return combined


def find_common_cases(experiment_npz_dirs: list[Path]) -> list[str]:
    """Return the sorted intersection of case stems across all experiment dirs.

    Each dir contains `<case>.npz` files; we expect every experiment to have
    produced softmax for every case in the test set.
    """
    sets: list[set[str]] = []
    for d in experiment_npz_dirs:
        sets.append({p.name.removesuffix(".npz") for p in d.glob("*.npz")})
    if not sets:
        return []
    common = set.intersection(*sets)
    return sorted(common)


def learn_weights_from_val(
    val_npz_dirs: list[Path],
    val_gt_dir: Path,
    case_ids: list[str],
    *,
    n_steps: int = 50,
    lr: float = 0.05,
    val_case_limit: int | None = None,
    val_random_seed: int = 42,
) -> list[float]:
    """Tiny gradient-free weight search on val: maximise mean val Dice.

    Uses a coordinate-ascent style sweep on the simplex `{w >= 0, sum w = 1}`.
    Returns a list of weights aligned with `val_npz_dirs`.

    Loads all softmax NPZs into RAM once (~30 GB working set for the
    11-member ensemble pool) instead of re-reading from network storage
    on each eval step, trading disk I/O for RAM to avoid stalls on
    transient shared-filesystem contention.

    Args:
        val_case_limit: If set (and less than len(case_ids)), randomly
            subsample this many cases before caching. Needed when running
            on OOF val (1356 cases × 12 models × ~57 MB ≈ 780 GB - won't
            fit); a random 100-case subsample brings the cache to ~68 GB
            for 12 models. `val_random_seed` fixes reproducibility.
    """
    import random
    import sys
    import time

    import SimpleITK as sitk

    n_models = len(val_npz_dirs)

    # --- Optional subsample of case_ids (needed for OOF at scale). ---
    if val_case_limit is not None and val_case_limit < len(case_ids):
        rng_local = random.Random(val_random_seed)
        sampled = rng_local.sample(sorted(case_ids), val_case_limit)
        print(
            f"[coord] subsampled {val_case_limit}/{len(case_ids)} val cases "
            f"(seed={val_random_seed}, sorted-input for reproducibility)",
            flush=True,
        )
        case_ids = sampled
    n_cases = len(case_ids)

    # --- Phase 1: cache all softmax + GT arrays in memory. ---
    # `cache[sid] = (list_of_softmax_arrays_per_experiment, gt_bool)`.
    print(
        f"[coord] caching {n_cases} cases × {n_models} experiments = "
        f"{n_cases * n_models} NPZs into RAM (this replaces per-eval disk reads)...",
        flush=True,
    )
    t0 = time.time()
    cache: dict[str, tuple[list[np.ndarray], np.ndarray]] = {}
    for i, sid in enumerate(case_ids):
        probs: list[np.ndarray] = []
        for d in val_npz_dirs:
            npz_path = d / f"{sid}.npz"
            with np.load(str(npz_path)) as f:
                probs.append(f["probabilities"].astype(np.float32))
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(val_gt_dir / f"{sid}.nii.gz"))).astype(np.uint8) > 0
        # Every experiment's softmax must share the same spatial shape; skip
        # if we detect a mismatch so eval_weights doesn't crash later.
        base_shape = probs[0].shape
        if any(p.shape != base_shape for p in probs):
            print(
                f"[coord]   WARN skipping {sid}: mismatched softmax shapes across experiments",
                file=sys.stderr,
                flush=True,
            )
            continue
        cache[sid] = (probs, gt)
        if (i + 1) % max(1, n_cases // 10) == 0 or (i + 1) == n_cases:
            print(f"[coord]   cached {i + 1}/{n_cases} cases", flush=True)
    print(
        f"[coord] cache complete in {time.time() - t0:.1f}s "
        f"({len(cache)} usable cases; ~{n_models * len(cache) * 30 / 1024:.1f} GB working set)",
        flush=True,
    )
    if not cache:
        # No usable val cases - return uniform weights as a safe default.
        return [1.0 / n_models] * n_models
    usable_case_ids = [sid for sid in case_ids if sid in cache]

    # --- Phase 2: coord ascent over the cached tensors. ---
    def eval_weights(w: np.ndarray) -> float:
        w_clip = np.clip(w, 0.0, None)
        if w_clip.sum() == 0:
            return 0.0
        w_norm = w_clip / w_clip.sum()  # sums to 1.0
        scores = []
        for sid in usable_case_ids:
            probs, gt_bool = cache[sid]
            combined = np.zeros_like(probs[0])
            for i, prob in enumerate(probs):
                combined += float(w_norm[i]) * prob
            pred_bool = combined[1] > 0.5
            denom = int(pred_bool.sum()) + int(gt_bool.sum())
            scores.append(2 * int((pred_bool & gt_bool).sum()) / denom if denom > 0 else 1.0)
        return float(np.mean(scores)) if scores else 0.0

    # Start uniform.
    weights = np.ones(n_models, dtype=np.float32) / n_models
    best_w = weights.copy()
    t_step0 = time.time()
    best_score = eval_weights(best_w)
    print(
        f"[coord] baseline (uniform) Dice={best_score:.4f}  eval_time={time.time() - t_step0:.2f}s",
        flush=True,
    )
    rng = np.random.default_rng(42)
    n_accepted = 0
    for step in range(n_steps):
        t_step = time.time()
        idx = int(rng.integers(0, n_models))
        candidate = best_w.copy()
        candidate[idx] += lr * (1.0 - 2.0 * rng.random())
        score = eval_weights(candidate)
        accepted = score > best_score
        if accepted:
            best_w = candidate
            best_score = score
            n_accepted += 1
        # Log every step - 51 lines total for the default n_steps=50 is
        # cheap and gives real-time observability on a workload that used
        # to be silent for 100 min.
        print(
            f"[coord] step {step + 1}/{n_steps}  "
            f"idx={idx:2d}  cand={score:.4f}  best={best_score:.4f}  "
            f"acc={n_accepted}  step_dt={time.time() - t_step:.2f}s  "
            f"{'ACCEPT' if accepted else ''}",
            flush=True,
        )
    # Final normalise to simplex.
    w = np.clip(best_w, 0.0, None)
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / n_models
    print(
        f"[coord] done - accepted {n_accepted}/{n_steps} moves, final best Dice={best_score:.4f}",
        flush=True,
    )
    return w.tolist()


__all__ = ["ensemble_one_case", "find_common_cases", "learn_weights_from_val"]
