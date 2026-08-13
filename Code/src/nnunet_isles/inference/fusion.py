"""Detection-oriented ensemble fusion + never-empty rescue (Pillar 1).

An oracle probe showed that ~54% of missed small lesions carry
recoverable signal (max fg-prob > 0.05, 30% already > 0.4) that dies under
mean-softmax @ 0.5. These fusion rules preserve a lesion that fires strongly in
a *minority* of members, and the never-empty rescue guarantees a non-empty mask
(chronic-stroke GT is never empty, so an empty prediction is always wrong and
torches Dice / lesion-F1 / count-F1 / HD95 at once).

Pure functions over foreground-probability maps `(prob_fg,)` or a stack
`(M, *spatial)` of M members. Reuses `cc_postproc.apply_cc_filter` for size
gating. No I/O, no globals - unit-tested in Code/tests/test_fusion.py.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label as _ndi_label

from nnunet_isles.inference.cc_postproc import apply_cc_filter


def _stack(probs) -> np.ndarray:
    arr = np.asarray(probs, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError("expected a stack (M, *spatial) of >=1 member prob maps")
    return arr


def mean_prob(probs) -> np.ndarray:
    """Plain mean-softmax fusion (the current baseline)."""
    return _stack(probs).mean(axis=0)


def noisy_or(probs) -> np.ndarray:
    """Noisy-OR fusion: p = 1 - prod_m (1 - p_m).

    A lesion firing high in even one member survives - the opposite of mean's
    dilution. Raises the whole probability field, so pair with a higher decision
    threshold and/or CC gating to control false positives.
    """
    s = _stack(probs)
    return 1.0 - np.prod(1.0 - np.clip(s, 0.0, 1.0), axis=0)


def k_of_n_mask(probs, k: int, member_threshold: float = 0.5) -> np.ndarray:
    """Binary mask where >= k of N members individually exceed `member_threshold`.

    A robust middle ground between mean (k=N/2-ish behaviour) and noisy-OR (k=1):
    tolerates a couple of confident members without accepting singletons.
    """
    s = _stack(probs)
    votes = (s > member_threshold).sum(axis=0)
    return (votes >= int(k)).astype(np.uint8)


def threshold_mask(prob_fg: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(prob_fg) > threshold).astype(np.uint8)


def _cc_structure(ndim: int, connectivity: int) -> np.ndarray:
    """Structuring element matching ``cc_postproc.apply_cc_filter`` semantics."""
    if ndim == 3:
        if connectivity == 26:
            return np.ones((3, 3, 3), dtype=bool)
        if connectivity == 18:
            struct = np.ones((3, 3, 3), dtype=bool)
            for corner in [
                (0, 0, 0),
                (0, 0, 2),
                (0, 2, 0),
                (2, 0, 0),
                (0, 2, 2),
                (2, 0, 2),
                (2, 2, 0),
                (2, 2, 2),
            ]:
                struct[corner] = False
            return struct
        if connectivity == 6:
            struct = np.zeros((3, 3, 3), dtype=bool)
            struct[1, 1, :] = True
            struct[1, :, 1] = True
            struct[:, 1, 1] = True
            return struct
        raise ValueError(f"connectivity must be 6, 18 or 26 for 3D; got {connectivity}")
    # 2D (or other-dim) fallback - full neighbourhood.
    return np.ones((3,) * ndim, dtype=bool)


def largest_prob_component(prob_fg: np.ndarray, min_prob: float, connectivity: int = 26) -> np.ndarray:
    """Highest-probability-mass connected component among voxels > `min_prob`.

    Returns an all-zero mask only if no voxel exceeds `min_prob`. The
    ``connectivity`` argument (6 / 18 / 26 in 3D) controls the structuring
    element passed to :func:`scipy.ndimage.label` - semantics mirror those of
    :func:`cc_postproc.apply_cc_filter`.
    """
    prob_arr = np.asarray(prob_fg)
    cand = (prob_arr > min_prob).astype(np.uint8)
    if cand.sum() == 0:
        return np.zeros_like(cand)
    struct = _cc_structure(cand.ndim, connectivity)
    lbl, n = _ndi_label(cand, structure=struct)
    if n <= 1:
        return (lbl > 0).astype(np.uint8)
    best_i, best_mass = 1, -1.0
    for i in range(1, n + 1):
        mass = float(prob_arr[lbl == i].sum())
        if mass > best_mass:
            best_mass, best_i = mass, i
    return (lbl == best_i).astype(np.uint8)


def never_empty_mask(
    prob_fg: np.ndarray,
    threshold: float = 0.5,
    min_voxels: int = 0,
    rescue_min_prob: float = 0.10,
    connectivity: int = 26,
) -> np.ndarray:
    """Threshold + optional CC-size filter, but never return an all-zero mask.

    If the thresholded (and size-filtered) mask is empty, recover the single
    highest-probability-mass component among voxels > `rescue_min_prob`; if even
    that is empty, keep the single argmax voxel. Because chronic-stroke GT is
    always non-empty, this is pure upside on the catastrophic near-0 cases.
    """
    prob_fg = np.asarray(prob_fg, dtype=np.float32)
    mask = threshold_mask(prob_fg, threshold)
    if min_voxels > 1 and mask.sum() > 0:
        mask = apply_cc_filter(mask, min_voxels, connectivity=connectivity)
    if mask.sum() > 0:
        return mask.astype(np.uint8)

    rescued = largest_prob_component(prob_fg, rescue_min_prob, connectivity=connectivity)
    if rescued.sum() > 0:
        return rescued
    out = np.zeros_like(prob_fg, dtype=np.uint8)
    out[np.unravel_index(int(np.argmax(prob_fg)), prob_fg.shape)] = 1
    return out


def apply_decision_layer(
    probs: np.ndarray,
    *,
    mode: str = "mean",
    threshold: float = 0.5,
    k: int | None = None,
    member_threshold: float = 0.5,
    never_empty: bool = False,
    rescue_min_prob: float = 0.10,
    min_voxels: int = 0,
    connectivity: int = 26,
) -> np.ndarray:
    """Compose fusion + threshold/CC + optional never-empty into one final binary mask.

    `probs` is either a single fg-probability map `(*spatial)` or a member stack
    `(M, *spatial)`. `mode` in {"mean", "noisy_or", "k_of_n"}; single-map input is
    treated as its own fg field regardless of mode. All parameters must be chosen on
    leak-free OOF (never tuned on the test holdout) - except `never_empty`, which is a
    parameter-free domain rule (chronic-stroke GT is never empty).
    """
    arr = np.asarray(probs, dtype=np.float32)
    is_stack = arr.ndim == 4

    if mode == "k_of_n":
        if not is_stack:
            raise ValueError("k_of_n requires a member stack (M, *spatial)")
        kk = k if k is not None else max(1, int(np.ceil(0.5 * arr.shape[0])))
        mask = k_of_n_mask(arr, kk, member_threshold)
        if min_voxels > 1 and mask.sum() > 0:
            mask = apply_cc_filter(mask, min_voxels, connectivity=connectivity)
        if never_empty and mask.sum() == 0:
            mask = never_empty_mask(mean_prob(arr), threshold, min_voxels, rescue_min_prob, connectivity)
        return mask.astype(np.uint8)

    fg = (noisy_or(arr) if mode == "noisy_or" else mean_prob(arr)) if is_stack else arr
    if never_empty:
        return never_empty_mask(fg, threshold, min_voxels, rescue_min_prob, connectivity)
    mask = threshold_mask(fg, threshold)
    if min_voxels > 1 and mask.sum() > 0:
        mask = apply_cc_filter(mask, min_voxels, connectivity=connectivity)
    return mask.astype(np.uint8)


def best_per_case_threshold(
    prob_fg: np.ndarray, gt: np.ndarray, candidates: np.ndarray | None = None
) -> tuple[float, float]:
    """Oracle per-case threshold (uses GT - for ceiling analysis only, never at test time).

    Returns (best_threshold, best_dice).
    """
    from nnunet_isles.inference.threshold_tuner import dice_at

    if candidates is None:
        candidates = np.round(np.arange(0.05, 0.96, 0.05), 3)
    best_t, best_d = 0.5, -1.0
    for t in candidates:
        d = dice_at(prob_fg, gt, float(t))
        if d > best_d:
            best_d, best_t = d, float(t)
    return best_t, best_d


def mean_prob_weighted(
    probs: np.ndarray,
    weights: list[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Weighted mean of a stack ``(M, *spatial)``.

    Each member is clipped to ``[0, 1]`` BEFORE weighting, mirroring the safety
    net that ``noisy_or`` already applied but that plain ``mean_prob`` did not
    (see the non-clip inconsistency in ``mean_prob``). Weights are normalised
    to sum to 1, so ``weights=None`` (uniform) reduces exactly to the arithmetic
    mean of the clipped members.

    Raises
    ------
    ValueError
        If any weight is negative, if the weight vector's length does not match
        the number of members, or if the weights sum to zero (degenerate).
    """
    arr = _stack(probs)
    # In-place clip: for large (M, Z, Y, X) inputs (e.g. 8×512×512×325 fp32 = 2.7 GB)
    # a `arr = np.clip(...)` allocation OOMs on a ~15 GB host. The clip is
    # idempotent so mutating the input is safe - downstream operations here only
    # read from arr, and if the caller passes the same array to another fusion op
    # it will already be [0,1]-clipped (no observable side effect).
    np.clip(arr, 0.0, 1.0, out=arr)
    n_members = arr.shape[0]
    if weights is None:
        return arr.mean(axis=0)

    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.shape[0] != n_members:
        raise ValueError(f"weights must be a 1-D vector of length M={n_members}; got shape {w.shape}")
    if np.any(w < 0.0):
        raise ValueError(f"weights must all be >= 0; got {list(map(float, w))}")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError(f"weights must sum to > 0; got {list(map(float, w))}")
    w = (w / total).astype(np.float32)
    # tensordot contracts the member axis and returns (*spatial). Cast weights
    # to float32 BEFORE the contraction - a float64 w promotes the entire
    # (Z, Y, X) output to float64, which for a high-res case (512×512×325 =
    # 85 M voxels) allocates ~5 GB and OOMs on a ~15 GB host. With w
    # already float32 the output stays float32 (~2.5 GB peak).
    return np.tensordot(w, arr, axes=1)


def apply_policy(
    probs: np.ndarray,
    policy,  # DecisionPolicy - imported lazily inside to avoid a circular import
) -> np.ndarray:
    """Compose weighted fusion + adaptive threshold + FP cleanup + never-empty rescue.

    ``probs`` is either a single fg-probability map ``(*spatial)`` or a member
    stack ``(M, *spatial)``. Returns a ``uint8`` binary mask.

    Pipeline:

      1. Fuse the stack according to ``policy.mode``:

         * ``'mean'`` - :func:`mean_prob_weighted` with ``policy.weights``.
         * ``'noisy_or'`` - :func:`noisy_or` (unweighted; a lesion firing
           strongly in any member survives).
         * ``'k_of_n'`` - :func:`k_of_n_mask` returns a binary mask directly.
           A weighted-mean fg field is still computed from the same stack via
           :func:`mean_prob_weighted` with ``policy.weights`` so that:

             (a) the adaptive bucket (``min_voxels``) can be resolved via
                 :meth:`DecisionPolicy.pick_threshold_for_case` - only the
                 bucket-selected ``min_voxels`` and per-CC gates apply here;
                 the bucket's ``threshold`` is unused because the k_of_n mask
                 is not derived from a fused fg threshold;
             (b) the same :func:`drop_low_confidence_ccs` gates
                 (``min_max_prob`` / ``min_mean_prob`` / ``min_prob_mass`` +
                 ``min_voxels``) that ``mean`` / ``noisy_or`` apply also
                 apply to the k_of_n mask, using the weighted-mean fg as the
                 probability field;
             (c) any never-empty rescue in step 5 uses that same weighted
                 mean fg field.

         A single-map input is treated as its own fg field regardless of mode.

      2. Nominal-pass binarise at threshold 0.5 (inside
         :meth:`DecisionPolicy.pick_threshold_for_case`) to estimate the
         predicted lesion volume.
      3. Bucket the predicted volume and read the adaptive
         ``(threshold, min_voxels, bucket_label)`` triple from the policy.
         For ``k_of_n`` only ``min_voxels`` (and the per-CC gates) are used;
         the bucket threshold is a no-op there.
      4. Re-binarise ``fg > threshold`` (for ``mean`` / ``noisy_or``; skipped
         for ``k_of_n``) and apply :func:`drop_low_confidence_ccs` with the
         policy's per-CC gates.
      5. If ``policy.never_empty`` and the surviving mask is empty, replace
         it with :func:`never_empty_mask` on the fg field (weighted mean for
         ``k_of_n``).
      6. Return the mask as ``uint8``.
    """
    # Lazy import to avoid a policy <-> fusion circular import.
    from nnunet_isles.inference.cc_stats import drop_low_confidence_ccs
    from nnunet_isles.inference.policy import DecisionPolicy  # noqa: F401

    arr = np.asarray(probs, dtype=np.float32)
    is_stack = arr.ndim == 4  # (M, *spatial-3D); single map is 3D

    # ------------------------------------------------------------------ k_of_n short-circuit
    if is_stack and policy.mode == "k_of_n":
        kk = policy.k if policy.k is not None else max(1, int(np.ceil(0.5 * arr.shape[0])))
        mask = k_of_n_mask(arr, kk, policy.member_threshold)
        # Build the fused fg once so the same field powers (a) bucket selection,
        # (b) per-CC low-confidence gating, and (c) any never-empty rescue.
        fg_for_gates = mean_prob_weighted(arr, policy.weights)
        # For k_of_n the bucket threshold is not used to re-binarise (the mask
        # is already binary); only the bucket-selected min_voxels is applied.
        _threshold, min_vox, _ = policy.pick_threshold_for_case(fg_for_gates)
        mask = drop_low_confidence_ccs(
            fg_for_gates,
            mask,
            min_max_prob=policy.min_max_prob,
            min_mean_prob=policy.min_mean_prob,
            min_prob_mass=policy.min_prob_mass,
            min_voxels=min_vox,
            connectivity=policy.connectivity,
        )
        if policy.never_empty and int(mask.sum()) == 0:
            mask = never_empty_mask(
                fg_for_gates,
                _threshold,
                min_vox,
                policy.rescue_min_prob,
                policy.connectivity,
            )
        return mask.astype(np.uint8)

    # ------------------------------------------------------------------ fuse -> fg
    if is_stack:
        fg = noisy_or(arr) if policy.mode == "noisy_or" else mean_prob_weighted(arr, policy.weights)
    else:
        fg = arr

    # ------------------------------------------------------------------ adaptive threshold + FP cleanup
    threshold, min_voxels, _ = policy.pick_threshold_for_case(fg)
    mask = threshold_mask(fg, threshold)
    mask = drop_low_confidence_ccs(
        fg,
        mask,
        min_max_prob=policy.min_max_prob,
        min_mean_prob=policy.min_mean_prob,
        min_prob_mass=policy.min_prob_mass,
        min_voxels=min_voxels,
        connectivity=policy.connectivity,
    )

    # ------------------------------------------------------------------ never-empty rescue
    if policy.never_empty and int(mask.sum()) == 0:
        mask = never_empty_mask(
            fg,
            threshold,
            min_voxels,
            policy.rescue_min_prob,
            policy.connectivity,
        )

    return mask.astype(np.uint8)


__all__ = [
    "mean_prob",
    "mean_prob_weighted",
    "noisy_or",
    "k_of_n_mask",
    "threshold_mask",
    "largest_prob_component",
    "never_empty_mask",
    "apply_decision_layer",
    "apply_policy",
    "best_per_case_threshold",
]
