"""Plans-space ensemble fusion for giant volumes.

Preprocessing once per case and running fusion in plans space cuts per-case
wall time by ~8x on the same GPU compared with per-member native-space fusion.

The per-member native-space path preprocesses raw -> plans-space, runs the
sliding window on plans-space, inverse-resamples the softmax back to native,
and writes a per-member NPZ. Fusion then loads all NPZs and combines in
native space. On the 6 outlier SOOP volumes (348x1008x1008 native, 353 M raw
voxels), the inverse-resample from plans-space (11 M voxels,
native-shape-independent) to native (353 M voxels x 2 channels x fp32 =
2.83 GB) plus the compressed NPZ write dominates per-member wall-clock:
~60-90 s per member x 8 members ~= 720 s just in resample+I/O.

This module offers a fast path that:

  1. Preprocesses ONCE (all members share nnUNetPlans_iso10 for Pillar-1).
  2. Runs each member's sliding window on the shared plans-space tensor.
  3. Accumulates fused LOGITS in plans-space (streaming, no M-slab stack).
     LOGITS-space fusion is used (not fg-space) because the native-space
     path applies softmax AFTER the inverse resample
     (`convert_predicted_logits_to_segmentation_with_correct_shape` line 40
     in nnU-Net's export_prediction.py). Accumulating logits then softmaxing
     after ONE inverse-resample matches that ordering exactly, modulo the
     change from softmax(inv_resample(logits_m)) across members to
     inv_resample(mean_w(logits_m)); the difference is bounded by softmax's
     non-linearity at prob boundaries, and can be verified via a golden-diff
     before shipping.
  4. Inverse-resamples the fused LOGITS to native ONCE using nnU-Net's own
     `convert_predicted_logits_to_segmentation_with_correct_shape`, so bbox
     reversal, transpose_backward, and softmax nonlinearity all reuse the
     tested upstream code path.

Peak RAM drops from ~24 GB (11.3 GB stack + resample transients) to ~5 GB
(single plans-space logits accumulator ~350 MB + one member logits ~350 MB
+ one final native-space probs after the single inverse resample ~2.83 GB).
"""

from __future__ import annotations

import hashlib
import json
import sys
from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:
    import torch


class MixedPlansError(RuntimeError):
    """Raised when ensemble members disagree on plans metadata - caller should
    fall back to the legacy per-member path."""


def plans_signature(predictor) -> str:
    """Hash of the plans metadata that MUST match across ensemble members for
    plans-space fusion to be arithmetically well-defined. Any mismatch means
    plans-space shapes / resampling / channel ordering differ and the fused
    accumulator can't be reused."""
    cm = predictor.configuration_manager
    pm = predictor.plans_manager
    lm = predictor.label_manager
    # ConfigurationManager doesn't expose a stable name attribute - read the
    # underlying dict directly. The plans_name (from PlansManager) + the
    # resampling function's qualname pin the arithmetic path.
    resamp_fn = cm.resampling_fn_probabilities
    resamp_name = getattr(resamp_fn, "__qualname__", None) or repr(resamp_fn)
    key = {
        "plans_name": pm.plans_name,
        "spacing": [float(x) for x in cm.spacing],
        "patch_size": [int(x) for x in cm.patch_size],
        "transpose_forward": [int(x) for x in pm.transpose_forward],
        "transpose_backward": [int(x) for x in pm.transpose_backward],
        "num_segmentation_heads": int(lm.num_segmentation_heads),
        "resampling_fn_probabilities": resamp_name,
    }
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()


def verify_plans_parity(predictors: Iterable) -> str:
    """Return the common plans_signature across ``predictors`` or raise
    :class:`MixedPlansError`. Result is a plain hash string suitable for logs."""
    sigs = {plans_signature(p) for p in predictors}
    if len(sigs) != 1:
        raise MixedPlansError(
            f"ensemble members disagree on plans: {len(sigs)} distinct signatures"
        )
    return sigs.pop()


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def fuse_in_plans_space(
    predictors: list,
    weights,
    staged_image_files: list[str],
    *,
    autocast_dtype=None,
    mode: str = "logits",
    verbose: bool = True,
) -> tuple[np.ndarray, dict, object]:
    """Fast per-case ensemble inference: preprocess once, sliding-window per
    member, accumulate in plans-space, inverse-resample ONCE.

    Two fusion modes:

    * ``mode="logits"`` (arithmetically DIFFERENT from native-space fusion)
      Accumulates raw logits across members: ``softmax(inv_resample(mean_w(logits_m)))``.
      Non-linear reordering of softmax and member-mean; for heterogeneous
      ensembles (different logit magnitudes across trainers/architectures)
      this over-weights the "loudest" member and can dramatically drop
      Dice / LF1 on the 91-case holdout (~-0.10 vs native-space measured).

    * ``mode="fg"`` (RECOMMENDED, arithmetically equivalent to native-space fg fusion)
      Applies softmax per member in plans-space, accumulates only the FG
      channel, then inverse-resamples the 2-channel [1-fg, fg] via
      ``resampling_fn_probabilities`` + ``revert_cropping_on_probabilities``
      + transpose_backward. Because resample is linear, this equals
      ``resample(mean_w(softmax(logits_m)))`` = ``mean_w(resample(softmax(logits_m)))``
      = per-member ``resample(softmax(logits_m))`` averaged in native.
      Only difference vs the native-space path is the AMORTIZATION of N
      resamples into 1 - pure speed win, no arithmetic change.

    Parameters
    ----------
    predictors
        List of :class:`~nnunet_isles.inference.predictor.IslesPredictor`. All
        must have already been initialised from their model folders AND share
        plans metadata. Order MUST match ``weights``.
    weights
        Per-member policy weights. Sum > 0; non-negative. Will be normalised
        to sum 1 in fp32 before folding.
    staged_image_files
        One-element list containing the path to the staged nnU-Net-style input
        NIfTI (``<case>_0000.nii.gz``). Passed straight to the preprocessor.
    autocast_dtype
        ``torch.dtype | None`` - when set, wrap each per-member sliding-window
        call in ``torch.autocast(cuda, autocast_dtype)``.
    verbose
        Emit progress lines to stderr.

    Returns
    -------
    fg_native : np.ndarray (float32, native geometry)
        Fused foreground probability field in native geometry (spacing +
        transpose reverted). Ready for :func:`fusion.apply_policy`.
    properties : dict
        The nnU-Net data-properties dict from the shared preprocessing step
        (needed by downstream code for geometry / bbox metadata).
    ref_predictor : IslesPredictor
        The first predictor - its plans_manager / label_manager are used for
        the ONE inverse-resample call and downstream geometry.

    Raises
    ------
    MixedPlansError
        If plans metadata differs across ``predictors``. The caller should fall
        back to a per-member NPZ path.
    """
    import torch
    from nnunetv2.inference.export_prediction import (
        convert_predicted_logits_to_segmentation_with_correct_shape,
    )

    if len(predictors) == 0:
        raise ValueError("predictors is empty")

    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.shape[0] != len(predictors):
        raise ValueError(
            f"weights shape {w.shape} != n_members {len(predictors)}"
        )
    if np.any(w < 0.0):
        raise ValueError(f"weights must all be >= 0; got {list(map(float, w))}")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError(f"weights must sum to > 0; got {list(map(float, w))}")
    w = (w / total).astype(np.float32)

    sig = verify_plans_parity(predictors)
    if verbose:
        print(f"[plans-fuser] plans_signature = {sig[:12]}...", file=sys.stderr, flush=True)

    ref = predictors[0]

    # ------------------------------------------------------------ preprocess ONCE
    preprocessed, properties = ref.preprocess_case(staged_image_files)
    if verbose:
        print(
            f"[plans-fuser] preprocessed shape={tuple(preprocessed.shape)} "
            f"dtype={preprocessed.dtype}",
            file=sys.stderr,
            flush=True,
        )

    # ------------------------------------------------------------ streaming LOGITS accumulator
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else _NullContext()
    )

    if mode not in ("logits", "fg"):
        raise ValueError(f"mode must be 'logits' or 'fg'; got {mode!r}")

    fused_acc: torch.Tensor | None = None  # accumulator: logits stack (2,Z',Y',X') OR fg (Z',Y',X')
    for i, pred in enumerate(predictors):
        with autocast_ctx:
            logits = pred.predict_logits_plans_space(preprocessed)
        if not isinstance(logits, torch.Tensor):
            logits = torch.as_tensor(logits)
        logits = logits.to(dtype=torch.float32)

        if mode == "logits":
            if fused_acc is None:
                fused_acc = torch.zeros_like(logits)
            fused_acc.add_(logits, alpha=float(w[i]))
        else:  # mode == "fg" - softmax per member in plans-space, accumulate fg only
            # logits is (C, Z', Y', X'); class dim is 0. Apply softmax then take fg (channel 1).
            probs_plans = torch.softmax(logits, dim=0)
            fg_plans = probs_plans[1] if probs_plans.shape[0] >= 2 else probs_plans[0]
            if fused_acc is None:
                fused_acc = torch.zeros_like(fg_plans)
            fused_acc.add_(fg_plans, alpha=float(w[i]))
            del probs_plans, fg_plans

        del logits
        if verbose:
            print(
                f"[plans-fuser] accumulated member [{i + 1:>2}/{len(predictors)}] "
                f"w={float(w[i]):.4f}  mode={mode}",
                file=sys.stderr,
                flush=True,
            )

    # Free the preprocessed volume before the memory-heavy inverse-resample step.
    del preprocessed

    # ------------------------------------------------------------ ONE inverse resample + geometry revert
    if mode == "logits":
        # convert_predicted_logits_to_segmentation_with_correct_shape:
        # resample logits → softmax → argmax + revert cropping + transpose_backward.
        seg_ignore, probs_native = convert_predicted_logits_to_segmentation_with_correct_shape(
            fused_acc,
            ref.plans_manager,
            ref.configuration_manager,
            ref.label_manager,
            properties,
            return_probabilities=True,
        )
        del seg_ignore, fused_acc
    else:  # mode == "fg" - resample fg directly, then revert cropping + transpose_backward
        # Build a 2-channel [1-fg, fg] plans-space tensor. Since resample is linear,
        # resample([1-fg, fg]) = [1 - resample(fg), resample(fg)] - the fg channel
        # is exactly resample(fg_plans), matching per-member native-space fg resample.
        fg_plans = fused_acc.clamp_(0.0, 1.0)
        plans_probs = torch.stack([1.0 - fg_plans, fg_plans], dim=0)  # (2, Z', Y', X')
        del fg_plans, fused_acc

        cm = ref.configuration_manager
        pm = ref.plans_manager
        spacing_transposed = [properties["spacing"][i] for i in pm.transpose_forward]
        current_spacing = (
            cm.spacing
            if len(cm.spacing) == len(properties["shape_after_cropping_and_before_resampling"])
            else [spacing_transposed[0], *cm.spacing]
        )
        probs_native_cropped = cm.resampling_fn_probabilities(
            plans_probs,
            properties["shape_after_cropping_and_before_resampling"],
            current_spacing,
            spacing_transposed,
        )
        del plans_probs

        # revert cropping (insert into bbox-reverted native shape)
        probs_native = ref.label_manager.revert_cropping_on_probabilities(
            probs_native_cropped,
            properties["bbox_used_for_cropping"],
            properties["shape_before_cropping"],
        )
        del probs_native_cropped

        # to numpy + transpose_backward: same as convert_predicted_logits_to_segmentation_with_correct_shape
        if isinstance(probs_native, torch.Tensor):
            probs_native = probs_native.cpu().numpy()
        probs_native = probs_native.transpose([0] + [i + 1 for i in pm.transpose_backward])

    # probs_native is (C, Z, Y, X); pull the fg channel.
    if probs_native.ndim == 4 and probs_native.shape[0] >= 2:
        fg_native = probs_native[1].astype(np.float32, copy=False)
    else:
        fg_native = probs_native.astype(np.float32, copy=False)
    del probs_native

    if verbose:
        print(
            f"[plans-fuser] fg_native shape={fg_native.shape} dtype={fg_native.dtype}",
            file=sys.stderr,
            flush=True,
        )

    return fg_native, properties, ref


__all__ = [
    "MixedPlansError",
    "plans_signature",
    "verify_plans_parity",
    "fuse_in_plans_space",
]
