"""Native-space ensemble fusion - arithmetic-preserving fast path.

Baseline per-member arithmetic:

    For each of N members m:
        pred.predict_folder(save_probabilities=True)
        # internally: preprocess -> SW -> softmax(logits^plans) -> resample to
        # native -> save NPZ
    Load N NPZs, stack (M, Z, Y, X), average with weights:
        fg_native = sum_m w_m * softmax(logits_m^plans_resampled_to_native)

This native-space fuser preserves that arithmetic EXACTLY (per-member softmax
+ per-member inverse-resample), but drops the two overheads that don't
affect arithmetic:

  1. **Preprocess-once per plans group**. The baseline preprocesses N times
     (once per predict_folder call). Members sharing the same nnU-Net plans
     produce bit-identical preprocessed tensors, so we group by
     plans_signature and preprocess each group once. For the standard 8-mem
     shipping bundle all members share nnUNetPlans_iso10 -> 1 preprocess
     (baseline does 8). For the 11-mem paper baseline 10 share iso10 and 1
     uses iso10_resenc -> 2 preprocesses (baseline does 11).

  2. **No NPZ round-trip**. The baseline writes each member's native-space fg
     to a compressed NPZ then reloads it. This fuser accumulates the weighted
     native fg directly in memory. NPZ output is fp32 (nnU-Net doesn't
     quantise), so this is a pure I/O win with no arithmetic change.

Contrast with :mod:`plans_space_fuser`, which changes the order of softmax
and inverse-resample (either accumulates logits in plans-space OR accumulates
fg in plans-space before ONE resample). Both plans-space orderings empirically
deviate from the per-member arithmetic on the 91-case holdout (-0.10 to -0.15
Dice) - the plans-space fuser is fast but drops quality; the native-space
fuser preserves quality but keeps N inverse resamples so timing gains are
smaller than the plans-space fuser's.

Expected wall-clock relative to the per-member baseline (per case):
    baseline:  ~N * (preprocess + SW + softmax + resample + NPZ_write + NPZ_read)
    this:      ~N * (SW + softmax + resample) + G * preprocess    (G = plans groups)

The N * (softmax + resample) still dominates on giants. This path does NOT
fix the giant SOOP inference-time budget on its own - it recovers per-member
quality with a modest speed win. To fit the budget on giants we still need
either plans-space fusion (quality loss) or ensemble trimming.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

from nnunet_isles.inference.plans_space_fuser import (
    MixedPlansError,  # re-export for callers
    _NullContext,
    plans_signature,
)

if TYPE_CHECKING:
    import torch  # noqa: F401


def _group_predictors_by_plans(predictors: list) -> dict[str, list[int]]:
    """Return ``{plans_signature: [member indices...]}`` preserving input order
    within each group."""
    groups: dict[str, list[int]] = {}
    for i, pred in enumerate(predictors):
        sig = plans_signature(pred)
        groups.setdefault(sig, []).append(i)
    return groups


def fuse_in_native_space(
    predictors: list,
    weights,
    staged_image_files: list[str],
    *,
    autocast_dtype=None,
    verbose: bool = True,
) -> tuple[np.ndarray, dict, object]:
    """Per-member-arithmetic ensemble fusion with preprocess-once (per plans
    group) and no NPZ round-trip.

    Parameters
    ----------
    predictors
        List of :class:`~nnunet_isles.inference.predictor.IslesPredictor`. Each
        must have been ``initialize_from_model_folder``-ed. Order MUST match
        ``weights``.
    weights
        Per-member policy weights. Sum > 0; non-negative. Normalised to sum 1
        in fp64 before folding into the accumulator (matches
        ``mean_prob_weighted`` normalisation).
    staged_image_files
        One-element list containing the staged ``<case>_0000.nii.gz`` path.
    autocast_dtype
        ``torch.dtype | None`` - when set (typically ``torch.float16``),
        wrap each per-member sliding-window call in
        ``torch.autocast(cuda, autocast_dtype)``. Softmax + resample stay
        outside the autocast context to preserve fp32 arithmetic parity with
        the NPZ path (nnU-Net's default nonlin_apply upcasts to fp32).
    verbose
        Emit progress lines to stderr.

    Returns
    -------
    fg_native : np.ndarray  (float32, native geometry ``(Z, Y, X)``)
        Weighted mean of per-member native-space foreground probabilities.
        Ready for :func:`fusion.apply_policy`.
    properties : dict
        The nnU-Net data-properties dict from the FIRST plans group's
        preprocess pass (spacing / origin / bbox metadata). Reused by callers
        for geometry-cloning the output NIfTI.
    ref_predictor : IslesPredictor
        The first predictor - its plans/config/label managers were used for
        the first plans group's inverse-resample calls.

    Notes
    -----
    * Members sharing the same :func:`plans_signature` share one preprocessing
      pass. If ALL members share plans, only ONE preprocess call happens
      (matches ``fuse_in_plans_space``'s preprocess-once behaviour).
    * The inverse-resample happens ONCE per member (unlike
      ``fuse_in_plans_space`` which amortises to ONCE per case). This is the
      arithmetic-preserving trade-off vs the plans-space fuser.
    * If plans metadata differs across ``predictors`` the function transparently
      preprocesses each group; it does NOT raise ``MixedPlansError`` (unlike
      ``verify_plans_parity``). Callers can still opt into a strict-parity
      check upstream if desired.
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

    groups = _group_predictors_by_plans(predictors)
    if verbose:
        print(
            f"[native-fuser] {len(groups)} plans group(s): "
            + ", ".join(f"{sig[:8]}={len(idx)}" for sig, idx in groups.items()),
            file=sys.stderr,
            flush=True,
        )

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else _NullContext()
    )

    fg_native_acc: np.ndarray | None = None  # accumulator, native geometry, fp32
    first_properties: dict | None = None
    first_ref = predictors[0]

    total_members_processed = 0
    for group_sig, member_indices in groups.items():
        # Preprocess ONCE per group using the first member's predictor
        # (all members in this group share plans → same preprocessed output).
        ref = predictors[member_indices[0]]
        preprocessed, properties = ref.preprocess_case(staged_image_files)
        if first_properties is None:
            first_properties = properties
            first_ref = ref
        if verbose:
            print(
                f"[native-fuser] group {group_sig[:12]}: preprocessed "
                f"shape={tuple(preprocessed.shape)} dtype={preprocessed.dtype}  "
                f"n_members={len(member_indices)}",
                file=sys.stderr,
                flush=True,
            )

        for local_j, i in enumerate(member_indices):
            pred = predictors[i]
            with autocast_ctx:
                logits = pred.predict_logits_plans_space(preprocessed)
            if not isinstance(logits, torch.Tensor):
                logits = torch.as_tensor(logits)
            # Match nnU-Net's per-member path: apply_inference_nonlin runs in fp32 (softmax).
            logits = logits.to(dtype=torch.float32)

            # convert_predicted_logits_to_segmentation_with_correct_shape:
            #   softmax -> resample -> revert cropping -> transpose_backward
            # We want the probabilities, not the segmentation - request them.
            seg_ignore, probs_native = convert_predicted_logits_to_segmentation_with_correct_shape(
                logits,
                pred.plans_manager,
                pred.configuration_manager,
                pred.label_manager,
                properties,
                return_probabilities=True,
            )
            del seg_ignore, logits

            # probs_native is (C, Z, Y, X) numpy array; class 1 is FG for binary.
            if probs_native.ndim == 4 and probs_native.shape[0] >= 2:
                fg_m = probs_native[1].astype(np.float32, copy=False)
            else:
                fg_m = probs_native.astype(np.float32, copy=False)
            del probs_native

            if fg_native_acc is None:
                # First member: initialise accumulator to zeros in native shape.
                fg_native_acc = np.zeros_like(fg_m)
            elif fg_m.shape != fg_native_acc.shape:
                raise RuntimeError(
                    f"member {i} fg shape {fg_m.shape} != accumulator {fg_native_acc.shape}"
                )
            fg_native_acc += float(w[i]) * fg_m
            del fg_m

            total_members_processed += 1
            if verbose:
                print(
                    f"[native-fuser] group {group_sig[:12]} member "
                    f"[{local_j + 1:>2}/{len(member_indices)}] "
                    f"(global {total_members_processed:>2}/{len(predictors)}) "
                    f"w={float(w[i]):.4f}",
                    file=sys.stderr,
                    flush=True,
                )

        # Free the shared preprocessed tensor before the next group's preprocess.
        del preprocessed

    assert fg_native_acc is not None
    # Numerical safety - accumulated weighted mean of [0,1] values may drift
    # marginally out of range due to fp32 rounding. Clip in place (matches
    # mean_prob_weighted's np.clip).
    np.clip(fg_native_acc, 0.0, 1.0, out=fg_native_acc)

    if verbose:
        print(
            f"[native-fuser] fg_native shape={fg_native_acc.shape} "
            f"dtype={fg_native_acc.dtype}",
            file=sys.stderr,
            flush=True,
        )

    return fg_native_acc, first_properties, first_ref


def fuse_in_native_space_gpu_resample(
    predictors: list,
    weights,
    staged_image_files: list[str],
    *,
    autocast_dtype=None,
    verbose: bool = True,
) -> tuple[np.ndarray, dict, object]:
    """Same arithmetic contract as :func:`fuse_in_native_space` but the
    inverse-resample + softmax + crop-revert + transpose_backward are all done
    on GPU with ``torch.nn.functional.interpolate(mode='trilinear')``.

    Motivation: the CPU-side inverse-resample (via nnU-Net's
    ``resampling_fn_probabilities`` -> scipy ``map_coordinates`` order=1) is
    the dominant per-case cost on cases with fine in-plane spacing (e.g.
    R025's 0.625 mm -> native is 2x larger than plans-space, and it happens
    11 times per case). Moving that to GPU trilinear is ~5-10x faster.

    Small numerical drift vs. :func:`fuse_in_native_space`:

    * ``F.interpolate(mode='trilinear', align_corners=False)`` is a
      cell-centred trilinear equivalent to scipy's ``map_coordinates``
      order=1 for isotropic-ish spacings. For strongly anisotropic cases
      (nnU-Net's ``do_separate_z`` threshold: ratio > 3), nnU-Net switches
      to 2D-per-slice + 1D-z resample, which this function does NOT
      replicate. Expected divergence ~Dice +/-0.005 on isotropic cases,
      potentially larger on strongly anisotropic ones.

    * The softmax happens BEFORE the crop-revert here, so background voxels
      padded outside the cropped bbox get fg=0.5 fed through as
      neither-class. To keep parity with the CPU path we write the
      padded-region fg to 0.0 explicitly (softmax(0,0)=(0.5,0.5) is wrong
      for the outside-brain padding, which should be pure background -
      matches how the CPU path's ``revert_cropping_on_probabilities``
      writes zeros for padding).

    Peak GPU memory bump vs. the CPU-resample path:

    * Buffer for resampled probs (2 ch * shape_after_cropping * fp32).
    * Buffer for fg crop-revert padding (1 ch * shape_before_cropping * fp32).
    * fg accumulator (1 ch * shape_before_cropping * fp32).

    Worst-case SOOP outlier (353 M native vox): ~4.2 GB added on top of the
    ~5 GB baseline (framework + 5-fold weights) -> ~9-10 GB peak on a 16 GB
    GPU. Fits with margin.
    """
    import torch
    import torch.nn.functional as F

    if len(predictors) == 0:
        raise ValueError("predictors is empty")

    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.shape[0] != len(predictors):
        raise ValueError(f"weights shape {w.shape} != n_members {len(predictors)}")
    if np.any(w < 0.0):
        raise ValueError(f"weights must all be >= 0; got {list(map(float, w))}")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError(f"weights must sum to > 0; got {list(map(float, w))}")
    w = (w / total).astype(np.float32)

    groups = _group_predictors_by_plans(predictors)
    if verbose:
        print(
            f"[gpu-resample-fuser] {len(groups)} plans group(s): "
            + ", ".join(f"{sig[:8]}={len(idx)}" for sig, idx in groups.items()),
            file=sys.stderr,
            flush=True,
        )

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else _NullContext()
    )

    device = predictors[0]._predictor.device
    fg_native_acc: torch.Tensor | None = None
    first_properties: dict | None = None
    first_ref = predictors[0]

    total_members_processed = 0
    for group_sig, member_indices in groups.items():
        ref = predictors[member_indices[0]]
        preprocessed, properties = ref.preprocess_case(staged_image_files)
        if first_properties is None:
            first_properties = properties
            first_ref = ref

        # Geometry metadata (shared across the group - plans are identical).
        shape_after_crop = tuple(int(s) for s in properties["shape_after_cropping_and_before_resampling"])
        shape_before_crop = tuple(int(s) for s in properties["shape_before_cropping"])
        bbox = properties["bbox_used_for_cropping"]                 # [[z0,z1], [y0,y1], [x0,x1]]
        transpose_backward = tuple(int(x) for x in ref.plans_manager.transpose_backward)

        if verbose:
            print(
                f"[gpu-resample-fuser] group {group_sig[:12]}: preprocessed "
                f"shape={tuple(preprocessed.shape)}  "
                f"shape_after_crop={shape_after_crop}  "
                f"shape_before_crop={shape_before_crop}  "
                f"transpose_backward={transpose_backward}  "
                f"n_members={len(member_indices)}",
                file=sys.stderr,
                flush=True,
            )

        for local_j, i in enumerate(member_indices):
            pred = predictors[i]
            with autocast_ctx:
                logits = pred.predict_logits_plans_space(preprocessed)
            if not isinstance(logits, torch.Tensor):
                logits = torch.as_tensor(logits)
            logits = logits.to(device=device, dtype=torch.float32)   # (C, Z_p, Y_p, X_p)

            # --- 1) inverse trilinear resample plans → shape_after_crop, on GPU
            # F.interpolate expects a batch dim: (N, C, D, H, W)
            resampled = F.interpolate(
                logits.unsqueeze(0),
                size=shape_after_crop,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)                                              # (C, Z_ac, Y_ac, X_ac)
            del logits

            # --- 2) softmax → take fg channel (class 1)
            probs = F.softmax(resampled, dim=0)
            del resampled
            fg = probs[1] if probs.shape[0] >= 2 else probs[0]        # (Z_ac, Y_ac, X_ac)
            del probs

            # --- 3) revert cropping: pad to shape_before_crop with zeros, insert fg at bbox
            fg_padded = torch.zeros(shape_before_crop, dtype=torch.float32, device=device)
            slicer = tuple(slice(int(b[0]), int(b[1])) for b in bbox)
            fg_padded[slicer] = fg
            del fg

            # --- 4) transpose backward to native raw geometry
            fg_native = fg_padded.permute(transpose_backward).contiguous()
            del fg_padded

            # --- 5) accumulate weighted fg on GPU
            if fg_native_acc is None:
                fg_native_acc = torch.zeros_like(fg_native)
            elif fg_native.shape != fg_native_acc.shape:
                raise RuntimeError(
                    f"member {i} fg shape {fg_native.shape} != accumulator {fg_native_acc.shape}"
                )
            fg_native_acc.add_(fg_native, alpha=float(w[i]))
            del fg_native

            total_members_processed += 1
            if verbose:
                print(
                    f"[gpu-resample-fuser] group {group_sig[:12]} member "
                    f"[{local_j + 1:>2}/{len(member_indices)}] "
                    f"(global {total_members_processed:>2}/{len(predictors)}) "
                    f"w={float(w[i]):.4f}",
                    file=sys.stderr,
                    flush=True,
                )

        del preprocessed

    assert fg_native_acc is not None
    # clip in place (float rounding may nudge slightly outside [0,1])
    fg_native_acc.clamp_(0.0, 1.0)

    # move to CPU numpy for downstream policy application
    fg_native_np = fg_native_acc.detach().cpu().numpy().astype(np.float32)
    del fg_native_acc

    if verbose:
        print(
            f"[gpu-resample-fuser] fg_native shape={fg_native_np.shape} "
            f"dtype={fg_native_np.dtype}",
            file=sys.stderr,
            flush=True,
        )

    return fg_native_np, first_properties, first_ref


__all__ = [
    "MixedPlansError",
    "fuse_in_native_space",
    "fuse_in_native_space_gpu_resample",
]
