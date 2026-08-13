"""Frozen decision policy for the Pillar-1 inference stack.

A `DecisionPolicy` is a fully self-describing bundle of knobs that turns per-case
foreground softmax volumes into the final binary mask. It captures every
tunable parameter the Pillar-1 stack exposes:

  * fusion mode across ensemble members (mean / noisy-OR / k-of-N + weights);
  * an adaptive threshold selected from the predicted lesion volume via
    `bucket_edges_ml` (never uses GT - the bucket is inferred from the
    thresholded prediction volume);
  * a per-CC low-confidence FP cleanup (max/mean/mass gates);
  * the never-empty rescue for the domain rule that chronic-stroke GT is
    never empty;
  * bookkeeping needed to translate voxel counts into millilitres.

The `DecisionPolicy` is tuner-output and applier-input: the tuner searches over
these parameters on OOF val cases and writes the winner to JSON; the applier
loads that JSON at inference time and runs the pipeline in exactly one way.
Keeping the whole policy in one dataclass makes the search space explicit and
guarantees reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from nnunet_isles.losses._volume_weights import bucket_for_volume

_VALID_MODES = ("mean", "noisy_or", "k_of_n")
_VALID_CONNECTIVITIES = (6, 18, 26)
_CANONICAL_EDGES = (0.5, 5.0, 50.0)
_SCHEMA_VERSION = "1.0"


@dataclass
class DecisionPolicy:
    """See module docstring. All fields are documented on the class body."""

    schema_version: str = _SCHEMA_VERSION

    # Fusion
    mode: str = "mean"  # 'mean' | 'noisy_or' | 'k_of_n'
    k: int | None = None  # for k_of_n
    member_threshold: float = 0.5  # for k_of_n
    weights: list[float] | None = None  # per-member weights; None => uniform

    # Adaptive threshold - bucket by PREDICTED lesion volume (never GT).
    # `bucket_edges_ml` partitions the total-foreground predicted volume (ml)
    # into `len(bucket_edges_ml) + 1` buckets; `threshold_by_bucket` and
    # `min_voxels_by_bucket` are aligned per-bucket lists of the same length.
    bucket_edges_ml: list[float] = field(default_factory=list)
    threshold_by_bucket: list[float] = field(default_factory=lambda: [0.5])
    min_voxels_by_bucket: list[int] = field(default_factory=lambda: [0])

    # Per-CC low-confidence FP cleanup (see cc_stats.drop_low_confidence_ccs).
    min_max_prob: float = 0.0
    min_mean_prob: float = 0.0
    min_prob_mass: float = 0.0

    # Never-empty rescue.
    never_empty: bool = True
    rescue_min_prob: float = 0.10

    # Misc.
    connectivity: int = 26
    voxel_volume_ml: float = 0.001  # 1x1x1 mm3 = 0.001 ml (nnU-Net iso10 plans-space)

    # Smooth threshold interpolation between adjacent buckets on log-vol axis.
    # Off by default (backward-compatible with policies tuned before this
    # existed). Turn on to defuse catastrophic bucket-misassignment
    # regressions when the predicted volume lands near an edge (the hard-bucket
    # tuner had a -0.44 Dice case triggered by exactly this mode of failure).
    soft_bucket_boundary: bool = False

    def __post_init__(self) -> None:
        # Coerce tuples to lists so JSON round-trips and equality behave.
        if isinstance(self.bucket_edges_ml, tuple):
            self.bucket_edges_ml = list(self.bucket_edges_ml)
        if isinstance(self.threshold_by_bucket, tuple):
            self.threshold_by_bucket = list(self.threshold_by_bucket)
        if isinstance(self.min_voxels_by_bucket, tuple):
            self.min_voxels_by_bucket = list(self.min_voxels_by_bucket)
        if isinstance(self.weights, tuple):
            self.weights = list(self.weights)
        self.validate()

    # ------------------------------------------------------------------ I/O

    def to_json(self, path: str | Path) -> None:
        """Write the policy to `path` as human-readable JSON."""
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> DecisionPolicy:
        """Load a policy from JSON.

        Raises `ValueError` on schema-version mismatch - only "1.0" is accepted
        for now. Bump `_SCHEMA_VERSION` (and add a migration) when the on-disk
        layout changes.
        """
        payload = json.loads(Path(path).read_text())
        version = payload.get("schema_version", "unknown")
        if version != _SCHEMA_VERSION:
            raise ValueError(
                f"DecisionPolicy schema_version mismatch: file has {version!r}, "
                f"this build accepts {_SCHEMA_VERSION!r} only. "
                "Regenerate the policy with the current tuner, or add a migration."
            )
        return cls(**payload)

    # ---------------------------------------------------------- Adaptive threshold

    def pick_threshold_for_case(
        self,
        prob_fg: np.ndarray,
        *,
        nominal_threshold: float = 0.5,
    ) -> tuple[float, int, str]:
        """Pick (threshold, min_voxels, bucket_label) from the case's predicted volume.

        Two-pass:
          1. Binarise `prob_fg` at `nominal_threshold` to estimate the case's
             predicted lesion volume.
          2. Convert to millilitres via `voxel_volume_ml`.
          3. Assign to a bucket by counting how many `bucket_edges_ml` entries
             are strictly less than the predicted volume.

        Never touches the ground truth, so this is safe to use at test time.
        """
        binarised = np.asarray(prob_fg) > nominal_threshold
        pred_vol_ml = float(binarised.sum()) * float(self.voxel_volume_ml)
        idx = self._bucket_index(pred_vol_ml)
        label = self._bucket_label(idx)
        if self.soft_bucket_boundary and len(self.bucket_edges_ml) > 0:
            threshold = self._smooth_threshold(pred_vol_ml)
        else:
            threshold = float(self.threshold_by_bucket[idx])
        return (
            threshold,
            int(self.min_voxels_by_bucket[idx]),
            label,
        )

    def _smooth_threshold(self, pred_vol_ml: float) -> float:
        """Linear interpolation of threshold in log10-volume space between
        adjacent bucket centres.

        Bucket centres in log10 space are (a) half-a-decade below the lowest
        edge, (b) the geometric mean of adjacent edges for interior buckets,
        and (c) half-a-decade above the highest edge. This gives a smooth
        threshold curve without a hard step at each bucket edge - the exact
        failure mode that produced the hard-bucket tuner's -0.44 Dice regression on
        `sub-r027s052` (a 5.1 mL predicted case that got the 5-50 mL bucket's
        stricter threshold despite sitting essentially on the boundary).

        Assumes `self.bucket_edges_ml` is non-empty (caller guarantees this).
        """
        edges = self.bucket_edges_ml
        thrs = self.threshold_by_bucket
        # Half-decade offsets for the outermost bucket centres so the
        # interpolation still has a defined slope near vol == 0 / vol -> inf.
        log_edges = [float(np.log10(e)) for e in edges]
        log_centres = [log_edges[0] - 0.5]
        for i in range(len(log_edges) - 1):
            log_centres.append(0.5 * (log_edges[i] + log_edges[i + 1]))
        log_centres.append(log_edges[-1] + 0.5)

        log_pv = float(np.log10(max(pred_vol_ml, 1e-6)))
        if log_pv <= log_centres[0]:
            return float(thrs[0])
        if log_pv >= log_centres[-1]:
            return float(thrs[-1])
        for i in range(len(log_centres) - 1):
            lo, hi = log_centres[i], log_centres[i + 1]
            if lo <= log_pv <= hi:
                t = (log_pv - lo) / (hi - lo)
                return float((1.0 - t) * thrs[i] + t * thrs[i + 1])
        return float(thrs[-1])  # unreachable

    # ---------------------------------------------------------- Validation

    def validate(self) -> None:
        """Structural checks on the policy - raises `ValueError` with a message
        that names the offending field."""
        if self.mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}; got {self.mode!r}")

        if self.connectivity not in _VALID_CONNECTIVITIES:
            raise ValueError(f"connectivity must be one of {_VALID_CONNECTIVITIES}; got {self.connectivity}")

        n_buckets = len(self.bucket_edges_ml) + 1
        if len(self.threshold_by_bucket) != n_buckets:
            raise ValueError(
                f"threshold_by_bucket has {len(self.threshold_by_bucket)} entries; "
                f"expected {n_buckets} (len(bucket_edges_ml) + 1)."
            )
        if len(self.min_voxels_by_bucket) != n_buckets:
            raise ValueError(
                f"min_voxels_by_bucket has {len(self.min_voxels_by_bucket)} entries; "
                f"expected {n_buckets} (len(bucket_edges_ml) + 1)."
            )

        for i in range(1, len(self.bucket_edges_ml)):
            if self.bucket_edges_ml[i] <= self.bucket_edges_ml[i - 1]:
                raise ValueError(f"bucket_edges_ml must be strictly ascending; got {self.bucket_edges_ml}.")

        if self.mode == "k_of_n" and self.k is None:
            raise ValueError("k must be set for k_of_n mode.")
        if self.mode == "k_of_n" and self.k is not None and self.k < 1:
            raise ValueError(f"k must be >= 1 for k_of_n; got {self.k}")

        if self.weights is not None:
            if len(self.weights) < 1:
                raise ValueError("weights must be a non-empty list when provided.")
            if any(w < 0 for w in self.weights):
                raise ValueError(f"weights must all be >= 0; got {self.weights}")

    # ---------------------------------------------------------- Helpers

    def _bucket_index(self, pred_vol_ml: float) -> int:
        """Return the bucket index for a predicted volume in millilitres.

        Convention: the bucket is `sum(edge <= vol for edge in bucket_edges_ml)`,
        i.e. the number of edges less than or equal to the volume. With edges
        `[0.5, 5.0, 50.0]`:
          * vol < 0.5     -> 0
          * 0.5 <= vol<5  -> 1
          * 5   <= vol<50 -> 2
          * vol >= 50     -> 3
        This matches `bucket_for_volume` (which uses `vol < edge`) at every
        exact edge as well as on the interior of each bucket.
        """
        idx = int(sum(1 for edge in self.bucket_edges_ml if edge <= pred_vol_ml))
        # Clamp defensively (should be unreachable after validate()).
        return max(0, min(idx, len(self.threshold_by_bucket) - 1))

    def _bucket_label(self, idx: int) -> str:
        """Human-readable label for bucket `idx`.

        Uses the canonical `bucket_for_volume` labels when
        `bucket_edges_ml == [0.5, 5.0, 50.0]`; otherwise synthesises labels
        from the edges as `'<E0'`, `'E0-E1'`, ..., `'>=EN-1'`.
        """
        edges = self.bucket_edges_ml
        if tuple(edges) == _CANONICAL_EDGES:
            # Query the canonical function at a volume that lands in bucket `idx`.
            if idx == 0:
                probe = 0.0
            elif idx == len(edges):
                # Push strictly past the last edge so bucket_for_volume returns
                # the top bucket even if it ever moves to a strict-inequality check.
                probe = edges[-1] + 1.0
            else:
                probe = edges[idx - 1]
            return bucket_for_volume(probe)

        if not edges:
            return "all"
        if idx == 0:
            return f"<{_fmt(edges[0])}"
        if idx == len(edges):
            return f">={_fmt(edges[-1])}"
        return f"{_fmt(edges[idx - 1])}-{_fmt(edges[idx])}"


def _fmt(x: float) -> str:
    """Compact float formatting for bucket labels - trims a trailing `.0`."""
    s = f"{x:g}"
    return s


def default_policy(n_members: int = 1) -> DecisionPolicy:
    """Return a sensible defaults-only policy.

    Uniform weights, one bucket at threshold 0.5, no FP cleanup, never_empty on.
    A good starting point for the tuner. `n_members` sets `len(weights)` when
    greater than 1; otherwise `weights` is `None` (implying uniform).
    """
    weights: list[float] | None = [1.0 / float(n_members)] * int(n_members) if n_members > 1 else None
    return DecisionPolicy(
        schema_version=_SCHEMA_VERSION,
        mode="mean",
        k=None,
        member_threshold=0.5,
        weights=weights,
        bucket_edges_ml=[],
        threshold_by_bucket=[0.5],
        min_voxels_by_bucket=[0],
        min_max_prob=0.0,
        min_mean_prob=0.0,
        min_prob_mass=0.0,
        never_empty=True,
        rescue_min_prob=0.10,
        connectivity=26,
        voxel_volume_ml=0.001,
    )


__all__ = ["DecisionPolicy", "default_policy"]
