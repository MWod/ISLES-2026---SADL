"""Tests for curriculum-by-lesion-size sampling helpers."""

from __future__ import annotations

import numpy as np
from nnunet_isles.dataloading.curriculum_loader import (
    compute_curriculum_probabilities,
    compute_curriculum_visible_threshold,
)


def _build_volumes(
    n: int, vol_range: tuple[float, float] = (0.1, 100.0)
) -> tuple[list[str], dict[str, float]]:
    """Build n cases with volumes log-spaced from `vol_range[0]` to `vol_range[1]`."""
    log_vols = np.linspace(np.log(vol_range[0]), np.log(vol_range[1]), n)
    vols = np.exp(log_vols)
    indices = [f"case_{i:04d}" for i in range(n)]
    return indices, dict(zip(indices, vols.tolist(), strict=True))


def test_at_epoch_zero_only_floor_percentile_visible():
    """At epoch 0 with floor=90, only the top-10% should have positive probability."""
    indices, vols = _build_volumes(30)
    probs = compute_curriculum_probabilities(
        vols, indices, current_epoch=0, warmup_epochs=150, floor_percentile=90.0
    )
    visible = (probs > 0).sum()
    # Top 10% of 30 cases = 3 cases (round-trip via percentile may include the boundary).
    assert 3 <= visible <= 4


def test_at_warmup_end_all_visible():
    """At `current_epoch >= warmup_epochs`, all cases visible with uniform probability."""
    indices, vols = _build_volumes(30)
    probs = compute_curriculum_probabilities(
        vols, indices, current_epoch=150, warmup_epochs=150, floor_percentile=90.0
    )
    assert (probs > 0).sum() == 30
    assert np.allclose(probs, 1.0 / 30)


def test_after_warmup_still_uniform():
    indices, vols = _build_volumes(30)
    probs = compute_curriculum_probabilities(
        vols, indices, current_epoch=300, warmup_epochs=150, floor_percentile=90.0
    )
    assert np.allclose(probs, 1.0 / 30)


def test_midway_pool_grows():
    """At epoch warmup_epochs/2 with floor=90, threshold = 45th percentile;
    visible count = top-55% of cases."""
    indices, vols = _build_volumes(100)
    probs = compute_curriculum_probabilities(
        vols, indices, current_epoch=75, warmup_epochs=150, floor_percentile=90.0
    )
    visible = (probs > 0).sum()
    # ~55 cases (give some slack for percentile rounding).
    assert 50 <= visible <= 60


def test_visible_cases_are_the_largest():
    """The cases that ARE visible at epoch 0 are exactly the top-10% by volume."""
    indices, vols = _build_volumes(20)
    probs = compute_curriculum_probabilities(
        vols, indices, current_epoch=0, warmup_epochs=150, floor_percentile=90.0
    )
    visible_ids = [cid for cid, p in zip(indices, probs, strict=True) if p > 0]
    visible_vols = [vols[cid] for cid in visible_ids]
    # All visible volumes are above the 90th percentile of the full volume distribution.
    p90 = np.percentile(list(vols.values()), 90.0)
    assert all(v >= p90 for v in visible_vols)


def test_threshold_decreases_monotonically_with_epoch():
    """Threshold should be non-increasing as the curriculum opens up."""
    _, vols = _build_volumes(50)
    vols_arr = np.array(list(vols.values()))
    thresholds = [
        compute_curriculum_visible_threshold(
            vols_arr, current_epoch=e, warmup_epochs=100, floor_percentile=90.0
        )
        for e in range(0, 101, 10)
    ]
    for i in range(1, len(thresholds)):
        assert thresholds[i] <= thresholds[i - 1] + 1e-9


def test_zero_warmup_means_no_curriculum():
    indices, vols = _build_volumes(10)
    probs = compute_curriculum_probabilities(
        vols, indices, current_epoch=0, warmup_epochs=0, floor_percentile=90.0
    )
    assert np.allclose(probs, 1.0 / 10)


def test_empty_indices_returns_empty():
    out = compute_curriculum_probabilities({}, [], current_epoch=0, warmup_epochs=150, floor_percentile=90.0)
    assert out.shape == (0,)


def test_floor_percentile_100_gives_no_visible_initially_falls_back_uniform():
    """With floor=100, threshold > max(vols) → 0 cases visible → uniform fallback."""
    indices, vols = _build_volumes(10)
    probs = compute_curriculum_probabilities(
        vols, indices, current_epoch=0, warmup_epochs=10, floor_percentile=100.0
    )
    # Either 0 visible (fallback uniform) or 1 visible (boundary). Both valid.
    assert probs.shape == (10,)
    assert abs(probs.sum() - 1.0) < 1e-9


def test_shared_epoch_counter_propagates_to_get_indices(monkeypatch):
    """The shared `multiprocessing.Value` epoch counter must drive get_indices'
    sampling. This is the path that breaks if we use a plain Python attribute
    and the loader is forked into worker processes.

    We construct a minimal CurriculumDataLoader3D, simulate parent-process
    `set_current_epoch(N)`, then call `get_indices()` and verify that the
    sample pool grew (more visible cases at epoch N than at epoch 0)."""
    # Build a thin shim that mimics the upstream nnUNetDataLoader interface
    # without needing the heavy `dataset_class` machinery: we monkeypatch the
    # parent's __init__ to a no-op and stub `super().get_indices` to return
    # the indices vector verbatim so we can inspect what got sampled from.
    import numpy as np
    from nnunet_isles.dataloading import curriculum_loader as cl

    indices_list = [f"c{i:03d}" for i in range(20)]
    vols = {cid: float(i) for i, cid in enumerate(indices_list)}  # 0..19 mL

    class _FakeBase:
        def __init__(self, *a, **k):
            self.indices = indices_list
            self.sampling_probabilities = None
            self.batch_size = 2
            self.infinite = True

        def get_indices(self):
            # Mimic upstream: read sampling_probabilities, return sampled indices.
            p = self.sampling_probabilities
            assert p is not None
            return list(
                np.random.default_rng(0).choice(self.indices, size=self.batch_size, replace=True, p=p)
            )

    monkeypatch.setattr(cl, "nnUNetDataLoader", _FakeBase)

    # Need to redefine the class to pick up the monkeypatched base.
    class _Loader(_FakeBase):
        # Reuse the curriculum logic by composition rather than inheritance -
        # cleanest with monkeypatch.
        pass

    # Easiest: re-import after monkeypatch.
    import importlib

    importlib.reload(cl)

    dl = cl.CurriculumDataLoader3D.__new__(cl.CurriculumDataLoader3D)
    _FakeBase.__init__(dl)
    dl._case_volumes_ml = vols
    dl._warmup_epochs = 100
    dl._floor_percentile = 90.0
    import multiprocessing as _mp

    dl._epoch_counter = _mp.Value("i", 0)
    dl._cached_epoch_for_probs = -1
    dl._refresh_sampling_probabilities()

    visible_at_0 = (dl.sampling_probabilities > 0).sum()
    # Simulate parent calling set_current_epoch deep into warmup.
    dl.set_current_epoch(75)
    visible_at_75 = (dl.sampling_probabilities > 0).sum()

    # Critically: get_indices() must trigger refresh on its own (covering the
    # case where the parent updates the shared counter but didn't call
    # _refresh_sampling_probabilities explicitly - e.g., the worker process).
    dl._epoch_counter.value = 100  # simulate parent bumping shared counter
    dl._cached_epoch_for_probs = 75  # stale cache from prior epoch
    _ = dl.get_indices()
    visible_after_get_indices = (dl.sampling_probabilities > 0).sum()

    assert visible_at_75 > visible_at_0
    # After warmup_epochs=100, all 20 cases should be visible.
    assert visible_after_get_indices == 20
