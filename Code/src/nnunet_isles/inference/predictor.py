"""IslesPredictor - thin wrapper around nnU-Net v2's nnUNetPredictor.

Used by docker/inference.py (Grand Challenge invoke API) and by
scripts/evaluate.py (per-fold + ensemble test passes).
"""

from __future__ import annotations

from pathlib import Path


class IslesPredictor:
    """Wraps nnUNetPredictor with our default TTA / ensemble settings."""

    def __init__(
        self,
        *,
        use_mirroring: bool = True,
        allowed_mirroring_axes: tuple[int, ...] | None = None,
        tile_step_size: float = 0.5,
        use_gaussian: bool = True,
        perform_everything_on_device: bool = True,
        device: str = "cuda",
        verbose: bool = False,
        use_tent: bool = False,
        tent_n_steps: int = 3,
        tent_lr: float = 1.0e-4,
    ) -> None:
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        self._predictor = nnUNetPredictor(
            tile_step_size=tile_step_size,
            use_mirroring=use_mirroring,
            perform_everything_on_device=perform_everything_on_device,
            device=torch.device(device),
            verbose=verbose,
            verbose_preprocessing=verbose,
            allow_tqdm=False,
        )
        # allowed_mirroring_axes contract: None => keep whatever the loaded plans baked
        # (typically (0,1,2) => 8x mirror TTA). A tuple explicitly overrides the
        # checkpoint's baked-in axes after initialize_from_model_folder (wire-up
        # applied there). Passing (0,) enables sagittal-only TTA (2x per patch).
        self.allowed_mirroring_axes = allowed_mirroring_axes
        self.use_gaussian = use_gaussian
        # TENT (test-time entropy minimization) - installed in initialize_from_model_folder.
        self.use_tent = bool(use_tent)
        self.tent_n_steps = int(tent_n_steps)
        self.tent_lr = float(tent_lr)
        self._tent_adapter = None

    def initialize_from_model_folder(
        self,
        model_folder: str | Path,
        use_folds: tuple[int, ...] = (0, 1, 2, 3, 4),
        checkpoint_name: str = "checkpoint_best.pth",
    ) -> None:
        # Upstream `initialize_from_trained_model_folder` looks the trainer up via
        # `recursive_find_python_class` over the vendored nnunetv2 tree. Our
        # IslesTrainer / IslesTrainerResEnc / IslesTrainerLesionOversample live
        # in `nnunet_isles.trainers` and aren't discoverable that way, so
        # inference fails for any checkpoint trained with a custom trainer.
        # Monkey-patch the lookup just for the duration of the call: try the
        # upstream walk first, fall back to our registry.
        from nnunetv2.inference import predict_from_raw_data as _prd

        import nnunet_isles.trainers  # noqa: F401 - registers IslesTrainer et al.
        from nnunet_isles.registry import TRAINER_REGISTRY

        _original_find = _prd.recursive_find_python_class

        def _patched_find(folder: str, class_name: str, current_module: str):
            cls = _original_find(folder, class_name, current_module)
            if cls is not None:
                return cls
            try:
                return TRAINER_REGISTRY.get(class_name)
            except KeyError:
                return None

        _prd.recursive_find_python_class = _patched_find
        try:
            self._predictor.initialize_from_trained_model_folder(
                str(model_folder), use_folds=use_folds, checkpoint_name=checkpoint_name
            )
        finally:
            _prd.recursive_find_python_class = _original_find

        # Wire up mirror-axis override if the caller asked for one. nnUNetPredictor
        # sets allowed_mirroring_axes from the checkpoint's inference_allowed_mirroring_axes
        # (typically (0,1,2)); overriding here lets the caller request (0,) (sagittal
        # only, 2x TTA per patch). Default None means no override.
        if self.allowed_mirroring_axes is not None:
            self._predictor.allowed_mirroring_axes = tuple(self.allowed_mirroring_axes)

        if self.use_tent:
            self._install_tent_adapter()

    # ------------------------------------------------------------------ accessors
    # The plans-space fuser reaches into these; expose as first-class
    # properties instead of `_predictor.<attr>` grope for stability.

    @property
    def plans_manager(self):
        return self._predictor.plans_manager

    @property
    def configuration_manager(self):
        return self._predictor.configuration_manager

    @property
    def label_manager(self):
        return self._predictor.label_manager

    @property
    def dataset_json(self):
        return self._predictor.dataset_json

    def preprocess_case(self, image_files: list[str]):
        """Preprocess ONE case without a Pool - return (torch.FloatTensor,
        properties). Reused across all N ensemble members when they share plans."""
        import torch

        pred = self._predictor
        preprocessor = pred.configuration_manager.preprocessor_class(
            verbose=pred.verbose_preprocessing
        )
        data, _seg, properties = preprocessor.run_case(
            image_files, None, pred.plans_manager,
            pred.configuration_manager, pred.dataset_json,
        )
        tensor = torch.from_numpy(data).to(
            dtype=torch.float32, memory_format=torch.contiguous_format
        )
        return tensor, properties

    def predict_logits_plans_space(self, preprocessed):
        """Run the 5-fold averaged sliding window on an already-preprocessed
        volume; return LOGITS in plans-space. No inverse-resample, no NPZ."""
        return self._predictor.predict_logits_from_preprocessed_data(preprocessed)

    def _install_tent_adapter(self) -> None:
        """Wrap nnUNetPredictor.predict_logits_from_preprocessed_data so that
        each per-case prediction first runs N TENT adapt steps on IN affine
        params, then resets γ/β to the checkpoint snapshot.

        InstanceNorm has no running stats; we adapt only the affine (γ, β)
        params via entropy minimization on the preprocessed input volume.
        """
        from nnunet_isles.inference.tent_adapter import TentAdapter

        # Move the network to the predictor's device BEFORE snapshotting γ/β so
        # the snapshot lives on the right device and adapt_step inputs match.
        # nnUNetPredictor.predict_sliding_window_return_logits would normally do
        # this lazily, but our monkey-patch below skips that path on the first
        # call.
        device = self._predictor.device
        self._predictor.network = self._predictor.network.to(device)
        network = self._predictor.network

        adapter = TentAdapter(network, lr=self.tent_lr, momentum=0.9)
        adapter.prepare()
        self._tent_adapter = adapter

        # Wrap predict_logits_from_preprocessed_data so adaptation happens
        # ONCE per case (this method is the per-case entry point in nnU-Net).
        original_predict = self._predictor.predict_logits_from_preprocessed_data
        n_steps = self.tent_n_steps
        # Use the plans' patch size for the TENT adapt-step forward - feeding
        # the full preprocessed volume to the network breaks the U-Net's
        # encoder/decoder skip-shape symmetry whenever spatial dims aren't
        # clean multiples of prod(strides). The plans' patch_size is by
        # construction valid for the trained network.
        patch_size = tuple(int(s) for s in self._predictor.configuration_manager.patch_size)

        from nnunet_isles.inference.tent_adapter import center_pad_crop

        def predict_with_tent(data):
            import torch

            patch = center_pad_crop(data, patch_size).to(device).unsqueeze(0)  # (1, C_in, *patch_size)
            for _ in range(n_steps):
                with torch.enable_grad():
                    adapter.adapt_step(patch)
            # Now produce the actual prediction; reset γ/β afterwards.
            network.eval()
            logits = original_predict(data)
            adapter.reset()
            return logits

        self._predictor.predict_logits_from_preprocessed_data = predict_with_tent

    def predict_folder(
        self,
        input_folder: str | Path,
        output_folder: str | Path,
        *,
        save_probabilities: bool = False,
        num_processes_preprocessing: int = 2,
        num_processes_segmentation_export: int = 2,
    ) -> None:
        self._predictor.predict_from_files(
            str(input_folder),
            str(output_folder),
            save_probabilities=save_probabilities,
            overwrite=True,
            num_processes_preprocessing=num_processes_preprocessing,
            num_processes_segmentation_export=num_processes_segmentation_export,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )
