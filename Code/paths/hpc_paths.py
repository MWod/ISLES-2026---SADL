"""HPC / SLURM cluster paths for MICCAI 2026 ISLES.

This module contains PLACEHOLDER values for the public release. On the authors'
HPC / SLURM cluster, all data-heavy artifacts (raw NIfTIs, nnU-Net cache,
preprocessed volumes, results, GT segmentations) lived under
``<project>/Data/`` so the project root only held source code + scripts + small
outputs (Logs, Checkpoints).

Fill each ``None`` with a ``pathlib.Path`` pointing at the corresponding
directory on your own cluster before running the HPC entrypoints. The
auto-selector in ``paths/__init__.py`` chooses this module over
``pc_paths.py`` iff ``raw_data_path.exists()`` on the machine, so the
placeholder ``None`` here also makes the auto-selector fall back to the local
paths on machines that don't have your HPC filesystem mounted.
"""

# user: fill in the Path values below for your own cluster.

project_path = None  # user: set to your HPC project root directory (pathlib.Path)
venv_path = None  # user: set to your HPC virtualenv directory (pathlib.Path); build on a compute node if your login node lacks the target architecture

# All data-heavy artifacts live under <project>/Data/ on the HPC filesystem.
data_path = None  # user: set to your HPC data root directory (pathlib.Path)
# ATLAS v2 raw NIfTIs (V2 supersedes V1).
raw_data_path = None  # user: set to your HPC raw ATLAS v2 data directory (pathlib.Path)
parsed_data_path = None  # user: set to your HPC parsed data directory (pathlib.Path)
splits_path = None  # user: set to your HPC Splits/ directory (pathlib.Path)

# nnU-Net env-var targets - also under Data/.
nnunet_raw = None  # user: set to your HPC nnUNet_raw directory (pathlib.Path)
nnunet_preprocessed = None  # user: set to your HPC nnUNet_preprocessed directory (pathlib.Path)
nnunet_results = None  # user: set to your HPC nnUNet_results directory (pathlib.Path)

# Outputs. "Results" is the canonical user-facing name on HPC; pulled to local
# Evaluation_Results/ by a sync script.
checkpoints_path = None  # user: set to your HPC Checkpoints/ directory (pathlib.Path)
logs_path = None  # user: set to your HPC Logs/ directory (pathlib.Path)
hpc_logs_path = None  # user: set to your HPC SLURM stdout/stderr directory (pathlib.Path)
evaluation_results_path = None  # user: set to your HPC Results/ directory (pathlib.Path)

third_party_path = None  # user: set to your HPC third_party/ directory (pathlib.Path)
