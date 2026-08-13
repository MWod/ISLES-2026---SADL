"""Local (workstation) paths for MICCAI 2026 ISLES.

This module contains PLACEHOLDER values for the public release. Fill each `None`
with a ``pathlib.Path`` pointing at the corresponding directory on your machine
before running any of the training / evaluation entrypoints.

Original layout on the authors' workstation (kept for reference in comments):

- ``project_path``      = ``<checkout>``  (this repository)
- ``venv_path``         = ``<virtualenv>`` (Python 3.10 venv with the pinned deps)
- ``raw_data_path``     = ATLAS v2 raw NIfTIs (BIDS-style; V2 = 1453 sessions, 55 sites)
- ``parsed_data_path``  = per-session preprocessed volumes (Parsed/)
- ``nnunet_{raw,preprocessed,results}`` = the three nnU-Net env-var targets
- ``checkpoints_path``  = trainer checkpoints
- ``logs_path``         = TensorBoard + hydra run logs
- ``evaluation_results_path`` = per-experiment eval outputs
- ``third_party_path``  = vendored nnU-Net at pinned tag
"""

# user: fill in the Path values below for your own machine.

project_path = None  # user: set to your project checkout directory (pathlib.Path)
venv_path = None  # user: set to your Python virtualenv directory (pathlib.Path)

# ATLAS v2 raw NIfTIs (BIDS-style; V2 supersedes V1).
raw_data_path = None  # user: set to your raw ATLAS v2 data directory (pathlib.Path)

parsed_data_path = None  # user: set to your parsed/preprocessed data directory (pathlib.Path)
splits_path = None  # user: set to your Splits/ output directory (pathlib.Path)

# nnU-Net env-var targets - keep these on a fast disk to avoid I/O bottlenecks.
nnunet_raw = None  # user: set to your nnUNet_raw directory (pathlib.Path)
nnunet_preprocessed = None  # user: set to your nnUNet_preprocessed directory (pathlib.Path)
nnunet_results = None  # user: set to your nnUNet_results directory (pathlib.Path)

# Outputs the user inspects.
checkpoints_path = None  # user: set to your Checkpoints/ output directory (pathlib.Path)
logs_path = None  # user: set to your Logs/ output directory (pathlib.Path)
evaluation_results_path = None  # user: set to your Evaluation_Results/ output directory (pathlib.Path)

# Vendored nnU-Net v2.7.0 (commit 566198c4f3f8190b0d52a10172c84a5cd8f78db2).
third_party_path = None  # user: set to your third_party/ directory (pathlib.Path)
