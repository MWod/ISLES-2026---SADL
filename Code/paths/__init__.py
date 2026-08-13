"""Auto-select local or HPC paths based on environment.

Checks whether the HPC data path exists on disk. If so, uses HPC paths;
otherwise falls back to local PC paths. All downstream code should import
paths from this module, never directly from pc_paths or hpc_paths.
"""

from paths import hpc_paths, pc_paths

is_hpc = hpc_paths.raw_data_path.exists()
_paths = hpc_paths if is_hpc else pc_paths

project_path = _paths.project_path
raw_data_path = _paths.raw_data_path
parsed_data_path = _paths.parsed_data_path
splits_path = _paths.splits_path
venv_path = _paths.venv_path
checkpoints_path = _paths.checkpoints_path
logs_path = _paths.logs_path
evaluation_results_path = _paths.evaluation_results_path
third_party_path = _paths.third_party_path

# nnU-Net env-var targets - also auto-selected.
nnunet_raw = _paths.nnunet_raw
nnunet_preprocessed = _paths.nnunet_preprocessed
nnunet_results = _paths.nnunet_results

hpc_logs_path = getattr(_paths, "hpc_logs_path", None)
