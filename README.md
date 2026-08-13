# ISLES'26 MICCAI Challenge - SanoAGH Submission

**Team:** SanoAGH

**Authors:**
Marek Wodzinski¹² · Gniewosz Drwiega¹ · Wojciech Szymanski¹²

¹ Sano Centre for Computational Personalised Medicine International Research Foundation, Krakow, Poland
² AGH University of Krakow, Krakow, Poland

---

## 1. Overview

Chronic stroke lesion segmentation on T1-weighted MRI (ISLES 2026). The shipped
algorithm is a **SADL / Pillar-1 decision layer** over an **11-member nnU-Net
ensemble**: the members are heterogeneous 3D full-resolution nnU-Net v2
configurations (baseline / ResEnc / DA5 / bucket-weighted / cohort-aware /
curriculum / MoE / SWAD / diffusion-aug / two temperature-mixed-loss variants),
each trained under a 5-fold site-disjoint split; a single frozen policy
(`policy.json`) turns the 11 softmax outputs into a lesion mask with a fixed
weight vector, per-cohort thresholds, and connected-component post-processing.
The algorithm is described in the SWITCH+ 2026 workshop paper (Wodzinski,
Drwiega, Szymanski; *to appear*).

## 2. Repository layout

```
Code/src/nnunet_isles/            trainers, losses, dataloaders, inference logic (SADL / Pillar-1)
Code/scripts/                     entry points (train, finalize, evaluate, apply_pillar1_test, ...)
Code/configs/                     Hydra configs for the 11 ensemble members + shared defaults
Code/tests/                       pytest suite
Code/paths/                       path resolver stubs (users fill in for their environment)
Splits/v2_site_disjoint_test3/    the 5-fold site-disjoint split JSON used for the final submission
docker/                           Grand Challenge "invoke" API container (Docker v4)
third_party/nnUNet/               vendored nnU-Net v2.7.0 (Apache 2.0)
requirements*.txt                 training / inference / HPC dependency lists
pyproject.toml
```

## 3. Requirements

- Python **>= 3.10**
- CUDA-capable GPU (training and inference)
- PyTorch **>= 2.0** (pins in `requirements.txt`)
- Linux; the Docker container targets Grand Challenge's `nvidia/cuda:12.x` base.

## 4. Installation

```bash
python3.10 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e third_party/nnUNet
pip install -e .
```

For inference only (no training extras):

```bash
pip install -r requirements_inference.txt
```

## 5. Set your paths

The pipeline never hardcodes paths: everything routes through `Code/paths/`
(auto-selects local vs HPC by filesystem check) and the three nnU-Net env vars.

Export nnU-Net env vars:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

Then open `Code/paths/pc_paths.py` (and `hpc_paths.py` if you are on a cluster)
and fill in the module-level constants. The release ships with them set to
`None`, so first invocations will raise until you point them at your own
directories:

```python
# Code/paths/pc_paths.py
project_path   = None   # /your/repo/checkout
venv_path      = None   # /your/venv
raw_data_path  = None   # <root>/RAW/Training_Raw_V2
parsed_data_path = None
nnunet_raw          = None
nnunet_preprocessed = None
nnunet_results      = None
```

## 6. Data preparation

ISLES 2026 training pool: V2 release (1453 sessions, 55 sites). Three-step
chain:

```bash
# 1. Convert BIDS-style RAW into an nnU-Net raw dataset (Dataset510_AtlasV2_V2).
python Code/scripts/convert_atlas_v2_to_nnunet.py

# 2. Run nnU-Net planning + preprocessing (iso10 plans + resenc plans).
python Code/scripts/preprocess_dataset.py dataset=Dataset510_AtlasV2_V2 plans=nnUNetPlans_iso10
python Code/scripts/preprocess_dataset.py dataset=Dataset510_AtlasV2_V2 plans=nnUNetPlans_iso10_resenc

# 3. (Optional) Regenerate the 5-fold site-disjoint split used in the paper.
#     The shipped split is Splits/v2_site_disjoint_test3/ (checked in verbatim).
python Code/scripts/generate_splits.py split=v2_site_disjoint_test3
```

## 7. Training a single member

All training goes through `Code/scripts/train.py` (Hydra). Example: one fold
of the bucket-weighted member:

```bash
python Code/scripts/train.py \
    experiment=hpcv7_bucketweighted_v2_dicetopk_500ep \
    fold=0
```

Sweep the 5 folds:

```bash
for fold in 0 1 2 3 4; do
  python Code/scripts/train.py \
      experiment=hpcv7_bucketweighted_v2_dicetopk_500ep \
      fold=${fold}
done
python Code/scripts/finalize.py experiment=hpcv7_bucketweighted_v2_dicetopk_500ep
```

## 8. Reproducing the full 11-member ensemble

The canonical member list is `docker/manifest_pillar1_v2.json`. The order is
1:1 with `policy.weights` in the frozen policy.

| # | Experiment YAML                                | Trainer class                     | Plans                     | Epochs |
|---|------------------------------------------------|-----------------------------------|---------------------------|--------|
| 1 | `hpcv7_bucketweighted_v2_dicetopk_500ep`       | `IslesTrainerBucketWeighted`      | `nnUNetPlans_iso10`       | 500    |
| 2 | `hpcv7_da5_v2_dicetopk_700ep`                  | `nnUNetTrainerDA5`                | `nnUNetPlans_iso10`       | 700    |
| 3 | `hpcv7_moe_cohort_v2_dicetopk_500ep`           | `IslesTrainerCohortMoE`           | `nnUNetPlans_iso10`       | 500    |
| 4 | `hpcv7_cohortaware_v2_dicetopk_500ep`          | `IslesTrainerCohortBalanced`      | `nnUNetPlans_iso10`       | 500    |
| 5 | `hpcv7_curriculum_v2_dicetopk_500ep`           | `IslesTrainerCurriculum`          | `nnUNetPlans_iso10`       | 500    |
| 6 | `hpcv7_diffusionaug_v2_dicetopk_500ep`         | `IslesTrainer`                    | `nnUNetPlans_iso10`       | 500    |
| 7 | `hpcv6_baseline_v2_dicetopk_500ep`             | `IslesTrainer`                    | `nnUNetPlans_iso10`       | 500    |
| 8 | `hpcv6_resenc_v2_dicetopk_500ep`               | `IslesTrainerResEnc`              | `nnUNetPlans_iso10_resenc`| 500    |
| 9 | `hpcv8_bucketweighted_swa_v2_dicetopk_700ep`   | `IslesTrainerBucketWeightedSWA`   | `nnUNetPlans_iso10`       | 700    |
|10 | `hpcv8_bucketweighted_tml2_v2_dicetopk_500ep`  | `IslesTrainerBucketWeighted`      | `nnUNetPlans_iso10`       | 500    |
|11 | `hpcv8_bucketweighted_tml3_v2_dicetopk_500ep`  | `IslesTrainerBucketWeighted`      | `nnUNetPlans_iso10`       | 500    |

A full reproduction needs **11 x 5 = 55 checkpoints**. Member 9 uses `swa.pth`;
all others use `checkpoint_best.pth` (see the manifest). Train each row with
the loop from Section 7 and then finalize.

### Pre-trained checkpoints

The 55 trained checkpoints and the frozen `policy.json` are **not shipped in
this repository** (~9 GB packed, too large for GitHub). To obtain the
checkpoints that back the ISLES 2026 submission, please **contact the authors
directly** (see the maintainer list in `pyproject.toml`). They will be provided
along with the manifest and policy JSON so the Docker container in `docker/`
can be built and run against the exact ensemble used in the challenge.

## 9. Running inference

Two paths are supported. Both consume the frozen `policy.json` (Pillar-1
decision layer).

**In-container / GC "invoke" API:**

```bash
python docker/inference.py    # entry point used by the Grand Challenge image
```

**Standalone (local, on any nnU-Net-formatted test set):**

```bash
python Code/scripts/apply_pillar1_test.py \
    manifest=docker/manifest_pillar1_v2.json \
    policy=docker/model/policy/policy.json \
    input_dir=/path/to/nnUNet_raw/DatasetXXX/imagesTs \
    output_dir=/path/to/predictions
```

## 10. Building the Docker container

```bash
bash docker/do_build.sh          # build image
bash docker/do_save.sh           # export tar.gz for Grand Challenge upload
bash docker/stage_test_input.sh  # populate docker/test/ with a sample case
```

**Note:** the 55-checkpoint bundle (`model.tar.gz`) is uploaded to Grand
Challenge separately as a **Model** resource; it is *not* baked into the
container image. The image reads it at runtime from the GC-provided model
mount.

## 11. Internal holdout results

Site-disjoint 91-case holdout (R018 + R027 + R047), scored with the official
ISLES 2026 evaluator (panoptica-based):

| Metric        | Value  |
|---------------|--------|
| Dice          | 0.7322 |
| Lesion-F1     | 0.7461 |
| PR-AUC        | 0.8430 |

Policy hash: `b4c86ddf234203be` (see `docker/manifest_pillar1_v2.json`).

## 12. Citation

The SWITCH+ 2026 workshop paper describing this submission is in preparation.
The BibTeX entry below will be finalised at publication:

```bibtex
@inproceedings{wodzinski2026sadl,
  title     = {SADL: A Size-Adaptive Decision Layer for Chronic Stroke Lesion Segmentation in ISLES 2026},
  author    = {Wodzinski, Marek and Drwiega, Gniewosz and Szymanski, Wojciech},
  booktitle = {MICCAI SWITCH+ Workshop},
  year      = {2026}
}
```

## 13. Acknowledgements

This work was supported by the National Science Centre, Poland, under Grant
"MultiGeoMed" No. 2024/55/D/ST6/02081. We gratefully acknowledge the Polish
high-performance computing infrastructure PLGrid, HPC Center: ACK Cyfronet
AGH, for providing computational resources and support within computational
grant No. PLG/2025/018770. The work was partially supported by the Excellence
Initiative - Research University program at the AGH University of Krakow.

## 14. License

Apache License 2.0. Vendored code under `third_party/nnUNet/` is redistributed
under its original Apache 2.0 license (see `third_party/nnUNet/LICENSE`).
