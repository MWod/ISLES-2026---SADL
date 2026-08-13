"""Generate outer + inner splits and write manifests under Splits/<split_name>/.

Usage:
    python scripts/generate_splits.py --config-name config split=site_disjoint_test3
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))

from scripts._autopath_resolver import register_autopath_resolver  # noqa: E402

register_autopath_resolver()

import hydra  # noqa: E402
import pandas as pd  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402


def _load_sessions(sessions_tsv: Path) -> pd.DataFrame:
    if not sessions_tsv.exists():
        raise FileNotFoundError(
            f"Expected per-session table at {sessions_tsv} "
            "(columns: `session_id`, `site`, `lesion_bucket`, ...). "
            "Regenerate it with your EDA pipeline or provide the file manually."
        )
    if sessions_tsv.suffix in (".parquet", ".pq"):
        return pd.read_parquet(sessions_tsv)
    return pd.read_csv(sessions_tsv, sep="\t")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> int:
    from nnunet_isles.splits.inner import write_inner_splits
    from nnunet_isles.splits.manifest import write_outer_manifest
    from nnunet_isles.splits.outer import build_outer_split
    from nnunet_isles.utils import current_git_sha

    sessions_tsv = Path(cfg.data.sessions_tsv)
    df = _load_sessions(sessions_tsv)
    print(f"[generate_splits] loaded {len(df)} sessions from {sessions_tsv}")

    manifest_dir = Path(cfg.split.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    git_sha = current_git_sha(Path(cfg.paths.project_path))

    # `cfg.split.params` is an OmegaConf node; nested values (e.g. `test_sites`
    # in the V2 site_disjoint_fixed strategy) survive as ListConfig if we only
    # do `dict(...)`. Recursively unwrap to plain Python so the manifest writer
    # can JSON-serialise it.
    params = OmegaConf.to_container(cfg.split.params, resolve=True) if cfg.split.params else {}
    outer = build_outer_split(
        df,
        split_name=cfg.split.name,
        strategy=cfg.split.strategy,
        params=params,
        git_sha=git_sha,
    )
    outer_path = write_outer_manifest(outer, manifest_dir)
    print(f"[generate_splits] wrote outer manifest → {outer_path}")

    if outer.strategy == "loso":
        print(
            f"[generate_splits] LOSO strategy: {len(outer.stats['iterations'])} outer iterations "
            "(no single inner CV; consult outer.json:stats.iterations to drive the loop)"
        )
        return 0

    train_df = df[df["session_id"].isin(outer.train_session_ids)].copy()
    splits_path, meta_path = write_inner_splits(
        train_df,
        out_dir=str(manifest_dir),
        split_name=cfg.split.name,
        n_folds=int(cfg.split.inner_n_folds),
        seed=int(cfg.split.inner_seed),
        group_by=cfg.split.inner_group_by,
        git_sha=git_sha,
    )
    print(f"[generate_splits] wrote inner manifest → {splits_path}")
    print(f"[generate_splits] wrote inner meta     → {meta_path}")

    # nnU-Net reads splits_final.json from nnUNet_preprocessed/<dataset>/.
    nnunet_target_dir = Path(cfg.paths.nnunet_preprocessed) / cfg.nnunet_dataset_name
    nnunet_target_dir.mkdir(parents=True, exist_ok=True)
    target = nnunet_target_dir / "splits_final.json"
    target.write_text(Path(splits_path).read_text())
    print(f"[generate_splits] copied inner manifest → {target} (for nnU-Net consumption)")

    # Summary printout
    print(
        json.dumps(
            {
                "split_name": outer.split_name,
                "n_train": outer.n_train,
                "n_test": outer.n_test,
                "train_per_site_top5": dict(list(outer.stats.get("train_per_site", {}).items())[:5]),
                "test_per_site": outer.stats.get("test_per_site", {}),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
