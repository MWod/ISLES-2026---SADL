"""Evaluation entrypoint.

CLI:
    python scripts/evaluate.py experiment_name=foo split=site_disjoint_test3 \\
        report=all mode=both fold=all

Report x mode x fold semantics:

    report:
      - per_fold    single-fold outputs (requires fold=N where N is one of 0..4)
      - cv_summary  aggregate mean +/- std over all folds
      - ensemble    5-fold soft-vote ensemble on the held-out test set
                    (only valid with mode=test)
      - all         run every applicable combination

    mode:
      - val         evaluate on the internal validation fold
      - test        evaluate on the held-out outer test set
      - both        run both val and test

    fold:
      - N in 0..4   restrict to a single fold (required for report=per_fold)
      - all         iterate over every fold that exists on disk
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))

from scripts._autopath_resolver import register_autopath_resolver  # noqa: E402

register_autopath_resolver()

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

VALID_REPORTS = {"per_fold", "cv_summary", "ensemble", "all"}
VALID_MODES = {"val", "test", "both"}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> int:
    report = str(cfg.get("report", "all"))
    mode = str(cfg.get("mode", "both"))
    fold = str(cfg.get("fold", "all"))

    if report not in VALID_REPORTS:
        print(
            f"[evaluate] invalid report={report!r} (must be one of {sorted(VALID_REPORTS)})", file=sys.stderr
        )
        return 2
    if mode not in VALID_MODES:
        print(f"[evaluate] invalid mode={mode!r} (must be one of {sorted(VALID_MODES)})", file=sys.stderr)
        return 2
    if report == "ensemble" and mode == "val":
        print("[evaluate] report=ensemble is only valid with mode=test", file=sys.stderr)
        return 2

    eval_root = Path(cfg.paths.evaluation_results_path) / cfg.experiment_name
    eval_root.mkdir(parents=True, exist_ok=True)

    print(f"[evaluate] experiment={cfg.experiment_name} report={report} mode={mode} fold={fold}")
    print(f"[evaluate] eval_root={eval_root}")
    print(OmegaConf.to_yaml({"experiment_name": cfg.experiment_name, "split": cfg.split.name}, resolve=True))

    # Skeleton - the actual report generation calls into
    # nnunet_isles.evaluation.reports.* once predictions exist.
    print(f"[evaluate] (skeleton; report={report} mode={mode} fold={fold} not yet executed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
