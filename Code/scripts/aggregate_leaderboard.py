"""Walk Evaluation_Results/*/leaderboard_row.json → leaderboard.csv (sorted by ensemble Dice)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))


CSV_COLUMNS = (
    "experiment_name",
    "dataset_version",
    "git_sha",
    "config_hash",
    "split_name",
    "ensemble_dice",
    "ensemble_lesion_f1",
    "ensemble_hd95",
    "ensemble_avd",
    "ensemble_lesion_count_f1",
    "dice_cv_mean",
    "dice_cv_std",
    "timestamp",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()

    if args.root is None:
        from paths import evaluation_results_path

        root = evaluation_results_path
    else:
        root = args.root

    rows: list[dict] = []
    for path in sorted(root.glob("*/leaderboard_row.json")):
        try:
            row = json.loads(path.read_text())
        except Exception as e:
            print(f"[aggregate] WARN: failed to read {path}: {e}", file=sys.stderr)
            continue
        # Existing V1 rows lack `dataset_version`; default to "v1".
        row.setdefault("dataset_version", "v1")
        rows.append({col: row.get(col, "") for col in CSV_COLUMNS})

    rows.sort(key=lambda r: -float(r.get("ensemble_dice", 0.0) or 0.0))

    out_path = root / "leaderboard.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write(",".join(CSV_COLUMNS) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(col, "")) for col in CSV_COLUMNS) + "\n")
    print(f"[aggregate] wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
