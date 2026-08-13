"""Merge official ISLES 2026 scores with our internal leaderboard.

For every experiment (and Pillar-1) that has been re-scored by
``evaluate_official_isles26.py`` under a common output root, we join the
official mean Dice / Lesion-F1 / Volume-diff / Lesion-Count-diff / PR-AUC
against the values in ``Results/leaderboard.csv`` (our internal metric
set) and pretty-print a side-by-side ranking table.

Outputs:
* stdout - Markdown table sorted by official Dice.
* ``--out-csv`` (optional) - merged CSV.
* ``--out-json`` (optional) - structured summary + ranking-change report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _fmt(x, prec=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{prec}f}"
    return str(x)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--scoring-root",
        type=Path,
        required=True,
        help="Root produced by run_all_scores.sh - contains <exp>/official_metrics_isles26.json.",
    )
    p.add_argument(
        "--internal-leaderboard",
        type=Path,
        default=None,
        help="Path to our leaderboard.csv (default: Results/leaderboard.csv).",
    )
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    import pandas as pd

    internal_path = args.internal_leaderboard or Path("Results/leaderboard.csv")
    internal = pd.read_csv(internal_path) if internal_path.exists() else None

    # Discover all official-score directories.
    rows = []
    for sub in sorted(args.scoring_root.iterdir()):
        summary_path = sub / "official_metrics_isles26.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        row = {
            "experiment": sub.name,
            "source_mode": summary.get("source_mode"),
            "n_cases": summary.get("n_cases_scored"),
            "off_dice": summary["overall_means"].get("dice"),
            "off_lesion_f1": summary["overall_means"].get("lesion_f1"),
            "off_lcd": summary["overall_means"].get("abs_lesion_count_diff"),
            "off_avd_ml": summary["overall_means"].get("abs_volume_diff_ml"),
            "off_pr_auc": summary["overall_means"].get("pr_auc"),
        }
        if internal is not None:
            # Prefer an exact experiment_name match, fall back to any name that
            # equals the sub.name (handles pillar1 / ensemble labels).
            hit = internal[internal["experiment_name"] == sub.name]
            if not hit.empty:
                r = hit.iloc[0]
                row.update(
                    {
                        "int_dice": float(r.get("ensemble_dice", float("nan"))),
                        "int_lesion_f1": float(r.get("ensemble_lesion_f1", float("nan"))),
                        "int_hd95": float(r.get("ensemble_hd95", float("nan"))),
                        "int_avd": float(r.get("ensemble_avd", float("nan"))),
                        "int_lesion_count_f1": float(r.get("ensemble_lesion_count_f1", float("nan"))),
                    }
                )
        rows.append(row)

    if not rows:
        print(f"[compare] no rows found under {args.scoring_root}", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows).sort_values("off_dice", ascending=False).reset_index(drop=True)

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"[compare] wrote merged CSV to {args.out_csv}")

    # -------- Markdown table --------------------------------------------------
    cols_show = [
        ("experiment", "Experiment"),
        ("off_dice", "OFF Dice"),
        ("off_lesion_f1", "OFF LF1"),
        ("off_pr_auc", "OFF PR-AUC"),
        ("off_avd_ml", "OFF ΔVol mL"),
        ("off_lcd", "OFF ΔCount"),
        ("int_dice", "ours Dice"),
        ("int_lesion_f1", "ours LF1"),
    ]
    header = "| " + " | ".join(h for _, h in cols_show) + " |"
    sep = "|" + "|".join("---" for _ in cols_show) + "|"
    print(header)
    print(sep)
    for _, r in df.iterrows():
        line_items = []
        for key, _label in cols_show:
            if key == "experiment":
                line_items.append(str(r[key]))
            elif key.startswith(("off_lcd", "off_avd", "int_avd")):
                line_items.append(_fmt(r.get(key), 2))
            else:
                line_items.append(_fmt(r.get(key), 4))
        print("| " + " | ".join(line_items) + " |")

    # -------- ranking-change summary -----------------------------------------
    print()
    print("### Ranking summary")

    # Rank by official Dice and by our internal Dice; find leader diff.
    rank_off = df[["experiment", "off_dice"]].sort_values("off_dice", ascending=False).reset_index(drop=True)
    if "int_dice" in df.columns:
        rank_int = (
            df[["experiment", "int_dice"]]
            .dropna(subset=["int_dice"])
            .sort_values("int_dice", ascending=False)
            .reset_index(drop=True)
        )
    else:
        rank_int = None

    off_leader = rank_off.iloc[0]["experiment"]
    print(f"* Best experiment by OFFICIAL Dice: **{off_leader}** ({rank_off.iloc[0]['off_dice']:.4f}).")
    if rank_int is not None and not rank_int.empty:
        int_leader = rank_int.iloc[0]["experiment"]
        print(f"* Best experiment by OUR Dice:      **{int_leader}** ({rank_int.iloc[0]['int_dice']:.4f}).")
        if off_leader != int_leader:
            print("* **Leader changes under the official metric.**")
        else:
            print("* Leader is unchanged.")

    # Ranking rank-difference table (top-6 by official).
    if rank_int is not None:
        print()
        print("| rank | Off leader | Ours leader | Off Dice | Ours Dice |")
        print("|---|---|---|---|---|")
        int_rank_map = {e: i for i, e in enumerate(rank_int["experiment"])}
        for i in range(min(6, len(rank_off))):
            off_e = rank_off.iloc[i]["experiment"]
            int_e = rank_int.iloc[i]["experiment"] if i < len(rank_int) else "-"
            off_d = rank_off.iloc[i]["off_dice"]
            our_d = rank_int.iloc[i]["int_dice"] if i < len(rank_int) else float("nan")
            shift = int_rank_map.get(off_e)
            shift_str = "" if shift is None else f" (was #{shift + 1})"
            print(f"| {i + 1} | {off_e}{shift_str} | {int_e} | {off_d:.4f} | {our_d:.4f} |")

    if args.out_json:
        args.out_json.write_text(
            json.dumps(
                {
                    "off_leader": off_leader,
                    "int_leader": (
                        rank_int.iloc[0]["experiment"]
                        if rank_int is not None and not rank_int.empty
                        else None
                    ),
                    "top_by_official": rank_off.head(15).to_dict(orient="records"),
                    "top_by_internal": (
                        rank_int.head(15).to_dict(orient="records") if rank_int is not None else None
                    ),
                },
                indent=2,
                default=str,
            )
        )
        print(f"\n[compare] wrote {args.out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
