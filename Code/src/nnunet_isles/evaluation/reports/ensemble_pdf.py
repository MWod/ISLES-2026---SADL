"""test_ensemble.pdf - canonical submitted-number report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def _title_page(pdf: PdfPages, header: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    lines = ["ISLES 2026 - Held-out Test, Ensemble", ""]
    for k, v in header.items():
        lines.append(f"{k}: {v}")
    ax.text(0.05, 0.95, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def _headline_page(pdf: PdfPages, summary: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    rows = [["metric", "mean", "std", "p10", "p50", "p90"]]
    for m, s in summary.items():
        if not isinstance(s, dict) or "mean" not in s:
            continue
        rows.append(
            [
                m,
                f"{s['mean']:.4f}",
                f"{s.get('std', float('nan')):.4f}",
                f"{s.get('p10', float('nan')):.4f}",
                f"{s.get('p50', float('nan')):.4f}",
                f"{s.get('p90', float('nan')):.4f}",
            ]
        )
    table = ax.table(cellText=rows, loc="upper left", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title("Headline ensemble metrics")
    pdf.savefig(fig)
    plt.close(fig)


def _per_site_bar(pdf: PdfPages, summary: dict[str, Any]) -> None:
    per_site = summary.get("per_site_dice") or {}
    if not per_site:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    sites = sorted(per_site.keys())
    values = [per_site[s] for s in sites]
    ax.bar(range(len(sites)), values)
    ax.set_xticks(range(len(sites)))
    ax.set_xticklabels(sites, rotation=90, fontsize=8)
    ax.set_ylabel("Dice")
    ax.set_title("Ensemble Dice per held-out test site")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def build_ensemble_pdf(
    output_path: str | Path,
    header: dict[str, Any],
    per_case_df: pd.DataFrame,
    summary: dict[str, Any],
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        _title_page(pdf, header)
        _headline_page(pdf, summary)
        _per_site_bar(pdf, summary)
    return output_path
