"""cv_summary.pdf - 5-fold cross-validation summary report.

Uses matplotlib.backends.backend_pdf.PdfPages so the Docker image doesn't need reportlab.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def _title_page(pdf: PdfPages, header: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    lines = ["ISLES 2026 - CV Summary", ""]
    for k, v in header.items():
        lines.append(f"{k}: {v}")
    ax.text(0.05, 0.95, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def _metric_table_page(pdf: PdfPages, cv_summary: dict[str, dict[str, float]]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    rows = [["metric", "mean", "std"]]
    for m, s in cv_summary.items():
        rows.append([m, f"{s.get('mean', float('nan')):.4f}", f"{s.get('std', float('nan')):.4f}"])
    table = ax.table(cellText=rows, loc="upper left", cellLoc="left", colWidths=[0.3, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    ax.set_title("Headline metric table (CV)", fontsize=13)
    pdf.savefig(fig)
    plt.close(fig)


def _per_fold_bar_page(pdf: PdfPages, per_case_per_fold: list[pd.DataFrame], metric: str = "dice") -> None:
    means, stds = [], []
    for df in per_case_per_fold:
        s = df[metric].replace([np.inf, -np.inf], np.nan).dropna() if metric in df.columns else pd.Series([])
        means.append(float(s.mean()) if len(s) else float("nan"))
        stds.append(float(s.std(ddof=0)) if len(s) else 0.0)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.bar(range(len(means)), means, yerr=stds, capsize=4)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels([f"fold_{i}" for i in range(len(means))])
    ax.set_ylabel(metric)
    ax.set_title(f"Per-fold mean {metric} ± std")
    pdf.savefig(fig)
    plt.close(fig)


def build_cv_summary_pdf(
    output_path: str | Path,
    header: dict[str, Any],
    per_case_per_fold: list[pd.DataFrame],
    cv_summary: dict[str, dict[str, float]],
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        _title_page(pdf, header)
        _metric_table_page(pdf, cv_summary)
        _per_fold_bar_page(pdf, per_case_per_fold, metric="dice")
        _per_fold_bar_page(pdf, per_case_per_fold, metric="lesion_f1")
    return output_path
