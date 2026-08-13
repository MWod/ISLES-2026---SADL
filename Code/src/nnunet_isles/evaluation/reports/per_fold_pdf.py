"""test_per_fold/fold_<N>.pdf - single-fold held-out-test report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def _title_page(pdf: PdfPages, fold: int, header: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    lines = [f"ISLES 2026 - Held-out Test, fold {fold}", ""]
    for k, v in header.items():
        lines.append(f"{k}: {v}")
    ax.text(0.05, 0.95, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def _summary_table_page(pdf: PdfPages, ensemble_summary: dict[str, dict[str, float]]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    rows = [["metric", "mean", "std", "p10", "p50", "p90"]]
    for m, s in ensemble_summary.items():
        rows.append(
            [
                m,
                f"{s.get('mean', float('nan')):.4f}",
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
    ax.set_title("Per-fold test metrics")
    pdf.savefig(fig)
    plt.close(fig)


def build_per_fold_pdf(
    output_path: str | Path,
    fold: int,
    header: dict[str, Any],
    per_case_df: pd.DataFrame,
    summary: dict[str, dict[str, float]],
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        _title_page(pdf, fold, header)
        _summary_table_page(pdf, summary)
    return output_path
