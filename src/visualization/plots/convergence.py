"""Leaf plots for the convergence (split-vs-metric) workflow.

Three figure builders shared across per-model and combined views: a
single-model metric curve with a ±1σ band, a multi-model curve, and an
overlay of training-history time series.

These functions write directly to absolute output paths because the
convergence orchestrator builds a per-model directory tree
(``<analysis_root>/<model>/...`` and ``<analysis_root>/complete/...``)
that does not match the flat ``output_dir + subdir`` policy used by
``finalize_figure``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..labels import TEX_R2

_METRIC_LABEL = {
    "r2": TEX_R2,
    "rmse": "RMSE",
    "mae": "MAE",
    "mape": "MAPE",
}


def _metric_label(metric: str) -> str:
    return _METRIC_LABEL.get(metric, metric.upper())


def plot_metric_curve(df: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    agg = (
        df.groupby("train_split")[metric]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("train_split")
    )
    x = agg["train_split"].to_numpy(dtype=float)
    y = agg["mean"].to_numpy(dtype=float)
    yerr = agg["std"].fillna(0.0).to_numpy(dtype=float)
    ax.plot(x, y, marker="o", linewidth=2)
    ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)
    ax.set_xlabel("Train split")
    ax.set_ylabel(_metric_label(metric))
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_metric_curve_all_models(
    df: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
    model_sequence: Sequence[str],
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for model_name in model_sequence:
        subset = df[df["model"] == model_name]
        if subset.empty:
            continue
        agg = (
            subset.groupby("train_split")[metric]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("train_split")
        )
        x = agg["train_split"].to_numpy(dtype=float)
        y = agg["mean"].to_numpy(dtype=float)
        yerr = agg["std"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=2.5, label=model_name.upper())
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.15)
    ax.set_xlabel("Train split")
    ax.set_ylabel(_metric_label(metric))
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_training_histories(
    history_records: List[Dict[str, Any]],
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for record in sorted(history_records, key=lambda item: (item["train_split"], item.get("run_index", 0))):
        values = np.asarray(record["history_values"], dtype=float)
        if values.size == 0:
            continue
        x = np.arange(1, values.size + 1)
        label = f"{record['model']}: split={record['train_split']:.2f}, run={record.get('run_index', 0)}"
        ax.plot(x, values, linewidth=1.0, alpha=0.75, label=label)
    ylabel = history_records[0]["history_label"] if history_records else "history"
    ax.set_xlabel("epoch/step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if history_records:
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
