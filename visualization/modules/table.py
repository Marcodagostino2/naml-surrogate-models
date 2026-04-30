"""Metrics table visualization for multiple trained models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils import FIG_STYLE, apply_style, compute_metrics, finalize_figure, predict_mean

if TYPE_CHECKING:
    from ..config import VisualizationConfig

_METRICS = ("r2", "rmse", "mae", "mape", "max_error")


def _predict_split(model: Any, data: dict[str, Any]) -> np.ndarray:
    return predict_mean(
        model,
        data["X_test"],
        data["input_scaler_mean"],
        data["input_scaler_std"],
        data["target_scaler_mean"],
        data["target_scaler_std"],
    )


def _build_metrics_dataframe(models: dict[str, Any], data: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for model_name, model in models.items():
        y_pred = _predict_split(model, data)
        y_true = np.asarray(data["y_test"]).reshape(-1)
        row = {metric: compute_metrics(y_true, y_pred)[metric] for metric in _METRICS}
        row["model"] = model_name
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def _print_table(df: pd.DataFrame) -> None:
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    print("=" * 80)
    print("MODEL METRICS SUMMARY")
    print("=" * 80)
    if tabulate is None:
        print(df.to_string(float_format=lambda value: f"{value:.4f}"))
    else:
        print(tabulate(df.reset_index(), headers="keys", tablefmt="rounded_outline", floatfmt=".4f", showindex=False))


def _best_row_mask(df: pd.DataFrame, metric: str) -> pd.Series:
    return df[metric] == df[metric].max() if metric == "r2" else df[metric] == df[metric].min()


def _plot_table(df: pd.DataFrame, config: "VisualizationConfig", mode: str) -> None:
    fig, ax = plt.subplots(
        figsize=(max(6.0, len(df.index) * float(FIG_STYLE["table_col_width"])), len(df.columns) * float(FIG_STYLE["table_row_height"]) + float(FIG_STYLE["table_height_offset"])),
    )
    ax.axis("off")
    table = ax.table(cellText=np.round(df.values, 4), colLabels=df.columns.tolist(), rowLabels=df.index.tolist(), loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(int(FIG_STYLE["table_font_size"]))
    for column_index, metric in enumerate(df.columns):
        mask = _best_row_mask(df, metric)
        for row_index, is_best in enumerate(mask.tolist()):
            if is_best:
                cell = table[(row_index + 1, column_index)]
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#d9f0d3")
    apply_style(ax, "Metrics Table", "Model", "Metric")
    finalize_figure(fig, "metrics_table_all_models.png", config, mode)


def run_table(models: dict[str, Any], data: dict[str, Any], config: "VisualizationConfig", mode: str) -> None:
    """Print and plot the metrics table when all models are available."""

    df = _build_metrics_dataframe(models, data)
    _print_table(df)
    _plot_table(df, config, mode)
