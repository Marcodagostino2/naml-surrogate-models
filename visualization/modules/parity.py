"""Parity and residual visualizations for a single model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from ..utils import (
    FIG_STYLE,
    PALETTE,
    apply_style,
    compute_metrics,
    finalize_figure,
    predict_mean,
)

if TYPE_CHECKING:
    from ..config import VisualizationConfig

_SPLITS = ("train", "val", "test")
_SPLIT_MARKERS = {"train": "o", "val": "s", "test": "^"}


def _predict_split(model: Any, data: dict[str, Any], split: str) -> tuple[np.ndarray, np.ndarray]:
    return predict_mean(
        model,
        data[f"X_{split}"],
        data["input_scaler_mean"],
        data["input_scaler_std"],
        data["target_scaler_mean"],
        data["target_scaler_std"],
    ), np.asarray(data[f"y_{split}"]).reshape(-1)


def _annotate_metrics(ax: plt.Axes, split: str, metrics: dict[str, float], index: int) -> None:
    text = f"{split}: R²={metrics['r2']:.3f}\nRMSE={metrics['rmse']:.3f}\nMAPE={metrics['mape']:.2f}%"
    ax.text(
        0.02,
        0.98 - index * 0.15,
        text,
        transform=ax.transAxes,
        fontsize=int(FIG_STYLE["annotation_size"]),
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": PALETTE[split], "alpha": 0.7},
    )


def _plot_parity(model: Any, model_name: str, data: dict[str, Any], config: "VisualizationConfig", mode: str) -> None:
    fig, ax = plt.subplots(figsize=(float(FIG_STYLE["figure_width"]), float(FIG_STYLE["figure_height"])))
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    for split in _SPLITS:
        y_pred, y_true = _predict_split(model, data, split)
        all_true.append(y_true)
        all_pred.append(y_pred)
        ax.scatter(
            y_true,
            y_pred,
            alpha=float(FIG_STYLE["scatter_alpha"]),
            s=float(FIG_STYLE["marker_size"]),
            marker=_SPLIT_MARKERS[split],
            color=PALETTE[split],
            label=split,
        )
        _annotate_metrics(ax, split, compute_metrics(y_true, y_pred), _SPLITS.index(split))

    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)
    min_value = float(np.min([y_true_all.min(), y_pred_all.min()]))
    max_value = float(np.max([y_true_all.max(), y_pred_all.max()]))
    ax.plot([min_value, max_value], [min_value, max_value], linestyle="--", linewidth=1.0, color=PALETTE["ref"], label="y = x")
    apply_style(ax, f"Parity Plot — {model_name}", data["output_name"], data["output_name"])
    ax.legend()
    finalize_figure(fig, f"parity_{model_name}.png", config, mode)


def _plot_residuals(model: Any, model_name: str, data: dict[str, Any], config: "VisualizationConfig", mode: str) -> None:
    fig, ax = plt.subplots(figsize=(float(FIG_STYLE["figure_width"]), float(FIG_STYLE["figure_height"])))
    residual_limits: list[np.ndarray] = []

    for split in _SPLITS:
        y_pred, y_true = _predict_split(model, data, split)
        residuals = y_true - y_pred
        residual_limits.append(residuals)
        bins = max(10, int(np.sqrt(residuals.size)))
        ax.hist(residuals, bins=bins, density=True, alpha=float(FIG_STYLE["hist_alpha"]), color=PALETTE[split], label=f"{split} hist")
        if residuals.size > 1:
            kde = gaussian_kde(residuals)
            x_grid = np.linspace(float(residuals.min()), float(residuals.max()), 200)
            ax.plot(x_grid, kde(x_grid), color=PALETTE[split], linewidth=1.5, label=f"{split} KDE")
        stats_text = f"{split}: mean={residuals.mean():.3f}\nstd={residuals.std():.3f}\nMAE={np.mean(np.abs(residuals)):.3f}"
        ax.text(
            0.02,
            0.98 - _SPLITS.index(split) * 0.15,
            stats_text,
            transform=ax.transAxes,
            fontsize=int(FIG_STYLE["annotation_size"]),
            va="top",
            ha="left",
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": PALETTE[split], "alpha": 0.7},
        )

    combined = np.concatenate(residual_limits)
    limit = float(np.max(np.abs(combined))) if combined.size else 1.0
    ax.axvline(0.0, linestyle="--", linewidth=1.0, color=PALETTE["ref"], label="x = 0")
    ax.set_xlim(-limit, limit)
    apply_style(ax, f"Residual Distribution — {model_name}", f"Residual ({data['output_name']})", "Density")
    ax.legend()
    finalize_figure(fig, f"residuals_{model_name}.png", config, mode)


def run_parity(model: Any, model_name: str, data: dict[str, Any], config: "VisualizationConfig", mode: str) -> None:
    """Generate parity and residual visualizations for one model."""

    _plot_parity(model, model_name, data, config, mode)
    if config.parity.show_residuals:
        _plot_residuals(model, model_name, data, config, mode)
