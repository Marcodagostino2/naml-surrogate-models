"""Parity, residual, and MAPE-distribution visualizations for a single model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from ..utils import (
    FIG_STYLE,
    PALETTE,
    apply_style,
    axis_label,
    compute_metrics,
    finalize_figure,
    get_output_display,
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


def _annotate_box(ax: plt.Axes, text: str, edge: str, *, position: tuple[float, float] = (0.02, 0.98)) -> None:
    ax.text(
        position[0],
        position[1],
        text,
        transform=ax.transAxes,
        fontsize=int(FIG_STYLE["annotation_size"]),
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": edge, "alpha": 0.7},
    )


def _plot_parity(model: Any, model_name: str, data: dict[str, Any], config: "VisualizationConfig", mode: str) -> None:
    output_name, output_unit = get_output_display(data)
    label = axis_label(output_name, output_unit)
    fig, ax = plt.subplots(figsize=(float(FIG_STYLE["figure_width"]), float(FIG_STYLE["figure_height"])))
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    for index, split in enumerate(_SPLITS):
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
        metrics = compute_metrics(y_true, y_pred)
        annotation = (
            f"{split}: $R^2$={metrics['r2']:.3f}\n"
            f"RMSE={metrics['rmse']:.3f}\n"
            f"MAPE={metrics['mape']:.2f}%"
        )
        _annotate_box(ax, annotation, PALETTE[split], position=(0.02, 0.98 - index * 0.18))

    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)
    min_value = float(np.min([y_true_all.min(), y_pred_all.min()]))
    max_value = float(np.max([y_true_all.max(), y_pred_all.max()]))
    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        linewidth=float(FIG_STYLE["reference_line_width"]),
        color=PALETTE["ref"],
        label="$y = x$",
    )
    apply_style(ax, f"Parity Plot — {model_name}", f"True {label}", f"Predicted {label}")
    ax.legend()
    finalize_figure(fig, f"parity_{model_name}.png", config, mode)


def _plot_residuals(model: Any, model_name: str, data: dict[str, Any], config: "VisualizationConfig", mode: str) -> None:
    output_name, output_unit = get_output_display(data)
    residual_label = axis_label(f"Residual ({output_name})", output_unit)

    residuals_per_split: dict[str, np.ndarray] = {}
    for split in _SPLITS:
        y_pred, y_true = _predict_split(model, data, split)
        residuals_per_split[split] = y_true - y_pred

    combined = np.concatenate(list(residuals_per_split.values()))
    limit = float(np.max(np.abs(combined))) if combined.size else 1.0

    fig, axes = plt.subplots(
        1,
        len(_SPLITS),
        figsize=(float(FIG_STYLE["figure_width"]), float(FIG_STYLE["figure_height"]) * 0.6),
        sharey=True,
    )

    for ax, split in zip(axes, _SPLITS):
        residuals = residuals_per_split[split]
        bins = max(10, int(np.sqrt(residuals.size)))
        ax.hist(
            residuals,
            bins=bins,
            density=True,
            alpha=float(FIG_STYLE["hist_alpha"]),
            color=PALETTE[split],
            label=f"{split} hist",
        )
        if residuals.size > 1 and float(residuals.std()) > 1e-12:
            kde = gaussian_kde(residuals)
            x_grid = np.linspace(float(residuals.min()), float(residuals.max()), 200)
            ax.plot(x_grid, kde(x_grid), color=PALETTE[split], linewidth=float(FIG_STYLE["line_width"]), label="KDE")
        ax.axvline(
            0.0,
            linestyle="--",
            linewidth=float(FIG_STYLE["reference_line_width"]),
            color=PALETTE["ref"],
        )
        ax.set_xlim(-limit, limit)
        stats = (
            f"mean={residuals.mean():.3f}\n"
            f"std={residuals.std():.3f}\n"
            f"MAE={np.mean(np.abs(residuals)):.3f}"
        )
        _annotate_box(ax, stats, PALETTE[split])
        apply_style(ax, split, residual_label, "Density")

    fig.suptitle(f"Residual Distributions — {model_name}", fontsize=int(FIG_STYLE["title_size"]), fontweight="bold")
    fig.tight_layout()
    finalize_figure(fig, f"residuals_{model_name}.png", config, mode)


def _plot_mape(model: Any, model_name: str, data: dict[str, Any], config: "VisualizationConfig", mode: str) -> None:
    """Plot per-sample MAPE distributions for train/val/test as 3 subplots."""

    mape_per_split: dict[str, np.ndarray] = {}
    for split in _SPLITS:
        y_pred, y_true = _predict_split(model, data, split)
        denom = np.where(np.abs(y_true) > 1e-12, np.abs(y_true), np.nan)
        ape = np.abs(y_true - y_pred) / denom * 100.0
        mape_per_split[split] = ape[np.isfinite(ape)]

    finite = np.concatenate([m for m in mape_per_split.values() if m.size]) if any(m.size for m in mape_per_split.values()) else np.array([0.0])
    upper = float(np.percentile(finite, 99)) if finite.size else 1.0
    upper = max(upper, 1e-3)

    fig, axes = plt.subplots(
        1,
        len(_SPLITS),
        figsize=(float(FIG_STYLE["figure_width"]), float(FIG_STYLE["figure_height"]) * 0.6),
        sharey=True,
    )

    for ax, split in zip(axes, _SPLITS):
        values = mape_per_split[split]
        if values.size == 0:
            ax.text(0.5, 0.5, "no finite MAPE", transform=ax.transAxes, ha="center", va="center")
            apply_style(ax, split, "MAPE [%]", "Density")
            continue
        bins = max(10, int(np.sqrt(values.size)))
        ax.hist(
            values,
            bins=bins,
            range=(0.0, upper),
            density=True,
            alpha=float(FIG_STYLE["hist_alpha"]),
            color=PALETTE[split],
            label=f"{split} hist",
        )
        if values.size > 1 and float(values.std()) > 1e-12:
            kde = gaussian_kde(values)
            x_grid = np.linspace(0.0, upper, 200)
            ax.plot(x_grid, kde(x_grid), color=PALETTE[split], linewidth=float(FIG_STYLE["line_width"]), label="KDE")
        stats = (
            f"mean={values.mean():.2f}%\n"
            f"median={np.median(values):.2f}%\n"
            f"p95={np.percentile(values, 95):.2f}%"
        )
        _annotate_box(ax, stats, PALETTE[split])
        ax.set_xlim(0.0, upper)
        apply_style(ax, split, "MAPE [%]", "Density")

    fig.suptitle(f"MAPE Distributions — {model_name}", fontsize=int(FIG_STYLE["title_size"]), fontweight="bold")
    fig.tight_layout()
    finalize_figure(fig, f"mape_{model_name}.png", config, mode)


def run_parity(model: Any, model_name: str, data: dict[str, Any], config: "VisualizationConfig", mode: str) -> None:
    """Generate parity, residual, and MAPE-distribution visualizations for one model."""

    _plot_parity(model, model_name, data, config, mode)
    if config.parity.show_residuals:
        _plot_residuals(model, model_name, data, config, mode)
    if config.parity.show_mape:
        _plot_mape(model, model_name, data, config, mode)
