"""Gaussian-process uncertainty visualizations for one model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, norm

from ..utils import FIG_STYLE, PALETTE, apply_style, finalize_figure, predict_mean_std

if TYPE_CHECKING:
    from ..config import VisualizationConfig


def _predict_sigma(model: Any, data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return predict_mean_std(
        model,
        data["X_test"],
        data["input_scaler_mean"],
        data["input_scaler_std"],
        data["target_scaler_mean"],
        data["target_scaler_std"],
    )


def _plot_calibration(sigma: np.ndarray, error: np.ndarray, model_name: str, config: "VisualizationConfig", mode: str) -> None:
    fig, ax = plt.subplots(figsize=(float(FIG_STYLE["figure_width"]), float(FIG_STYLE["figure_height"])))
    ax.hexbin(sigma, error, gridsize=30, cmap="Blues", mincnt=1)
    ax.scatter(sigma, error, alpha=0.4, color=PALETTE["test"], s=float(FIG_STYLE["marker_size"]))
    if sigma.size > 1 and not np.allclose(sigma, sigma[0]):
        slope, intercept = np.polyfit(sigma, error, 1)
        line_x = np.linspace(float(sigma.min()), float(sigma.max()), 200)
        ax.plot(line_x, slope * line_x + intercept, color=PALETTE["ref"], linewidth=1.5, label="fit")
    else:
        slope = 0.0
    ref_limit = float(max(sigma.max(), error.max())) if sigma.size and error.size else 1.0
    ax.plot([0.0, ref_limit], [0.0, ref_limit], linestyle="--", linewidth=1.0, color=PALETTE["ref"], label="y = x")
    correlation = float(np.corrcoef(sigma, error)[0, 1]) if sigma.size > 1 else 0.0
    ax.text(0.02, 0.98, f"Pearson r = {correlation:.3f}\nslope = {slope:.3f}", transform=ax.transAxes, va="top", ha="left", fontsize=int(FIG_STYLE["annotation_size"]), bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7})
    apply_style(ax, f"Calibration — {model_name}", "Predicted σ", "Absolute error")
    ax.legend()
    finalize_figure(fig, f"uncertainty_calibration_{model_name}.png", config, mode)


def _plot_reliability(sigma: np.ndarray, error: np.ndarray, model_name: str, config: "VisualizationConfig", mode: str) -> None:
    alphas = np.linspace(0.05, 0.95, 19)
    expected = 1.0 - alphas
    empirical = np.array([np.mean(error <= norm.ppf(1.0 - alpha / 2.0) * sigma) for alpha in alphas])
    fig, ax = plt.subplots(figsize=(float(FIG_STYLE["figure_width"]), float(FIG_STYLE["figure_height"])))
    ax.plot(expected, empirical, color=PALETTE["test"], marker="o", linewidth=1.5, label="empirical")
    ax.plot(expected, expected, linestyle="--", linewidth=1.0, color=PALETTE["ref"], label="ideal")
    ax.fill_between(expected, expected, empirical, color=PALETTE["band"], alpha=float(FIG_STYLE["density_alpha"]))
    apply_style(ax, f"Reliability — {model_name}", "Expected coverage", "Empirical coverage")
    ax.legend()
    finalize_figure(fig, f"uncertainty_reliability_{model_name}.png", config, mode)


def _plot_sigma_vs_error_dist(sigma: np.ndarray, error: np.ndarray, model_name: str, config: "VisualizationConfig", mode: str) -> None:
    fig, ax = plt.subplots(figsize=(float(FIG_STYLE["figure_width"]), float(FIG_STYLE["figure_height"])))
    for values, label, color in ((sigma, "σ", PALETTE["train"]), (error, "|error|", PALETTE["test"])):
        bins = max(10, int(np.sqrt(values.size)))
        ax.hist(values, bins=bins, density=True, alpha=float(FIG_STYLE["hist_alpha"]), color=color, label=f"{label} hist")
        if values.size > 1 and np.std(values) > 0:
            kde = gaussian_kde(values)
            x_grid = np.linspace(float(values.min()), float(values.max()), 200)
            ax.plot(x_grid, kde(x_grid), color=color, linewidth=1.5, label=f"{label} KDE")
    apply_style(ax, f"Uncertainty vs Error Distribution — {model_name}", "Value", "Density")
    ax.legend()
    finalize_figure(fig, f"uncertainty_distributions_{model_name}.png", config, mode)


def run_uncertainty_gp(model: Any, model_name: str, data: dict[str, Any], config: "VisualizationConfig", mode: str) -> None:
    """Generate calibration and uncertainty diagnostics for one GP model."""

    y_pred, sigma = _predict_sigma(model, data)
    error = np.abs(y_pred - np.asarray(data["y_test"]).reshape(-1))
    _plot_calibration(sigma, error, model_name, config, mode)
    _plot_reliability(sigma, error, model_name, config, mode)
    _plot_sigma_vs_error_dist(sigma, error, model_name, config, mode)
