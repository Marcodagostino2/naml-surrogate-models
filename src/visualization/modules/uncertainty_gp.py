"""Gaussian-process uncertainty visualizations for one model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, norm

from src.utils import setup_logger
from ..utils import FIG_STYLE, PALETTE, apply_style, finalize_figure, predict_mean_std

logger = setup_logger(__name__)

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
    
    # Compute correlation safely with guard against degenerate data
    correlation = 0.0
    if sigma.size > 1 and error.size > 1:
        sigma_std = np.std(sigma)
        error_std = np.std(error)
        if sigma_std > 1e-10 and error_std > 1e-10:
            correlation = float(np.corrcoef(sigma, error)[0, 1])
            if not np.isfinite(correlation):
                correlation = 0.0
    
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
    """
    Plot 2D joint distribution of sigma and error with KDE.
    
    Skips KDE and falls back to scatter if data has near-zero variance,
    which is common early in training when the MLP outputs uniform sigma.
    """
    fig, ax = plt.subplots(figsize=(float(FIG_STYLE["figure_width"]), float(FIG_STYLE["figure_height"])))
    
    # Stack data for KDE: (2, n)
    values = np.stack([sigma, error], axis=0)
    
    # Check variance of each dimension
    row_stds = np.std(values, axis=1)
    has_degenerate_data = np.any(row_stds < 1e-10) or np.any(~np.isfinite(row_stds))
    
    if has_degenerate_data:
        logger.warning(
            "Skipping sigma-vs-error KDE plot: data has near-zero or invalid variance "
            f"(stds: sigma={row_stds[0]:.2e}, error={row_stds[1]:.2e}). "
            "This is expected if the MLP outputs uniform sigma values."
        )
        # Fallback: plot scatter without KDE
        ax.scatter(sigma, error, alpha=0.5, color=PALETTE["test"], s=float(FIG_STYLE["marker_size"]))
        apply_style(ax, f"Σ vs Error Distribution — {model_name}", "Predicted σ", "Absolute error")
        finalize_figure(fig, f"uncertainty_distributions_{model_name}.png", config, mode)
        return
    
    # Sufficient variance: use 2D KDE
    try:
        kde = gaussian_kde(values)
        
        # Create grid for KDE evaluation
        sigma_grid = np.linspace(float(sigma.min()), float(sigma.max()), 100)
        error_grid = np.linspace(float(error.min()), float(error.max()), 100)
        sigma_mesh, error_mesh = np.meshgrid(sigma_grid, error_grid)
        
        # Evaluate KDE on grid
        grid_points = np.vstack([sigma_mesh.ravel(), error_mesh.ravel()])
        kde_values = kde(grid_points).reshape(sigma_mesh.shape)
        
        # Plot density contours and scatter
        contour = ax.contourf(sigma_mesh, error_mesh, kde_values, levels=15, cmap="Blues", alpha=0.7)
        ax.contour(sigma_mesh, error_mesh, kde_values, levels=5, colors="black", alpha=0.3, linewidths=0.5)
        ax.scatter(sigma, error, alpha=0.3, color=PALETTE["test"], s=float(FIG_STYLE["marker_size"]), label="data")
        plt.colorbar(contour, ax=ax, label="Density")
        
    except Exception as e:
        logger.warning(
            f"Failed to compute 2D KDE for {model_name}: {e}. Falling back to scatter plot."
        )
        ax.scatter(sigma, error, alpha=0.5, color=PALETTE["test"], s=float(FIG_STYLE["marker_size"]))
    
    apply_style(ax, f"Σ vs Error Distribution — {model_name}", "Predicted σ", "Absolute error")
    ax.legend()
    finalize_figure(fig, f"uncertainty_distributions_{model_name}.png", config, mode)


def run_uncertainty_gp(model: Any, model_name: str, data: dict[str, Any], config: "VisualizationConfig", mode: str) -> None:
    """Generate calibration and uncertainty diagnostics for one GP model."""

    y_pred, sigma = _predict_sigma(model, data)
    error = np.abs(y_pred - np.asarray(data["y_test"]).reshape(-1))
    _plot_calibration(sigma, error, model_name, config, mode)
    _plot_reliability(sigma, error, model_name, config, mode)
    _plot_sigma_vs_error_dist(sigma, error, model_name, config, mode)
