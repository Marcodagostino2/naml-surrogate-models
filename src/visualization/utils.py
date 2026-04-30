"""Shared utilities for visualization modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING
import warnings
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from gpytorch.utils.warnings import GPInputWarning

if TYPE_CHECKING:
    from .config import VisualizationConfig

PALETTE: dict[str, str] = {
    "train": "#4C72B0",
    "val": "#DD8452",
    "test": "#55A868",
    "band": "#999999",
    "ref": "#000000",
}

FIG_STYLE: dict[str, float] = {
    "title_size": 14,
    "label_size": 12,
    "tick_size": 10,
    "annotation_size": 9,
    "figure_width": 12,
    "figure_height": 8,
    "surface_width": 12,
    "surface_height": 8,
    "table_font_size": 10,
    "table_col_width": 2.0,
    "table_row_height": 0.6,
    "table_height_offset": 1.0,
    "contour_levels": 20,
    "contour_line_levels": 6,
    "hist_alpha": 0.4,
    "band_alpha": 0.2,
    "density_alpha": 0.15,
    "dpi": 300,
    "line_width": 1.5,
    "reference_line_width": 1.0,
    "scatter_alpha": 0.6,
    "marker_size": 24.0,
}


def _as_1d_array(values: Any) -> np.ndarray:
    array = np.asarray(values).reshape(-1)
    if array.size == 0:
        raise ValueError("Expected a non-empty array")
    return array


def normalize_features(features: Any, mean: Any, std: Any) -> np.ndarray:
    """Normalize feature arrays with a stored mean and standard deviation."""

    features_array = np.asarray(features, dtype=float)
    mean_array = np.asarray(mean, dtype=float)
    std_array = np.asarray(std, dtype=float)
    std_array = np.where(np.abs(std_array) < 1e-12, 1.0, std_array)
    return (features_array - mean_array) / std_array


def sanitize_filename_component(value: Any) -> str:
    """Convert arbitrary text into a filesystem-safe filename fragment."""

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "plot"


def is_gp_model(model: Any) -> bool:
    """Return True when the model behaves like a Gaussian process regressor."""

    return bool(hasattr(model, "likelihood") and hasattr(model, "covar_module"))


def predict_mean(
    model: Any,
    features: Any,
    input_mean: Any,
    input_std: Any,
    target_mean: Any,
    target_std: Any,
) -> np.ndarray:
    """Predict denormalized means for a batch of feature vectors."""

    mean, _ = predict_mean_std(model, features, input_mean, input_std, target_mean, target_std)
    return mean


def predict_mean_std(
    model: Any,
    features: Any,
    input_mean: Any,
    input_std: Any,
    target_mean: Any,
    target_std: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict denormalized means and standard deviations for a batch."""

    normalized_features = normalize_features(features, input_mean, input_std)
    model.eval()
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        device = torch.device("cpu")

    tensor_features = torch.as_tensor(normalized_features, dtype=torch.get_default_dtype(), device=device)

    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", GPInputWarning)
        if is_gp_model(model):
            posterior = model.likelihood(model(tensor_features))
            mean = posterior.mean.detach().cpu().numpy()
            std = posterior.stddev.detach().cpu().numpy()
        else:
            mean = model(tensor_features).detach().cpu().numpy()
            std = np.zeros_like(mean)

    mean_array = _as_1d_array(mean)
    std_array = _as_1d_array(std)
    target_std_array = _as_1d_array(target_std)
    target_mean_array = _as_1d_array(target_mean)
    target_scale = float(target_std_array.reshape(-1)[0])
    target_shift = float(target_mean_array.reshape(-1)[0])
    return np.asarray(mean_array * target_scale + target_shift), np.asarray(std_array * abs(target_scale))


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return R², RMSE, MAE, MAPE, and MaxError for a single split."""

    y_true_array = _as_1d_array(y_true)
    y_pred_array = _as_1d_array(y_pred)
    if y_true_array.shape != y_pred_array.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    residuals = y_true_array - y_pred_array
    squared = residuals ** 2
    abs_true = np.abs(y_true_array)
    with np.errstate(divide="ignore", invalid="ignore"):
        percentage = np.where(abs_true > 0.0, np.abs(residuals) / abs_true, np.nan)
    mape = float(np.nanmean(percentage) * 100.0) if np.isfinite(np.nanmean(percentage)) else 0.0
    ss_res = float(np.sum(squared))
    centered = y_true_array - float(np.mean(y_true_array))
    ss_tot = float(np.sum(centered ** 2))
    r2 = 1.0 if ss_tot == 0.0 and ss_res == 0.0 else 0.0 if ss_tot == 0.0 else 1.0 - (ss_res / ss_tot)
    return {
        "r2": float(r2),
        "rmse": float(np.sqrt(np.mean(squared))),
        "mae": float(np.mean(np.abs(residuals))),
        "mape": float(mape),
        "max_error": float(np.max(np.abs(residuals))),
    }


def finalize_figure(fig: plt.Figure, filename: str, config: "VisualizationConfig", mode: str) -> None:
    """Save and/or show a figure using the shared visualization policy."""

    if mode not in {"train", "test"}:
        raise ValueError("mode must be 'train' or 'test'")

    if config.save_plots:
        output_path = Path(config.output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=int(FIG_STYLE["dpi"]), bbox_inches="tight")

    if mode == "test":
        plt.show(block=False)
        plt.pause(0.001)

    plt.close(fig)


def apply_style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply the shared title, labels, grid, and tick styling."""

    ax.set_title(title, fontsize=int(FIG_STYLE["title_size"]), fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=int(FIG_STYLE["label_size"]))
    ax.set_ylabel(ylabel, fontsize=int(FIG_STYLE["label_size"]))
    ax.tick_params(axis="both", labelsize=int(FIG_STYLE["tick_size"]))
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
