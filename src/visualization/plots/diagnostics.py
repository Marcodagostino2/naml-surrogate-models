"""Training-history curves and uncertainty bands. Parity/residual/MAPE live in ``visualization.modules``."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.utils import setup_logger

logger = setup_logger(__name__)

DEFAULT_HISTORY_FIGSIZE: Tuple[int, int] = (10, 6)
DEFAULT_UNCERTAINTY_FIGSIZE: Tuple[int, int] = (14, 6)


def _series_to_array(series: list | None) -> np.ndarray:
    """``None`` entries become NaN so matplotlib draws a gap instead of raising."""
    if series is None:
        return np.array([], dtype=float)
    return np.array([np.nan if v is None else float(v) for v in series], dtype=float)


def plot_training_history(
    train_losses: list,
    val_losses: list,
    title: str = "Training History",
    figsize: Tuple[int, int] = DEFAULT_HISTORY_FIGSIZE,
    output_path: Optional[Path | str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    train_arr = _series_to_array(train_losses)
    val_arr = _series_to_array(val_losses)

    plotted_anything = False
    if train_arr.size and np.isfinite(train_arr).any():
        ax.plot(np.arange(1, train_arr.size + 1), train_arr, label="Training Loss", marker="o", markersize=3)
        plotted_anything = True
    if val_arr.size and np.isfinite(val_arr).any():
        ax.plot(np.arange(1, val_arr.size + 1), val_arr, label="Validation Loss", marker="s", markersize=3)
        plotted_anything = True

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    if plotted_anything:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No finite loss values to plot", ha="center", va="center", transform=ax.transAxes)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_uncertainty_panel(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    title: str,
) -> None:
    sort_idx = np.argsort(y_pred)
    x_axis = np.arange(y_pred.size)
    ax.scatter(x_axis, y_true[sort_idx], label="True", marker="o", alpha=0.6, s=24, color="#4C72B0")
    ax.plot(x_axis, y_pred[sort_idx], label="Predicted", color="#C44E52", linewidth=1.5)
    ax.fill_between(
        x_axis,
        y_pred[sort_idx] - 2 * y_std[sort_idx],
        y_pred[sort_idx] + 2 * y_std[sort_idx],
        alpha=0.2,
        color="#C44E52",
        label=r"$\pm 2\sigma$",
    )
    ax.set_xlabel("Sample index (sorted)")
    ax.set_ylabel("Output")
    ax.set_title(title)
    ax.legend(loc="best")


def plot_uncertainty_combined(
    y_val_true: np.ndarray,
    y_val_pred: np.ndarray,
    y_val_std: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    y_test_std: np.ndarray,
    title: str = "Predictions with Uncertainty",
    figsize: Tuple[int, int] = DEFAULT_UNCERTAINTY_FIGSIZE,
    output_path: Optional[Path | str] = None,
) -> None:
    """Val and test predictions side-by-side with shared y-axis."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    _plot_uncertainty_panel(
        axes[0],
        np.asarray(y_val_true).squeeze(),
        np.asarray(y_val_pred).squeeze(),
        np.asarray(y_val_std).squeeze(),
        "Validation",
    )
    _plot_uncertainty_panel(
        axes[1],
        np.asarray(y_test_true).squeeze(),
        np.asarray(y_test_pred).squeeze(),
        np.asarray(y_test_std).squeeze(),
        "Test",
    )
    fig.suptitle(title)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generate_diagnostics(
    y_val_true: np.ndarray,
    y_val_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    y_val_std: Optional[np.ndarray] = None,
    y_test_std: Optional[np.ndarray] = None,
    train_losses: Optional[list] = None,
    val_losses: Optional[list] = None,
    output_dir: Optional[Path | str] = None,
) -> None:
    output_dir = Path(output_dir) if output_dir else None

    if train_losses is not None and val_losses is not None and len(train_losses) > 0:
        plot_training_history(
            train_losses,
            val_losses,
            output_path=output_dir / "training_history.png" if output_dir else None,
        )

    if y_val_std is not None and y_test_std is not None:
        plot_uncertainty_combined(
            y_val_true=y_val_true,
            y_val_pred=y_val_pred,
            y_val_std=y_val_std,
            y_test_true=y_test_true,
            y_test_pred=y_test_pred,
            y_test_std=y_test_std,
            output_path=output_dir / "uncertainty_val_test.png" if output_dir else None,
        )
