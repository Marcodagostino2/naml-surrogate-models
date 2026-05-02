"""Legacy training-time diagnostics: training-history curves and prediction-with-uncertainty bands.

Parity, residual, and MAPE plots are produced by the modular visualization
package (``src.visualization.modules.parity``). This module now only handles
artifacts that are not yet covered there: training-history curves and the
combined val/test uncertainty figure for GP-style models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.utils import setup_logger
from src.visualization.utils import set_plot_style

logger = setup_logger(__name__)
set_plot_style()


def plot_training_history(
    train_losses: list,
    val_losses: list,
    title: str = "Training History",
    figsize: Tuple[int, int] = (10, 6),
    output_path: Optional[Path | str] = None,
) -> None:
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=figsize)
    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Training Loss", marker="o", markersize=3)
    ax.plot(epochs, val_losses, label="Validation Loss", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()

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
    figsize: Tuple[int, int] = (14, 6),
    output_path: Optional[Path | str] = None,
) -> None:
    """Render val + test predictions-with-uncertainty as a single 2-subplot figure."""
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
    fig.suptitle(title, fontweight="bold")
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
    """Save training-history and (optionally) combined val/test uncertainty plots."""
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
