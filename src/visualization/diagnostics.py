"""
Plotting and diagnostics module for visualization of model predictions.

Generates:
- Parity plots (predicted vs. actual)
- Residual diagnostic plots
- Training history plots
- Uncertainty plots (for GP models)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.utils import setup_logger

logger = setup_logger(__name__)


def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Parity Plot",
    figsize: Tuple[int, int] = (10, 8),
    output_path: Optional[Path | str] = None,
) -> None:
    """
    Create a parity (predicted vs. actual) plot.

    Parameters
    ----------
    y_true : np.ndarray
        True values (n,) or (n, 1).
    y_pred : np.ndarray
        Predicted values, same shape as y_true.
    title : str, optional
        Plot title. Default: "Parity Plot".
    figsize : Tuple[int, int], optional
        Figure size (width, height). Default: (10, 8).
    output_path : Path or str, optional
        If provided, save plot to this path.
    """
    y_true = np.asarray(y_true).squeeze()
    y_pred = np.asarray(y_pred).squeeze()

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidth=0.5)

    # Ideal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal")

    ax.set_xlabel("True Values", fontsize=12)
    ax.set_ylabel("Predicted Values", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Parity plot saved to {output_path}")

    plt.close(fig)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Plot",
    figsize: Tuple[int, int] = (12, 5),
    output_path: Optional[Path | str] = None,
) -> None:
    """
    Create residual diagnostic plots (residuals vs. predicted, histogram).

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    title : str, optional
        Plot title prefix. Default: "Residual Plot".
    figsize : Tuple[int, int], optional
        Figure size. Default: (12, 5).
    output_path : Path or str, optional
        If provided, save plot to this path.
    """
    y_true = np.asarray(y_true).squeeze()
    y_pred = np.asarray(y_pred).squeeze()
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Residuals vs. predicted
    axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidth=0.5)
    axes[0].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[0].set_xlabel("Predicted Values", fontsize=11)
    axes[0].set_ylabel("Residuals", fontsize=11)
    axes[0].set_title(f"{title} (vs. Predicted)", fontsize=12, fontweight="bold")
    axes[0].grid(alpha=0.3)

    # Residuals histogram
    axes[1].hist(residuals, bins=30, edgecolor="k", alpha=0.7)
    axes[1].set_xlabel("Residuals", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].set_title(f"{title} (Distribution)", fontsize=12, fontweight="bold")
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Residual plot saved to {output_path}")

    plt.close(fig)


def plot_training_history(
    train_losses: list,
    val_losses: list,
    title: str = "Training History",
    figsize: Tuple[int, int] = (10, 6),
    output_path: Optional[Path | str] = None,
) -> None:
    """
    Plot training and validation loss curves.

    Parameters
    ----------
    train_losses : list
        Training loss history.
    val_losses : list
        Validation loss history.
    title : str, optional
        Plot title. Default: "Training History".
    figsize : Tuple[int, int], optional
        Figure size. Default: (10, 6).
    output_path : Path or str, optional
        If provided, save plot to this path.
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Training Loss", marker="o", markersize=3)
    ax.plot(epochs, val_losses, label="Validation Loss", marker="s", markersize=3)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Training history plot saved to {output_path}")

    plt.close(fig)


def plot_uncertainty(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    title: str = "Predictions with Uncertainty",
    figsize: Tuple[int, int] = (12, 6),
    output_path: Optional[Path | str] = None,
) -> None:
    """
    Plot predictions with uncertainty bands (for GP models).

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted means.
    y_std : np.ndarray
        Predicted standard deviations.
    title : str, optional
        Plot title. Default: "Predictions with Uncertainty".
    figsize : Tuple[int, int], optional
        Figure size. Default: (12, 6).
    output_path : Path or str, optional
        If provided, save plot to this path.
    """
    y_true = np.asarray(y_true).squeeze()
    y_pred = np.asarray(y_pred).squeeze()
    y_std = np.asarray(y_std).squeeze()

    fig, ax = plt.subplots(figsize=figsize)

    # Sort by predictions for better visualization
    sort_idx = np.argsort(y_pred)
    y_true_sorted = y_true[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    y_std_sorted = y_std[sort_idx]

    x_axis = np.arange(len(y_pred_sorted))

    # Plot true values
    ax.scatter(x_axis, y_true_sorted, label="True Values", marker="o", alpha=0.6, s=30, color="blue")

    # Plot predictions with uncertainty band
    ax.plot(x_axis, y_pred_sorted, label="Predictions", color="red", linewidth=1.5)
    ax.fill_between(
        x_axis,
        y_pred_sorted - 2 * y_std_sorted,
        y_pred_sorted + 2 * y_std_sorted,
        alpha=0.2,
        color="red",
        label="±2σ Uncertainty",
    )

    ax.set_xlabel("Sample Index (sorted)", fontsize=12)
    ax.set_ylabel("Output Value", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Uncertainty plot saved to {output_path}")

    plt.close(fig)


def generate_diagnostics(
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
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
    """
    Generate a comprehensive set of diagnostic plots.

    Parameters
    ----------
    y_train_true, y_train_pred : np.ndarray
        Training data (true and predicted).
    y_val_true, y_val_pred : np.ndarray
        Validation data (true and predicted).
    y_test_true, y_test_pred : np.ndarray
        Test data (true and predicted).
    y_val_std, y_test_std : np.ndarray, optional
        Uncertainty estimates (for GP models).
    train_losses, val_losses : list, optional
        Training history.
    output_dir : Path or str, optional
        Output directory for plots. If None, plots are not saved.
    """
    output_dir = Path(output_dir) if output_dir else None

    # Parity plots
    plot_parity(
        y_train_true,
        y_train_pred,
        title="Train Parity Plot",
        output_path=output_dir / "01_train_parity.png" if output_dir else None,
    )
    plot_parity(
        y_val_true,
        y_val_pred,
        title="Validation Parity Plot",
        output_path=output_dir / "02_val_parity.png" if output_dir else None,
    )
    plot_parity(
        y_test_true,
        y_test_pred,
        title="Test Parity Plot",
        output_path=output_dir / "03_test_parity.png" if output_dir else None,
    )

    # Residual plots
    plot_residuals(
        y_train_true,
        y_train_pred,
        title="Train Residuals",
        output_path=output_dir / "04_train_residuals.png" if output_dir else None,
    )
    plot_residuals(
        y_test_true,
        y_test_pred,
        title="Test Residuals",
        output_path=output_dir / "05_test_residuals.png" if output_dir else None,
    )

    # Training history
    if train_losses is not None and val_losses is not None:
        plot_training_history(
            train_losses,
            val_losses,
            output_path=output_dir / "06_training_history.png" if output_dir else None,
        )

    # Uncertainty plots (if available)
    if y_val_std is not None:
        plot_uncertainty(
            y_val_true,
            y_val_pred,
            y_val_std,
            title="Validation Predictions with Uncertainty",
            output_path=output_dir / "07_val_uncertainty.png" if output_dir else None,
        )

    if y_test_std is not None:
        plot_uncertainty(
            y_test_true,
            y_test_pred,
            y_test_std,
            title="Test Predictions with Uncertainty",
            output_path=output_dir / "08_test_uncertainty.png" if output_dir else None,
        )

    logger.info("Diagnostic plots generated.")
