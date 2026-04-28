"""
Evaluation metrics for regression models.

Provides:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² score
- Mean Absolute Percentage Error (MAPE)
"""

from __future__ import annotations

import numpy as np
import torch


def _to_numpy_1d_or_2d(value: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert tensors/arrays to a NumPy array with at least one dimension."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    array = np.asarray(value)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim > 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {array.shape}")
    return array


def mse(y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor) -> float:
    """
    Compute Mean Squared Error.

    Parameters
    ----------
    y_true : np.ndarray or torch.Tensor
        True values.
    y_pred : np.ndarray or torch.Tensor
        Predicted values.

    Returns
    -------
    float
        MSE value.
    """
    y_true = _to_numpy_1d_or_2d(y_true)
    y_pred = _to_numpy_1d_or_2d(y_pred)

    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    return np.mean((y_true - y_pred) ** 2)


def mae(y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor) -> float:
    """
    Compute Mean Absolute Error.

    Parameters
    ----------
    y_true : np.ndarray or torch.Tensor
        True values.
    y_pred : np.ndarray or torch.Tensor
        Predicted values.

    Returns
    -------
    float
        MAE value.
    """
    y_true = np.squeeze(_to_numpy_1d_or_2d(y_true))
    y_pred = np.squeeze(_to_numpy_1d_or_2d(y_pred))

    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor) -> float:
    """
    Compute R² score.

    Parameters
    ----------
    y_true : np.ndarray or torch.Tensor
        True values.
    y_pred : np.ndarray or torch.Tensor
        Predicted values.

    Returns
    -------
    float
        R² score (1.0 is perfect, 0.0 is baseline, negative is worse than baseline).
    """
    y_true = np.squeeze(_to_numpy_1d_or_2d(y_true))
    y_pred = np.squeeze(_to_numpy_1d_or_2d(y_pred))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return 1.0 - (ss_res / ss_tot)


def rmse(y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor) -> float:
    """
    Compute Root Mean Squared Error.

    Parameters
    ----------
    y_true : np.ndarray or torch.Tensor
        True values.
    y_pred : np.ndarray or torch.Tensor
        Predicted values.

    Returns
    -------
    float
        RMSE value.
    """
    return np.sqrt(mse(y_true, y_pred))


def mape(y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor) -> float:
    """
    Compute Mean Absolute Percentage Error.

    Parameters
    ----------
    y_true : np.ndarray or torch.Tensor
        True values.
    y_pred : np.ndarray or torch.Tensor
        Predicted values.

    Returns
    -------
    float
        MAPE value (as percentage).
    """
    y_true = np.squeeze(_to_numpy_1d_or_2d(y_true))
    y_pred = np.squeeze(_to_numpy_1d_or_2d(y_pred))

    mask = y_true != 0
    if not mask.any():
        return 0.0

    return 100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def compute_all_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
) -> dict:
    """
    Compute all standard regression metrics.

    Parameters
    ----------
    y_true : np.ndarray or torch.Tensor
        True values.
    y_pred : np.ndarray or torch.Tensor
        Predicted values.

    Returns
    -------
    dict
        Dictionary with keys: "mse", "mae", "rmse", "r2", "mape".
    """
    return {
        "mse": mse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }
