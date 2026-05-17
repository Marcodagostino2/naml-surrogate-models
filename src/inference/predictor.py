"""Unified prediction interface for MLP / GP / MLPGP models, with denormalization."""

from __future__ import annotations

from typing import Optional, Tuple
import warnings

import numpy as np
import torch
from gpytorch.utils.warnings import GPInputWarning

from src.models import GPRegressionModel, MLPGPModel, MLPRegressor
from src.utils import resolve_device, setup_logger

logger = setup_logger(__name__)


def _validate_input_matrix(X: np.ndarray) -> np.ndarray:
    X_array = np.asarray(X)
    if X_array.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {X_array.shape}")
    return X_array


def _as_column_matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    return array.reshape(-1, 1) if array.ndim == 1 else array


class Predictor:
    """Run a trained model and undo target normalization."""

    def __init__(
        self,
        model: MLPRegressor | GPRegressionModel | MLPGPModel,
        target_scaler_mean: np.ndarray,
        target_scaler_std: np.ndarray,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.target_scaler_mean = target_scaler_mean
        self.target_scaler_std = target_scaler_std
        self.device = device or resolve_device()

        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Predictor initialized on device: {self.device}")

    def predict_mlp(self, X: np.ndarray, denormalize: bool = True) -> np.ndarray:
        X_t = torch.from_numpy(_validate_input_matrix(X)).to(self.device).to(torch.get_default_dtype())
        with torch.no_grad():
            y_pred = self.model(X_t).cpu().numpy()
        return y_pred * self.target_scaler_std + self.target_scaler_mean if denormalize else y_pred

    def predict_gp(self, X: np.ndarray, denormalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        X_t = torch.from_numpy(_validate_input_matrix(X)).to(self.device).to(torch.get_default_dtype())
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore", GPInputWarning)
            posterior = self.model.likelihood(self.model(X_t))
            mean = _as_column_matrix(posterior.mean.cpu().numpy())
            std = _as_column_matrix(posterior.stddev.cpu().numpy())

        if denormalize:
            # Mean shifts and scales; std only scales (linear in y).
            mean = mean * self.target_scaler_std + self.target_scaler_mean
            std = std * self.target_scaler_std
        return np.asarray(mean), np.asarray(std)

    def predict(
        self,
        X: np.ndarray,
        denormalize: bool = True,
        return_uncertainty: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """Auto-dispatch on model type. ``return_uncertainty`` only meaningful for GP-family models."""
        if isinstance(self.model, MLPRegressor):
            return self.predict_mlp(X, denormalize=denormalize)
        if isinstance(self.model, (GPRegressionModel, MLPGPModel)):
            mean, std = self.predict_gp(X, denormalize=denormalize)
            return (mean, std) if return_uncertainty else mean
        raise TypeError(f"Unknown model type: {type(self.model)}")
