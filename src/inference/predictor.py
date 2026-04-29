"""
Inference module for making predictions with trained models.

Supports:
- MLP predictions
- GP predictions with uncertainty quantification
- Denormalization of outputs
"""

from __future__ import annotations

from typing import Optional, Tuple
import warnings

import gpytorch
import numpy as np
import torch
from gpytorch.utils.warnings import GPInputWarning

from src.models import GPRegressionModel, MLPGPModel, MLPRegressor
from src.utils import resolve_device, setup_logger

logger = setup_logger(__name__)


class Predictor:
    """
    Unified interface for making predictions with trained models.

    Supports MLP and GP models with automatic denormalization.
    """

    def __init__(
        self,
        model: MLPRegressor | GPRegressionModel | MLPGPModel,
        target_scaler_mean: np.ndarray,
        target_scaler_std: np.ndarray,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize Predictor.

        Parameters
        ----------
        model : MLPRegressor or GPRegressionModel or MLPGPModel
            Trained model.
        target_scaler_mean : np.ndarray
            Mean used for target normalization.
        target_scaler_std : np.ndarray
            Std used for target normalization.
        device : torch.device, optional
            Device to use. If None, auto-detected.
        """
        self.model = model
        self.target_scaler_mean = target_scaler_mean
        self.target_scaler_std = target_scaler_std
        self.device = device or resolve_device()

        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Predictor initialized on device: {self.device}")

    def predict_mlp(self, X: np.ndarray, denormalize: bool = True) -> np.ndarray:
        """
        Make MLP predictions.

        Parameters
        ----------
        X : np.ndarray
            Input features (n, n_features). Assumed to be already normalized.
        denormalize : bool, optional
            If True, denormalize predictions. Default: True.

        Returns
        -------
        np.ndarray
            Predictions (n, n_outputs).
        """
        X_t = torch.from_numpy(X).to(self.device).to(torch.get_default_dtype())

        with torch.no_grad():
            y_pred_normalized = self.model(X_t).cpu().numpy()

        if denormalize:
            y_pred = y_pred_normalized * self.target_scaler_std + self.target_scaler_mean
        else:
            y_pred = y_pred_normalized

        return y_pred

    def predict_gp(
        self,
        X: np.ndarray,
        denormalize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make GP predictions with uncertainty.

        Parameters
        ----------
        X : np.ndarray
            Input features (n, n_features). Assumed to be already normalized.
        denormalize : bool, optional
            If True, denormalize predictions. Default: True.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (mean, std) of predictions. Both shape (n, n_outputs).
        """
        X_t = torch.from_numpy(X).to(self.device).to(torch.get_default_dtype())

        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore", GPInputWarning)
            posterior = self.model.likelihood(self.model(X_t))
            mean_normalized = posterior.mean.cpu().numpy()
            std_normalized = posterior.stddev.cpu().numpy()

        mean_normalized = np.asarray(mean_normalized).reshape(-1)
        std_normalized = np.asarray(std_normalized).reshape(-1)

        if denormalize:
            mean = mean_normalized * self.target_scaler_std + self.target_scaler_mean
            # Uncertainty scales linearly
            std = std_normalized * self.target_scaler_std
        else:
            mean = mean_normalized
            std = std_normalized

        return np.asarray(mean).reshape(-1, 1) if np.asarray(mean).ndim == 1 else np.asarray(mean), np.asarray(std).reshape(-1, 1) if np.asarray(std).ndim == 1 else np.asarray(std)

    def predict(
        self,
        X: np.ndarray,
        denormalize: bool = True,
        return_uncertainty: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with automatic model type detection.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        denormalize : bool, optional
            If True, denormalize predictions. Default: True.
        return_uncertainty : bool, optional
            If True and model is GP, return (mean, std). Otherwise, return mean. Default: False.

        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            Predictions, or (mean, std) if return_uncertainty=True.
        """
        if isinstance(self.model, MLPRegressor):
            return self.predict_mlp(X, denormalize=denormalize)
        elif isinstance(self.model, (GPRegressionModel, MLPGPModel)):
            mean, std = self.predict_gp(X, denormalize=denormalize)
            if return_uncertainty:
                return mean, std
            else:
                return mean
        else:
            raise TypeError(f"Unknown model type: {type(self.model)}")
