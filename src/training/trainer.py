"""
Training routines for MLP and GP models.

Supports:
- MLP training with early stopping
- GP hyperparameter optimization via Exact Marginal Log-Likelihood
- Checkpointing and model saving
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gpytorch
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models import GPRegressionModel, MLPRegressor
from src.training.metrics import compute_all_metrics
from src.utils import resolve_device, setup_logger

logger = setup_logger(__name__)


class MLPTrainer:
    """
    Trainer for Multi-Layer Perceptron models.

    Features:
    - Configurable optimizer and loss function
    - Early stopping with patience
    - Checkpoint saving
    - Automatic device selection
    """

    def __init__(
        self,
        model: MLPRegressor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        training_cfg: Dict[str, Any],
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize MLPTrainer.

        Parameters
        ----------
        model : MLPRegressor
            The MLP model to train.
        train_loader : DataLoader
            Training data loader.
        val_loader : DataLoader
            Validation data loader.
        training_cfg : Dict[str, Any]
            Training configuration with keys: optimizer, learning_rate,
            n_epochs, loss_function, early_stopping_patience, etc.
        device : torch.device, optional
            Device to use. If None, auto-detected.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.training_cfg = training_cfg
        self.device = device or resolve_device(training_cfg.get("device", "auto"))

        self.model.to(self.device)

        # Setup optimizer
        optimizer_name = training_cfg.get("optimizer", "adam").lower()
        lr = float(training_cfg.get("learning_rate", 0.001))

        if optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        elif optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Setup loss function
        loss_name = training_cfg.get("loss_function", "mse").lower()
        if loss_name == "mse":
            self.loss_fn = torch.nn.MSELoss()
        elif loss_name == "mae":
            self.loss_fn = torch.nn.L1Loss()
        elif loss_name == "huber":
            self.loss_fn = torch.nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

        # Early stopping
        self.early_stopping_patience = int(training_cfg.get("early_stopping_patience", 20))
        self.early_stopping_min_delta = float(training_cfg.get("early_stopping_min_delta", 1e-5))
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # History
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []

        logger.info(f"MLPTrainer initialized on device: {self.device}")

    def _train_epoch(self) -> float:
        """
        Run a single training epoch.

        Returns
        -------
        float
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0

        for X_batch, y_batch in self.train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss = self.loss_fn(y_pred, y_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * X_batch.shape[0]

        avg_loss = total_loss / len(self.train_loader.dataset)
        return avg_loss

    def _validate(self) -> Tuple[float, Dict[str, float]]:
        """
        Run validation.

        Returns
        -------
        Tuple[float, Dict[str, float]]
            (validation_loss, metrics_dict)
        """
        self.model.eval()
        y_true_list = []
        y_pred_list = []
        total_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                y_pred = self.model(X_batch)
                loss = self.loss_fn(y_pred, y_batch)

                total_loss += loss.item() * X_batch.shape[0]
                y_true_list.append(y_batch.cpu().numpy())
                y_pred_list.append(y_pred.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader.dataset)
        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)

        metrics = compute_all_metrics(y_true, y_pred)
        return avg_loss, metrics

    def train(self) -> Dict[str, Any]:
        """
        Run the complete training loop.

        Returns
        -------
        Dict[str, Any]
            Training history and final metrics.
        """
        n_epochs = int(self.training_cfg.get("n_epochs", 200))
        logger.info(f"Starting MLP training for {n_epochs} epochs")

        start_time = time.time()

        for epoch in range(1, n_epochs + 1):
            train_loss = self._train_epoch()
            val_loss, val_metrics = self._validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)

            # Early stopping check
            if val_loss < self.best_val_loss - self.early_stopping_min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}/{n_epochs} - Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}, Val R²: {val_metrics['r2']:.4f}"
                )

            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed:.2f} seconds")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_metrics": self.val_metrics,
            "elapsed_time": elapsed,
        }


class GPTrainer:
    """
    Trainer for Gaussian Process models (via GPyTorch).

    Optimizes hyperparameters by maximizing Exact Marginal Log-Likelihood (EMLL).
    """

    def __init__(
        self,
        model: GPRegressionModel,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        training_cfg: Dict[str, Any],
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize GPTrainer.

        Parameters
        ----------
        model : GPRegressionModel
            The GP model to train.
        X_train : torch.Tensor
            Training inputs.
        y_train : torch.Tensor
            Training targets.
        training_cfg : Dict[str, Any]
            Training configuration.
        device : torch.device, optional
            Device to use. If None, auto-detected.
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.training_cfg = training_cfg
        self.device = device or resolve_device(training_cfg.get("device", "auto"))

        self.model.to(self.device)
        self.X_train = self.X_train.to(self.device)
        self.y_train = self.y_train.to(self.device)

        # Setup optimizer
        optimizer_name = training_cfg.get("optimizer", "adam").lower()
        lr = float(training_cfg.get("learning_rate", 0.1))

        if optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # MLL
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        # History
        self.mll_history = []
        self.hyperparams_history = []

        logger.info(f"GPTrainer initialized on device: {self.device}")

    def train(self) -> Dict[str, Any]:
        """
        Run GP hyperparameter optimization.

        Returns
        -------
        Dict[str, Any]
            Optimization history and final hyperparameters.
        """
        n_steps = int(self.training_cfg.get("n_training_steps", 200))
        abs_tol = float(self.training_cfg.get("abs_tol", 1e-5))
        rel_tol = float(self.training_cfg.get("rel_tol", 1e-3))
        convergence_check_interval = int(self.training_cfg.get("convergence_check_interval", 10))

        logger.info(f"Starting GP optimization for {n_steps} steps")

        start_time = time.time()

        for step in range(1, n_steps + 1):
            self.optimizer.zero_grad()

            # Forward pass
            with gpytorch.settings.cholesky_jitter(1e-1):
                output = self.model(self.X_train)
                loss = -self.mll(output, self.y_train)

            loss.backward()
            self.optimizer.step()

            self.mll_history.append(-loss.item())

            if step % 10 == 0:
                logger.info(f"Step {step}/{n_steps} - MLL: {-loss.item():.6f}")

            # Convergence check
            if step % convergence_check_interval == 0 and step > convergence_check_interval:
                mll_prev = self.mll_history[-convergence_check_interval - 1]
                mll_curr = self.mll_history[-1]
                rel_change = abs(mll_curr - mll_prev) / (abs(mll_prev) + 1e-10)

                if rel_change < rel_tol and abs(mll_curr - mll_prev) < abs_tol:
                    logger.info(f"Converged at step {step} (rel_change={rel_change:.6f})")
                    break

        elapsed = time.time() - start_time
        logger.info(f"GP optimization completed in {elapsed:.2f} seconds")

        return {
            "mll_history": self.mll_history,
            "elapsed_time": elapsed,
        }
