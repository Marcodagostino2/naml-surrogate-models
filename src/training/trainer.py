"""
Training routines for MLP, GP, and MLPGP models.

Supports:
- MLP training with early stopping
- GP hyperparameter optimization via Exact Marginal Log-Likelihood
- MLPGP joint optimization (MLP + GP kernel parameters)
- Checkpointing and model saving
"""

from __future__ import annotations

import time
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gpytorch
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models import GPRegressionModel, MLPRegressor
from src.models.mlpgp import MLPGPModel
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
                # Save best model state
                try:
                    self._best_model_state = copy.deepcopy(self.model.state_dict())
                except Exception:
                    self._best_model_state = None
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

        # Restore best model state if available
        if hasattr(self, "_best_model_state") and self._best_model_state is not None:
            try:
                self.model.load_state_dict(self._best_model_state)
                logger.info("Restored MLP model to best validation state")
            except Exception:
                logger.warning("Failed to restore best MLP model state_dict")

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
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
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
        self.X_val = X_val.to(self.device) if X_val is not None else None
        self.y_val = y_val.to(self.device) if y_val is not None else None
        self.objective_name = str(training_cfg.get("objective", "concentrated")).lower()
        self.concentrated_cholesky_jitter = float(training_cfg.get("concentrated_cholesky_jitter", 1e-8))

        if self.objective_name not in {"exact_mll", "concentrated"}:
            raise ValueError(f"Unknown GP objective: {self.objective_name}")

        if self.objective_name == "concentrated":
            self.model.mean_module.coefficients.requires_grad_(False)
            self.model.covar_module.raw_outputscale.requires_grad_(False)
            self.model.likelihood.raw_noise.requires_grad_(False)

        # Setup optimizer
        optimizer_name = training_cfg.get("optimizer", "lbfgs").lower()
        lr = float(training_cfg.get("learning_rate", 0.1))

        optimizable_parameters = self._optimizable_parameters()

        if optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(optimizable_parameters, lr=lr)
        elif optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(optimizable_parameters, lr=lr)
        elif optimizer_name == "lbfgs":
            self.optimizer = torch.optim.LBFGS(
                optimizable_parameters,
                lr=lr,
                max_iter=int(training_cfg.get("lbfgs_max_iter", 20)),
                line_search_fn="strong_wolfe",
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        self.optimizer_name = optimizer_name

        # MLL
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        # History
        self.mll_history = []
        self.hyperparams_history = []
        self.last_profiled_sigma2: Optional[torch.Tensor] = None
        self.last_profiled_mean: Optional[torch.Tensor] = None

        logger.info(f"GPTrainer initialized on device: {self.device}")

        # Best-validation tracking
        self.best_val_loss = float("inf")
        self._best_model_state: Optional[Dict[str, torch.Tensor]] = None

    def _optimizable_parameters(self):
        if self.objective_name == "concentrated":
            base_kernel = self.model.covar_module.base_kernel
            return [base_kernel.raw_theta, base_kernel.raw_power_exponent, base_kernel.raw_nugget]
        return list(self.model.parameters())

    def _compute_concentrated_loss(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_kernel = self.model.covar_module.base_kernel
        y = self.y_train.reshape(-1, 1)
        F = self.model.mean_module._design_matrix(self.X_train)

        with gpytorch.settings.cholesky_jitter(self.concentrated_cholesky_jitter):
            R = base_kernel(self.X_train, self.X_train).evaluate()

        eye = torch.eye(R.shape[0], device=R.device, dtype=R.dtype)
        R = R + self.concentrated_cholesky_jitter * eye

        chol = torch.linalg.cholesky(R)
        R_inv_F = torch.cholesky_solve(F, chol)
        R_inv_y = torch.cholesky_solve(y, chol)

        term1 = F.transpose(0, 1) @ R_inv_F
        term2 = F.transpose(0, 1) @ R_inv_y
        mean_coefficients = torch.linalg.solve(term1, term2)

        residual = y - F @ mean_coefficients
        R_inv_residual = torch.cholesky_solve(residual, chol)
        sigma2 = (residual.transpose(0, 1) @ R_inv_residual) / residual.shape[0]
        sigma2 = sigma2.squeeze()

        loss = 0.5 * (residual.shape[0] * torch.log(sigma2) + 2.0 * torch.sum(torch.log(torch.diag(chol))))
        return loss, sigma2, mean_coefficients.squeeze(-1)

    def _apply_profiled_parameters(self) -> None:
        if self.last_profiled_sigma2 is not None:
            self.model.covar_module.outputscale = self.last_profiled_sigma2.detach().clamp_min(1e-10)
        if self.last_profiled_mean is not None:
            self.model.mean_module.coefficients.data.copy_(self.last_profiled_mean.detach())

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
            if self.optimizer_name == "lbfgs":
                def closure() -> torch.Tensor:
                    self.optimizer.zero_grad()
                    if self.objective_name == "concentrated":
                        loss, sigma2, mean_coefficients = self._compute_concentrated_loss()
                        self.last_profiled_sigma2 = sigma2
                        self.last_profiled_mean = mean_coefficients
                    else:
                        with gpytorch.settings.cholesky_jitter(self.concentrated_cholesky_jitter):
                            output = self.model(self.X_train)
                            loss = -self.mll(output, self.y_train)
                    loss.backward()
                    return loss

                loss = self.optimizer.step(closure)
            else:
                self.optimizer.zero_grad()

                if self.objective_name == "concentrated":
                    loss, sigma2, mean_coefficients = self._compute_concentrated_loss()
                    self.last_profiled_sigma2 = sigma2
                    self.last_profiled_mean = mean_coefficients
                else:
                    with gpytorch.settings.cholesky_jitter(self.concentrated_cholesky_jitter):
                        output = self.model(self.X_train)
                        loss = -self.mll(output, self.y_train)

                loss.backward()
                self.optimizer.step()

            loss_value = float(loss.item()) if torch.is_tensor(loss) else float(loss)
            self.mll_history.append(-loss_value)

            if self.objective_name == "concentrated":
                self._apply_profiled_parameters()

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

            # Validation check and best-state save (if validation provided)
            if self.X_val is not None and self.y_val is not None:
                try:
                    self.model.eval()
                    with torch.no_grad():
                        out = self.model(self.X_val)
                        mean = out.mean
                        val_loss = torch.nn.functional.mse_loss(mean, self.y_val).item()

                    if val_loss < self.best_val_loss - 1e-12:
                        self.best_val_loss = val_loss
                        try:
                            self._best_model_state = copy.deepcopy(self.model.state_dict())
                            logger.info(f"Saved best GP model at step {step} (val_loss={val_loss:.6e})")
                        except Exception:
                            self._best_model_state = None
                except Exception:
                    logger.debug("Validation evaluation failed during GP training step")

        elapsed = time.time() - start_time

        if self.objective_name == "concentrated":
            self._apply_profiled_parameters()

        logger.info(f"GP optimization completed in {elapsed:.2f} seconds")

        # Restore best validation state if available
        if self._best_model_state is not None:
            try:
                self.model.load_state_dict(self._best_model_state)
                logger.info("Restored GP model to best validation state")
            except Exception:
                logger.warning("Failed to restore best GP model state_dict")

        return {
            "mll_history": self.mll_history,
            "elapsed_time": elapsed,
        }


@dataclass(slots=True)
class MLPGPTrainingConfig:
    """Typed configuration for MLPGP joint training."""

    optimizer: str
    objective: str
    mlp_learning_rate: float
    gp_learning_rate: float
    n_training_steps: int
    lbfgs_max_iter: int
    abs_tol: float
    rel_tol: float
    convergence_check_interval: int
    device: str
    concentrated_cholesky_jitter: float

    @classmethod
    def from_dict(cls, training_cfg: Dict[str, Any]) -> "MLPGPTrainingConfig":
        cfg = cls(
            optimizer=str(training_cfg.get("optimizer", "adam")).lower(),
            objective=str(training_cfg.get("objective", "concentrated")).lower(),
            mlp_learning_rate=float(training_cfg.get("mlp_learning_rate", training_cfg.get("learning_rate", 1e-3))),
            gp_learning_rate=float(training_cfg.get("gp_learning_rate", training_cfg.get("learning_rate", 1e-2))),
            n_training_steps=int(training_cfg.get("n_training_steps", 300)),
            lbfgs_max_iter=int(training_cfg.get("lbfgs_max_iter", 20)),
            abs_tol=float(training_cfg.get("abs_tol", 1e-5)),
            rel_tol=float(training_cfg.get("rel_tol", 1e-3)),
            convergence_check_interval=int(training_cfg.get("convergence_check_interval", 10)),
            device=str(training_cfg.get("device", "auto")),
            concentrated_cholesky_jitter=float(training_cfg.get("concentrated_cholesky_jitter", 1e-6)),
        )
        if cfg.objective not in {"concentrated", "exact_mll"}:
            raise ValueError("MLPGP objective must be 'concentrated' or 'exact_mll'.")
        return cfg


class MLPGPTrainer:
    """Trainer for end-to-end MLPGP models."""

    def __init__(
        self,
        model: MLPGPModel,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        training_cfg: Dict[str, Any],
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.config = MLPGPTrainingConfig.from_dict(training_cfg)
        self.device = device or resolve_device(self.config.device)

        self.model.to(self.device)
        self.X_train = self.X_train.to(self.device)
        self.y_train = self.y_train.to(self.device)
        self.X_val = X_val.to(self.device) if X_val is not None else None
        self.y_val = y_val.to(self.device) if y_val is not None else None

        self.objective_name = self.config.objective
        if self.objective_name == "concentrated" and not self._supports_concentrated_objective():
            logger.warning(
                "Concentrated objective requested but model structure is not compatible. "
                "Falling back to exact_mll."
            )
            self.objective_name = "exact_mll"

        if self.objective_name == "concentrated":
            self.model.mean_module.coefficients.requires_grad_(False)
            self.model.covar_module.raw_outputscale.requires_grad_(False)
            self.model.likelihood.raw_noise.requires_grad_(False)

        self.optimizer, self.optimizer_name = self._build_optimizer()
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        self.mll_history: List[float] = []
        self.last_profiled_sigma2: Optional[torch.Tensor] = None
        self.last_profiled_mean: Optional[torch.Tensor] = None

        # Best-validation tracking
        self.best_val_loss = float("inf")
        self._best_model_state: Optional[Dict[str, torch.Tensor]] = None

        logger.info(f"MLPGPTrainer initialized on device: {self.device}, objective={self.objective_name}")

    def _supports_concentrated_objective(self) -> bool:
        """
        Concentrated likelihood remains valid when covariance is tau^2 * R(psi)
        and trend is linear in coefficients. Here R(psi) includes MLP-dependent
        sigma(x) and l(x).
        """
        has_mean_design = hasattr(self.model.mean_module, "_design_matrix")
        has_mean_coeff = hasattr(self.model.mean_module, "coefficients")
        has_base_kernel = hasattr(self.model.covar_module, "base_kernel")
        return bool(has_mean_design and has_mean_coeff and has_base_kernel)

    def _parameter_groups(self) -> List[Dict[str, Any]]:
        base_kernel = self.model.covar_module.base_kernel

        gp_params: List[torch.nn.Parameter] = [
            base_kernel.base_kernel.raw_theta,
            base_kernel.base_kernel.raw_power_exponent,
            base_kernel.base_kernel.raw_nugget,
        ]

        if self.objective_name == "exact_mll":
            gp_params.append(self.model.covar_module.raw_outputscale)

        return [
            {
                "params": self.model.feature_extractor.parameters(),
                "lr": float(self.config.mlp_learning_rate),
            },
            {
                "params": gp_params,
                "lr": float(self.config.gp_learning_rate),
            },
        ]

    def _build_optimizer(self) -> Tuple[torch.optim.Optimizer, str]:
        optimizer_name = self.config.optimizer
        parameter_groups = self._parameter_groups()

        if optimizer_name == "adam":
            return torch.optim.Adam(parameter_groups), optimizer_name
        if optimizer_name == "adamw":
            return torch.optim.AdamW(parameter_groups), optimizer_name
        if optimizer_name == "sgd":
            return torch.optim.SGD(parameter_groups), optimizer_name
        if optimizer_name == "lbfgs":
            return (
                torch.optim.LBFGS(
                    parameter_groups,
                    lr=max(self.config.mlp_learning_rate, self.config.gp_learning_rate),
                    max_iter=int(self.config.lbfgs_max_iter),
                    line_search_fn="strong_wolfe",
                ),
                optimizer_name,
            )
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _compute_concentrated_loss(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = self.y_train.reshape(-1, 1)
        F = self.model.mean_module._design_matrix(self.X_train)
        base_kernel = self.model.covar_module.base_kernel

        with gpytorch.settings.cholesky_jitter(self.config.concentrated_cholesky_jitter):
            R = base_kernel(self.X_train, self.X_train).evaluate()

        eye = torch.eye(R.shape[0], device=R.device, dtype=R.dtype)
        R = R + self.config.concentrated_cholesky_jitter * eye

        chol = torch.linalg.cholesky(R)
        R_inv_F = torch.cholesky_solve(F, chol)
        R_inv_y = torch.cholesky_solve(y, chol)

        term1 = F.transpose(0, 1) @ R_inv_F
        term2 = F.transpose(0, 1) @ R_inv_y
        mean_coefficients = torch.linalg.solve(term1, term2)

        residual = y - F @ mean_coefficients
        R_inv_residual = torch.cholesky_solve(residual, chol)
        sigma2 = (residual.transpose(0, 1) @ R_inv_residual) / residual.shape[0]
        sigma2 = sigma2.squeeze()

        # Profiled negative log-likelihood (up to additive constants).
        loss = 0.5 * (residual.shape[0] * torch.log(sigma2) + 2.0 * torch.sum(torch.log(torch.diag(chol))))
        return loss, sigma2, mean_coefficients.squeeze(-1)

    def _compute_exact_mll_loss(self) -> torch.Tensor:
        with gpytorch.settings.cholesky_jitter(self.config.concentrated_cholesky_jitter):
            output = self.model(self.X_train)
            return -self.mll(output, self.y_train)

    def _apply_profiled_parameters(self) -> None:
        if self.last_profiled_sigma2 is not None:
            self.model.covar_module.outputscale = self.last_profiled_sigma2.detach().clamp_min(1e-10)
        if self.last_profiled_mean is not None:
            self.model.mean_module.coefficients.data.copy_(self.last_profiled_mean.detach())

    def _closure(self) -> torch.Tensor:
        self.optimizer.zero_grad()

        if self.objective_name == "concentrated":
            try:
                loss, sigma2, mean_coefficients = self._compute_concentrated_loss()
                self.last_profiled_sigma2 = sigma2
                self.last_profiled_mean = mean_coefficients
            except RuntimeError as exc:
                logger.warning(f"Concentrated objective failed ({exc}). Falling back to exact_mll.")
                self.objective_name = "exact_mll"
                loss = self._compute_exact_mll_loss()
        else:
            loss = self._compute_exact_mll_loss()

        loss.backward()
        return loss

    def train(self) -> Dict[str, Any]:
        n_steps = int(self.config.n_training_steps)
        abs_tol = float(self.config.abs_tol)
        rel_tol = float(self.config.rel_tol)
        check_every = int(self.config.convergence_check_interval)

        logger.info(f"Starting MLPGP optimization for {n_steps} steps")
        start_time = time.time()

        for step in range(1, n_steps + 1):
            if self.optimizer_name == "lbfgs":
                loss = self.optimizer.step(self._closure)
            else:
                loss = self._closure()
                self.optimizer.step()

            if self.objective_name == "concentrated":
                self._apply_profiled_parameters()

            loss_value = float(loss.item()) if torch.is_tensor(loss) else float(loss)
            mll_value = -loss_value
            self.mll_history.append(mll_value)

            if step % 10 == 0:
                logger.info(f"Step {step}/{n_steps} - MLL: {mll_value:.6f}")

            if step % check_every == 0 and step > check_every:
                mll_prev = self.mll_history[-check_every - 1]
                mll_curr = self.mll_history[-1]
                rel_change = abs(mll_curr - mll_prev) / (abs(mll_prev) + 1e-10)
                if rel_change < rel_tol and abs(mll_curr - mll_prev) < abs_tol:
                    logger.info(f"Converged at step {step} (rel_change={rel_change:.6f})")
                    break

            # Validation check and best-state save (if validation provided)
            if self.X_val is not None and self.y_val is not None and step % check_every == 0:
                try:
                    self.model.eval()
                    with torch.no_grad():
                        out = self.model(self.X_val)
                        mean = out.mean
                        val_loss = torch.nn.functional.mse_loss(mean, self.y_val).item()

                    if val_loss < self.best_val_loss - 1e-12:
                        self.best_val_loss = val_loss
                        try:
                            self._best_model_state = copy.deepcopy(self.model.state_dict())
                            logger.info(f"Saved best MLPGP model at step {step} (val_loss={val_loss:.6e})")
                        except Exception:
                            self._best_model_state = None
                except Exception:
                    logger.debug("Validation evaluation failed during MLPGP training step")

        if self.objective_name == "concentrated":
            self._apply_profiled_parameters()

        elapsed = time.time() - start_time
        logger.info(f"MLPGP optimization completed in {elapsed:.2f} seconds")
        # Restore best validation state if available
        if self._best_model_state is not None:
            try:
                self.model.load_state_dict(self._best_model_state)
                logger.info("Restored MLPGP model to best validation state")
            except Exception:
                logger.warning("Failed to restore best MLPGP model state_dict")

        return {
            "mll_history": self.mll_history,
            "elapsed_time": elapsed,
            "objective_used": self.objective_name,
        }
