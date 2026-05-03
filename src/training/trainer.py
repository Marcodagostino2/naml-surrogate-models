"""
Training routines for MLP, GP, and MLPGP models.

Uses large epoch/step budgets with interval-based residual checks on train and
validation splits for early stopping.
"""

from __future__ import annotations

import copy
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import gpytorch
import torch
from torch.utils.data import DataLoader

from src.models import GPRegressionModel, MLPRegressor, MLPGPModel
from src.training.metrics import compute_all_metrics
from src.utils import resolve_device, setup_logger

logger = setup_logger(__name__)


def _check_interval(step: int, interval: int) -> bool:
    return interval > 0 and step % interval == 0

def _to_prediction_tensor(output: Any) -> torch.Tensor:
    if hasattr(output, "mean"):
        return output.mean
    return output

def _evaluate_regression_model(model: Any, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, Dict[str, float]]:
    model.eval()
    likelihood = getattr(model, "likelihood", None)
    if likelihood is not None:
        likelihood.eval()
    with torch.no_grad():
        output = model(X)
        prediction = _to_prediction_tensor(output)
        target = y.view_as(prediction)
        mse_value = float(torch.nn.functional.mse_loss(prediction, target).item())
        metrics = compute_all_metrics(target.detach().cpu().numpy(), prediction.detach().cpu().numpy())
    return mse_value, metrics

def _current_likelihood_noise(model: Any) -> float:
    likelihood = getattr(model, "likelihood", None)
    if likelihood is None or not hasattr(likelihood, "noise"):
        return float("nan")
    return float(likelihood.noise.detach().cpu().item())

def _build_scheduler(optimizer: torch.optim.Optimizer, training_cfg: Dict[str, Any]) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Build a common LR scheduler from training config.

    Config keys shared in common_training_config.yaml:
    - scheduler_step_size
    - scheduler_gamma
    - scheduler_type  (optional: "steplr" or "none")
    """
    scheduler_type = str(training_cfg.get("scheduler_type", "steplr")).lower()
    if scheduler_type == "none":
        return None

    step_size = int(training_cfg.get("scheduler_step_size", 50))
    gamma = float(training_cfg.get("scheduler_gamma", 0.5))

    if scheduler_type == "steplr":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

if TYPE_CHECKING:
    import optuna

class MLPTrainer:
    """Trainer for MLP regression models (residual-based early stopping + richer logging)."""

    def __init__(
        self,
        model: MLPRegressor,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        training_cfg: Dict[str, Any],
        device: Optional[torch.device] = None,
        trial: Optional["optuna.Trial"] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.training_cfg = training_cfg
        self.trial = trial

        self.device = device or resolve_device(str(training_cfg.get("device", "auto")))
        self.model.to(self.device)

        self.n_epochs = int(training_cfg.get("n_epochs", 5000))
        self.learning_rate = float(training_cfg.get("learning_rate", 1e-3))
        self.weight_decay = float(training_cfg.get("weight_decay", 0.0))
        self.optimizer_name = str(training_cfg.get("optimizer", "adam")).lower()

        # residual-based stopping (v2 idea)
        self.train_residual_check_interval = int(training_cfg.get("train_residual_check_interval", 1))
        self.validation_check_interval = int(training_cfg.get("validation_check_interval", 10))
        self.train_residual_tol = float(training_cfg.get("train_residual_tol", 1e-4))
        self.validation_residual_tol = float(training_cfg.get("validation_residual_tol", 1e-4))

        if self.train_residual_check_interval <= 0:
            raise ValueError("train_residual_check_interval must be a positive integer")
        if self.validation_check_interval <= 0:
            raise ValueError("validation_check_interval must be a positive integer")

        # loss function configurabile come in v1
        loss_name = str(training_cfg.get("loss_function", "mse")).lower()
        if loss_name == "mse":
            self.criterion = torch.nn.MSELoss()
        elif loss_name == "mae":
            self.criterion = torch.nn.L1Loss()
        elif loss_name == "huber":
            self.criterion = torch.nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

        self.optimizer = self._build_optimizer()
        self.scheduler = _build_scheduler(self.optimizer, training_cfg)
        self.best_val_loss = float("inf")
        self._best_model_state: Optional[Dict[str, Any]] = None

        logger.info(f"MLPTrainer initialized on device: {self.device}")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = self.model.parameters()
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.optimizer_name == "adam":
            return torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    def _evaluate_loader(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        losses: List[float] = []
        y_true: List[torch.Tensor] = []
        y_pred: List[torch.Tensor] = []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                prediction = self.model(X_batch)
                target = y_batch.view_as(prediction)
                losses.append(float(self.criterion(prediction, target).item()))
                y_true.append(target.detach().cpu())
                y_pred.append(prediction.detach().cpu())
        y_true_tensor = torch.cat(y_true, dim=0)
        y_pred_tensor = torch.cat(y_pred, dim=0)
        metrics = compute_all_metrics(y_true_tensor.numpy(), y_pred_tensor.numpy())
        return float(sum(losses) / max(len(losses), 1)), metrics

    def train(self) -> Dict[str, Any]:
        logger.info(f"Starting MLP training for up to {self.n_epochs} epochs")
        start_time = time.time()
        train_losses: List[float] = []
        val_losses: List[Optional[float]] = []
        train_metrics_history: List[Optional[Dict[str, float]]] = []
        val_metrics_history: List[Optional[Dict[str, float]]] = []
        train_residual_ok = False
        val_residual_ok = False

        for epoch in range(1, self.n_epochs + 1):
            self.model.train()
            epoch_losses: List[float] = []
            for X_batch, y_batch in self.train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                prediction = self.model(X_batch)
                target = y_batch.view_as(prediction)
                loss = self.criterion(prediction, target)

                if not torch.isfinite(loss):
                    logger.warning(f"Non-finite loss at epoch {epoch}: {loss.item()}")
                    break

                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    try:
                        self.scheduler.step()
                    except Exception:
                        logger.warning("Scheduler step failed; continuing without scheduler")
                        self.scheduler = None
                epoch_losses.append(float(loss.item()))

            if not epoch_losses:
                break

            train_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
            train_losses.append(train_loss)

            train_metrics: Optional[Dict[str, float]] = None
            if _check_interval(epoch, self.train_residual_check_interval):
                _, train_metrics = self._evaluate_loader(self.train_loader)
                train_metrics_history.append(train_metrics)
                rmse_train = float(train_metrics.get("rmse", float("inf")))
                train_residual_ok = torch.isfinite(torch.tensor(rmse_train)) and rmse_train <= self.train_residual_tol
            else:
                train_metrics_history.append(None)

            val_loss: Optional[float] = None
            val_metrics: Optional[Dict[str, float]] = None
            if self.val_loader is not None:
                val_loss, raw_val_metrics = self._evaluate_loader(self.val_loader)

                if val_loss is not None and not torch.isfinite(torch.tensor(val_loss)):
                    logger.warning(f"Non-finite val_loss at epoch {epoch}: {val_loss}")
                    val_loss = None

                if _check_interval(epoch, self.validation_check_interval):
                    val_metrics = raw_val_metrics
                    rmse_val = float(val_metrics.get("rmse", float("inf")))
                    val_residual_ok = torch.isfinite(torch.tensor(rmse_val)) and rmse_val <= self.validation_residual_tol
                    val_metrics_history.append(val_metrics)
                else:
                    val_metrics_history.append(None)
            else:
                val_metrics_history.append(None)

            val_losses.append(val_loss)

            # logging stile v1 ogni 10 epoche
            if epoch % 10 == 0:
                if val_metrics is not None:
                    logger.info(
                        f"Epoch {epoch}/{self.n_epochs} - "
                        f"Train Loss: {train_loss:.6f}, "
                        f"Val Loss: {val_loss if val_loss is not None else float('nan'):.6f}, "
                        f"Val R²: {val_metrics.get('r2', float('nan')):.4f}"
                    )
                else:
                    logger.info(
                        f"Epoch {epoch}/{self.n_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss}"
                    )

            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._best_model_state = {"model": copy.deepcopy(self.model.state_dict())}

            if self.trial is not None and val_loss is not None:
                self.trial.report(val_loss, step=epoch)
                if self.trial.should_prune():
                    import optuna
                    logger.info(f"Trial pruned at epoch {epoch}")
                    raise optuna.TrialPruned()

            if epoch >= max(self.train_residual_check_interval, self.validation_check_interval) and train_residual_ok and val_residual_ok:
                logger.info(f"Stopping MLP training early at epoch {epoch} (residuals below tolerance)")
                break

        elapsed = time.time() - start_time
        logger.info(f"MLP training completed in {elapsed:.2f} seconds")
        if self._best_model_state is not None:
            self.model.load_state_dict(self._best_model_state["model"])
            logger.info("Restored MLP model to best validation state")

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_metrics_history": train_metrics_history,
            "val_metrics_history": val_metrics_history,
            "elapsed_time": elapsed,
        }

class GPTrainer:
    """Trainer for exact GP regression models (EMLL + residual-based early stopping)."""

    def __init__(
        self,
        model: GPRegressionModel,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        training_cfg: Dict[str, Any],
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        trial: Optional["optuna.Trial"] = None,
    ) -> None:
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.training_cfg = training_cfg
        self.trial = trial

        self.device = device or resolve_device(str(training_cfg.get("device", "auto")))
        self.model.to(self.device)
        self.X_train = self.X_train.to(self.device)
        self.y_train = self.y_train.to(self.device)
        self.X_val = self.X_val.to(self.device) if self.X_val is not None else None
        self.y_val = self.y_val.to(self.device) if self.y_val is not None else None

        self.n_training_steps = int(training_cfg.get("n_training_steps", 5000))
        self.learning_rate = float(training_cfg.get("learning_rate", 1e-2))
        self.optimizer_name = str(training_cfg.get("optimizer", "adam")).lower()

        self.train_residual_check_interval = int(training_cfg.get("train_residual_check_interval", 1))
        self.validation_check_interval = int(training_cfg.get("validation_check_interval", 10))
        self.train_residual_tol = float(training_cfg.get("train_residual_tol", 1e-4))
        self.validation_residual_tol = float(training_cfg.get("validation_residual_tol", 1e-4))

        if self.train_residual_check_interval <= 0:
            raise ValueError("train_residual_check_interval must be a positive integer")
        if self.validation_check_interval <= 0:
            raise ValueError("validation_check_interval must be a positive integer")

        params = self._optimizable_parameters()
        if self.optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(params, lr=self.learning_rate)
        elif self.optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(params, lr=self.learning_rate)
        elif self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        elif self.optimizer_name == "lbfgs":
            self.optimizer = torch.optim.LBFGS(
                params,
                lr=self.learning_rate,
                max_iter=int(training_cfg.get("lbfgs_max_iter", 20)),
                line_search_fn="strong_wolfe",
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=int(training_cfg.get("scheduler_step_size", 50)),
            gamma=float(training_cfg.get("scheduler_gamma", 0.5)),
        )

        self.scheduler = _build_scheduler(self.optimizer, training_cfg)

        self.mll_history: List[float] = []
        self.val_metrics_history: List[Optional[Dict[str, float]]] = []
        self.likelihood_noise_history: List[float] = []
        self.best_val_loss = float("inf")
        self._best_model_state: Optional[Dict[str, Any]] = None

        logger.info(f"GPTrainer initialized on device: {self.device}")
        logger.info(f"Initial GP likelihood noise: {_current_likelihood_noise(self.model):.6e}")

        self._setup_cholesky_jitter(training_cfg)

    def _optimizable_parameters(self):
        return list(self.model.parameters())

    def _setup_cholesky_jitter(self, training_cfg: Dict[str, Any]) -> None:
        jitter_float32 = float(training_cfg.get("cholesky_jitter_float32", 1e-6))
        jitter_float64 = float(training_cfg.get("cholesky_jitter_float64", 1e-8))

        if jitter_float32 < 0:
            raise ValueError(f"cholesky_jitter_float32 must be non-negative, got {jitter_float32}")
        if jitter_float64 < 0:
            raise ValueError(f"cholesky_jitter_float64 must be non-negative, got {jitter_float64}")

        self._previous_jitter_state = {
            "float32": gpytorch.settings.cholesky_jitter.value(torch.float32),
            "float64": gpytorch.settings.cholesky_jitter.value(torch.float64),
        }

        gpytorch.settings.cholesky_jitter._set_value(
            float_value=jitter_float32,
            double_value=jitter_float64,
            half_value=None,
        )

        logger.info(
            f"Cholesky jitter configured: float32={jitter_float32:.2e}, float64={jitter_float64:.2e}"
        )

    def restore_previous_jitter(self) -> None:
        if not hasattr(self, "_previous_jitter_state"):
            logger.warning("No previous jitter state saved; skipping restoration")
            return
        prev_state = self._previous_jitter_state
        gpytorch.settings.cholesky_jitter._set_state(
            float_value=prev_state["float32"],
            double_value=prev_state["float64"],
        )
        logger.info(
            f"Restored previous Cholesky jitter: float32={prev_state['float32']:.2e}, "
            f"float64={prev_state['float64']:.2e}"
        )

    def train(self) -> Dict[str, Any]:
        logger.info(f"Starting GP training for up to {self.n_training_steps} steps")
        start_time = time.time()
        train_losses: List[float] = []
        val_losses: List[Optional[float]] = []
        train_metrics_history: List[Optional[Dict[str, float]]] = []
        likelihood_noise_history: List[float] = []
        train_residual_ok = False
        val_residual_ok = False

        for step in range(1, self.n_training_steps + 1):
            self.model.train()
            self.model.likelihood.train()

            def _closure() -> torch.Tensor:
                self.optimizer.zero_grad()
                self.model.train()
                output = self.model(self.X_train)
                loss = -self.mll(output, self.y_train)
                if not torch.isfinite(loss):
                    logger.warning(f"Non-finite GP loss at step {step}: {loss.item()}")
                loss.backward()
                return loss

            if self.optimizer_name == "lbfgs":
                loss = self.optimizer.step(_closure)
            else:
                self.optimizer.zero_grad()
                output = self.model(self.X_train)
                loss = -self.mll(output, self.y_train)
                if not torch.isfinite(loss):
                    logger.warning(f"Non-finite GP loss at step {step}: {loss.item()}")
                    break
                loss.backward()
                self.optimizer.step()

            if self.scheduler is not None:
                try:
                    self.scheduler.step()
                except Exception:
                    logger.warning("Scheduler step failed; disabling scheduler for the rest of training")
                    self.scheduler = None


            loss_value = float(loss.item()) if torch.is_tensor(loss) else float(loss)
            mll_value = -loss_value
            train_losses.append(loss_value)          # negativo della MLL
            self.mll_history.append(mll_value)       # MLL positiva

            current_noise = _current_likelihood_noise(self.model)
            self.likelihood_noise_history.append(current_noise)
            likelihood_noise_history.append(current_noise)

            if _check_interval(step, self.train_residual_check_interval):
                _, train_metrics = _evaluate_regression_model(self.model, self.X_train, self.y_train)
                train_metrics_history.append(train_metrics)
                rmse_train = float(train_metrics.get("rmse", float("inf")))
                train_residual_ok = torch.isfinite(torch.tensor(rmse_train)) and rmse_train <= self.train_residual_tol
            else:
                train_metrics_history.append(None)

            val_loss: Optional[float] = None
            val_metrics: Optional[Dict[str, float]] = None
            if self.X_val is not None and self.y_val is not None:
                val_loss, raw_val_metrics = _evaluate_regression_model(self.model, self.X_val, self.y_val)

                if val_loss is not None and not torch.isfinite(torch.tensor(val_loss)):
                    logger.warning(f"Non-finite GP val_loss at step {step}: {val_loss}")
                    val_loss = None

                if _check_interval(step, self.validation_check_interval):
                    val_metrics = raw_val_metrics
                    rmse_val = float(val_metrics.get("rmse", float("inf")))
                    val_residual_ok = torch.isfinite(torch.tensor(rmse_val)) and rmse_val <= self.validation_residual_tol
                    self.val_metrics_history.append(val_metrics)
                else:
                    self.val_metrics_history.append(None)
            else:
                self.val_metrics_history.append(None)

            val_losses.append(val_loss)

            if step % 10 == 0:
                val_loss_display = f"{val_loss:.6f}" if val_loss is not None else "None"
                if val_metrics is not None:
                    logger.info(
                        f"Step {step}/{self.n_training_steps} - "
                        f"Train Loss (MLL): {mll_value:.6f}, "
                        f"Val MSE: {val_loss_display}, "
                        f"Val R²: {val_metrics.get('r2', float('nan')):.4f}, "
                        f"Val MAE: {val_metrics.get('mae', float('nan')):.4f}, "
                        f"Likelihood Noise: {current_noise:.6e}"
                    )
                else:
                    logger.info(
                        f"Step {step}/{self.n_training_steps} - "
                        f"Train Loss (MLL): {mll_value:.6f}, "
                        f"Val MSE: {val_loss_display}, "
                        f"Likelihood Noise: {current_noise:.6e}"
                    )

            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._best_model_state = {"model": copy.deepcopy(self.model.state_dict())}

            if self.trial is not None and val_loss is not None:
                self.trial.report(val_loss, step=step)
                if self.trial.should_prune():
                    import optuna
                    logger.info(f"Trial pruned at step {step}")
                    raise optuna.TrialPruned()

            if step >= max(self.train_residual_check_interval, self.validation_check_interval) and train_residual_ok and val_residual_ok:
                logger.info(f"Stopping GP training early at step {step} (residuals below tolerance)")
                break

        elapsed = time.time() - start_time
        logger.info(f"GP training completed in {elapsed:.2f} seconds")

        if self._best_model_state is not None:
            self.model.load_state_dict(self._best_model_state["model"])
            logger.info("Restored GP model to best validation state")

        return {
            "train_losses": train_losses,          # negative mll
            "mll_history": self.mll_history,       # positive mll
            "val_losses": val_losses,
            "train_metrics_history": train_metrics_history,
            "val_metrics_history": self.val_metrics_history,
            "likelihood_noise_history": likelihood_noise_history,
            "elapsed_time": elapsed,
        }

class MLPGPTrainer:
    """Trainer for end-to-end MLPGP models (EMLL + residual-based early stopping)."""

    def __init__(
        self,
        model: MLPGPModel,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        training_cfg: Dict[str, Any],
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        trial: Optional["optuna.Trial"] = None,
    ) -> None:
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.training_cfg = training_cfg
        self.trial = trial

        self.device = device or resolve_device(str(training_cfg.get("device", "auto")))
        self.model.to(self.device)
        self.X_train = self.X_train.to(self.device)
        self.y_train = self.y_train.to(self.device)
        self.X_val = self.X_val.to(self.device) if self.X_val is not None else None
        self.y_val = self.y_val.to(self.device) if self.y_val is not None else None

        self.n_training_steps = int(training_cfg.get("n_training_steps", 5000))
        self.optimizer_name = str(training_cfg.get("optimizer", "adam")).lower()
        self.mlp_learning_rate = float(training_cfg.get("mlp_learning_rate", training_cfg.get("learning_rate", 1e-3)))
        self.gp_learning_rate = float(training_cfg.get("gp_learning_rate", training_cfg.get("learning_rate", 1e-2)))
        self.mlp_weight_decay = float(training_cfg.get("mlp_weight_decay", 1e-4))

        self.train_residual_check_interval = int(training_cfg.get("train_residual_check_interval", 1))
        self.validation_check_interval = int(training_cfg.get("validation_check_interval", 10))
        self.train_residual_tol = float(training_cfg.get("train_residual_tol", 1e-4))
        self.validation_residual_tol = float(training_cfg.get("validation_residual_tol", 1e-4))

        if self.train_residual_check_interval <= 0:
            raise ValueError("train_residual_check_interval must be a positive integer")
        if self.validation_check_interval <= 0:
            raise ValueError("validation_check_interval must be a positive integer")

        self.optimizer = self._build_optimizer()
        self.scheduler = _build_scheduler(self.optimizer, training_cfg)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        self.mll_history: List[float] = []
        self.likelihood_noise_history: List[float] = []
        self.val_metrics_history: List[Optional[Dict[str, float]]] = []
        self.best_val_loss = float("inf")
        self._best_model_state: Optional[Dict[str, Any]] = None

        logger.info(f"MLPGPTrainer initialized on device: {self.device}")
        logger.info(f"Initial MLPGP likelihood noise: {_current_likelihood_noise(self.model):.6e}")

        self._setup_cholesky_jitter(training_cfg)

    def _parameter_groups(self) -> List[Dict[str, Any]]:
        return [
            {
                "params": list(self.model.feature_extractor_parameters()),
                "lr": self.mlp_learning_rate,
                "weight_decay": self.mlp_weight_decay,
            },
            {
                "params": list(self.model.gp_parameters()),
                "lr": self.gp_learning_rate,
                "weight_decay": 0.0,
            },
        ]

    def _build_optimizer(self) -> torch.optim.Optimizer:
        parameter_groups = self._parameter_groups()
        if self.optimizer_name == "adam":
            return torch.optim.Adam(parameter_groups)
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(parameter_groups)
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(parameter_groups)
        raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    def _setup_cholesky_jitter(self, training_cfg: Dict[str, Any]) -> None:
        jitter_float32 = float(training_cfg.get("cholesky_jitter_float32", 1e-6))
        jitter_float64 = float(training_cfg.get("cholesky_jitter_float64", 1e-8))

        if jitter_float32 < 0:
            raise ValueError(f"cholesky_jitter_float32 must be non-negative, got {jitter_float32}")
        if jitter_float64 < 0:
            raise ValueError(f"cholesky_jitter_float64 must be non-negative, got {jitter_float64}")

        self._previous_jitter_state = {
            "float32": gpytorch.settings.cholesky_jitter.value(torch.float32),
            "float64": gpytorch.settings.cholesky_jitter.value(torch.float64),
        }

        gpytorch.settings.cholesky_jitter._set_value(
            float_value=jitter_float32,
            double_value=jitter_float64,
            half_value=None,
        )

        logger.info(
            f"Cholesky jitter configured: float32={jitter_float32:.2e}, float64={jitter_float64:.2e}"
        )

    def restore_previous_jitter(self) -> None:
        if not hasattr(self, "_previous_jitter_state"):
            logger.warning("No previous jitter state saved; skipping restoration")
            return
        prev_state = self._previous_jitter_state
        gpytorch.settings.cholesky_jitter._set_state(
            float_value=prev_state["float32"],
            double_value=prev_state["float64"],
        )
        logger.info(
            f"Restored previous Cholesky jitter: float32={prev_state['float32']:.2e}, "
            f"float64={prev_state['float64']:.2e}"
        )

    def train(self) -> Dict[str, Any]:
        architecture = getattr(self.model, "architecture_name", "unknown")
        logger.info(
            f"Starting MLPGP training for up to {self.n_training_steps} steps (architecture: {architecture})"
        )

        if architecture == "dkl" and hasattr(self.model, "dkl_feature_extractor"):
            output_dim = self.model.dkl_feature_extractor.output_dim
            logger.info(f"DKL feature extractor output_dim: {output_dim}")

        start_time = time.time()
        train_losses: List[float] = []
        val_losses: List[Optional[float]] = []
        train_metrics_history: List[Optional[Dict[str, float]]] = []
        likelihood_noise_history: List[float] = []
        train_residual_ok = False
        val_residual_ok = False

        for step in range(1, self.n_training_steps + 1):
            self.model.train()
            self.model.likelihood.train()
            self.optimizer.zero_grad()

            output = self.model(self.X_train)
            loss = -self.mll(output, self.y_train)
            if not torch.isfinite(loss):
                logger.warning(f"Non-finite MLPGP loss at step {step}: {loss.item()}")
                break
            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                try:
                    self.scheduler.step()
                except Exception:
                    logger.warning("Scheduler step failed; disabling scheduler for the rest of training")
                    self.scheduler = None


                loss_value = float(loss.item()) if torch.is_tensor(loss) else float(loss)
                mll_value = -loss_value
                train_losses.append(loss_value)          # negativo della MLL
                self.mll_history.append(mll_value)       # MLL positiva

                current_noise = _current_likelihood_noise(self.model)
                self.likelihood_noise_history.append(current_noise)
                likelihood_noise_history.append(current_noise)

                if _check_interval(step, self.train_residual_check_interval):
                    _, train_metrics = _evaluate_regression_model(self.model, self.X_train, self.y_train)
                    train_metrics_history.append(train_metrics)
                    rmse_train = float(train_metrics.get("rmse", float("inf")))
                    train_residual_ok = torch.isfinite(torch.tensor(rmse_train)) and rmse_train <= self.train_residual_tol
                else:
                    train_metrics_history.append(None)

                val_loss: Optional[float] = None
                val_metrics: Optional[Dict[str, float]] = None
                if self.X_val is not None and self.y_val is not None:
                    val_loss, raw_val_metrics = _evaluate_regression_model(self.model, self.X_val, self.y_val)

                    if val_loss is not None and not torch.isfinite(torch.tensor(val_loss)):
                        logger.warning(f"Non-finite MLPGP val_loss at step {step}: {val_loss}")
                        val_loss = None

                    if _check_interval(step, self.validation_check_interval):
                        val_metrics = raw_val_metrics
                        rmse_val = float(val_metrics.get("rmse", float("inf")))
                        val_residual_ok = torch.isfinite(torch.tensor(rmse_val)) and rmse_val <= self.validation_residual_tol
                        self.val_metrics_history.append(val_metrics)
                    else:
                        self.val_metrics_history.append(None)
                else:
                    self.val_metrics_history.append(None)

                val_losses.append(val_loss)

            if step % 10 == 0:
                val_loss_display = f"{val_loss:.6f}" if val_loss is not None else "None"
                if val_metrics is not None:
                    logger.info(
                        f"Step {step}/{self.n_training_steps} - "
                        f"MLL: {mll_value:.6f}, "
                        f"Val MSE: {val_loss_display}, "
                        f"Val R²: {val_metrics.get('r2', float('nan')):.4f}, "
                        f"Val MAE: {val_metrics.get('mae', float('nan')):.4f}, "
                        f"Likelihood Noise: {current_noise:.6e}"
                    )
                else:
                    logger.info(
                        f"Step {step}/{self.n_training_steps} - "
                        f"MLL: {mll_value:.6f}, "
                        f"Val MSE: {val_loss_display}, "
                        f"Likelihood Noise: {current_noise:.6e}"
                    )

            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._best_model_state = {"model": copy.deepcopy(self.model.state_dict())}

            if self.trial is not None and val_loss is not None:
                self.trial.report(val_loss, step=step)
                if self.trial.should_prune():
                    import optuna
                    logger.info(f"Trial pruned at step {step}")
                    raise optuna.TrialPruned()

            if step >= max(self.train_residual_check_interval, self.validation_check_interval) and train_residual_ok and val_residual_ok:
                logger.info(f"Stopping MLPGP training early at step {step} (residuals below tolerance)")
                break

        elapsed = time.time() - start_time
        logger.info(f"MLPGP training completed in {elapsed:.2f} seconds")

        if self._best_model_state is not None:
            self.model.load_state_dict(self._best_model_state["model"])
            logger.info("Restored MLPGP model to best validation state")

        return {
            "train_losses": train_losses,          # negative mll
            "mll_history": self.mll_history,       # positive mll
            "val_losses": val_losses,
            "train_metrics_history": train_metrics_history,
            "val_metrics_history": self.val_metrics_history,
            "likelihood_noise_history": likelihood_noise_history,
            "elapsed_time": elapsed,
        }

__all__ = ["MLPTrainer", "GPTrainer", "MLPGPTrainer"]
