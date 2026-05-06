"""
Training routines for MLP, GP, and MLPGP models.

Uses large epoch/step budgets with interval-based residual checks on train and
validation splits for early stopping.

Note on Scheduler Semantics:
    StepLR scheduler.step() is called after each epoch (not per batch).
    This decays the learning rate every `scheduler_step_size` epochs by
    a factor of `scheduler_gamma`.
"""

from __future__ import annotations

import copy
import contextlib
import time
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import gpytorch
import torch
from gpytorch.utils.warnings import GPInputWarning
from torch.utils.data import DataLoader

from linear_operator.utils.warnings import NumericalWarning

# NotPSDError is raised by linear_operator's psd_safe_cholesky when even the
# escalated jitter (up to 100x cholesky_jitter) cannot make K positive
# definite. We handle it as a graceful end-of-training signal rather than a
# crash. Older versions of the stack expose it under different paths, so we
# fall back to plain RuntimeError if the import fails.
try:
    from linear_operator.utils.errors import NotPSDError as _NotPSDError
    _CHOLESKY_FAIL_EXCEPTIONS: Tuple[type, ...] = (_NotPSDError, RuntimeError)
except ImportError:  # linear_operator not installed or moved
    _CHOLESKY_FAIL_EXCEPTIONS = (RuntimeError,)

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

def _evaluate_regression_model(
    model: Any,
    X: torch.Tensor,
    y: torch.Tensor,
    force_cholesky: bool = False,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    likelihood = getattr(model, "likelihood", None)
    if likelihood is not None:
        likelihood.eval()
    if force_cholesky:
        ctx = contextlib.ExitStack()
        ctx.enter_context(gpytorch.settings.max_cholesky_size(float("inf")))
        ctx.enter_context(
            gpytorch.settings.fast_computations(
                covar_root_decomposition=False,
                log_prob=False,
                solves=False,
            )
        )
    else:
        ctx = contextlib.nullcontext()
    with torch.no_grad(), ctx, warnings.catch_warnings():
        warnings.simplefilter("ignore", GPInputWarning)
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

    Config keys shared in training_common_config.yaml:
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



def _is_cholesky_failure(exc: BaseException) -> bool:
    """Return True if `exc` looks like a Cholesky / non-PSD failure.

    We catch both linear_operator's NotPSDError (clear signal) and torch's
    plain RuntimeError when its message mentions cholesky / PSD. The string
    match keeps us from swallowing unrelated RuntimeErrors raised from inside
    the forward pass (e.g. shape mismatches, CUDA OOM).
    """
    if _CHOLESKY_FAIL_EXCEPTIONS and isinstance(exc, _CHOLESKY_FAIL_EXCEPTIONS[0]) \
            and _CHOLESKY_FAIL_EXCEPTIONS[0] is not RuntimeError:
        return True  # NotPSDError - definitive
    msg = str(exc).lower()
    return ("cholesky" in msg) or ("not positive definite" in msg) or ("psd" in msg)


def _handle_cholesky_failure(
    exc: BaseException,
    *,
    model_name: str,
    step: int,
    phase: str,
    trial: Optional["optuna.Trial"],
) -> None:
    """Common handling for a Cholesky / non-PSD failure inside training.

    Re-raises `exc` if it is not actually a Cholesky failure (so unrelated
    RuntimeErrors like CUDA OOM are not swallowed).

    If `trial` is provided (Optuna sweep), raise optuna.TrialPruned so the
    sampler treats this hyperparameter region as unviable rather than scoring
    the trial by whatever partial best_val_loss it accumulated. Pruned trials
    are recorded distinctly from completed ones, which keeps Optuna's posterior
    over the hyperparameter space honest.

    Outside an Optuna trial, just log a warning and return; the caller is
    expected to `break` out of the training loop. The post-loop logic in each
    trainer then restores `_best_model_state` as usual.
    """
    if not _is_cholesky_failure(exc):
        raise exc  # unrelated RuntimeError - don't swallow

    logger.warning(
        f"{model_name} {phase} stopped at step {step} due to non-PSD covariance "
        f"(jitter exhausted). Underlying error: {exc}"
    )

    if trial is not None:
        import optuna
        logger.info(
            f"Pruning Optuna trial: hyperparameters caused a non-PSD covariance "
            f"at step {step}, marking trial as unviable."
        )
        raise optuna.TrialPruned()

    # No trial: caller will break and restore best state.

_DEPRECATED_KEYS = {
    "train_residual_tol": "train_residual_tol no longer used; replaced by rel_tol/abs_tol on the loss",
    "validation_residual_tol": "validation_residual_tol no longer used; replaced by rel_tol/abs_tol on best_val_loss",
    "train_residual_check_interval": "train_residual_check_interval no longer used; replaced by train_metrics_log_interval (logging only)",
}


def _warn_deprecated_keys(training_cfg: Dict[str, Any]) -> None:
    """One-shot deprecation warning for keys whose meaning changed.

    These keys used to control stopping; now the stopping logic lives in
    `ConvergenceMonitor` and reads different keys. Other unknown keys are
    silently ignored (consistent with the rest of the trainer's `dict.get`
    style) — only these specific names get a warning because their meaning
    changed and silent defaults would be misleading.
    """
    for key, message in _DEPRECATED_KEYS.items():
        if key in training_cfg:
            logger.warning(f"[deprecated config key] {message}")


class ConvergenceMonitor:
    """Unified convergence + early-stopping monitor shared by all trainers.

    Two independent stopping criteria are tracked; training stops as soon as
    EITHER fires (logical OR), per the user's spec.

    1. Train-loss convergence (scipy.optimize.least_squares style)
       Stop when the loss change between consecutive epochs satisfies
            |L_{t-1} - L_t| <= abs_tol + rel_tol * |L_{t-1}|
       for `train_patience` consecutive epochs. The mixed abs+rel form is the
       standard one used by NumPy's `isclose` and most numerical optimizers;
       requiring it to hold for k consecutive steps avoids firing on a single
       noisy step (same idea as L-BFGS' termination history).

    2. Validation-loss patience (Prechelt 1998, "Early Stopping - But When?")
       Track best_val_loss; an epoch counts as an improvement only if
            best_val_loss - val_loss > abs_tol + rel_tol * |best_val_loss|
       Stop after `val_patience` consecutive non-improving epochs. The
       min-delta is built from the same (abs_tol, rel_tol) pair so the
       config has a single, consistent meaning of "tolerance".

    Both criteria are skipped during a configurable warmup so noisy initial
    epochs don't cause premature stops.

    Config keys read from training_cfg:
        rel_tol         (default 1e-4)
        abs_tol         (default 1e-6)
        train_patience  (default 5)
        val_patience    (default 20)
        warmup_epochs   (default 10)
    """

    def __init__(self, training_cfg: Dict[str, Any]) -> None:
        # Tolerances - shared meaning across both criteria
        self.rel_tol = float(training_cfg.get("rel_tol", 1e-4))
        self.abs_tol = float(training_cfg.get("abs_tol", 1e-6))

        # Per-criterion patience
        self.train_patience = int(training_cfg.get("train_patience", 5))
        self.val_patience = int(training_cfg.get("val_patience", 20))

        # Warmup: don't evaluate either criterion before this epoch/step
        self.warmup_epochs = int(training_cfg.get("warmup_epochs", 10))

        if self.rel_tol < 0 or self.abs_tol < 0:
            raise ValueError("rel_tol and abs_tol must be non-negative")
        if self.train_patience < 1 or self.val_patience < 1:
            raise ValueError("train_patience and val_patience must be >= 1")

        # State
        self._prev_train_loss: Optional[float] = None
        self._train_converged_streak: int = 0
        self.best_val_loss: float = float("inf")
        self._val_no_improve_streak: int = 0

        # Reason for the stop, set when should_stop() returns True
        self.stop_reason: Optional[str] = None

    def _is_close(self, current: float, previous: float) -> bool:
        """numpy.isclose-style mixed abs+rel tolerance test."""
        return abs(previous - current) <= self.abs_tol + self.rel_tol * abs(previous)

    def step(self, epoch: int, train_loss: float, val_loss: Optional[float]) -> bool:
        """Update state and return True if training should stop.

        Streaks are frozen during warmup so that noisy initial epochs cannot
        accumulate patience counts that fire immediately after warmup ends.
        best_val_loss is still updated during warmup for checkpointing purposes.
        """
        in_warmup = epoch < self.warmup_epochs

        # Train-loss convergence streak — frozen during warmup
        if not in_warmup and self._prev_train_loss is not None and torch.isfinite(torch.tensor(train_loss)):
            if self._is_close(train_loss, self._prev_train_loss):
                self._train_converged_streak += 1
            else:
                self._train_converged_streak = 0
        self._prev_train_loss = train_loss

        # Validation improvement streak — frozen during warmup
        if val_loss is not None and torch.isfinite(torch.tensor(val_loss)):
            if torch.isfinite(torch.tensor(self.best_val_loss)):
                improvement_threshold = self.abs_tol + self.rel_tol * abs(self.best_val_loss)
                if self.best_val_loss - val_loss > improvement_threshold:
                    self.best_val_loss = val_loss
                    if not in_warmup:
                        self._val_no_improve_streak = 0
                else:
                    # Strictly lower but not by min_delta: update best for
                    # checkpointing without resetting the patience counter.
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                    if not in_warmup:
                        self._val_no_improve_streak += 1
            else:
                # First finite val_loss seen
                self.best_val_loss = val_loss
                self._val_no_improve_streak = 0

        if in_warmup:
            return False

        if self._train_converged_streak >= self.train_patience:
            self.stop_reason = (
                f"train-loss converged: |dL| <= {self.abs_tol:.2e} + "
                f"{self.rel_tol:.2e}*|L| for {self._train_converged_streak} consecutive epochs"
            )
            return True

        if self._val_no_improve_streak >= self.val_patience:
            self.stop_reason = (
                f"validation loss did not improve (min_delta = abs_tol + rel_tol*|best|) "
                f"for {self._val_no_improve_streak} consecutive epochs "
                f"(best_val_loss={self.best_val_loss:.6e})"
            )
            return True

        return False

    @property
    def state(self) -> Dict[str, Any]:
        """Diagnostic snapshot, useful for logging."""
        return {
            "train_converged_streak": self._train_converged_streak,
            "val_no_improve_streak": self._val_no_improve_streak,
            "best_val_loss": self.best_val_loss,
        }


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

        # Logging cadence (cheap-to-eval losses are computed every epoch;
        # full metric dicts only at these intervals, purely for logging).
        self.validation_check_interval = int(training_cfg.get("validation_check_interval", 10))
        self.train_metrics_log_interval = int(training_cfg.get("train_metrics_log_interval", 10))

        if self.validation_check_interval <= 0:
            raise ValueError("validation_check_interval must be a positive integer")
        if self.train_metrics_log_interval <= 0:
            raise ValueError("train_metrics_log_interval must be a positive integer")

        # Unified convergence + early-stopping monitor. Reads rel_tol, abs_tol,
        # train_patience, val_patience, warmup_epochs from training_cfg.
        _warn_deprecated_keys(training_cfg)
        self.convergence_monitor = ConvergenceMonitor(training_cfg)

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

            # Train-set evaluation: kept only on logging ticks (matches original
            # `_check_interval` gating; metrics_history entry is None off-tick).
            train_metrics: Optional[Dict[str, float]] = None
            if _check_interval(epoch, self.train_metrics_log_interval):
                _, train_metrics = self._evaluate_loader(self.train_loader)
            train_metrics_history.append(train_metrics)

            # Validation: a single evaluation per epoch; val_loss is always
            # collected (needed by ConvergenceMonitor), val_metrics dict is
            # only retained on logging ticks (same single-call pattern as
            # original).
            val_loss: Optional[float] = None
            val_metrics: Optional[Dict[str, float]] = None
            if self.val_loader is not None:
                val_loss, raw_val_metrics = self._evaluate_loader(self.val_loader)
                if val_loss is not None and not torch.isfinite(torch.tensor(val_loss)):
                    logger.warning(f"Non-finite val_loss at epoch {epoch}: {val_loss}")
                    val_loss = None
                if _check_interval(epoch, self.validation_check_interval):
                    val_metrics = raw_val_metrics
            val_metrics_history.append(val_metrics)
            val_losses.append(val_loss)

            # Convergence monitor: single source of truth for early stopping.
            should_stop = self.convergence_monitor.step(epoch, train_loss, val_loss)

            # Log training progress every 10 epochs
            if epoch % 10 == 0:
                state = self.convergence_monitor.state
                if val_metrics is not None:
                    logger.info(
                        f"Epoch {epoch}/{self.n_epochs} - "
                        f"Train Loss: {train_loss:.6f}, "
                        f"Val Loss: {val_loss if val_loss is not None else float('nan'):.6f}, "
                        f"Val R²: {val_metrics.get('r2', float('nan')):.4f} | "
                        f"train_streak={state['train_converged_streak']}, "
                        f"val_no_improve={state['val_no_improve_streak']}"
                    )
                else:
                    logger.info(
                        f"Epoch {epoch}/{self.n_epochs} - Train Loss: {train_loss:.6f}, "
                        f"Val Loss: {val_loss} | "
                        f"train_streak={state['train_converged_streak']}, "
                        f"val_no_improve={state['val_no_improve_streak']}"
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

            if should_stop:
                logger.info(
                    f"Stopping MLP training early at epoch {epoch}: "
                    f"{self.convergence_monitor.stop_reason}"
                )
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

class _BaseGPTrainer:
    def __init__(self, training_cfg: Dict[str, Any]) -> None:
        self._force_cholesky = bool(training_cfg.get("force_cholesky", True))
        logger.info(f"Cholesky forced: {self._force_cholesky}")

        self._jitter_float32 = float(training_cfg.get("cholesky_jitter_float32", 1e-6))
        self._jitter_float64 = float(training_cfg.get("cholesky_jitter_float64", 1e-8))

        if self._jitter_float32 < 0:
            raise ValueError(f"cholesky_jitter_float32 must be non-negative, got {self._jitter_float32}")
        if self._jitter_float64 < 0:
            raise ValueError(f"cholesky_jitter_float64 must be non-negative, got {self._jitter_float64}")

        logger.info(
            f"Cholesky jitter configured: float32={self._jitter_float32:.2e}, "
            f"float64={self._jitter_float64:.2e}"
        )

    @contextlib.contextmanager
    def _cholesky_context(self):
        jitter_ctx = gpytorch.settings.cholesky_jitter(
            self._jitter_float32, self._jitter_float64
        )
        with warnings.catch_warnings(), jitter_ctx:
            warnings.filterwarnings("ignore", category=NumericalWarning)
            if self._force_cholesky:
                with gpytorch.settings.max_cholesky_size(float("inf")), gpytorch.settings.fast_computations(
                    covar_root_decomposition=False,
                    log_prob=False,
                    solves=False,
                ):
                    yield
                return
            yield

class GPTrainer(_BaseGPTrainer):
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
        super().__init__(training_cfg)

        self.device = device or resolve_device(str(training_cfg.get("device", "auto")))
        self.model.to(self.device)
        self.X_train = self.X_train.to(self.device)
        self.y_train = self.y_train.to(self.device)
        self.X_val = self.X_val.to(self.device) if self.X_val is not None else None
        self.y_val = self.y_val.to(self.device) if self.y_val is not None else None

        self.n_training_steps = int(training_cfg.get("n_training_steps", 5000))
        self.learning_rate = float(training_cfg.get("learning_rate", 1e-2))
        self.optimizer_name = str(training_cfg.get("optimizer", "adam")).lower()

        self.validation_check_interval = int(training_cfg.get("validation_check_interval", 10))
        self.train_metrics_log_interval = int(training_cfg.get("train_metrics_log_interval", 10))

        if self.validation_check_interval <= 0:
            raise ValueError("validation_check_interval must be a positive integer")
        if self.train_metrics_log_interval <= 0:
            raise ValueError("train_metrics_log_interval must be a positive integer")

        _warn_deprecated_keys(training_cfg)
        self.convergence_monitor = ConvergenceMonitor(training_cfg)

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
        self.scheduler = _build_scheduler(self.optimizer, training_cfg)

        self.mll_history: List[float] = []
        self.val_metrics_history: List[Optional[Dict[str, float]]] = []
        self.likelihood_noise_history: List[float] = []
        self.best_val_loss = float("inf")
        self._best_model_state: Optional[Dict[str, Any]] = None

        logger.info(f"GPTrainer initialized on device: {self.device}")
        logger.info(f"Initial GP likelihood noise: {_current_likelihood_noise(self.model):.6e}")

    def _optimizable_parameters(self):
        return list(self.model.parameters())

    def train(self) -> Dict[str, Any]:
        logger.info(f"Starting GP training for up to {self.n_training_steps} steps")
        start_time = time.time()
        train_losses: List[float] = []
        val_losses: List[Optional[float]] = []
        train_metrics_history: List[Optional[Dict[str, float]]] = []
        likelihood_noise_history: List[float] = []

        with self._cholesky_context():
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

                try:
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
                except _CHOLESKY_FAIL_EXCEPTIONS as exc:
                    _handle_cholesky_failure(
                        exc, model_name="GP", step=step, phase="training", trial=self.trial
                    )
                    break

                if self.scheduler is not None:
                    try:
                        self.scheduler.step()
                    except Exception:
                        logger.warning("Scheduler step failed; disabling scheduler for the rest of training")
                        self.scheduler = None


                loss_value = float(loss.item()) if torch.is_tensor(loss) else float(loss)
                mll_value = -loss_value
                current_noise = _current_likelihood_noise(self.model)

                # Compute eval results into locals first; commit *all* per-step
                # appends together at the end of the iteration, so a Cholesky
                # failure inside the eval block leaves every history list at the
                # same length (=step-1) and plots stay aligned.
                train_metrics: Optional[Dict[str, float]] = None
                val_loss: Optional[float] = None
                val_metrics: Optional[Dict[str, float]] = None
                try:
                    # Train-set evaluation: kept only on logging ticks.
                    if _check_interval(step, self.train_metrics_log_interval):
                        _, train_metrics = _evaluate_regression_model(
                            self.model,
                            self.X_train,
                            self.y_train,
                            force_cholesky=self._force_cholesky,
                        )

                    # Validation: single evaluation per step; val_metrics retained only
                    # on logging ticks (matches original single-call pattern).
                    if self.X_val is not None and self.y_val is not None:
                        val_loss, raw_val_metrics = _evaluate_regression_model(
                            self.model,
                            self.X_val,
                            self.y_val,
                            force_cholesky=self._force_cholesky,
                        )
                        if val_loss is not None and not torch.isfinite(torch.tensor(val_loss)):
                            logger.warning(f"Non-finite GP val_loss at step {step}: {val_loss}")
                            val_loss = None
                        if _check_interval(step, self.validation_check_interval):
                            val_metrics = raw_val_metrics
                except _CHOLESKY_FAIL_EXCEPTIONS as exc:
                        _handle_cholesky_failure(
                            exc, model_name="GP", step=step, phase="evaluation", trial=self.trial
                        )
                        break

                # Atomic commit: every list grows by exactly one entry per
                # successful step.
                train_losses.append(loss_value)          # negative MLL
                self.mll_history.append(mll_value)       # positive MLL
                self.likelihood_noise_history.append(current_noise)
                likelihood_noise_history.append(current_noise)
                train_metrics_history.append(train_metrics)
                self.val_metrics_history.append(val_metrics)
                val_losses.append(val_loss)

                should_stop = self.convergence_monitor.step(step, loss_value, val_loss)

                if step % 10 == 0:
                    val_loss_display = f"{val_loss:.6f}" if val_loss is not None else "None"
                    state = self.convergence_monitor.state
                    if val_metrics is not None:
                        logger.info(
                            f"Step {step}/{self.n_training_steps} - "
                            f"Train Loss (MLL): {mll_value:.6f}, "
                            f"Val MSE: {val_loss_display}, "
                            f"Val R²: {val_metrics.get('r2', float('nan')):.4f}, "
                            f"Val MAE: {val_metrics.get('mae', float('nan')):.4f}, "
                            f"Noise: {current_noise:.6e} | "
                            f"train_streak={state['train_converged_streak']}, "
                            f"val_no_improve={state['val_no_improve_streak']}"
                        )
                    else:
                        logger.info(
                            f"Step {step}/{self.n_training_steps} - "
                            f"Train Loss (MLL): {mll_value:.6f}, "
                            f"Val MSE: {val_loss_display}, "
                            f"Noise: {current_noise:.6e} | "
                            f"train_streak={state['train_converged_streak']}, "
                            f"val_no_improve={state['val_no_improve_streak']}"
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

                if should_stop:
                    logger.info(
                        f"Stopping GP training early at step {step}: "
                        f"{self.convergence_monitor.stop_reason}"
                    )
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

class MLPGPTrainer(_BaseGPTrainer):
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
        super().__init__(training_cfg)

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

        self.validation_check_interval = int(training_cfg.get("validation_check_interval", 10))
        self.train_metrics_log_interval = int(training_cfg.get("train_metrics_log_interval", 10))

        if self.validation_check_interval <= 0:
            raise ValueError("validation_check_interval must be a positive integer")
        if self.train_metrics_log_interval <= 0:
            raise ValueError("train_metrics_log_interval must be a positive integer")

        _warn_deprecated_keys(training_cfg)
        self.convergence_monitor = ConvergenceMonitor(training_cfg)

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

        with self._cholesky_context():
            for step in range(1, self.n_training_steps + 1):
                self.model.train()
                self.model.likelihood.train()

                try:
                    self.optimizer.zero_grad()
                    output = self.model(self.X_train)
                    loss = -self.mll(output, self.y_train)
                    if not torch.isfinite(loss):
                        logger.warning(f"Non-finite MLPGP loss at step {step}: {loss.item()}")
                        break
                    loss.backward()
                    self.optimizer.step()
                except _CHOLESKY_FAIL_EXCEPTIONS as exc:
                    _handle_cholesky_failure(
                        exc, model_name="MLPGP", step=step, phase="training", trial=self.trial
                    )
                    break

                if self.scheduler is not None:
                    try:
                        self.scheduler.step()
                    except Exception:
                        logger.warning("Scheduler step failed; disabling scheduler for the rest of training")
                        self.scheduler = None

                loss_value = float(loss.item()) if torch.is_tensor(loss) else float(loss)
                mll_value = -loss_value
                current_noise = _current_likelihood_noise(self.model)

                # Compute eval results into locals; commit *all* per-step appends
                # together at the end so a Cholesky failure during eval leaves
                # every history list at length=step-1.
                train_metrics: Optional[Dict[str, float]] = None
                val_loss: Optional[float] = None
                val_metrics: Optional[Dict[str, float]] = None
                try:
                    # Train-set evaluation: kept only on logging ticks.
                    if _check_interval(step, self.train_metrics_log_interval):
                        _, train_metrics = _evaluate_regression_model(
                            self.model,
                            self.X_train,
                            self.y_train,
                            force_cholesky=self._force_cholesky,
                        )

                    # Validation: single evaluation per step; val_metrics retained only
                    # on logging ticks.
                    if self.X_val is not None and self.y_val is not None:
                        val_loss, raw_val_metrics = _evaluate_regression_model(
                            self.model,
                            self.X_val,
                            self.y_val,
                            force_cholesky=self._force_cholesky,
                        )
                        if val_loss is not None and not torch.isfinite(torch.tensor(val_loss)):
                            logger.warning(f"Non-finite MLPGP val_loss at step {step}: {val_loss}")
                            val_loss = None
                        if _check_interval(step, self.validation_check_interval):
                            val_metrics = raw_val_metrics
                except _CHOLESKY_FAIL_EXCEPTIONS as exc:
                    _handle_cholesky_failure(
                        exc, model_name="MLPGP", step=step, phase="evaluation", trial=self.trial
                    )
                    break

                # Atomic commit: every list grows by exactly one entry per step.
                train_losses.append(loss_value)  # negative mll
                self.mll_history.append(mll_value)  # positive mll
                self.likelihood_noise_history.append(current_noise)
                likelihood_noise_history.append(current_noise)
                train_metrics_history.append(train_metrics)
                self.val_metrics_history.append(val_metrics)
                val_losses.append(val_loss)

                should_stop = self.convergence_monitor.step(step, loss_value, val_loss)

                if step % 10 == 0:
                    val_loss_display = f"{val_loss:.6f}" if val_loss is not None else "None"
                    state = self.convergence_monitor.state
                    if val_metrics is not None:
                        logger.info(
                            f"Step {step}/{self.n_training_steps} - "
                            f"MLL: {mll_value:.6f}, "
                            f"Val MSE: {val_loss_display}, "
                            f"Val R²: {val_metrics.get('r2', float('nan')):.4f}, "
                            f"Val MAE: {val_metrics.get('mae', float('nan')):.4f}, "
                            f"Noise: {current_noise:.6e} | "
                            f"train_streak={state['train_converged_streak']}, "
                            f"val_no_improve={state['val_no_improve_streak']}"
                        )
                    else:
                        logger.info(
                            f"Step {step}/{self.n_training_steps} - "
                            f"MLL: {mll_value:.6f}, "
                            f"Val MSE: {val_loss_display}, "
                            f"Noise: {current_noise:.6e} | "
                            f"train_streak={state['train_converged_streak']}, "
                            f"val_no_improve={state['val_no_improve_streak']}"
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

                if should_stop:
                    logger.info(
                        f"Stopping MLPGP training early at step {step}: "
                        f"{self.convergence_monitor.stop_reason}"
                    )
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