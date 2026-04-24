from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .metrics import regression_metrics


@dataclass
class TrainingResult:
    history: Dict[str, List[float]]
    best_epoch: int
    best_val_loss: float


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def _build_optimizer(model: nn.Module, training_cfg: Dict) -> torch.optim.Optimizer:
    optimizer_name = str(training_cfg.get("optimizer_name", "adam")).lower()
    learning_rate = float(training_cfg.get("learning_rate", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    optimizer_kwargs = dict(training_cfg.get("optimizer_kwargs", {}))

    optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "adagrad": torch.optim.Adagrad,
        "adadelta": torch.optim.Adadelta,
    }
    if optimizer_name not in optimizers:
        raise ValueError(
            f"Unsupported optimizer '{optimizer_name}'. Available: {sorted(optimizers)}"
        )

    optimizer_class = optimizers[optimizer_name]
    optimizer_arguments = {
        "lr": learning_rate,
        "weight_decay": weight_decay,
        **optimizer_kwargs,
    }

    if optimizer_name == "sgd" and "momentum" not in optimizer_arguments:
        optimizer_arguments["momentum"] = 0.0

    return optimizer_class(model.parameters(), **optimizer_arguments)


def _run_epoch(
    model: nn.Module,
    loader,
    loss_fn,
    device: torch.device,
    optimizer=None,
    show_batch_progress: bool = False,
    desc: str = "",
) -> float:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    num_samples = 0

    iterator = tqdm(loader, desc=desc, leave=False) if show_batch_progress else loader
    for features, targets in iterator:
        features = features.to(device)
        targets = targets.to(device)

        if is_training:
            optimizer.zero_grad()

        predictions = model(features)
        loss = loss_fn(predictions, targets)

        if is_training:
            loss.backward()
            optimizer.step()

        batch_size = features.shape[0]
        total_loss += float(loss.detach().cpu().item()) * batch_size
        num_samples += batch_size

    return total_loss / max(1, num_samples)


def train_model(model: nn.Module, data_bundle, training_cfg: Dict) -> tuple[TrainingResult, Dict]:
    device = _resolve_device(str(training_cfg.get("device", "auto")))
    model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = _build_optimizer(model, training_cfg)

    num_epochs = int(training_cfg.get("epochs", 200))
    patience = int(training_cfg.get("early_stopping_patience", 25))
    use_tqdm = bool(training_cfg.get("use_tqdm", True))
    progress_mode = str(training_cfg.get("progress_mode", "epoch")).lower()
    log_every_n_epochs = max(1, int(training_cfg.get("log_every_n_epochs", 1)))

    if progress_mode not in {"epoch", "batch", "off"}:
        raise ValueError("progress_mode must be one of: epoch, batch, off")

    if not use_tqdm:
        progress_mode = "off"

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    epoch_iterator = range(1, num_epochs + 1)
    show_epoch_progress = progress_mode == "epoch"
    if show_epoch_progress:
        epoch_iterator = tqdm(epoch_iterator, desc="training", unit="epoch")

    for epoch in epoch_iterator:
        train_loss = _run_epoch(
            model,
            data_bundle.train_loader,
            loss_fn,
            device,
            optimizer=optimizer,
            show_batch_progress=(progress_mode == "batch"),
            desc=f"epoch {epoch}/{num_epochs} train",
        )
        val_loss = _run_epoch(
            model,
            data_bundle.val_loader,
            loss_fn,
            device,
            optimizer=None,
            show_batch_progress=(progress_mode == "batch"),
            desc=f"epoch {epoch}/{num_epochs} val",
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale_epochs += 1

        if show_epoch_progress:
            epoch_iterator.set_postfix(
                train_loss=f"{train_loss:.6f}",
                val_loss=f"{val_loss:.6f}",
                best_val=f"{best_val_loss:.6f}",
                best_epoch=best_epoch,
                stale=stale_epochs,
            )

        should_log = (epoch % log_every_n_epochs == 0) or (epoch == 1) or (epoch == num_epochs)
        if should_log and progress_mode != "batch":
            logger = tqdm.write if show_epoch_progress else print
            logger(
                "[epoch {}/{}] train_loss={:.6f} val_loss={:.6f} best_val={:.6f} best_epoch={} stale={}/{}".format(
                    epoch,
                    num_epochs,
                    train_loss,
                    val_loss,
                    best_val_loss,
                    best_epoch,
                    stale_epochs,
                    patience,
                )
            )

        if stale_epochs >= patience:
            if progress_mode != "batch":
                logger = tqdm.write if show_epoch_progress else print
                logger(f"Early stopping triggered at epoch {epoch}.")
            break

    result = TrainingResult(
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
    )
    return result, best_state


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader,
    device: torch.device,
    target_scaler=None,
) -> Dict[str, float]:
    model.eval()
    y_true_batches = []
    y_pred_batches = []
    for features, targets in loader:
        features = features.to(device)
        predictions = model(features)
        y_pred_batches.append(predictions.cpu().numpy())
        y_true_batches.append(targets.cpu().numpy())

    y_true = np.concatenate(y_true_batches, axis=0)
    y_pred = np.concatenate(y_pred_batches, axis=0)

    if target_scaler is not None:
        y_true = target_scaler.inverse_transform(y_true)
        y_pred = target_scaler.inverse_transform(y_pred)

    return regression_metrics(y_true, y_pred)
