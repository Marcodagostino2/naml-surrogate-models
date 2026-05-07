"""Shared training pipeline: model/trainer factories, run-directory layout, ``train_model`` entry point."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import yaml

from src.data import DataBundle
from src.inference import Predictor
from src.models import (
    GPRegressionModel,
    MLPGPModel,
    MLPRegressor,
    MLPSVGPModel,
    SVGPRegressionModel,
)
from src.training import GPTrainer, MLPGPTrainer, MLPTrainer, SVGPTrainer, compute_all_metrics
from src.utils import OutputPathResolver, resolve_device, setup_logger
from src.visualization.diagnostics import generate_diagnostics

logger = setup_logger(__name__)

if TYPE_CHECKING:
    import optuna


# (model_class, trainer_class, supports_uncertainty)
_MODEL_REGISTRY: Dict[str, Tuple[type, type, bool]] = {
    "mlp": (MLPRegressor, MLPTrainer, False),
    "gp": (GPRegressionModel, GPTrainer, True),
    "mlpgp": (MLPGPModel, MLPGPTrainer, True),
}

# SVGP variants of the GP-family models, picked when model_cfg["inference"] == "svgp".
_SVGP_CLASSES: Dict[str, type] = {
    "gp": SVGPRegressionModel,
    "mlpgp": MLPSVGPModel,
}


def create_run_directory(output_resolver: OutputPathResolver) -> Path:
    run_dir = output_resolver.root
    run_dir.mkdir(parents=True, exist_ok=True)
    output_resolver.config_dir.mkdir(parents=True, exist_ok=True)
    output_resolver.plots_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_configs(
    output_resolver: OutputPathResolver,
    selected_model: str,
    main_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    viz_cfg: Dict[str, Any],
) -> None:
    """Persist the per-run config snapshot under ``run_dir/config/``."""
    config_out = output_resolver.config_dir
    entries: List[Tuple[Path, str, Optional[Dict[str, Any]]]] = [
        (config_out, "main_config.yaml", None),
        (config_out / "dataset", "dataset_config.yaml", dataset_cfg),
        (config_out / "model", f"{selected_model}_config.yaml", model_cfg),
        (config_out / "training", f"training_{selected_model}_config.yaml", training_cfg),
        (config_out / "visualization", "visualization_config.yaml", viz_cfg),
    ]
    patched_main_cfg = {**main_cfg, "model_to_train": selected_model}
    for subdir, filename, content in entries:
        subdir.mkdir(parents=True, exist_ok=True)
        payload = patched_main_cfg if filename == "main_config.yaml" else content
        with open(subdir / filename, "w", encoding="utf-8") as fh:
            yaml.dump(payload, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
    logger.debug(f"Run configs saved to {config_out}")


def save_summary_row(summary_csv_path: Path, run_info: Dict[str, Any]) -> None:
    """Append a row to a CSV, creating the file (with header) if missing."""
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = summary_csv_path.exists()
    with open(summary_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(run_info.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(run_info)
    logger.debug(f"Summary row appended to {summary_csv_path}")


def _resolve_gp_inference(model_name: str, model_cfg: Dict[str, Any]) -> str:
    """Return ``"exact"`` or ``"svgp"`` from ``model_cfg["inference"]``; raise on typos."""
    inference = str(model_cfg.get("inference", "exact")).lower()
    if inference not in {"exact", "svgp"}:
        raise ValueError(
            f"Unknown inference mode for {model_name!r}: {inference!r}. Expected 'exact' or 'svgp'."
        )
    return inference


def _gp_train_tensors(data_bundle: DataBundle) -> Tuple[torch.Tensor, torch.Tensor]:
    X_train_t = torch.from_numpy(data_bundle.X_train)
    y_train_t = torch.from_numpy(data_bundle.y_train)
    if y_train_t.ndim == 2 and y_train_t.shape[1] == 1:
        y_train_t = y_train_t.squeeze(-1)
    return X_train_t, y_train_t


def _build_model(model_name: str, data_bundle: DataBundle, model_cfg: Dict[str, Any]) -> Any:
    if model_name == "mlp":
        return MLPRegressor(
            input_dim=data_bundle.n_inputs,
            output_dim=data_bundle.n_outputs,
            model_cfg=model_cfg,
        )

    X_train_t, y_train_t = _gp_train_tensors(data_bundle)
    inference = _resolve_gp_inference(model_name, model_cfg)
    model_cls = _SVGP_CLASSES[model_name] if inference == "svgp" else _MODEL_REGISTRY[model_name][0]
    return model_cls(X_train=X_train_t, y_train=y_train_t, model_cfg=model_cfg)


def _build_trainer(
    model_name: str,
    model: Any,
    data_bundle: DataBundle,
    dataset_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    device: torch.device,
    trial: Optional["optuna.Trial"],
) -> Any:
    """Pick the trainer for ``model``. SVGP models route to ``SVGPTrainer`` regardless of ``model_name``."""
    if model_name == "mlp":
        batch_size = int(dataset_cfg.get("batch_size", 128))
        return MLPTrainer(
            model=model,
            train_loader=data_bundle.get_train_loader(batch_size=batch_size),
            val_loader=data_bundle.get_val_loader(batch_size=batch_size),
            training_cfg=training_cfg,
            device=device,
            trial=trial,
        )

    if isinstance(model, (SVGPRegressionModel, MLPSVGPModel)):
        batch_size = int(training_cfg.get("svgp_batch_size", dataset_cfg.get("batch_size", 1024)))
        return SVGPTrainer(
            model=model,
            train_loader=data_bundle.get_train_loader(batch_size=batch_size),
            val_loader=data_bundle.get_val_loader(batch_size=batch_size),
            training_cfg=training_cfg,
            device=device,
            trial=trial,
        )

    trainer_cls = _MODEL_REGISTRY[model_name][1]
    X_train_t, y_train_t = _gp_train_tensors(data_bundle)
    return trainer_cls(
        model=model,
        X_train=X_train_t,
        y_train=y_train_t,
        X_val=torch.from_numpy(data_bundle.X_val),
        y_val=torch.from_numpy(data_bundle.y_val),
        training_cfg=training_cfg,
        device=device,
        trial=trial,
    )


def _predict_split(predictor: Predictor, X: Any, supports_uncertainty: bool) -> Tuple[Any, Optional[Any]]:
    if supports_uncertainty:
        return predictor.predict(X, return_uncertainty=True)
    return predictor.predict(X), None


def train_model(
    model_name: str,
    data_bundle: DataBundle,
    dataset_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    output_resolver: OutputPathResolver,
    device: Optional[torch.device] = None,
    trial: Optional["optuna.Trial"] = None,
) -> Dict[str, Any]:
    """Train one model end-to-end: build, fit, evaluate, save diagnostics + checkpoint."""
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    _, _, supports_uncertainty = _MODEL_REGISTRY[model_name]
    logger.info("=" * 60)
    logger.info(f"Training {model_name.upper()} Model")
    logger.info("=" * 60)

    device = device or resolve_device(training_cfg.get("device", "auto"))
    model = _build_model(model_name, data_bundle, model_cfg)
    trainer = _build_trainer(model_name, model, data_bundle, dataset_cfg, training_cfg, device, trial)
    history = trainer.train()

    logger.info(f"Evaluating {model_name.upper()} on test set...")
    predictor = Predictor(
        model=model,
        target_scaler_mean=data_bundle.target_scaler_mean,
        target_scaler_std=data_bundle.target_scaler_std,
        device=device,
    )

    splits = {
        "train": (data_bundle.X_train, data_bundle.y_train),
        "val": (data_bundle.X_val, data_bundle.y_val),
        "test": (data_bundle.X_test, data_bundle.y_test),
    }

    predictions: Dict[str, Any] = {}
    uncertainties: Dict[str, Optional[Any]] = {}
    metrics: Dict[str, Dict[str, float]] = {}
    y_scale_mean = data_bundle.target_scaler_mean
    y_scale_std = data_bundle.target_scaler_std

    for split_name, (X_split, y_split_norm) in splits.items():
        y_pred, y_std = _predict_split(predictor, X_split, supports_uncertainty)
        y_true = y_split_norm * y_scale_std + y_scale_mean
        predictions[split_name] = y_pred
        uncertainties[split_name] = y_std
        metrics[split_name] = compute_all_metrics(y_true, y_pred)

    logger.info(f"Test R²: {metrics['test']['r2']:.6f}, Test RMSE: {metrics['test']['rmse']:.6f}")
    logger.info("=" * 70)
    logger.info(f"FINAL METRICS SUMMARY ({model_name.upper()})")
    logger.info("=" * 70)
    for split in ["train", "val", "test"]:
        m = metrics[split]
        logger.info(
            f"{split.upper()}: "
            f"MAE={m['mae']:.6f}, MSE={m['mse']:.6f}, RMSE={m['rmse']:.6f}, "
            f"R²={m['r2']:.6f}, MAPE={m['mape']:.2f}%"
        )
    logger.info("=" * 70)

    generate_diagnostics(
        y_val_true=splits["val"][1] * y_scale_std + y_scale_mean,
        y_val_pred=predictions["val"],
        y_test_true=splits["test"][1] * y_scale_std + y_scale_mean,
        y_test_pred=predictions["test"],
        train_losses=history.get("train_losses"),
        val_losses=history.get("val_losses"),
        y_val_std=uncertainties["val"],
        y_test_std=uncertainties["test"],
        output_dir=output_resolver.plots_dir,
    )

    checkpoint_path = output_resolver.checkpoint
    torch.save(model.state_dict(), checkpoint_path)
    logger.debug(f"Model checkpoint saved to {checkpoint_path}")

    return {
        "model": model,
        "predictor": predictor,
        "history": history,
        "train_metrics": metrics["train"],
        "val_metrics": metrics["val"],
        "test_metrics": metrics["test"],
    }


__all__ = [
    "create_run_directory",
    "save_run_configs",
    "save_summary_row",
    "train_model",
]
