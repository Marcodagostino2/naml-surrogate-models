"""Shared training pipeline: model/trainer factories, run-directory layout, ``train_model`` entry point."""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
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
from src.visualization.plots.diagnostics import generate_diagnostics

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


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _history_summary(history: Dict[str, Any]) -> Dict[str, Any]:
    train_losses = [float(x) for x in history.get("train_losses", []) if x is not None]
    val_losses_raw = history.get("val_losses", [])
    val_losses = [float(v) if v is not None else None for v in val_losses_raw]

    best_val_loss = None
    best_val_step = None
    finite_pairs = [
        (idx + 1, float(v)) for idx, v in enumerate(val_losses)
        if v is not None and np.isfinite(float(v))
    ]
    if finite_pairs:
        best_val_step, best_val_loss = min(finite_pairs, key=lambda item: item[1])

    return {
        "n_train_points": len(train_losses),
        "n_val_points": len(val_losses),
        "train_loss_final": train_losses[-1] if train_losses else None,
        "val_loss_final": next((v for v in reversed(val_losses) if v is not None), None),
        "best_val_loss": best_val_loss,
        "best_val_step": best_val_step,
    }


def save_run_artifacts(
    output_resolver: OutputPathResolver,
    *,
    mode: str,
    model_name: str,
    history: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    timings: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    run_root = output_resolver.root
    run_root.mkdir(parents=True, exist_ok=True)

    history_payload = _json_safe(history or {})
    history_path = run_root / "history.json"
    with open(history_path, "w", encoding="utf-8") as f_history:
        json.dump(history_payload, f_history, indent=2)

    artifacts = {
        "config_dir": str(output_resolver.config_dir),
        "plots_dir": str(output_resolver.plots_dir),
        "checkpoint": str(output_resolver.checkpoint),
        "summary_csv": str(output_resolver.summary_csv),
        "history_json": str(history_path),
    }
    if output_resolver._run_type_value == "optimization":
        artifacts["best_params"] = str(output_resolver.best_params)
        artifacts["all_trials_csv"] = str(output_resolver.all_trials_csv)

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "dataset": output_resolver.dataset_name,
        "mode": str(mode),
        "model": str(model_name),
        "metrics": _json_safe(metrics or {}),
        "timings": _json_safe(timings or {}),
        "history_summary": _history_summary(history or {}),
        "artifacts": artifacts,
        "extra": _json_safe(extra or {}),
    }
    manifest_path = run_root / "results_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f_manifest:
        json.dump(manifest, f_manifest, indent=2)
    logger.debug(f"Run artifacts saved: {manifest_path}, {history_path}")


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


def _benchmark_inference(
    predictor: Predictor,
    X: Any,
    supports_uncertainty: bool,
    n_warmup: int = 2,
    n_runs: int = 5,
) -> Dict[str, float]:
    """Mean wall-clock time of ``Predictor.predict`` on ``X`` over ``n_runs``
    calls, after ``n_warmup`` discarded calls. Reports total + per-sample.
    """
    n_samples = int(np.asarray(X).shape[0])
    if n_samples == 0 or n_runs <= 0:
        return {"inference_total_seconds": float("nan"), "inference_per_sample_seconds": float("nan"), "n_samples": n_samples}
    for _ in range(max(0, n_warmup)):
        _predict_split(predictor, X, supports_uncertainty)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timings: List[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _predict_split(predictor, X, supports_uncertainty)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - t0)
    total_mean = float(np.mean(timings))
    return {
        "inference_total_seconds": total_mean,
        "inference_per_sample_seconds": total_mean / n_samples,
        "n_samples": n_samples,
    }


def _predict_split(predictor: Predictor, X: Any, supports_uncertainty: bool) -> Tuple[Any, Optional[Any]]:
    if supports_uncertainty:
        return predictor.predict(X, return_uncertainty=True)
    return predictor.predict(X), None


def _evaluate_model(
    *,
    model: Any,
    model_name: str,
    data_bundle: DataBundle,
    output_resolver: OutputPathResolver,
    device: torch.device,
    supports_uncertainty: bool,
    history: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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

    train_elapsed = float((history or {}).get("elapsed_time", float("nan")))
    timings = {
        "training_seconds": train_elapsed,
        **_benchmark_inference(predictor, data_bundle.X_test, supports_uncertainty),
    }

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
    logger.info(
        f"TIMING: train={timings['training_seconds']:.2f}s | "
        f"infer(test, n={timings['n_samples']})={timings['inference_total_seconds'] * 1e3:.2f}ms "
        f"({timings['inference_per_sample_seconds'] * 1e6:.1f}us/sample)"
    )
    logger.info("=" * 70)

    generate_diagnostics(
        y_val_true=splits["val"][1] * y_scale_std + y_scale_mean,
        y_val_pred=predictions["val"],
        y_test_true=splits["test"][1] * y_scale_std + y_scale_mean,
        y_test_pred=predictions["test"],
        train_losses=(history or {}).get("train_losses"),
        val_losses=(history or {}).get("val_losses"),
        y_val_std=uncertainties["val"],
        y_test_std=uncertainties["test"],
        output_dir=output_resolver.plots_dir / "diagnostics",
    )

    return {
        "model": model,
        "predictor": predictor,
        "history": history or {},
        "train_metrics": metrics["train"],
        "val_metrics": metrics["val"],
        "test_metrics": metrics["test"],
        "timings": timings,
    }


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

    run_mode = output_resolver._run_type_value
    training_cfg_effective = dict(training_cfg)
    training_cfg_effective.setdefault("run_mode", run_mode)
    training_cfg_effective.setdefault("run_model", model_name)
    if trial is not None and hasattr(trial, "number"):
        training_cfg_effective["run_trial_number"] = int(trial.number) + 1

    device = device or resolve_device(training_cfg_effective.get("device", "auto"))
    model = _build_model(model_name, data_bundle, model_cfg)
    trainer = _build_trainer(
        model_name, model, data_bundle, dataset_cfg, training_cfg_effective, device, trial,
    )
    history = trainer.train()

    eval_results = _evaluate_model(
        model=model,
        model_name=model_name,
        data_bundle=data_bundle,
        output_resolver=output_resolver,
        device=device,
        supports_uncertainty=supports_uncertainty,
        history=history,
    )

    checkpoint_path = output_resolver.checkpoint
    torch.save(model.state_dict(), checkpoint_path)
    logger.debug(f"Model checkpoint saved to {checkpoint_path}")

    save_run_artifacts(
        output_resolver,
        mode=run_mode,
        model_name=model_name,
        history=history,
        metrics={
            "train": eval_results["train_metrics"],
            "val": eval_results["val_metrics"],
            "test": eval_results["test_metrics"],
        },
        timings=eval_results["timings"],
        extra={"supports_uncertainty": supports_uncertainty},
    )
    eval_results["history"] = history
    return eval_results


def test_model_from_checkpoint(
    model_name: str,
    data_bundle: DataBundle,
    dataset_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    checkpoint_path: Path,
    output_resolver: OutputPathResolver,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a checkpoint and run test/evaluation without re-training."""
    del dataset_cfg
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    _, _, supports_uncertainty = _MODEL_REGISTRY[model_name]
    logger.info("=" * 60)
    logger.info(f"Testing {model_name.upper()} Model from checkpoint")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info("=" * 60)

    device = device or resolve_device("auto")
    model = _build_model(model_name, data_bundle, model_cfg)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    eval_results = _evaluate_model(
        model=model,
        model_name=model_name,
        data_bundle=data_bundle,
        output_resolver=output_resolver,
        device=device,
        supports_uncertainty=supports_uncertainty,
        history={},
    )

    save_run_artifacts(
        output_resolver,
        mode=output_resolver._run_type_value,
        model_name=model_name,
        history={},
        metrics={
            "train": eval_results["train_metrics"],
            "val": eval_results["val_metrics"],
            "test": eval_results["test_metrics"],
        },
        timings=eval_results["timings"],
        extra={
            "supports_uncertainty": supports_uncertainty,
            "source_checkpoint": str(checkpoint_path),
            "tested_without_training": True,
        },
    )
    return eval_results


__all__ = [
    "create_run_directory",
    "save_run_configs",
    "save_run_artifacts",
    "save_summary_row",
    "test_model_from_checkpoint",
    "train_model",
]
