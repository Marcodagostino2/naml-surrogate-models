"""
Main entry point for the aerodynamic surrogate model training and evaluation pipeline.

Orchestrates:
1. Configuration loading (main, dataset, model, training)
2. Data loading and preprocessing
3. Model training (MLP or GP)
4. Evaluation and diagnostics
5. Output management and logging
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import yaml

import torch

from src.data import load_and_preprocess_data
from src.inference import Predictor
from src.models import GPRegressionModel, MLPGPModel, MLPRegressor
from src.plot import generate_diagnostics
from src.training import GPTrainer, MLPTrainer, MLPGPTrainer, compute_all_metrics
from src.utils import (
    load_yaml,
    resolve_device,
    save_yaml,
    set_global_seed,
    setup_logger,
    setup_torch_dtype,
)

logger = setup_logger(__name__)


def create_run_directory(base_output_dir: Path, model_name: str) -> Path:
    """
    Create a timestamped run directory for outputs.

    Parameters
    ----------
    base_output_dir : Path
        Base output directory (e.g., "outputs/mlp").
    model_name : str
        Model name (e.g., "mlp", "gp").

    Returns
    -------
    Path
        Timestamped run directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / model_name / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config").mkdir(exist_ok=True)
    (run_dir / "plot").mkdir(exist_ok=True)

    return run_dir


def save_run_configs(
    run_dir: Path,
    selected_model: str,
    main_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    viz_cfg: Dict[str, Any],
) -> None:
    """
    Persist all run configurations inside ``run_dir/config/``, mirroring the
    canonical source layout.  ``main_config.yaml`` is patched so that
    ``model_to_train`` always reflects the model actually trained in this run
    (never ``"all"``).

    Output layout
    -------------
    run_dir/
    └── config/
        ├── main_config.yaml
        ├── dataset/
        │   └── dataset_config.yaml
        ├── model/
        │   └── {selected_model}_config.yaml
        ├── training/
        │   └── training_{selected_model}_config.yaml
        └── visualization/
            └── visualization_config.yaml

    Parameters
    ----------
    run_dir : Path
        Root directory of the current run.
    selected_model : str
        Model identifier (``"mlp"``, ``"gp"``, or ``"mlpgp"``).
    main_cfg : Dict[str, Any]
        Loaded main configuration.
    dataset_cfg : Dict[str, Any]
        Loaded dataset configuration.
    model_cfg : Dict[str, Any]
        Loaded model configuration for ``selected_model``.
    training_cfg : Dict[str, Any]
        Loaded training configuration for ``selected_model``.
    viz_cfg : Dict[str, Any]
        Loaded visualization configuration.
    """
    config_out = run_dir / "config"

    # Subdirectory map: subdir → (filename, content)
    entries: List[Tuple[Path, str, Dict[str, Any]]] = [
        (config_out,               "main_config.yaml",                          None),
        (config_out / "dataset",   "dataset_config.yaml",                       dataset_cfg),
        (config_out / "model",     f"{selected_model}_config.yaml",             model_cfg),
        (config_out / "training",  f"training_{selected_model}_config.yaml",    training_cfg),
        (config_out / "visualization", "visualization_config.yaml",             viz_cfg),
    ]

    # Patch main_cfg: model_to_train must always be the specific model, never "all"
    patched_main_cfg = {**main_cfg, "model_to_train": selected_model}

    for subdir, filename, content in entries:
        subdir.mkdir(parents=True, exist_ok=True)
        payload = patched_main_cfg if filename == "main_config.yaml" else content
        with open(subdir / filename, "w", encoding="utf-8") as fh:
            yaml.dump(payload, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logger.debug(f"Run configs saved to {config_out}")


def save_summary_row(
    summary_csv_path: Path,
    run_info: Dict[str, Any],
) -> None:
    """
    Append a row to the summary CSV file.

    Parameters
    ----------
    summary_csv_path : Path
        Path to summary.csv.
    run_info : Dict[str, Any]
        Dictionary of metrics and info for this run.
    """
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists to determine if we need headers
    file_exists = summary_csv_path.exists()

    with open(summary_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(run_info.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(run_info)

    logger.info(f"Summary row appended to {summary_csv_path}")


def train_mlp(
    data_bundle,
    dataset_cfg: Dict,
    model_cfg: Dict,
    training_cfg: Dict,
    run_dir: Path,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Train standalone MLP model.

    Parameters
    ----------
    data_bundle
        DataBundle object.
    dataset_cfg : Dict
        Dataset configuration.
    model_cfg : Dict
        MLP model configuration.
    training_cfg : Dict
        MLP training configuration.
    run_dir : Path
        Run output directory.
    device : torch.device, optional
        Device to use.

    Returns
    -------
    Dict[str, Any]
        Results including metrics and model.
    """
    logger.info("=" * 60)
    logger.info("Training MLP Model")
    logger.info("=" * 60)

    device = device or resolve_device(training_cfg.get("device", "auto"))

    # Create model
    model = MLPRegressor(
        input_dim=data_bundle.n_inputs,
        output_dim=data_bundle.n_outputs,
        model_cfg=model_cfg,
    )

    # Create dataloaders
    batch_size = int(dataset_cfg.get("batch_size", 128))
    train_loader = data_bundle.get_train_loader(batch_size=batch_size)
    val_loader = data_bundle.get_val_loader(batch_size=batch_size)

    # Train
    trainer = MLPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_cfg=training_cfg,
        device=device,
    )

    history = trainer.train()

    # Evaluate
    logger.info("Evaluating MLP on test set...")

    predictor = Predictor(
        model=model,
        target_scaler_mean=data_bundle.target_scaler_mean,
        target_scaler_std=data_bundle.target_scaler_std,
        device=device,
    )

    y_train_pred = predictor.predict(data_bundle.X_train)
    y_val_pred = predictor.predict(data_bundle.X_val)
    y_test_pred = predictor.predict(data_bundle.X_test)

    # Denormalize for metrics
    y_train = data_bundle.y_train * data_bundle.target_scaler_std + data_bundle.target_scaler_mean
    y_val = data_bundle.y_val * data_bundle.target_scaler_std + data_bundle.target_scaler_mean
    y_test = data_bundle.y_test * data_bundle.target_scaler_std + data_bundle.target_scaler_mean

    train_metrics = compute_all_metrics(y_train, y_train_pred)
    val_metrics = compute_all_metrics(y_val, y_val_pred)
    test_metrics = compute_all_metrics(y_test, y_test_pred)

    logger.info(f"Test R²: {test_metrics['r2']:.6f}, Test RMSE: {test_metrics['rmse']:.6f}")

    # Plots
    generate_diagnostics(
        y_train_true=y_train,
        y_train_pred=y_train_pred,
        y_val_true=y_val,
        y_val_pred=y_val_pred,
        y_test_true=y_test,
        y_test_pred=y_test_pred,
        train_losses=history["train_losses"],
        val_losses=history["val_losses"],
        output_dir=run_dir / "plot",
    )

    # Save model
    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Model checkpoint saved to {checkpoint_path}")

    return {
        "model": model,
        "predictor": predictor,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
    }


def train_gp(
    data_bundle,
    model_cfg: Dict,
    training_cfg: Dict,
    run_dir: Path,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Train standalone GP (Kriging) model.

    Parameters
    ----------
    data_bundle
        DataBundle object.
    model_cfg : Dict
        GP model configuration.
    training_cfg : Dict
        GP training configuration.
    run_dir : Path
        Run output directory.
    device : torch.device, optional
        Device to use.

    Returns
    -------
    Dict[str, Any]
        Results including metrics and model.
    """
    logger.info("=" * 60)
    logger.info("Training GP (Kriging) Model")
    logger.info("=" * 60)

    device = device or resolve_device(training_cfg.get("device", "auto"))

    # Convert training data to tensors
    X_train_t = torch.from_numpy(data_bundle.X_train)
    y_train_t = torch.from_numpy(data_bundle.y_train)

    # Squeeze if single output
    if y_train_t.ndim == 2 and y_train_t.shape[1] == 1:
        y_train_t = y_train_t.squeeze(-1)

    # Create model
    likelihood = None  # Let GPRegressionModel create default
    model = GPRegressionModel(
        X_train=X_train_t,
        y_train=y_train_t,
        model_cfg=model_cfg,
        likelihood=likelihood,
    )

    # Train
    # Validation tensors
    X_val_t = torch.from_numpy(data_bundle.X_val)
    y_val_t = torch.from_numpy(data_bundle.y_val)

    trainer = GPTrainer(
        model=model,
        X_train=X_train_t,
        y_train=y_train_t,
        X_val=X_val_t,
        y_val=y_val_t,
        training_cfg=training_cfg,
        device=device,
    )

    history = trainer.train()

    # Evaluate
    logger.info("Evaluating GP on test set...")

    predictor = Predictor(
        model=model,
        target_scaler_mean=data_bundle.target_scaler_mean,
        target_scaler_std=data_bundle.target_scaler_std,
        device=device,
    )

    y_train_pred, y_train_std = predictor.predict(data_bundle.X_train, return_uncertainty=True)
    y_val_pred, y_val_std = predictor.predict(data_bundle.X_val, return_uncertainty=True)
    y_test_pred, y_test_std = predictor.predict(data_bundle.X_test, return_uncertainty=True)

    # Denormalize for metrics
    y_train = data_bundle.y_train * data_bundle.target_scaler_std + data_bundle.target_scaler_mean
    y_val = data_bundle.y_val * data_bundle.target_scaler_std + data_bundle.target_scaler_mean
    y_test = data_bundle.y_test * data_bundle.target_scaler_std + data_bundle.target_scaler_mean

    train_metrics = compute_all_metrics(y_train, y_train_pred)
    val_metrics = compute_all_metrics(y_val, y_val_pred)
    test_metrics = compute_all_metrics(y_test, y_test_pred)

    logger.info(f"Test R²: {test_metrics['r2']:.6f}, Test RMSE: {test_metrics['rmse']:.6f}")

    # Plots
    generate_diagnostics(
        y_train_true=y_train,
        y_train_pred=y_train_pred,
        y_val_true=y_val,
        y_val_pred=y_val_pred,
        y_test_true=y_test,
        y_test_pred=y_test_pred,
        y_val_std=y_val_std,
        y_test_std=y_test_std,
        output_dir=run_dir / "plot",
    )

    # Save model
    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Model checkpoint saved to {checkpoint_path}")

    return {
        "model": model,
        "predictor": predictor,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
    }


def train_mlpgp(
    data_bundle,
    model_cfg: Dict,
    training_cfg: Dict,
    run_dir: Path,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Train end-to-end MLPGP model.

    Parameters
    ----------
    data_bundle
        DataBundle object.
    model_cfg : Dict
        MLPGP model configuration.
    training_cfg : Dict
        MLPGP training configuration.
    run_dir : Path
        Run output directory.
    device : torch.device, optional
        Device to use.

    Returns
    -------
    Dict[str, Any]
        Results including metrics and model.
    """
    logger.info("=" * 60)
    logger.info("Training MLPGP Model")
    logger.info("=" * 60)

    device = device or resolve_device(training_cfg.get("device", "auto"))

    X_train_t = torch.from_numpy(data_bundle.X_train)
    y_train_t = torch.from_numpy(data_bundle.y_train)

    if y_train_t.ndim == 2 and y_train_t.shape[1] == 1:
        y_train_t = y_train_t.squeeze(-1)

    likelihood = None
    model = MLPGPModel(
        X_train=X_train_t,
        y_train=y_train_t,
        model_cfg=model_cfg,
        likelihood=likelihood,
    )

    X_val_t = torch.from_numpy(data_bundle.X_val)
    y_val_t = torch.from_numpy(data_bundle.y_val)

    trainer = MLPGPTrainer(
        model=model,
        X_train=X_train_t,
        y_train=y_train_t,
        X_val=X_val_t,
        y_val=y_val_t,
        training_cfg=training_cfg,
        device=device,
    )

    history = trainer.train()

    logger.info("Evaluating MLPGP on test set...")

    predictor = Predictor(
        model=model,
        target_scaler_mean=data_bundle.target_scaler_mean,
        target_scaler_std=data_bundle.target_scaler_std,
        device=device,
    )

    y_train_pred, y_train_std = predictor.predict(data_bundle.X_train, return_uncertainty=True)
    y_val_pred, y_val_std = predictor.predict(data_bundle.X_val, return_uncertainty=True)
    y_test_pred, y_test_std = predictor.predict(data_bundle.X_test, return_uncertainty=True)

    y_train = data_bundle.y_train * data_bundle.target_scaler_std + data_bundle.target_scaler_mean
    y_val = data_bundle.y_val * data_bundle.target_scaler_std + data_bundle.target_scaler_mean
    y_test = data_bundle.y_test * data_bundle.target_scaler_std + data_bundle.target_scaler_mean

    train_metrics = compute_all_metrics(y_train, y_train_pred)
    val_metrics = compute_all_metrics(y_val, y_val_pred)
    test_metrics = compute_all_metrics(y_test, y_test_pred)

    logger.info(f"Test R²: {test_metrics['r2']:.6f}, Test RMSE: {test_metrics['rmse']:.6f}")

    generate_diagnostics(
        y_train_true=y_train,
        y_train_pred=y_train_pred,
        y_val_true=y_val,
        y_val_pred=y_val_pred,
        y_test_true=y_test,
        y_test_pred=y_test_pred,
        y_val_std=y_val_std,
        y_test_std=y_test_std,
        output_dir=run_dir / "plot",
    )

    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Model checkpoint saved to {checkpoint_path}")

    return {
        "model": model,
        "predictor": predictor,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
    }


def main() -> None:
    """Main training pipeline orchestrator."""

    # ==================== Argument Parsing ====================
    parser = argparse.ArgumentParser(
        description="Train aerodynamic surrogate models (MLP, GP, or Hybrid MLP+GP)."
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Path to config directory. All configurations are loaded from this directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Base output directory.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config_base_dir = project_root / args.config_dir
    main_config_path = config_base_dir / "main_config.yaml" 
    output_base_dir = project_root / args.output_dir

    # ==================== Config Loading ====================
    logger.info("Loading configurations...")

    main_cfg = load_yaml(main_config_path)
    global_seed = int(main_cfg.get("global_seed", 42))
    model_to_train = str(main_cfg.get("model_to_train", "mlp")).lower()
    run_mode = str(main_cfg.get("run_mode", "training")).lower()
    precision = str(main_cfg.get("precision", "float32")).lower()

    set_global_seed(global_seed)
    logger.info(f"Global seed set to {global_seed}")

    setup_torch_dtype(precision)
    logger.info(f"PyTorch default dtype set to {torch.get_default_dtype()}")

    # Load dataset config
    dataset_cfg = load_yaml(config_base_dir / "dataset" / "dataset_config.yaml")

    if model_to_train not in {"mlp", "gp", "mlpgp", "all"}:
        raise ValueError(f"Unknown model_to_train: {model_to_train}")

    viz_cfg = load_yaml(config_base_dir / "visualization" / "visualization_config.yaml")

    logger.info(f"Model to train: {model_to_train.upper()}")
    logger.info(f"Run mode: {run_mode.upper()}")

    # ==================== Data Loading ====================
    logger.info("Loading and preprocessing data...")

    csv_path = project_root / dataset_cfg["csv_path"]
    data_bundle = load_and_preprocess_data(
        csv_path=csv_path,
        input_columns=dataset_cfg["input_columns"],
        target_columns=dataset_cfg["target_columns"],
        dropna=bool(dataset_cfg.get("dropna", True)),
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", True)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", True)),
        train_ratio=float(dataset_cfg.get("train_ratio", 0.70)),
        val_ratio=float(dataset_cfg.get("val_ratio", 0.15)),
        test_ratio=float(dataset_cfg.get("test_ratio", 0.15)),
        random_seed=global_seed,
        default_type=precision,
    )

    logger.info(
        f"Data shapes: "
        f"X_train={data_bundle.X_train.shape}, "
        f"X_val={data_bundle.X_val.shape}, "
        f"X_test={data_bundle.X_test.shape}"
    )

    selected_models = ["mlp", "gp", "mlpgp"] if model_to_train == "all" else [model_to_train]

    # ==================== Training Phase ====================
    if "training" in run_mode:
        for selected_model in selected_models:
            if selected_model == "mlp":
                model_cfg = load_yaml(config_base_dir / "model" / "mlp_config.yaml")
                training_cfg = load_yaml(config_base_dir / "training" / "training_mlp_config.yaml")
            elif selected_model == "gp":
                model_cfg = load_yaml(config_base_dir / "model" / "gp_config.yaml")
                training_cfg = load_yaml(config_base_dir / "training" / "training_gp_config.yaml")
            elif selected_model == "mlpgp":
                model_cfg = load_yaml(config_base_dir / "model" / "mlpgp_config.yaml")
                training_cfg = load_yaml(config_base_dir / "training" / "training_mlpgp_config.yaml")
            else:
                raise ValueError(f"Unknown model: {selected_model}")
            

            run_dir = create_run_directory(output_base_dir, selected_model)
            logger.info(f"Run directory created: {run_dir}")

            save_run_configs(
                run_dir=run_dir,
                selected_model=selected_model,
                main_cfg=main_cfg,
                dataset_cfg=dataset_cfg,
                model_cfg=model_cfg,
                training_cfg=training_cfg,
                viz_cfg=viz_cfg,
            )

            if selected_model == "mlp":
                results = train_mlp(
                    data_bundle=data_bundle,
                    dataset_cfg=dataset_cfg,
                    model_cfg=model_cfg,
                    training_cfg=training_cfg,
                    run_dir=run_dir,
                )
            elif selected_model == "gp":
                results = train_gp(
                    data_bundle=data_bundle,
                    model_cfg=model_cfg,
                    training_cfg=training_cfg,
                    run_dir=run_dir,
                )
            else:
                results = train_mlpgp(
                    data_bundle=data_bundle,
                    model_cfg=model_cfg,
                    training_cfg=training_cfg,
                    run_dir=run_dir,
                )

            summary_csv_path = output_base_dir / selected_model / "summary.csv"
            run_info = {
                "timestamp": datetime.now().isoformat(),
                "run_dir": str(run_dir),
                "test_r2": results["test_metrics"]["r2"],
                "test_rmse": results["test_metrics"]["rmse"],
                "test_mae": results["test_metrics"]["mae"],
                "test_mape": results["test_metrics"]["mape"],
                "val_r2": results["val_metrics"]["r2"],
                "val_rmse": results["val_metrics"]["rmse"],
            }
            save_summary_row(summary_csv_path, run_info)

            logger.info(f"Training completed. Results saved to {run_dir}")

    logger.info("=" * 60)
    logger.info("Pipeline Execution Completed Successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
