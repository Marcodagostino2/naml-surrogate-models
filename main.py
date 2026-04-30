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
from typing import TYPE_CHECKING, Any, Dict
import yaml

import torch

from src.data import load_data_from_config
from src.pipeline import create_run_directory, save_run_configs, save_summary_row, train_model
from src.utils import (
    OutputPathResolver,
    RunType,
    load_run_configs,
    set_global_seed,
    setup_logger,
    setup_torch_dtype,
)

logger = setup_logger(__name__)

if TYPE_CHECKING:
    import optuna


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
    output_base_dir = project_root / args.output_dir

    # ==================== Config Loading ====================
    logger.info("Loading configurations...")

    loaded_configs = load_run_configs(config_base_dir, "mlp", include_optimize=False)
    main_cfg = loaded_configs["main_cfg"]
    global_seed = int(main_cfg.get("global_seed", 42))
    model_to_train = str(main_cfg.get("model_to_train", "mlp")).lower()
    run_mode = str(main_cfg.get("run_mode", "training")).lower()
    precision = str(main_cfg.get("precision", "float32")).lower()

    set_global_seed(global_seed)
    logger.info(f"Global seed set to {global_seed}")

    setup_torch_dtype(precision)
    logger.info(f"PyTorch default dtype set to {torch.get_default_dtype()}")

    dataset_cfg = loaded_configs["dataset_cfg"]
    dataset_name = loaded_configs["dataset_name"]
    logger.info(f"Dataset: {dataset_name} - {dataset_cfg.get('description', '')}")

    if model_to_train not in {"mlp", "gp", "mlpgp", "all"}:
        raise ValueError(f"Unknown model_to_train: {model_to_train}")

    viz_cfg = loaded_configs["viz_cfg"]

    logger.info(f"Model to train: {model_to_train.upper()}")
    logger.info(f"Run mode: {run_mode.upper()}")

    # ==================== Data Loading ====================
    logger.info("Loading and preprocessing data...")

    data_bundle = load_data_from_config(project_root, dataset_cfg, global_seed, precision)

    logger.debug(
        f"Data shapes: "
        f"X_train={data_bundle.X_train.shape}, "
        f"X_val={data_bundle.X_val.shape}, "
        f"X_test={data_bundle.X_test.shape}"
    )

    selected_models = ["mlp", "gp", "mlpgp"] if model_to_train == "all" else [model_to_train]
    trained_models: Dict[str, Any] = {}

    # ==================== Training Phase ====================
    if "training" in run_mode:
        for selected_model in selected_models:
            model_configs = load_run_configs(config_base_dir, selected_model, include_optimize=False)
            model_cfg = model_configs["model_cfg"]
            training_cfg = model_configs["training_cfg"]
            

            output_resolver = OutputPathResolver(
                dataset_name=dataset_name,
                run_type=RunType.SINGLE_RUN,
                model_name=selected_model,
                output_base_dir=output_base_dir,
            )

            run_dir = create_run_directory(output_resolver)
            logger.debug(f"Run directory created: {run_dir}")

            save_run_configs(
                output_resolver=output_resolver,
                selected_model=selected_model,
                main_cfg=main_cfg,
                dataset_cfg=dataset_cfg,
                model_cfg=model_cfg,
                training_cfg=training_cfg,
                viz_cfg=viz_cfg,
            )

            results = train_model(
                model_name=selected_model,
                data_bundle=data_bundle,
                dataset_cfg=dataset_cfg,
                model_cfg=model_cfg,
                training_cfg=training_cfg,
                output_resolver=output_resolver,
            )
            trained_models[selected_model] = results["model"]

            summary_csv_path = output_resolver.summary_csv
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

            from src.visualization import run_visualization
            from src.visualization.config import VisualizationConfig

            viz_config = VisualizationConfig.from_mapping(viz_cfg, output_dir=str(output_resolver.plots_dir))
            run_visualization(
                models={selected_model: results["model"]},
                data={
                    "X_train": data_bundle.X_train * data_bundle.input_scaler_std + data_bundle.input_scaler_mean,
                    "y_train": data_bundle.y_train * data_bundle.target_scaler_std + data_bundle.target_scaler_mean,
                    "X_val": data_bundle.X_val * data_bundle.input_scaler_std + data_bundle.input_scaler_mean,
                    "y_val": data_bundle.y_val * data_bundle.target_scaler_std + data_bundle.target_scaler_mean,
                    "X_test": data_bundle.X_test * data_bundle.input_scaler_std + data_bundle.input_scaler_mean,
                    "y_test": data_bundle.y_test * data_bundle.target_scaler_std + data_bundle.target_scaler_mean,
                    "feature_names": data_bundle.feature_names,
                    "output_name": ", ".join(data_bundle.target_names),
                    "input_scaler_mean": data_bundle.input_scaler_mean,
                    "input_scaler_std": data_bundle.input_scaler_std,
                    "target_scaler_mean": data_bundle.target_scaler_mean,
                    "target_scaler_std": data_bundle.target_scaler_std,
                },
                config=viz_config,
                mode="train",
            )

            logger.info(f"Training completed. Results saved to {run_dir}")

        if len(trained_models) == len(selected_models):
            from src.visualization import run_visualization
            from src.visualization.config import VisualizationConfig

            for selected_model in selected_models:
                table_resolver = OutputPathResolver(
                    dataset_name=dataset_name,
                    run_type=RunType.SINGLE_RUN,
                    model_name=selected_model,
                    output_base_dir=output_base_dir,
                )
                table_viz_config = VisualizationConfig.from_mapping(
                    viz_cfg,
                    output_dir=str(table_resolver.plots_dir),
                )
                table_viz_config.parity.enabled = False
                table_viz_config.surface.enabled = False
                table_viz_config.uncertainty_gp.enabled = False
                table_viz_config.table.enabled = True
                run_visualization(
                    models=trained_models,
                    data={
                        "X_train": data_bundle.X_train * data_bundle.input_scaler_std + data_bundle.input_scaler_mean,
                        "y_train": data_bundle.y_train * data_bundle.target_scaler_std + data_bundle.target_scaler_mean,
                        "X_val": data_bundle.X_val * data_bundle.input_scaler_std + data_bundle.input_scaler_mean,
                        "y_val": data_bundle.y_val * data_bundle.target_scaler_std + data_bundle.target_scaler_mean,
                        "X_test": data_bundle.X_test * data_bundle.input_scaler_std + data_bundle.input_scaler_mean,
                        "y_test": data_bundle.y_test * data_bundle.target_scaler_std + data_bundle.target_scaler_mean,
                        "feature_names": data_bundle.feature_names,
                        "output_name": ", ".join(data_bundle.target_names),
                        "input_scaler_mean": data_bundle.input_scaler_mean,
                        "input_scaler_std": data_bundle.input_scaler_std,
                        "target_scaler_mean": data_bundle.target_scaler_mean,
                        "target_scaler_std": data_bundle.target_scaler_std,
                    },
                    config=table_viz_config,
                    mode="train",
                )

    logger.info("=" * 60)
    logger.info("Pipeline Execution Completed Successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
