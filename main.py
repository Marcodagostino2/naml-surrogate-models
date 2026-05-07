"""Workflow dispatcher.

Three modes (selected via ``--mode`` or ``main_config.yaml: mode``):

- ``single_run`` — one training/evaluation pass per configured model
- ``optimization`` — Optuna hyperparameter search
- ``convergence`` — repeated split/seed analysis

Implementations live in ``src.pipeline``, ``src.entrypoints.optimize``, and
``src.entrypoints.convergence``; this script only routes.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import torch

from src.data import load_data_from_config
from src.pipeline import create_run_directory, save_run_configs, save_summary_row, train_model
from src.utils import OutputPathResolver, RunType, load_run_configs, setup_logger
from src.utils.entrypoints import load_and_initialize_run_configs

logger = setup_logger(__name__)

if TYPE_CHECKING:
    import optuna  # noqa: F401


def _build_viz_data(data_bundle) -> Dict[str, Any]:
    """Denormalize data and assemble the dict consumed by ``run_visualization``."""
    return {
        "X_train": data_bundle.X_train * data_bundle.input_scaler_std + data_bundle.input_scaler_mean,
        "y_train": data_bundle.y_train * data_bundle.target_scaler_std + data_bundle.target_scaler_mean,
        "X_val": data_bundle.X_val * data_bundle.input_scaler_std + data_bundle.input_scaler_mean,
        "y_val": data_bundle.y_val * data_bundle.target_scaler_std + data_bundle.target_scaler_mean,
        "X_test": data_bundle.X_test * data_bundle.input_scaler_std + data_bundle.input_scaler_mean,
        "y_test": data_bundle.y_test * data_bundle.target_scaler_std + data_bundle.target_scaler_mean,
        "feature_names": data_bundle.feature_names,
        "output_name": ", ".join(data_bundle.target_names),
        "input_display_names": data_bundle.input_display_names,
        "input_units": data_bundle.input_units,
        "output_display_name": data_bundle.output_display_name,
        "output_unit": data_bundle.output_unit,
        "input_scaler_mean": data_bundle.input_scaler_mean,
        "input_scaler_std": data_bundle.input_scaler_std,
        "target_scaler_mean": data_bundle.target_scaler_mean,
        "target_scaler_std": data_bundle.target_scaler_std,
    }


def _run_single_run(config_dir: Path, output_dir: Path) -> None:
    logger.info("Loading configurations...")
    loaded_configs = load_and_initialize_run_configs(config_dir, "mlp", include_optimize=False)

    main_cfg = loaded_configs["main_cfg"]
    dataset_cfg = loaded_configs["dataset_cfg"]
    dataset_name = loaded_configs["dataset_name"]
    viz_cfg = loaded_configs["viz_cfg"]

    model_to_train = str(main_cfg.get("model_to_train", "mlp")).lower()
    run_mode = str(main_cfg.get("run_mode", "training")).lower()
    global_seed = int(main_cfg.get("global_seed", 42))
    precision = str(main_cfg.get("precision", "float32")).lower()

    logger.info(f"Global seed set to {global_seed}")
    logger.info(f"PyTorch default dtype set to {torch.get_default_dtype()}")
    logger.info(f"Dataset: {dataset_name} - {dataset_cfg.get('description', '')}")
    logger.info(f"Model to train: {model_to_train.upper()}")
    logger.info(f"Run mode: {run_mode.upper()}")

    if model_to_train not in {"mlp", "gp", "mlpgp", "all"}:
        raise ValueError(f"Unknown model_to_train: {model_to_train}")

    logger.info("Loading and preprocessing data...")
    data_bundle = load_data_from_config(config_dir.parent, dataset_cfg, global_seed, precision)
    logger.debug(
        f"Data shapes: X_train={data_bundle.X_train.shape}, "
        f"X_val={data_bundle.X_val.shape}, X_test={data_bundle.X_test.shape}"
    )

    selected_models = ["mlp", "gp", "mlpgp"] if model_to_train == "all" else [model_to_train]
    trained_models: Dict[str, Any] = {}

    if "training" not in run_mode:
        logger.info("=" * 60)
        logger.info("Pipeline Execution Completed Successfully!")
        logger.info("=" * 60)
        return

    for selected_model in selected_models:
        model_configs = load_run_configs(config_dir, selected_model, include_optimize=False)
        model_cfg = model_configs["model_cfg"]
        training_cfg = model_configs["training_cfg"]

        output_resolver = OutputPathResolver(
            dataset_name=dataset_name,
            run_type=RunType.SINGLE_RUN,
            model_name=selected_model,
            output_base_dir=output_dir,
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

        save_summary_row(
            output_resolver.summary_csv,
            {
                "timestamp": datetime.now().isoformat(),
                "run_dir": str(run_dir),
                "test_r2": results["test_metrics"]["r2"],
                "test_rmse": results["test_metrics"]["rmse"],
                "test_mae": results["test_metrics"]["mae"],
                "test_mape": results["test_metrics"]["mape"],
                "val_r2": results["val_metrics"]["r2"],
                "val_rmse": results["val_metrics"]["rmse"],
            },
        )

        from src.visualization import run_visualization
        from src.visualization.config import VisualizationConfig

        viz_config = VisualizationConfig.from_mapping(viz_cfg, output_dir=str(output_resolver.plots_dir))
        run_visualization(
            models={selected_model: results["model"]},
            data=_build_viz_data(data_bundle),
            config=viz_config,
            mode="train",
        )
        logger.info(f"Training completed. Results saved to {run_dir}")

    if len(trained_models) == len(selected_models):
        # When all models trained, render a comparison table per model output dir.
        from src.visualization import run_visualization
        from src.visualization.config import VisualizationConfig

        for selected_model in selected_models:
            table_resolver = OutputPathResolver(
                dataset_name=dataset_name,
                run_type=RunType.SINGLE_RUN,
                model_name=selected_model,
                output_base_dir=output_dir,
            )
            table_viz_config = VisualizationConfig.from_mapping(
                viz_cfg, output_dir=str(table_resolver.plots_dir),
            )
            table_viz_config.parity.enabled = False
            table_viz_config.surface.enabled = False
            table_viz_config.uncertainty_gp.enabled = False
            table_viz_config.table.enabled = True
            run_visualization(
                models=trained_models,
                data=_build_viz_data(data_bundle),
                config=table_viz_config,
                mode="train",
            )

    logger.info("=" * 60)
    logger.info("Pipeline Execution Completed Successfully!")
    logger.info("=" * 60)


def _run_optimization(
    config_dir: Path,
    output_dir: Path,
    model_override: str | None = None,
    n_trials_override: int | None = None,
    timeout_override: int | None = None,
) -> None:
    from src.entrypoints.optimize import run_optimization as run_optimization_workflow

    loaded_configs = load_run_configs(config_dir, "mlp", include_optimize=False)
    main_cfg = loaded_configs["main_cfg"]
    model_to_train = str(model_override or main_cfg.get("model_to_train", "mlp")).lower()

    if model_to_train == "all":
        for selected_model in ["mlp", "gp", "mlpgp"]:
            run_optimization_workflow(config_dir, selected_model, output_dir, n_trials_override, timeout_override)
    else:
        run_optimization_workflow(config_dir, model_to_train, output_dir, n_trials_override, timeout_override)


def _run_convergence(config_path: Path) -> None:
    from src.entrypoints.convergence import run_analysis
    run_analysis(config_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single entry point for surrogate-model workflows.")
    parser.add_argument(
        "--mode", type=str, default=None,
        choices=("single_run", "optimization", "convergence"),
        help="Workflow mode. Defaults to config/main_config.yaml.",
    )
    parser.add_argument("--config-dir", type=str, default="config", help="Path to the config directory.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Base output directory.")
    parser.add_argument(
        "--model", type=str, default=None,
        choices=["mlp", "gp", "mlpgp", "all"],
        help="Override the optimization model selection from config/main_config.yaml.",
    )
    parser.add_argument("--n-trials", type=int, default=None, help="Override the number of optimization trials.")
    parser.add_argument("--timeout", type=int, default=None, help="Override the optimization timeout in seconds.")
    parser.add_argument(
        "--config-path", type=str, default="config/convergence/convergence_config.yaml",
        help="Path to the convergence-analysis config file.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    project_root = Path(__file__).resolve().parent
    config_dir = project_root / args.config_dir
    output_dir = project_root / args.output_dir
    workflow_cfg = load_run_configs(config_dir, "mlp", include_optimize=False)["main_cfg"]
    workflow_mode = str(args.mode or workflow_cfg.get("mode", "single_run")).lower()

    if workflow_mode == "single_run":
        _run_single_run(config_dir, output_dir)
    elif workflow_mode == "optimization":
        _run_optimization(config_dir, output_dir, args.model, args.n_trials, args.timeout)
    elif workflow_mode == "convergence":
        _run_convergence(project_root / args.config_path)
    else:
        raise ValueError(f"Unknown workflow mode: {workflow_mode}")


if __name__ == "__main__":
    main()
