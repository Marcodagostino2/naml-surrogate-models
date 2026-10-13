"""Workflow dispatcher.

Two modes (selected via ``--mode`` or ``main_config.yaml: mode``):

- ``single_run`` — one training/evaluation pass per configured model
- ``convergence`` — repeated split/seed analysis (also hosts HPO via
  ``convergence_config.yaml: mode: optimization``)

Implementations live in ``src.pipeline`` and ``src.entrypoints.convergence``;
this script only routes.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import torch

from src.data import load_data_from_config
from src.pipeline import (
    create_run_directory,
    save_run_configs,
    save_summary_row,
    test_model_from_checkpoint,
    train_model,
)
from src.utils import OutputPathResolver, RunType, load_run_configs, load_yaml, make_run_id, setup_logger
from src.utils.entrypoints import load_and_initialize_run_configs

logger = setup_logger(__name__)

if TYPE_CHECKING:
    import optuna  # noqa: F401


CANONICAL_MODELS = ("mlp", "gp", "mlpgp_ns", "mlpgp_dkl")


def parse_models_list(value: Any) -> list[str]:
    """Accept a list, a comma-separated string, or the ``"all"`` alias; return the canonical list."""
    if value is None:
        return list(CANONICAL_MODELS)
    if isinstance(value, str):
        items = [item.strip().lower() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        items = [str(item).strip().lower() for item in value if str(item).strip()]
    else:
        raise ValueError(f"models_to_train must be a list or string, got {type(value).__name__}")
    if not items:
        return list(CANONICAL_MODELS)
    if "all" in items:
        return list(CANONICAL_MODELS)
    unknown = [name for name in items if name not in CANONICAL_MODELS]
    if unknown:
        raise ValueError(
            f"Unknown model(s) in models_to_train: {unknown}. "
            f"Valid entries: {list(CANONICAL_MODELS)} or 'all'."
        )
    return items


def _check_legacy_model_key(main_cfg: Dict[str, Any]) -> None:
    """Reject the legacy ``model_to_train`` scalar with a migration hint."""
    if "model_to_train" in main_cfg:
        raise ValueError(
            "`model_to_train` was renamed to `models_to_train` (list). Update "
            "config/main_config.yaml — e.g. `models_to_train: [\"gp\"]` or "
            "`models_to_train: [\"all\"]`."
        )


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


def _resolve_checkpoint_source(
    output_dir: Path,
    dataset_name: str,
    model_name: str,
    run_id: str | None,
) -> tuple[Path, str]:
    resolver = OutputPathResolver(
        dataset_name=dataset_name,
        run_type=RunType.SINGLE_RUN,
        model_name=model_name,
        output_base_dir=output_dir,
    )

    if run_id is not None:
        run_checkpoint = resolver.runs_root / run_id / "checkpoint.pt"
        if not run_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found for run_id='{run_id}' at {run_checkpoint}")
        return run_checkpoint, run_id

    if resolver.runs_root.exists():
        run_dirs = sorted([path for path in resolver.runs_root.iterdir() if path.is_dir()])
        for run_dir in reversed(run_dirs):
            candidate = run_dir / "checkpoint.pt"
            if candidate.exists():
                return candidate, run_dir.name

    legacy_checkpoint = resolver.model_root / "checkpoint.pt"
    if legacy_checkpoint.exists():
        return legacy_checkpoint, "legacy"

    raise FileNotFoundError(
        f"No checkpoint found for model='{model_name}' under {resolver.model_root}"
    )


def _run_single_run(
    config_dir: Path,
    output_dir: Path,
    run_id_override: str | None = None,
    resume_run_id_override: str | None = None,
) -> None:
    logger.info("Loading configurations...")
    loaded_configs = load_and_initialize_run_configs(config_dir, "mlp", include_optimize=False)

    main_cfg = loaded_configs["main_cfg"]
    dataset_cfg = loaded_configs["dataset_cfg"]
    dataset_name = loaded_configs["dataset_name"]
    viz_cfg = loaded_configs["viz_cfg"]

    _check_legacy_model_key(main_cfg)
    selected_models = parse_models_list(main_cfg.get("models_to_train"))
    run_mode = str(main_cfg.get("run_mode", "training")).lower()
    run_id_cfg = str(run_id_override or main_cfg.get("run_id", "")).strip() or None
    resume_run_id = str(resume_run_id_override or main_cfg.get("resume_run_id", "")).strip() or None
    global_seed = int(main_cfg.get("global_seed", 42))
    precision = str(main_cfg.get("precision", "float32")).lower()

    logger.info(f"Global seed set to {global_seed}")
    logger.info(f"PyTorch default dtype set to {torch.get_default_dtype()}")
    logger.info(f"Dataset: {dataset_name} - {dataset_cfg.get('description', '')}")
    logger.info(f"Models to train: {', '.join(model.upper() for model in selected_models)}")
    logger.info(f"Run mode: {run_mode.upper()}")

    normalized_mode = run_mode.replace("+", "_")
    training_enabled = "training" in normalized_mode
    testing_enabled = ("test" in normalized_mode) or ("testing" in normalized_mode)
    if not training_enabled and not testing_enabled:
        raise ValueError("run_mode must include 'training', 'test', or 'training+test'.")

    logger.info("Loading and preprocessing data...")
    data_bundle = load_data_from_config(config_dir.parent, dataset_cfg, global_seed, precision)
    logger.debug(
        f"Data shapes: X_train={data_bundle.X_train.shape}, "
        f"X_val={data_bundle.X_val.shape}, X_test={data_bundle.X_test.shape}"
    )
    trained_models: Dict[str, Any] = {}

    train_run_id = run_id_cfg or make_run_id("single")
    test_run_id = make_run_id("test")
    if training_enabled:
        logger.info(f"Training run_id: {train_run_id}")
    if testing_enabled and not training_enabled:
        logger.info(f"Testing run_id: {test_run_id}")

    total_models = len(selected_models)
    for model_index, selected_model in enumerate(selected_models, start=1):
        stage_tag = f"[Stage {model_index}/{total_models} | model={selected_model}]"
        logger.info(f"{stage_tag} starting")
        model_configs = load_run_configs(config_dir, selected_model, include_optimize=False)
        model_cfg = model_configs["model_cfg"]
        training_cfg = model_configs["training_cfg"]

        if training_enabled:
            output_resolver = OutputPathResolver(
                dataset_name=dataset_name,
                run_type=RunType.SINGLE_RUN,
                model_name=selected_model,
                output_base_dir=output_dir,
                run_id=train_run_id,
            )

            run_dir = create_run_directory(output_resolver)
            logger.debug(f"Run directory created: {run_dir}")
            contextual_training_cfg = {
                **training_cfg,
                "run_mode": "single_run",
                "run_phase": "training",
                "run_model": selected_model,
            }
            save_run_configs(
                output_resolver=output_resolver,
                selected_model=selected_model,
                main_cfg=main_cfg,
                dataset_cfg=dataset_cfg,
                model_cfg=model_cfg,
                training_cfg=contextual_training_cfg,
                viz_cfg=viz_cfg,
            )

            try:
                results = train_model(
                    model_name=selected_model,
                    data_bundle=data_bundle,
                    dataset_cfg=dataset_cfg,
                    model_cfg=model_cfg,
                    training_cfg=contextual_training_cfg,
                    output_resolver=output_resolver,
                )
            except (KeyboardInterrupt, MemoryError):
                raise
            except Exception:
                logger.exception(f"{stage_tag} training failed; continuing with next model")
                continue
            trained_models[selected_model] = results["model"]

            timings = results.get("timings", {}) or {}
            save_summary_row(
                output_resolver.summary_csv,
                {
                    "timestamp": datetime.now().isoformat(),
                    "run_id": train_run_id,
                    "run_dir": str(run_dir),
                    "test_r2": results["test_metrics"]["r2"],
                    "test_rmse": results["test_metrics"]["rmse"],
                    "test_mae": results["test_metrics"]["mae"],
                    "test_mape": results["test_metrics"]["mape"],
                    "val_r2": results["val_metrics"]["r2"],
                    "val_rmse": results["val_metrics"]["rmse"],
                    "training_seconds": timings.get("training_seconds"),
                    "inference_total_seconds": timings.get("inference_total_seconds"),
                    "inference_per_sample_seconds": timings.get("inference_per_sample_seconds"),
                },
            )

            from src.visualization import run_visualization
            from src.visualization.config import VisualizationConfig

            viz_config = VisualizationConfig.from_mapping(viz_cfg, output_dir=str(output_resolver.plots_dir))
            # Per-model viz only — the cross-model metrics table is rendered once
            # at the end of the batch (table-only run below).
            viz_config.table.enabled = False
            run_visualization(
                "single_run",
                models={selected_model: results["model"]},
                data=_build_viz_data(data_bundle),
                config=viz_config,
                dataset_name=dataset_name,
            )
            logger.info(f"Training completed. Results saved to {run_dir}")

        if testing_enabled and not training_enabled:
            checkpoint_path, source_run_id = _resolve_checkpoint_source(
                output_dir=output_dir,
                dataset_name=dataset_name,
                model_name=selected_model,
                run_id=resume_run_id,
            )

            source_model_cfg_path = checkpoint_path.parent / "config" / "model" / f"{selected_model}_config.yaml"
            source_training_cfg_path = checkpoint_path.parent / "config" / "training" / f"training_{selected_model}_config.yaml"
            source_model_cfg = load_yaml(source_model_cfg_path) if source_model_cfg_path.exists() else model_cfg
            source_training_cfg = load_yaml(source_training_cfg_path) if source_training_cfg_path.exists() else training_cfg

            test_resolver = OutputPathResolver(
                dataset_name=dataset_name,
                run_type=RunType.TESTING,
                model_name=selected_model,
                output_base_dir=output_dir,
                run_id=test_run_id,
            )
            create_run_directory(test_resolver)
            save_run_configs(
                output_resolver=test_resolver,
                selected_model=selected_model,
                main_cfg=main_cfg,
                dataset_cfg=dataset_cfg,
                model_cfg=source_model_cfg,
                training_cfg=source_training_cfg,
                viz_cfg=viz_cfg,
            )

            try:
                test_results = test_model_from_checkpoint(
                    model_name=selected_model,
                    data_bundle=data_bundle,
                    dataset_cfg=dataset_cfg,
                    model_cfg=source_model_cfg,
                    checkpoint_path=checkpoint_path,
                    output_resolver=test_resolver,
                )
            except (KeyboardInterrupt, MemoryError):
                raise
            except Exception:
                logger.exception(f"{stage_tag} testing failed; continuing with next model")
                continue

            from src.visualization import run_visualization
            from src.visualization.config import VisualizationConfig

            viz_config = VisualizationConfig.from_mapping(viz_cfg, output_dir=str(test_resolver.plots_dir))
            viz_config.table.enabled = False
            run_visualization(
                "single_run",
                models={selected_model: test_results["model"]},
                data=_build_viz_data(data_bundle),
                config=viz_config,
                dataset_name=dataset_name,
            )

            timings = test_results.get("timings", {}) or {}
            save_summary_row(
                test_resolver.summary_csv,
                {
                    "timestamp": datetime.now().isoformat(),
                    "run_id": test_run_id,
                    "source_run_id": source_run_id,
                    "run_dir": str(test_resolver.root),
                    "source_checkpoint": str(checkpoint_path),
                    "test_r2": test_results["test_metrics"]["r2"],
                    "test_rmse": test_results["test_metrics"]["rmse"],
                    "test_mae": test_results["test_metrics"]["mae"],
                    "test_mape": test_results["test_metrics"]["mape"],
                    "val_r2": test_results["val_metrics"]["r2"],
                    "val_rmse": test_results["val_metrics"]["rmse"],
                    "training_seconds": timings.get("training_seconds"),
                    "inference_total_seconds": timings.get("inference_total_seconds"),
                    "inference_per_sample_seconds": timings.get("inference_per_sample_seconds"),
                },
            )
            logger.info(
                f"Testing completed for {selected_model.upper()} from run '{source_run_id}'. "
                f"Results saved to {test_resolver.root}"
            )

    if training_enabled and trained_models:
        # Render the cross-model metrics table ONCE; save it under the first
        # trained model's plots/ dir for archival.
        from src.visualization import run_visualization
        from src.visualization.config import VisualizationConfig

        first_model = next(iter(trained_models))
        table_resolver = OutputPathResolver(
            dataset_name=dataset_name,
            run_type=RunType.SINGLE_RUN,
            model_name=first_model,
            output_base_dir=output_dir,
            run_id=train_run_id,
        )
        table_viz_config = VisualizationConfig.from_mapping(
            viz_cfg, output_dir=str(table_resolver.plots_dir),
        )
        table_viz_config.parity.enabled = False
        table_viz_config.surface.enabled = False
        table_viz_config.uncertainty_gp.enabled = False
        table_viz_config.hyddb.enabled = False
        table_viz_config.table.enabled = True
        run_visualization(
            "single_run",
            models=trained_models,
            data=_build_viz_data(data_bundle),
            config=table_viz_config,
            dataset_name=dataset_name,
        )

    logger.info("=" * 60)
    logger.info("Pipeline execution completed")
    logger.info("=" * 60)


def _run_convergence(config_path: Path) -> None:
    from src.entrypoints.convergence import run_analysis
    run_analysis(config_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single entry point for surrogate-model workflows.")
    parser.add_argument(
        "--mode", type=str, default=None,
        choices=("single_run", "convergence"),
        help="Workflow mode. Defaults to config/main_config.yaml. HPO is invoked "
             "via `--mode convergence` with `convergence_config.mode: optimization`.",
    )
    parser.add_argument("--config-dir", type=str, default="config", help="Path to the config directory.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Base output directory.")
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated model list (e.g. 'mlp,gp,mlpgp_ns'); 'all' expands to "
             "mlp,gp,mlpgp_ns,mlpgp_dkl. Overrides main_config.yaml: models_to_train.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier for output folder naming (otherwise timestamp-generated).",
    )
    parser.add_argument(
        "--resume-run-id",
        type=str,
        default=None,
        help="For single_run test mode: explicit source run_id to load checkpoint from.",
    )
    parser.add_argument(
        "--config-path", type=str, default="config/convergence/convergence_config.yaml",
        help="Path to the convergence-analysis config file.",
    )
    return parser


def main() -> None:
    try:
        args = _build_parser().parse_args()
        project_root = Path(__file__).resolve().parent
        config_dir = project_root / args.config_dir
        output_dir = project_root / args.output_dir
        workflow_cfg = load_run_configs(config_dir, "mlp", include_optimize=False)["main_cfg"]
        workflow_mode = str(args.mode or workflow_cfg.get("mode", "single_run")).lower()

        if workflow_mode == "single_run":
            _run_single_run(config_dir, output_dir, args.run_id, args.resume_run_id)
        elif workflow_mode == "convergence":
            _run_convergence(project_root / args.config_path)
        else:
            raise ValueError(f"Unknown workflow mode: {workflow_mode}")
    except (KeyboardInterrupt, MemoryError):
        raise
    except Exception:
        logger.exception("Unhandled error in main workflow; exiting non-zero")
        sys.exit(1)


if __name__ == "__main__":
    main()
