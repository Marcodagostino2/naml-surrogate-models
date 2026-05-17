"""Repeated split/seed analysis: per-(split, model, seed) runs and aggregated CSVs.

Plotting is delegated to ``src.visualization.suites.convergence`` via the
top-level ``run_visualization`` dispatcher.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from src.data import DataBundle, load_and_preprocess_data
from src.optimization_support import (
    _make_logging_callback,
    build_pruner,
    build_sampler,
    make_objective,
    save_optimization_results,
)
from src.pipeline import train_model
from src.utils import OutputPathResolver, RunType, load_run_configs, load_yaml, make_run_id, set_global_seed, setup_logger, setup_torch_dtype
from src.visualization import VisualizationConfig, run_visualization

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_SEQUENCE: Sequence[str] = ("mlp", "gp", "mlpgp_ns", "mlpgp_dkl")
ANALYSIS_ROOT_NAME = "convergence_analysis"

logger = setup_logger(__name__)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _run_count_for_model(config: Dict[str, Any], key: str, model_name: str, default: int = 1) -> int:
    raw = config.get(key, {})
    if isinstance(raw, dict):
        value = raw.get(model_name, raw.get("default", default))
    else:
        value = raw
    try:
        return max(1, int(value))
    except Exception:
        return default


def _normalize_split_value(value: float) -> float:
    return value / 100.0 if value > 1.0 else value


def _build_split_grid(raw_splits: Iterable[float]) -> List[float]:
    split_grid: List[float] = []
    for raw in raw_splits:
        train_ratio = _normalize_split_value(float(raw))
        if not 0.0 < train_ratio < 1.0:
            raise ValueError(f"Invalid train split {raw}. Must be in (0,1) or (0,100).")
        split_grid.append(round(train_ratio, 6))
    if not split_grid:
        raise ValueError("split_percentages cannot be empty.")
    return sorted(set(split_grid))


def _build_validation_ratios(train_plus_val_ratio: float, validation_train_ratio: float) -> tuple[float, float]:
    val_ratio = float(train_plus_val_ratio) * float(validation_train_ratio)
    train_ratio = train_plus_val_ratio - float(val_ratio) 
    test_ratio = 1.0 - float(train_plus_val_ratio)
    if val_ratio <= 0.0 or test_ratio <= 0.0:
        raise ValueError(
            f"Invalid ratios for train={train_ratio:.6f} and validation_train_ratio={validation_train_ratio:.6f}."
        )
    return train_ratio, val_ratio, test_ratio


def _split_tag(train_ratio: float) -> str:
    return f"split_{train_ratio:.4f}".replace(".", "p")


def _get_models_to_train(main_cfg: Dict[str, Any], all_models: Sequence[str] = MODEL_SEQUENCE) -> Sequence[str]:
    """Filter models to train based on main_cfg model_to_train setting.
    
    If model_to_train is 'all', return all models. Otherwise return just the specified model.
    """
    model_to_train = str(main_cfg.get("model_to_train", "all")).lower()
    if model_to_train == "all":
        return all_models
    if model_to_train in all_models:
        return (model_to_train,)
    logger.warning(f"Invalid model_to_train '{model_to_train}'. Using all models instead.")
    return all_models


def _load_contexts(config_dir: Path, mode: str, model_sequence: Sequence[str] = MODEL_SEQUENCE) -> Dict[str, Dict[str, Any]]:
    contexts: Dict[str, Dict[str, Any]] = {}
    include_optimize = mode == "optimization"
    for model_name in model_sequence:
        contexts[model_name] = load_run_configs(config_dir, model_name, include_optimize=include_optimize)
    return contexts


def _prepare_data_bundle(dataset_cfg: Dict[str, Any], train_plus_val_ratio: float, validation_train_ratio: float, seed: int) -> DataBundle:
    train_ratio, val_ratio, test_ratio = _build_validation_ratios(train_plus_val_ratio, validation_train_ratio)
    # Mirror the global torch default dtype set by ``setup_torch_dtype`` so
    # NumPy data and model params share precision (otherwise GP-family forwards
    # raise "expected Float/Double" mismatch errors).
    default_type = "float64" if torch.get_default_dtype() is torch.float64 else "float32"
    return load_and_preprocess_data(
        csv_path=PROJECT_ROOT / dataset_cfg["csv_path"],
        input_columns=dataset_cfg["input_columns"],
        target_columns=dataset_cfg["target_columns"],
        dropna=bool(dataset_cfg.get("dropna", True)),
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", True)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", True)),
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
        random_seed=int(seed),
        default_type=default_type,
    )


def _analysis_output_resolver(
    dataset_name: str,
    model_name: str,
    analysis_run_id: str,
    *,
    split_tag: Optional[str] = None,
    sub_tag: Optional[str] = None,
) -> OutputPathResolver:
    """Return output resolver for a single training under the convergence tree.

    ``split_tag`` and ``sub_tag`` (e.g. ``"hpo"``, ``"run_0"``) are inserted
    into the path so that artifacts from different splits / runs / phases
    don't overwrite each other. Final layout:
    ``outputs/<dataset>/convergence_analysis/<split>/<sub>/<run_type>/<model>/``.
    """
    parts = [dataset_name, ANALYSIS_ROOT_NAME]
    if split_tag:
        parts.append(split_tag)
    if sub_tag:
        parts.append(sub_tag)
    return OutputPathResolver(
        dataset_name="/".join(parts),
        run_type=RunType.CONVERGENCE_ANALYSIS,
        model_name=model_name,
        output_base_dir=PROJECT_ROOT / "outputs",
        run_id=analysis_run_id,
    )


def _extract_history_values(model_name: str, history: Dict[str, Any]) -> tuple[str, List[float]]:
    if model_name == "mlp":
        values = history.get("train_losses", [])
        label = "train_loss"
    else:
        values = history.get("mll_history", history.get("train_losses", []))
        label = "mll"
    return label, [float(value) for value in _as_list(values)]


def _train_best_model(
    *,
    dataset_name: str,
    model_name: str,
    data_bundle: DataBundle,
    dataset_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    analysis_run_id: str,
    split_tag: Optional[str] = None,
    sub_tag: Optional[str] = None,
) -> Dict[str, Any]:
    output_resolver = _analysis_output_resolver(
        dataset_name, model_name, analysis_run_id, split_tag=split_tag, sub_tag=sub_tag,
    )
    output_resolver.root.mkdir(parents=True, exist_ok=True)
    contextual_training_cfg = deepcopy(training_cfg)
    contextual_training_cfg["run_mode"] = "convergence"
    contextual_training_cfg["run_phase"] = "analysis"
    contextual_training_cfg["run_model"] = model_name
    if split_tag is not None:
        contextual_training_cfg["run_split_tag"] = split_tag
    if sub_tag is not None:
        contextual_training_cfg["run_index"] = sub_tag

    results = train_model(
        model_name=model_name,
        data_bundle=data_bundle,
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        training_cfg=contextual_training_cfg,
        output_resolver=output_resolver,
    )
    history = results.get("history", {}) or {}
    history_label, history_values = _extract_history_values(model_name, history)

    timings = results.get("timings", {}) or {}
    return {
        "model": model_name,
        "history_label": history_label,
        "history_values": history_values,
        "elapsed_time": float(history.get("elapsed_time", np.nan)),
        "training_seconds": float(timings.get("training_seconds", np.nan)),
        "inference_total_seconds": float(timings.get("inference_total_seconds", np.nan)),
        "inference_per_sample_seconds": float(timings.get("inference_per_sample_seconds", np.nan)),
        "train_metrics": {key: float(value) for key, value in results.get("train_metrics", {}).items()},
        "val_metrics": {key: float(value) for key, value in results.get("val_metrics", {}).items()},
        "test_metrics": {key: float(value) for key, value in results.get("test_metrics", {}).items()},
        "output_dir": str(output_resolver.root),
    }


def _apply_optimization_overrides(opt_cfg: Dict[str, Any], analysis_cfg: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(opt_cfg)
    overrides = analysis_cfg.get("optimization_overrides", {}) or {}
    study_overrides = overrides.get("study", overrides)
    if isinstance(study_overrides, dict):
        merged_study = merged.setdefault("study", {})
        for key, value in study_overrides.items():
            if key != "training_overrides":
                merged_study[key] = value
    training_overrides = overrides.get("training_overrides")
    if isinstance(training_overrides, dict):
        merged_training = merged.setdefault("training_overrides", {})
        merged_training.update(training_overrides)
    return merged


def _run_analysis_mode(
    *,
    analysis_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    dataset_name: str,
    model_cfg_map: Dict[str, Dict[str, Any]],
    training_cfg_map: Dict[str, Dict[str, Any]],
    analysis_run_id: str,
) -> tuple[pd.DataFrame, List[Dict[str, Any]], Path]:
    split_grid = _build_split_grid(analysis_cfg.get("split_percentages", [0.7]))
    validation_train_ratio = float(analysis_cfg.get("validation_train_ratio", 0.21428571428571427))
    global_seed = int(analysis_cfg.get("global_seed", 31))

    metrics_rows: List[Dict[str, Any]] = []
    histories: List[Dict[str, Any]] = []
    analysis_root = PROJECT_ROOT / "outputs" / dataset_name / ANALYSIS_ROOT_NAME / "runs" / analysis_run_id
    analysis_root.mkdir(parents=True, exist_ok=True)

    for split_index, train_plus_val_ratio in enumerate(split_grid):
        split_tag = _split_tag(round(train_plus_val_ratio, 6))
        for model_name in model_cfg_map.keys():
            n_runs = _run_count_for_model(analysis_cfg, "runs_per_model", model_name, default=1)
            for run_index in range(n_runs):
                seed = global_seed + split_index * 1000 + run_index * 10 + MODEL_SEQUENCE.index(model_name)
                set_global_seed(seed)
                data_bundle = _prepare_data_bundle(dataset_cfg, train_plus_val_ratio, validation_train_ratio, seed)
                model_cfg = deepcopy(model_cfg_map[model_name])
                training_cfg = deepcopy(training_cfg_map[model_name])

                result = _train_best_model(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    data_bundle=data_bundle,
                    dataset_cfg=dataset_cfg,
                    model_cfg=model_cfg,
                    training_cfg=training_cfg,
                    analysis_run_id=analysis_run_id,
                    split_tag=split_tag,
                    sub_tag=f"run_{run_index}",
                )

                train_ratio_actual = float(data_bundle.X_train.shape[0]) / float(data_bundle.X_train.shape[0] + data_bundle.X_val.shape[0] + data_bundle.X_test.shape[0])
                val_ratio_actual = float(data_bundle.X_val.shape[0]) / float(data_bundle.X_train.shape[0] + data_bundle.X_val.shape[0] + data_bundle.X_test.shape[0])
                test_ratio_actual = float(data_bundle.X_test.shape[0]) / float(data_bundle.X_train.shape[0] + data_bundle.X_val.shape[0] + data_bundle.X_test.shape[0])

                metrics_rows.append(
                    {
                        "mode": "analysis",
                        "model": model_name,
                        "train_split": train_ratio_actual,
                        "validation_ratio": val_ratio_actual,
                        "test_ratio": test_ratio_actual,
                        "n_train": int(data_bundle.X_train.shape[0]),
                        "n_val": int(data_bundle.X_val.shape[0]),
                        "n_test": int(data_bundle.X_test.shape[0]),
                        "seed": seed,
                        "run_index": run_index,
                        "elapsed_time": result["elapsed_time"],
                        "training_seconds": result["training_seconds"],
                        "inference_total_seconds": result["inference_total_seconds"],
                        "inference_per_sample_seconds": result["inference_per_sample_seconds"],
                        "r2": result["test_metrics"].get("r2"),
                        "rmse": result["test_metrics"].get("rmse"),
                        "mae": result["test_metrics"].get("mae"),
                        "mape": result["test_metrics"].get("mape"),
                        "train_r2": result["train_metrics"].get("r2"),
                        "val_r2": result["val_metrics"].get("r2"),
                        "output_dir": result["output_dir"],
                    }
                )
                histories.append(
                    {
                        "model": model_name,
                        "train_split": train_ratio_actual,
                        "run_index": run_index,
                        "history_label": result["history_label"],
                        "history_values": result["history_values"],
                    }
                )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.sort_values(["model", "train_split", "run_index"]).reset_index(drop=True)
    return metrics_df, histories, analysis_root


def _run_optimization_mode(
    *,
    analysis_cfg: Dict[str, Any],
    main_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    dataset_name: str,
    model_cfg_map: Dict[str, Dict[str, Any]],
    training_cfg_map: Dict[str, Dict[str, Any]],
    opt_cfg_map: Dict[str, Dict[str, Any]],
    analysis_run_id: str,
) -> tuple[pd.DataFrame, List[Dict[str, Any]], Path]:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError("optuna is required for convergence optimization mode.") from exc

    split_grid = _build_split_grid(analysis_cfg.get("split_percentages", [0.7]))
    validation_train_ratio = float(analysis_cfg.get("validation_train_ratio", 0.21428571428571427))
    global_seed = int(analysis_cfg.get("global_seed", 31))

    metrics_rows: List[Dict[str, Any]] = []
    histories: List[Dict[str, Any]] = []
    analysis_root = PROJECT_ROOT / "outputs" / dataset_name / ANALYSIS_ROOT_NAME / "runs" / analysis_run_id
    analysis_root.mkdir(parents=True, exist_ok=True)

    for split_index, train_ratio in enumerate(split_grid):
        split_tag = _split_tag(train_ratio)
        seed = global_seed + split_index * 1000
        set_global_seed(seed)
        data_bundle = _prepare_data_bundle(dataset_cfg, train_ratio, validation_train_ratio, seed)
        for model_name in model_cfg_map.keys():
            model_cfg = deepcopy(model_cfg_map[model_name])
            training_cfg = deepcopy(training_cfg_map[model_name])
            opt_cfg = _apply_optimization_overrides(opt_cfg_map[model_name], analysis_cfg)

            output_resolver = _analysis_output_resolver(
                dataset_name=dataset_name,
                model_name=model_name,
                analysis_run_id=analysis_run_id,
                split_tag=split_tag,
                sub_tag="hpo",
            )
            output_resolver.root.mkdir(parents=True, exist_ok=True)

            sampler = build_sampler(opt_cfg["study"])
            pruner = build_pruner(opt_cfg["study"])
            study = optuna.create_study(
                direction=opt_cfg["study"]["direction"],
                sampler=sampler,
                pruner=pruner,
            )

            from src.utils import resolve_device

            device = resolve_device(training_cfg.get("device", main_cfg.get("device", "auto")))
            objective = make_objective(
                model_name=model_name,
                data_bundle=data_bundle,
                base_model_cfg=model_cfg,
                base_training_cfg=training_cfg,
                dataset_cfg=dataset_cfg,
                opt_cfg=opt_cfg,
                device=device,
                seed=seed,
                n_seeds=_run_count_for_model(opt_cfg["study"], "n_seeds", model_name, default=3),
            )

            n_trials = _run_count_for_model(opt_cfg["study"], "n_trials", model_name, default=1)
            timeout = opt_cfg["study"].get("timeout")
            metric_name = str(opt_cfg.get("study", {}).get("metric", "objective"))
            callback_model_name = f"{model_name}|{split_tag}"
            log_every = int(opt_cfg.get("study", {}).get("log_every_n_trials", 5))
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=False,
                callbacks=[_make_logging_callback(callback_model_name, n_trials, metric_name, log_every)],
            )

            save_optimization_results(
                study=study,
                output_resolver=output_resolver,
                opt_cfg=opt_cfg,
                base_model_cfg=model_cfg,
                base_training_cfg=training_cfg,
                model_name=model_name,
            )
            opt_viz_config = VisualizationConfig.from_mapping(
                {}, output_dir=str(output_resolver.plots_dir),
            )
            run_visualization("optimization", config=opt_viz_config, study=study)

            best_model_cfg = load_yaml(output_resolver.config_dir / "model" / f"{model_name}_config.yaml")
            best_training_cfg = load_yaml(output_resolver.config_dir / "training" / f"training_{model_name}_config.yaml")
            optimized_kernel_cfg = deepcopy(best_model_cfg.get("kernel")) if isinstance(best_model_cfg.get("kernel"), dict) else None

            n_retrain_runs = _run_count_for_model(
                analysis_cfg,
                "optimization_runs_per_model",
                model_name,
                default=_run_count_for_model(analysis_cfg, "runs_per_model", model_name, default=1),
            )
            for run_index in range(n_retrain_runs):
                retrain_seed = seed + run_index
                set_global_seed(retrain_seed)
                retrain_data_bundle = _prepare_data_bundle(dataset_cfg, train_ratio, validation_train_ratio, retrain_seed)
                retrain_model_cfg = deepcopy(best_model_cfg)
                if optimized_kernel_cfg is not None:
                    retrain_model_cfg["kernel"] = deepcopy(optimized_kernel_cfg)
                retrain_training_cfg = deepcopy(best_training_cfg)

                best_result = _train_best_model(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    data_bundle=retrain_data_bundle,
                    dataset_cfg=dataset_cfg,
                    model_cfg=retrain_model_cfg,
                    training_cfg=retrain_training_cfg,
                    analysis_run_id=analysis_run_id,
                    split_tag=split_tag,
                    sub_tag=f"run_{run_index}",
                )

                histories.append(
                    {
                        "model": model_name,
                        "train_split": train_ratio,
                        "run_index": run_index,
                        "history_label": best_result["history_label"],
                        "history_values": best_result["history_values"],
                    }
                )

                metrics_rows.append(
                    {
                        "mode": "optimization",
                        "model": model_name,
                        "train_split": train_ratio,
                        "validation_ratio": float(data_bundle.X_val.shape[0]) / float(data_bundle.X_train.shape[0] + data_bundle.X_val.shape[0] + data_bundle.X_test.shape[0]),
                        "test_ratio": float(data_bundle.X_test.shape[0]) / float(data_bundle.X_train.shape[0] + data_bundle.X_val.shape[0] + data_bundle.X_test.shape[0]),
                        "n_train": int(data_bundle.X_train.shape[0]),
                        "n_val": int(data_bundle.X_val.shape[0]),
                        "n_test": int(data_bundle.X_test.shape[0]),
                        "seed": retrain_seed,
                        "run_index": run_index,
                        "kernel_init": json.dumps(retrain_model_cfg.get("kernel"), sort_keys=True) if isinstance(retrain_model_cfg.get("kernel"), dict) else None,
                        "best_value": float(study.best_value) if study.best_value is not None else np.nan,
                        "best_trial": int(study.best_trial.number) if study.best_trial is not None else -1,
                        "n_trials_completed": int(len(study.trials)),
                        "elapsed_time": best_result["elapsed_time"],
                        "training_seconds": best_result["training_seconds"],
                        "inference_total_seconds": best_result["inference_total_seconds"],
                        "inference_per_sample_seconds": best_result["inference_per_sample_seconds"],
                        "r2": best_result["test_metrics"].get("r2"),
                        "rmse": best_result["test_metrics"].get("rmse"),
                        "mae": best_result["test_metrics"].get("mae"),
                        "mape": best_result["test_metrics"].get("mape"),
                        "train_r2": best_result["train_metrics"].get("r2"),
                        "val_r2": best_result["val_metrics"].get("r2"),
                        "output_dir": best_result["output_dir"],
                        "best_params": json.dumps(study.best_params, sort_keys=True) if study.best_trial is not None else None,
                    }
                )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.sort_values(["model", "train_split", "run_index"]).reset_index(drop=True)
    return metrics_df, histories, analysis_root


def _summarize_and_save_outputs(metrics_df: pd.DataFrame, histories: List[Dict[str, Any]], analysis_root: Path, model_sequence: Sequence[str] = MODEL_SEQUENCE) -> None:
    analysis_root.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(analysis_root / "all_runs.csv", index=False)

    summary_rows: List[Dict[str, Any]] = []

    for model_name in model_sequence:
        subset = metrics_df[metrics_df["model"] == model_name].copy()
        if subset.empty:
            continue
        model_root = analysis_root / model_name
        model_root.mkdir(parents=True, exist_ok=True)

        subset.to_csv(model_root / "runs.csv", index=False)

        metrics_vs_split = (
            subset.groupby("train_split")[["r2", "rmse", "mae", "mape"]]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("train_split")
        )
        metrics_vs_split.columns = [
            "_".join([str(part) for part in col if part]).strip("_")
            for col in metrics_vs_split.columns.to_flat_index()
        ]
        metrics_vs_split.to_csv(model_root / "metrics_vs_split.csv", index=False)

        summary_rows.append(
            {
                "model": model_name,
                "best_r2_mean": float(subset.groupby("train_split")["r2"].mean().mean()),
                "best_rmse_mean": float(subset.groupby("train_split")["rmse"].mean().mean()),
                "best_mae_mean": float(subset.groupby("train_split")["mae"].mean().mean()),
                "best_mape_mean": float(subset.groupby("train_split")["mape"].mean().mean()),
            }
        )

        if "best_value" in subset.columns:
            summary_rows[-1]["best_value_mean"] = float(subset["best_value"].mean())

    pd.DataFrame(summary_rows).to_csv(analysis_root / "summary.csv", index=False)

    convergence_viz_config = VisualizationConfig.from_mapping(
        {}, output_dir=str(analysis_root),
    )
    run_visualization(
        "convergence",
        config=convergence_viz_config,
        convergence_records=(metrics_df, histories, model_sequence),
    )

    logger.info("=" * 70)
    logger.info("CONVERGENCE ANALYSIS RESULTS")
    logger.info("=" * 70)
    for model_name in model_sequence:
        model_data = metrics_df[metrics_df["model"] == model_name]
        if model_data.empty:
            continue
        logger.info(f"\n{model_name.upper()} Results:")
        logger.info("-" * 70)
        agg = model_data.groupby("train_split")[ ["r2", "rmse", "mae", "mape"]].agg(["mean", "std"]).reset_index()
        flat_columns = ["train_split"]
        for metric in ["r2", "rmse", "mae", "mape"]:
            flat_columns.extend([f"{metric}_mean", f"{metric}_std"])
        agg.columns = flat_columns
        for _, row in agg.iterrows():
            r2_std = float(row["r2_std"]) if not pd.isna(row["r2_std"]) else 0.0
            rmse_std = float(row["rmse_std"]) if not pd.isna(row["rmse_std"]) else 0.0
            logger.info(
                f"  Split={float(row['train_split']):.2f}: R²={float(row['r2_mean']):.6f} ± {r2_std:.6f}, "
                f"RMSE={float(row['rmse_mean']):.6f} ± {rmse_std:.6f}"
            )
    logger.info("=" * 70)


def run_analysis(config_path: Path) -> None:
    analysis_cfg = load_yaml(config_path)
    config_dir = PROJECT_ROOT / "config"
    mode = str(analysis_cfg.get("mode", "analysis")).lower()
    analysis_run_id = str(analysis_cfg.get("run_id", "")).strip() or make_run_id("conv")
    logger.info(f"Convergence run_id: {analysis_run_id}")
    if mode not in {"analysis", "optimization"}:
        raise ValueError("mode must be either 'analysis' or 'optimization'.")

    # Load main config to determine model filtering
    main_cfg_path = config_dir / "main_config.yaml"
    temp_main_cfg = load_yaml(main_cfg_path)
    models_to_train = _get_models_to_train(temp_main_cfg, MODEL_SEQUENCE)
    logger.info(f"Training models: {', '.join(models_to_train)}")

    contexts = _load_contexts(config_dir, mode, models_to_train)
    main_cfg = contexts[next(iter(models_to_train))]["main_cfg"]
    dataset_cfg = contexts[next(iter(models_to_train))]["dataset_cfg"]
    dataset_name = contexts[next(iter(models_to_train))]["dataset_name"]
    model_cfg_map = {name: contexts[name]["model_cfg"] for name in models_to_train}
    training_cfg_map = {name: contexts[name]["training_cfg"] for name in models_to_train}
    opt_cfg_map = {name: contexts[name].get("opt_cfg", {}) for name in models_to_train}

    setup_torch_dtype(str(main_cfg.get("precision", "float32")))

    if mode == "analysis":
        metrics_df, histories, analysis_root = _run_analysis_mode(
            analysis_cfg=analysis_cfg,
            dataset_cfg=dataset_cfg,
            dataset_name=dataset_name,
            model_cfg_map=model_cfg_map,
            training_cfg_map=training_cfg_map,
            analysis_run_id=analysis_run_id,
        )
    else:
        metrics_df, histories, analysis_root = _run_optimization_mode(
            analysis_cfg=analysis_cfg,
            main_cfg=main_cfg,
            dataset_cfg=dataset_cfg,
            dataset_name=dataset_name,
            model_cfg_map=model_cfg_map,
            training_cfg_map=training_cfg_map,
            opt_cfg_map=opt_cfg_map,
            analysis_run_id=analysis_run_id,
        )

    _summarize_and_save_outputs(metrics_df, histories, analysis_root, models_to_train)
    print(f"Done. Outputs written under {analysis_root}.")