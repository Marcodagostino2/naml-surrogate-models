"""Convergence analysis implementation moved into package namespace.

This is a copy of the previous top-level `convergence_analysis.py` implementation
so package-level imports can be used and the top-level script can be removed
after verification.
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import DataBundle, load_and_preprocess_data
from src.optimization_support import build_pruner, build_sampler, generate_optimization_plots, make_objective, save_optimization_results
from src.pipeline import train_model
from src.utils import OutputPathResolver, RunType, load_run_configs, load_yaml, set_global_seed, setup_logger, setup_torch_dtype

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_SEQUENCE: Sequence[str] = ("mlp", "gp", "mlpgp")
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


def _sample_from_bounds(bounds: Dict[str, Any], rng: random.Random) -> Any:
    low = float(bounds["low"])
    high = float(bounds["high"])
    if low > high:
        raise ValueError(f"Invalid bounds: low={low} > high={high}")
    return rng.uniform(low, high)


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


def _build_validation_ratios(train_ratio: float, validation_train_ratio: float) -> tuple[float, float]:
    val_ratio = float(train_ratio) * float(validation_train_ratio)
    test_ratio = 1.0 - float(train_ratio) - val_ratio
    if val_ratio <= 0.0 or test_ratio <= 0.0:
        raise ValueError(
            f"Invalid ratios for train={train_ratio:.6f} and validation_train_ratio={validation_train_ratio:.6f}."
        )
    return val_ratio, test_ratio


def _split_tag(train_ratio: float) -> str:
    return f"split_{train_ratio:.4f}".replace(".", "p")


def _run_tag(seed: int, repeat_index: int, kernel_index: int | None = None) -> str:
    if kernel_index is None:
        return f"seed_{seed}_run_{repeat_index}"
    return f"seed_{seed}_run_{repeat_index}_k_{kernel_index}"


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


def _prepare_data_bundle(dataset_cfg: Dict[str, Any], train_ratio: float, validation_train_ratio: float, seed: int) -> DataBundle:
    val_ratio, test_ratio = _build_validation_ratios(train_ratio, validation_train_ratio)
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
    )


def _kernel_init_for_model(analysis_cfg: Dict[str, Any], model_name: str, seed: int) -> Dict[str, Any] | None:
    bounds_cfg = analysis_cfg.get("kernel_init_bounds", {})
    if not isinstance(bounds_cfg, dict):
        return None

    model_bounds = bounds_cfg.get(model_name, bounds_cfg.get("default"))
    if not isinstance(model_bounds, dict):
        return None

    rng = random.Random(seed)
    sampled: Dict[str, Any] = {}
    for key, value in model_bounds.items():
        if key == "type":
            sampled[key] = value
        elif isinstance(value, dict) and "low" in value and "high" in value:
            sampled[key] = _sample_from_bounds(value, rng)
        else:
            sampled[key] = value
    return sampled


def _set_kernel_seeded_values(model_cfg: Dict[str, Any], kernel_init: Dict[str, Any] | None) -> Dict[str, Any]:
    if kernel_init is None:
        return deepcopy(model_cfg)
    updated = deepcopy(model_cfg)
    kernel_cfg = updated.get("kernel", {})
    if not isinstance(kernel_cfg, dict):
        kernel_cfg = {}
    kernel_cfg.update(kernel_init)
    updated["kernel"] = kernel_cfg
    return updated


def _analysis_output_resolver(dataset_name: str, model_name: str) -> OutputPathResolver:
    """Return output resolver for convergence analysis intermediate training outputs.
    
    Uses a flattened path structure without split/run tags to avoid cluttering the results directory.
    """
    synthetic_dataset_name = f"{dataset_name}/{ANALYSIS_ROOT_NAME}/training"
    return OutputPathResolver(
        dataset_name=synthetic_dataset_name,
        run_type=RunType.CONVERGENCE_ANALYSIS,
        model_name=model_name,
        output_base_dir=PROJECT_ROOT / "outputs",
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
) -> Dict[str, Any]:
    output_resolver = _analysis_output_resolver(dataset_name, model_name)
    output_resolver.root.mkdir(parents=True, exist_ok=True)
    results = train_model(
        model_name=model_name,
        data_bundle=data_bundle,
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        output_resolver=output_resolver,
    )
    history = results.get("history", {}) or {}
    history_label, history_values = _extract_history_values(model_name, history)

    return {
        "model": model_name,
        "history_label": history_label,
        "history_values": history_values,
        "elapsed_time": float(history.get("elapsed_time", np.nan)),
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
    main_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    dataset_name: str,
    model_cfg_map: Dict[str, Dict[str, Any]],
    training_cfg_map: Dict[str, Dict[str, Any]],
) -> tuple[pd.DataFrame, List[Dict[str, Any]], Path]:
    split_grid = _build_split_grid(analysis_cfg.get("split_percentages", [0.7]))
    validation_train_ratio = float(analysis_cfg.get("validation_train_ratio", 0.21428571428571427))
    global_seed = int(analysis_cfg.get("global_seed", 31))

    metrics_rows: List[Dict[str, Any]] = []
    histories: List[Dict[str, Any]] = []
    analysis_root = PROJECT_ROOT / "outputs" / dataset_name / ANALYSIS_ROOT_NAME
    analysis_root.mkdir(parents=True, exist_ok=True)

    for split_index, train_ratio in enumerate(split_grid):
        split_tag = _split_tag(train_ratio)
        for model_name in model_cfg_map.keys():
            n_runs = _run_count_for_model(analysis_cfg, "runs_per_model", model_name, default=1)
            for run_index in range(n_runs):
                seed = global_seed + split_index * 1000 + run_index * 10 + MODEL_SEQUENCE.index(model_name)
                set_global_seed(seed)
                data_bundle = _prepare_data_bundle(dataset_cfg, train_ratio, validation_train_ratio, seed)
                kernel_init = _kernel_init_for_model(analysis_cfg, model_name, seed)
                model_cfg = _set_kernel_seeded_values(model_cfg_map[model_name], kernel_init)
                training_cfg = deepcopy(training_cfg_map[model_name])

                result = _train_best_model(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    data_bundle=data_bundle,
                    dataset_cfg=dataset_cfg,
                    model_cfg=model_cfg,
                    training_cfg=training_cfg,
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
                        "kernel_init": json.dumps(kernel_init, sort_keys=True) if kernel_init is not None else None,
                        "elapsed_time": result["elapsed_time"],
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
    analysis_root = PROJECT_ROOT / "outputs" / dataset_name / ANALYSIS_ROOT_NAME
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
            )

            n_trials = int(opt_cfg["study"].get("n_trials", 1))
            timeout = opt_cfg["study"].get("timeout")
            study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

            save_optimization_results(
                study=study,
                output_resolver=output_resolver,
                opt_cfg=opt_cfg,
                base_model_cfg=model_cfg,
                base_training_cfg=training_cfg,
                model_name=model_name,
            )
            generate_optimization_plots(study, output_resolver.plots_dir)

            best_model_cfg = load_yaml(output_resolver.config_dir / "model" / f"{model_name}_config.yaml")
            best_training_cfg = load_yaml(output_resolver.config_dir / "training" / f"training_{model_name}_config.yaml")

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
                retrain_training_cfg = deepcopy(best_training_cfg)

                best_result = _train_best_model(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    data_bundle=retrain_data_bundle,
                    dataset_cfg=dataset_cfg,
                    model_cfg=retrain_model_cfg,
                    training_cfg=retrain_training_cfg,
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


def _aggregate_results(metrics_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    numeric_cols = [col for col in ["r2", "rmse", "mae", "mape", "elapsed_time", "train_r2", "val_r2"] if col in metrics_df.columns]
    agg = metrics_df.groupby(["model", "train_split"], as_index=False)[numeric_cols].agg(["mean", "std"])
    agg.columns = ["_".join([str(part) for part in col if part]) for col in agg.columns.to_flat_index()]
    return agg.reset_index(drop=True)


def _plot_metric_curve(df: pd.DataFrame, metric: str, output_path: Path, title: str, model_name: str | None = None) -> None:
    """Plot metric curve for a single model or combined models.
    
    If model_name is specified, filter to that model. Otherwise plot all models in df.
    """
    plt.figure(figsize=(9, 6))
    if model_name is not None:
        df = df[df["model"] == model_name]
    agg = df.groupby("train_split")[metric].agg(["mean", "std"]).reset_index().sort_values("train_split")
    x = agg["train_split"].to_numpy(dtype=float)
    y = agg["mean"].to_numpy(dtype=float)
    yerr = agg["std"].fillna(0.0).to_numpy(dtype=float)
    plt.plot(x, y, marker="o", linewidth=2)
    plt.fill_between(x, y - yerr, y + yerr, alpha=0.2)

    plt.xlabel("Train split")
    plt.ylabel(metric.upper())
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_metric_curve_all_models(df: pd.DataFrame, metric: str, output_path: Path, title: str, model_sequence: Sequence[str] = MODEL_SEQUENCE) -> None:
    """Plot metric curve showing all models together with uncertainty bands."""
    plt.figure(figsize=(11, 6))
    for model_name in model_sequence:
        subset = df[df["model"] == model_name]
        if subset.empty:
            continue
        agg = subset.groupby("train_split")[metric].agg(["mean", "std"]).reset_index().sort_values("train_split")
        x = agg["train_split"].to_numpy(dtype=float)
        y = agg["mean"].to_numpy(dtype=float)
        yerr = agg["std"].fillna(0.0).to_numpy(dtype=float)
        plt.plot(x, y, marker="o", linewidth=2.5, label=model_name.upper())
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.15)

    plt.xlabel("Train split")
    plt.ylabel(metric.upper())
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_training_histories(history_records: List[Dict[str, Any]], output_path: Path, title: str, model_name: str | None = None) -> None:
    """Plot training histories for a single model or all models.
    
    If model_name is specified, filter to that model. Otherwise plot all models.
    """
    plt.figure(figsize=(10, 6))
    if model_name is not None:
        history_records = [record for record in history_records if record["model"] == model_name]
    for record in sorted(history_records, key=lambda item: (item["train_split"], item.get("run_index", 0))):
        values = np.asarray(record["history_values"], dtype=float)
        if values.size == 0:
            continue
        x = np.arange(1, values.size + 1)
        label = f"{record['model']}: split={record['train_split']:.2f}, run={record.get('run_index', 0)}"
        plt.plot(x, values, linewidth=1.0, alpha=0.75, label=label)

    ylabel = history_records[0]["history_label"] if history_records else "history"
    plt.xlabel("epoch/step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if history_records:
        plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _summarize_and_save_outputs(metrics_df: pd.DataFrame, histories: List[Dict[str, Any]], analysis_root: Path, model_sequence: Sequence[str] = MODEL_SEQUENCE) -> None:
    analysis_root.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(analysis_root / "all_runs.csv", index=False)

    summary_rows: List[Dict[str, Any]] = []
    
    # Save per-model results
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

        model_histories = [record for record in histories if record["model"] == model_name]
        _plot_metric_curve(subset, "r2", model_root / "r2_vs_split.png", f"{model_name.upper()}: R2 vs train split", model_name=None)
        _plot_metric_curve(subset, "rmse", model_root / "rmse_vs_split.png", f"{model_name.upper()}: RMSE vs train split", model_name=None)
        _plot_metric_curve(subset, "mae", model_root / "mae_vs_split.png", f"{model_name.upper()}: MAE vs train split", model_name=None)
        _plot_metric_curve(subset, "mape", model_root / "mape_vs_split.png", f"{model_name.upper()}: MAPE vs train split", model_name=None)

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

    # Create complete folder with combined plots showing all models
    if len(model_sequence) > 1:
        complete_root = analysis_root / "complete"
        complete_root.mkdir(parents=True, exist_ok=True)
        _plot_metric_curve_all_models(metrics_df, "r2", complete_root / "r2_vs_split.png", "All Models: R² vs train split", model_sequence)
        _plot_metric_curve_all_models(metrics_df, "rmse", complete_root / "rmse_vs_split.png", "All Models: RMSE vs train split", model_sequence)
        _plot_metric_curve_all_models(metrics_df, "mae", complete_root / "mae_vs_split.png", "All Models: MAE vs train split", model_sequence)
        _plot_metric_curve_all_models(metrics_df, "mape", complete_root / "mape_vs_split.png", "All Models: MAPE vs train split", model_sequence)
        _plot_training_histories(histories, complete_root / "training_history.png", "All Models: training histories across splits", model_name=None)

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
            main_cfg=main_cfg,
            dataset_cfg=dataset_cfg,
            dataset_name=dataset_name,
            model_cfg_map=model_cfg_map,
            training_cfg_map=training_cfg_map,
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
        )

    _summarize_and_save_outputs(metrics_df, histories, analysis_root, models_to_train)
    print(f"Done. Outputs written under {analysis_root}.")