"""
Shared helpers for hyperparameter optimization.

This module keeps optimize.py thin by hosting the reusable Optuna helpers,
configuration translation, result persistence and plotting.
"""

from __future__ import annotations

import csv
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import yaml
from optuna.pruners import BasePruner, HyperbandPruner, MedianPruner, NopPruner
from optuna.samplers import BaseSampler, CmaEsSampler, RandomSampler, TPESampler

from src.pipeline import train_model
from src.utils import OutputPathResolver, setup_logger

logger = setup_logger(__name__)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


_HP_TO_MODEL_CFG: Dict[tuple[str, str], str] = {
    ("mlp", "dropout"): "dropout",
    ("mlp", "block1_layers"): "architecture",
    ("mlp", "block1_hidden_size"): "architecture",
    ("mlp", "block1_activation"): "architecture",
    ("mlp", "block2_layers"): "architecture",
    ("mlp", "block2_hidden_size"): "architecture",
    ("mlp", "block2_activation"): "architecture",
    ("mlp", "block3_layers"): "architecture",
    ("mlp", "block3_hidden_size"): "architecture",
    ("mlp", "block3_activation"): "architecture",
    ("gp", "outputscale_init"): "kernel",
    ("gp", "lengthscale_init"): "kernel",
    ("gp", "noise_var"): "kernel",
    ("mlpgp", "hidden_layers"): "feature_extractor",
    ("mlpgp", "hidden_layers_list"): "feature_extractor",
    ("mlpgp", "hidden_size"): "feature_extractor",
    ("mlpgp", "activation"): "feature_extractor",
    ("mlpgp", "dropout"): "feature_extractor",
    ("mlpgp", "outputscale_init"): "kernel",
    ("mlpgp", "lengthscale_init"): "kernel",
    ("mlpgp", "noise_var"): "kernel",
}


def suggest_from_config(trial: optuna.Trial, search_space: Dict[str, Any], target: str) -> Dict[str, Any]:
    """Convert a YAML search space section into Optuna suggestions."""
    if target not in search_space:
        return {}

    params: Dict[str, Any] = {}
    for param_name, param_spec in search_space[target].items():
        param_type = param_spec.get("type")
        if param_type == "int":
            params[param_name] = trial.suggest_int(
                param_name,
                int(param_spec.get("low", 1)),
                int(param_spec.get("high", 10)),
                step=int(param_spec.get("step", 1)),
            )
        elif param_type == "float":
            params[param_name] = trial.suggest_float(
                param_name,
                float(param_spec.get("low", 0.0)),
                float(param_spec.get("high", 1.0)),
                log=bool(param_spec.get("log", False)),
            )
        elif param_type == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, param_spec.get("choices", []))
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    return params


def overlay(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge patch into base without mutating either dictionary."""
    result = dict(base)
    for key, value in patch.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = overlay(result[key], value)
        else:
            result[key] = value
    return result


def _make_logging_callback(
    model_name: str,
    n_trials: int,
    metric_name: str,
    log_every: int = 5,
) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    """Create a compact Optuna callback that prints periodic progress."""
    # removed: per-trial logger spam, replaced by coarse-grained callback logging.
    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if log_every <= 0 or trial.number % log_every != 0:
            return
        best = study.best_value if study.best_trial else float("nan")
        current = trial.value if trial.value is not None else float("nan")
        pct = (trial.number + 1) / max(n_trials, 1) * 100
        print(
            f"[{model_name.upper()}] {trial.number + 1:>3}/{n_trials} ({pct:4.0f}%) | "
            f"current={current:.5f}  best={best:.5f}",
            flush=True,
        )

    return callback


def compute_val_nll(model: Any, X_val_t: torch.Tensor, y_val_t: torch.Tensor) -> float:
    """Compute negative log-likelihood on a validation set."""
    model.eval()
    model_device = next(model.parameters()).device
    X_val_t = X_val_t.to(model_device)
    y_val_t = y_val_t.to(model_device)
    with torch.no_grad(), torch.inference_mode():
        f_pred = model(X_val_t)
        likelihood = model.likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        nll = -mll(f_pred, y_val_t)
    return float(nll.item())


def build_sampler(study_cfg: Dict[str, Any]) -> BaseSampler:
    """Create an Optuna sampler from study configuration."""
    sampler_name = str(study_cfg.get("sampler", "tpe")).lower()
    seed = int(study_cfg.get("seed", 42))

    if sampler_name == "tpe":
        return TPESampler(seed=seed, multivariate=study_cfg.get("multivariate", True), n_startup_trials=study_cfg.get("n_startup_trials", 10))
    if sampler_name == "gp":
        from optuna.samplers import GPSampler
        return GPSampler(seed=seed)
    if sampler_name == "random":
        return RandomSampler(seed=seed)
    if sampler_name == "cmaes":
        return CmaEsSampler(seed=seed, restart_strategy=study_cfg.get("cmaes_restart", "ipop"))
    if sampler_name == "qmc":
        from optuna.samplers import QMCSampler
        return QMCSampler(seed=seed, qmc_type="sobol")
    if sampler_name == "grid":
        from optuna.samplers import GridSampler
        grid = study_cfg.get("grid_search_space")
        if grid is None:
            raise ValueError("GridSampler richiede 'grid_search_space' nel config")
        return GridSampler(grid)
    raise ValueError(f"Unknown sampler: {sampler_name}")


def build_pruner(study_cfg: Dict[str, Any]) -> BasePruner:
    """Create an Optuna pruner from study configuration."""
    pruner_name = study_cfg.get("pruner", "median")
    if pruner_name is None or pruner_name == "null":
        return NopPruner()
    if pruner_name == "median":
        return MedianPruner(
            n_startup_trials=study_cfg.get("n_startup_trials", 5),
            n_warmup_steps=study_cfg.get("n_warmup_steps", 0),
            interval_steps=study_cfg.get("pruner_interval", 1),
        )
    if pruner_name == "hyperband":
        return HyperbandPruner(
            min_resource=study_cfg.get("min_resource", 10),
            max_resource=study_cfg.get("max_resource", "auto"),
            reduction_factor=study_cfg.get("reduction_factor", 3),
        )
    raise ValueError(f"Unknown pruner: {pruner_name}")


def _apply_hp(model_name: str, scope: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Translate sampled hyperparameters into config patches via a registry."""
    patch: Dict[str, Any] = {}

    if model_name == "mlp" and scope == "model":
        architecture = []
        block_indices = sorted(
            {
                int(match.group(1))
                for key in params
                if (match := re.fullmatch(r"block(\d+)_(layers|hidden_size|activation)", key))
            }
        )
        for block_index in block_indices:
            layers_key = f"block{block_index}_layers"
            size_key = f"block{block_index}_hidden_size"
            activation_key = f"block{block_index}_activation"
            n_layers = int(params.get(layers_key, 0))
            hidden_size = params.get(size_key)
            activation = params.get(activation_key)
            if n_layers > 0 and hidden_size is not None and activation is not None:
                architecture.append({"layers": [int(hidden_size)] * n_layers, "activation": str(activation)})
        if architecture:
            patch["architecture"] = architecture
        if "dropout" in params:
            patch["dropout"] = float(params["dropout"])
        return patch

    if model_name == "mlpgp" and scope == "model":
        feature_extractor: Dict[str, Any] = {}
        if {"hidden_layers", "hidden_size", "activation"} & params.keys():
            hidden_layers = int(params.get("hidden_layers", 1))
            hidden_size = int(params.get("hidden_size", 64))
            activation = str(params.get("activation", "tanh"))
            feature_extractor["architecture"] = [
                {"layers": [hidden_size], "activation": activation} for _ in range(hidden_layers)
            ]
        if "dropout" in params:
            feature_extractor["dropout"] = float(params["dropout"])
        if feature_extractor:
            patch["feature_extractor"] = feature_extractor

    for hp_name, hp_value in params.items():
        mapped_scope = _HP_TO_MODEL_CFG.get((model_name, hp_name))
        if mapped_scope is None:
            continue
        if mapped_scope == "kernel":
            kernel_cfg = patch.setdefault("kernel", {})
            if hp_name == "outputscale_init":
                kernel_cfg["tau2"] = float(hp_value)
            elif hp_name == "lengthscale_init":
                kernel_cfg["theta"] = float(hp_value)
            elif hp_name == "noise_var":
                kernel_cfg["nugget"] = float(hp_value)
        elif mapped_scope == "feature_extractor" and hp_name == "dropout":
            feature_cfg = patch.setdefault("feature_extractor", {})
            feature_cfg["dropout"] = float(hp_value)

    if model_name == "mlpgp" and scope == "model":
        feature_cfg = patch.get("feature_extractor", {})
        if "architecture" not in feature_cfg:
            hidden_layers = int(params.get("hidden_layers", 1))
            hidden_size = int(params.get("hidden_size", 64))
            activation = str(params.get("activation", "tanh"))
            feature_cfg["architecture"] = [
                {"layers": [hidden_size], "activation": activation} for _ in range(hidden_layers)
            ]
            patch["feature_extractor"] = feature_cfg

    return patch


def _extract_metric(results: Dict[str, Any], metric_name: str, data_bundle: Any, model: Any) -> float:
    """Extract the target metric from validation metrics or validation NLL."""
    if metric_name == "val_nll":
        X_val_t = torch.from_numpy(data_bundle.X_val)
        y_val_t = torch.from_numpy(data_bundle.y_val)
        if y_val_t.ndim == 2 and y_val_t.shape[1] == 1:
            y_val_t = y_val_t.squeeze(-1)
        return compute_val_nll(model, X_val_t, y_val_t)

    metric_key = metric_name.replace("val_", "")
    value = results["val_metrics"].get(metric_key)
    if value is None:
        raise ValueError(f"Metric '{metric_key}' not in val_metrics")
    if metric_key == "r2":
        return 1.0 - float(value)
    return float(value)


def make_objective(
    model_name: str,
    data_bundle: Any,
    base_model_cfg: Dict[str, Any],
    base_training_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    opt_cfg: Dict[str, Any],
    device: torch.device,
) -> Callable[[optuna.Trial], float]:
    """Create an Optuna objective bound to the shared training pipeline."""

    def objective(trial: optuna.Trial) -> float:
        model_params = suggest_from_config(trial, opt_cfg["search_space"], "model")
        training_params = suggest_from_config(trial, opt_cfg["search_space"], "training")

        model_cfg = overlay(base_model_cfg, _apply_hp(model_name, "model", model_params))
        training_cfg = overlay(base_training_cfg, _apply_hp(model_name, "training", training_params))
        training_cfg = overlay(training_cfg, opt_cfg.get("training_overrides", {}))

        if model_name == "gp":
            optimizer_name = str(training_cfg.get("optimizer", "lbfgs")).lower()
            if optimizer_name == "lbfgs" and "learning_rate" in training_params:
                logger.warning(
                    "GP optimizer is lbfgs: suggested learning_rate is used as LBFGS step-size upper bound "
                    "during line search, not as a fixed SGD-like learning rate."
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            try:
                results = train_model(
                    model_name=model_name,
                    data_bundle=data_bundle,
                    dataset_cfg=dataset_cfg,
                    model_cfg=model_cfg,
                    training_cfg=training_cfg,
                    output_resolver=OutputPathResolver(
                        dataset_name="tmp",
                        run_type="single_run",
                        model_name=model_name,
                        output_base_dir=run_dir,
                    ),
                    device=device,
                    trial=trial,
                )

                metric_name = str(opt_cfg["study"]["metric"])
                metric_value = _extract_metric(results, metric_name, data_bundle, results["model"])
                trial.set_user_attr("objective_metric", metric_name)
                trial.set_user_attr("objective_value", float(metric_value))
                trial.set_user_attr(
                    "val_metrics",
                    {key: float(value) for key, value in results.get("val_metrics", {}).items()},
                )
                trial.set_user_attr(
                    "test_metrics",
                    {key: float(value) for key, value in results.get("test_metrics", {}).items()},
                )

                for metric_key, metric_val in results.get("val_metrics", {}).items():
                    trial.set_user_attr(f"val_{metric_key}", float(metric_val))
                for metric_key, metric_val in results.get("test_metrics", {}).items():
                    trial.set_user_attr(f"test_{metric_key}", float(metric_val))

                history = results.get("history", {})
                if isinstance(history.get("train_losses"), list):
                    trial.set_user_attr("train_losses", [float(x) for x in history["train_losses"]])
                if isinstance(history.get("val_losses"), list):
                    trial.set_user_attr("val_losses", [float(x) for x in history["val_losses"]])

                return float(metric_value)
            except optuna.TrialPruned:
                raise
            except Exception as exc:
                logger.warning(f"Trial {trial.number} failed: {exc}")
                return float("inf")

    return objective


def _extract_best_params(best_params: Dict[str, Any], search_space: Dict[str, Any], target: str) -> Dict[str, Any]:
    """Extract best parameters for a specific target."""
    if target not in search_space:
        return {}
    return {param_name: best_params[param_name] for param_name in search_space[target] if param_name in best_params}


def save_optimization_results(
    study: optuna.Study,
    output_resolver: OutputPathResolver,
    opt_cfg: Dict[str, Any],
    base_model_cfg: Dict[str, Any],
    base_training_cfg: Dict[str, Any],
    model_name: str,
) -> None:
    """Persist trials, best params and the optimized configs."""
    run_dir = output_resolver.root
    run_dir.mkdir(parents=True, exist_ok=True)

    trials_data = []
    for trial in study.trials:
        trials_data.append(
            {
                "trial_number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                **trial.params,
                **trial.user_attrs,
            }
        )

    trials_csv = output_resolver.all_trials_csv
    if trials_data:
        fieldnames = sorted({key for row in trials_data for key in row.keys()})
        with open(trials_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trials_data)
        logger.debug(f"Trials saved to {trials_csv}")

    if study.best_trial is not None:
        best_params_file = output_resolver.best_params
        with open(best_params_file, "w", encoding="utf-8") as f:
            json.dump(study.best_params, f, indent=2)
        logger.debug(f"Best params saved to {best_params_file}")

        model_params = _extract_best_params(study.best_params, opt_cfg["search_space"], "model")
        training_params = _extract_best_params(study.best_params, opt_cfg["search_space"], "training")
        best_model_cfg = overlay(base_model_cfg, _apply_hp(model_name, "model", model_params))
        best_training_cfg = overlay(overlay(base_training_cfg, _apply_hp(model_name, "training", training_params)), opt_cfg.get("training_overrides", {}))

        config_dir = output_resolver.config_dir
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "model").mkdir(exist_ok=True)
        (config_dir / "training").mkdir(exist_ok=True)

        with open(config_dir / "optimize_config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(opt_cfg, f, default_flow_style=False, allow_unicode=True)
        with open(config_dir / "model" / f"{model_name}_config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(best_model_cfg, f, default_flow_style=False, allow_unicode=True)
        with open(config_dir / "training" / f"training_{model_name}_config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(best_training_cfg, f, default_flow_style=False, allow_unicode=True)

        logger.debug(f"Best configs saved to {config_dir}")


# Plotting helpers are kept here so optimize.py can stay thin.

def _generate_summary_dashboard(study: optuna.Study, plots_dir: Path) -> None:
    """Generate a 2x3 dashboard with optimization and multi-metric diagnostics."""
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if not completed_trials:
        logger.warning("Skipping summary dashboard: no completed trials.")
        return

    trial_numbers = np.asarray([t.number for t in completed_trials], dtype=int)
    objective_values = np.asarray([float(t.value) for t in completed_trials], dtype=float)
    best_so_far = np.minimum.accumulate(objective_values)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax_convergence, ax_metrics, ax_hist = axes[0]
    ax_improvement, ax_top10, ax_losses = axes[1]

    ax_convergence.plot(trial_numbers, best_so_far, color="tab:blue", linewidth=2)
    ax_convergence.scatter(trial_numbers, objective_values, color="tab:blue", s=20, alpha=0.45)
    ax_convergence.set_title("Convergence (best so far)")
    ax_convergence.set_xlabel("Trial")
    ax_convergence.set_ylabel("Objective")
    ax_convergence.grid(alpha=0.3)

    metric_specs = [("val_rmse", "tab:red", "RMSE"), ("val_mae", "tab:orange", "MAE"), ("val_mape", "tab:green", "MAPE")]
    metric_r2 = np.asarray([float(t.user_attrs.get("val_r2", np.nan)) for t in completed_trials], dtype=float)

    has_primary = False
    for key, color, label in metric_specs:
        vals = np.asarray([float(t.user_attrs.get(key, np.nan)) for t in completed_trials], dtype=float)
        if np.isfinite(vals).any():
            has_primary = True
            ax_metrics.plot(trial_numbers, vals, marker="o", linewidth=1.5, color=color, label=label)

    ax_metrics_r2 = ax_metrics.twinx()
    has_r2 = np.isfinite(metric_r2).any()
    if has_r2:
        ax_metrics_r2.plot(trial_numbers, metric_r2, marker="x", linewidth=1.4, color="tab:purple", label="R2")
        ax_metrics_r2.set_ylabel("R2")

    ax_metrics.set_title("Val metrics over trials")
    ax_metrics.set_xlabel("Trial")
    ax_metrics.set_ylabel("Error metrics")
    ax_metrics.grid(alpha=0.3)
    if has_primary or has_r2:
        handles_1, labels_1 = ax_metrics.get_legend_handles_labels()
        handles_2, labels_2 = ax_metrics_r2.get_legend_handles_labels()
        ax_metrics.legend(handles_1 + handles_2, labels_1 + labels_2, loc="best", fontsize=8)
    else:
        ax_metrics.text(0.5, 0.5, "No val_* user attrs", ha="center", va="center", transform=ax_metrics.transAxes)

    ax_hist.hist(objective_values, bins=min(20, len(objective_values)), color="tab:cyan", edgecolor="black")
    ax_hist.axvline(float(study.best_value), color="tab:red", linestyle="--", linewidth=2)
    ax_hist.set_title("Distribution of values")
    ax_hist.set_xlabel("Objective")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(alpha=0.3)

    improvement = np.concatenate([[np.nan], np.diff(best_so_far)])
    ax_improvement.plot(trial_numbers, improvement, color="tab:green", marker="o", linewidth=1.5)
    ax_improvement.axhline(0.0, color="black", linewidth=1, linestyle=":")
    ax_improvement.set_title("Incremental improvement")
    ax_improvement.set_xlabel("Trial")
    ax_improvement.set_ylabel("Delta best")
    ax_improvement.grid(alpha=0.3)

    top_trials = sorted(completed_trials, key=lambda t: float(t.value))[:10]
    table_rows = [
        [str(t.number), f"{float(t.value):.5f}", f"{float(t.user_attrs.get('val_rmse', np.nan)):.4f}", f"{float(t.user_attrs.get('val_r2', np.nan)):.4f}"]
        for t in top_trials
    ]
    ax_top10.axis("off")
    ax_top10.set_title("Top-10 trials")
    table = ax_top10.table(cellText=table_rows, colLabels=["Trial", "Objective", "Val RMSE", "Val R2"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)

    best_trial = study.best_trial if study.best_trial is not None else None
    train_losses = best_trial.user_attrs.get("train_losses") if best_trial is not None else None
    val_losses = best_trial.user_attrs.get("val_losses") if best_trial is not None else None
    if isinstance(train_losses, list) and len(train_losses) > 0:
        epochs = np.arange(1, len(train_losses) + 1)
        ax_losses.plot(epochs, train_losses, color="tab:blue", linewidth=1.8, label="Train")
        if isinstance(val_losses, list) and len(val_losses) > 0:
            ax_losses.plot(np.arange(1, len(val_losses) + 1), val_losses, color="tab:orange", linewidth=1.8, label="Val")
        ax_losses.set_title("Train vs Val loss (best trial)")
        ax_losses.set_xlabel("Epoch")
        ax_losses.set_ylabel("Loss")
        ax_losses.grid(alpha=0.3)
        ax_losses.legend(loc="best", fontsize=8)
    else:
        ax_losses.text(0.5, 0.5, "No train/val loss history available", ha="center", va="center", transform=ax_losses.transAxes)
        ax_losses.set_title("Train vs Val loss (best trial)")
        ax_losses.set_xticks([])
        ax_losses.set_yticks([])

    fig.suptitle("Optimization Summary Dashboard", fontsize=14)
    fig.tight_layout()
    dashboard_path = plots_dir / "summary_dashboard.png"
    fig.savefig(dashboard_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.debug(f"Summary dashboard saved to {dashboard_path}")


def generate_optimization_plots(study: optuna.Study, plots_dir: Path) -> None:
    """Generate the optimization dashboard."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    _generate_summary_dashboard(study, plots_dir)


__all__ = [
    "suggest_from_config",
    "overlay",
    "_make_logging_callback",
    "compute_val_nll",
    "build_sampler",
    "build_pruner",
    "make_objective",
    "save_optimization_results",
    "generate_optimization_plots",
]
